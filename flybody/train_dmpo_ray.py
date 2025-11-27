"""Script for distributed reinforcement learning training with Ray.

This script trains fly RL tasks using a distributed version of the
DMPO agent. The training runs in an infinite loop until terminated.

For lightweight testing, run this script with --test argument. It will run
training with a single actor and print training statistics every 10 seconds.

You can select different fly tasks by passing --env, e.g.
  --env walk_on_ball
  --env walk_imitation
"""

# ruff: noqa: F821, E722, E402

# ----------------------------------------------------------------------
# 0. Start Ray cluster first, before heavy imports.
# ----------------------------------------------------------------------
import ray
try:
    # Try connecting to existing Ray cluster.
    ray_context = ray.init(
        address="auto",
        include_dashboard=True,
        dashboard_host="0.0.0.0",
    )
except Exception:
    # Spin up new Ray cluster.
    ray_context = ray.init(include_dashboard=True, dashboard_host="0.0.0.0")

# ----------------------------------------------------------------------
# 1. Standard imports
# ----------------------------------------------------------------------
import argparse
import time
import os

from acme import specs
from acme import wrappers
import sonnet as snt

import flybody
from flybody.agents.remote_as_local_wrapper import RemoteAsLocal
from flybody.agents.counting import PicklableCounter
from flybody.agents.network_factory import policy_loss_module_dmpo
from flybody.agents.losses_mpo import PenalizationCostRealActions

# 改成导入整个 fly_envs 模块，方便用名字选择不同环境
from flybody import fly_envs
from flybody.agents.network_factory import make_network_factory_dmpo

from flybody.agents.ray_distributed_dmpo import (
    DMPOConfig,
    ReplayServer,
    Learner,
    EnvironmentLoop,
)

PYHTONPATH = os.path.dirname(os.path.dirname(flybody.__file__))
LD_LIBRARY_PATH = (
    os.environ["LD_LIBRARY_PATH"] if "LD_LIBRARY_PATH" in os.environ else ""
)
# Defer specifying CUDA_VISIBLE_DEVICES to sub-processes.
# 主进程先屏蔽 GPU，只在 Ray 子进程里单独设置。
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# ----------------------------------------------------------------------
# 2. 命令行参数
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser()

parser.add_argument(
    "--env",
    type=str,
    default="walk_on_ball",
    help=(
        "Name of environment constructor in flybody.fly_envs, e.g. "
        "walk_on_ball, walk_imitation, flight_imitation, vision_*."
        " (must exist as flybody.fly_envs.<env>)"
    ),
)

parser.add_argument(
    "--test",
    "-t",
    help="Run job in test mode with one actor and output to current terminal.",
    action="store_true",
)

parser.add_argument(
    "--cpu-only",
    help="Force learner to run on CPU instead of GPU.",
    action="store_true",
)

parser.add_argument(
    "--num-actors",
    type=int,
    default=None,
    help="Override number of actors (default 32, or 1 in --test mode).",
)
# 在 parser.add_argument(...) 那几行后面加一个：
parser.add_argument(
    "--max-iters",
    type=int,
    default=None,
    help="How many times to call learner.run(). "
         "Each run does num_learner_steps gradient updates. "
         "Default: infinite loop.",
)


args = parser.parse_args()
is_test = args.test
cpu_only = args.cpu_only
manual_num_actors = args.num_actors
env_name = args.env
max_iters = args.max_iters
if is_test:
    print("\nRun job in test mode with one actor.")
    default_num_actors = 1
    log_every = 10           # seconds
    min_replay_size = 40
else:
    default_num_actors = 32
    log_every = 5 * 60       # seconds
    min_replay_size = 10_000

num_actors = manual_num_actors or default_num_actors

print("\nRay context:")
print(ray_context)

ray_resources = ray.available_resources()
print("\nAvailable Ray cluster resources:")
print(ray_resources)

print(f"\nUsing environment: flybody.fly_envs.{env_name}")
if not hasattr(fly_envs, env_name):
    raise ValueError(
        f"flybody.fly_envs has no attribute '{env_name}'. "
        "Please choose a valid environment name."
    )

# ----------------------------------------------------------------------
# 3. Environment & network factories
# ----------------------------------------------------------------------
def environment_factory(training: bool) -> "composer.Environment":
    """Creates replicas of environment for the agent."""
    del training  # Unused.
    env_ctor = getattr(fly_envs, env_name)
    env = env_ctor()
    env = wrappers.SinglePrecisionWrapper(env)
    env = wrappers.CanonicalSpecWrapper(env, clip=True)
    return env


# Create network factory for RL task.
network_factory = make_network_factory_dmpo()

# Dummy environment and network for quick use, deleted later.
dummy_env = environment_factory(training=True)
dummy_net = network_factory(dummy_env.action_spec())
# Get full environment specs.
environment_spec = specs.make_environment_spec(dummy_env)

# This callable will be calculating penalization cost by converting canonical
# actions to real (not wrapped) environment actions inside DMPO agent.
# Note that we need the action_spec of the underlying environment so we unwrap
# with dummy_env._environment.
penalization_cost = PenalizationCostRealActions(
    dummy_env._environment.action_spec()
)

# ----------------------------------------------------------------------
# 4. DMPO config
# ----------------------------------------------------------------------
dmpo_config = DMPOConfig(
    num_actors=num_actors,
    batch_size=256,
    prefetch_size=4,
    num_learner_steps=100,
    min_replay_size=min_replay_size,
    max_replay_size=4_000_000,
    samples_per_insert=15,
    n_step=5,
    num_samples=20,
    policy_loss_module=policy_loss_module_dmpo(
        epsilon=0.1,
        epsilon_mean=0.0025,
        epsilon_stddev=1e-7,
        action_penalization=True,
        epsilon_penalty=0.1,
        penalization_cost=penalization_cost,
    ),
    policy_optimizer=snt.optimizers.Adam(1e-4),
    critic_optimizer=snt.optimizers.Adam(1e-4),
    dual_optimizer=snt.optimizers.Adam(1e-3),
    target_critic_update_period=107,
    target_policy_update_period=101,
    actor_update_period=1000,
    log_every=log_every,
    logger_save_csv_data=False,
    checkpoint_max_to_keep=None,
    checkpoint_directory="~/ray-ckpts/",
    checkpoint_to_load=None,
    print_fn=print,
)

# Print full job config and full environment specs.
print("\n", dmpo_config)
print("\n", dummy_net)
print("\nobservation_spec:\n", dummy_env.observation_spec())
print("\naction_spec:\n", dummy_env.action_spec())
print("\ndiscount_spec:\n", dummy_env.discount_spec())
print("\nreward_spec:\n", dummy_env.reward_spec(), "\n")
del dummy_env
del dummy_net

# ----------------------------------------------------------------------
# 5. Ray runtime envs
# ----------------------------------------------------------------------
learner_num_gpus = 0 if cpu_only else 1
learner_cuda_visible = "-1" if cpu_only else "0"
learner_tf_allow_growth = "false" if cpu_only else "true"

# Environment variables for learner, actor, and replay buffer processes.
runtime_env_learner = {
    "env_vars": {
        "MUJOCO_GL": "egl",
        "CUDA_VISIBLE_DEVICES": learner_cuda_visible,
        "TF_FORCE_GPU_ALLOW_GROWTH": learner_tf_allow_growth,
        "PYTHONPATH": PYHTONPATH,
        "LD_LIBRARY_PATH": LD_LIBRARY_PATH,
    }
}
runtime_env_actor = {
    "env_vars": {
        "MUJOCO_GL": "egl",
        "MUJOCO_EGL_DEVICE_ID": "0",
        "CUDA_VISIBLE_DEVICES": "-1",  # CPU-actors don't use CUDA.
        "PYTHONPATH": PYHTONPATH,
        "LD_LIBRARY_PATH": LD_LIBRARY_PATH,
    }
}

# === Create Replay Server.
runtime_env_replay = {
    "env_vars": {
        "PYTHONPATH": PYHTONPATH,  # Also used for counter.
    }
}
ReplayServerRemote = ray.remote(
    num_gpus=0, runtime_env=runtime_env_replay
)(ReplayServer)
replay_server = ReplayServerRemote.remote(dmpo_config, environment_spec)
addr = ray.get(replay_server.get_server_address.remote())
print(f"Started Replay Server on {addr}")

# === Create Counter.
counter_remote = ray.remote(PicklableCounter)  # This is class (direct call to
# ray.remote decorator).
counter_remote = counter_remote.remote()  # Instantiate.
counter = RemoteAsLocal(counter_remote)

# === Create Learner.
LearnerRemote = ray.remote(
    num_gpus=learner_num_gpus, runtime_env=runtime_env_learner
)(Learner)
learner_remote = LearnerRemote.remote(
    replay_server.get_server_address.remote(),
    counter,
    environment_spec,
    dmpo_config,
    network_factory,
)
learner = RemoteAsLocal(learner_remote)

print("Waiting until learner is ready...")
learner.isready(block=True)

checkpointer_dir, snapshotter_dir = learner.get_checkpoint_dir()
print("Checkpointer directory:", checkpointer_dir)
print("Snapshotter directory:", snapshotter_dir)

# === Create Actors and Evaluator.

EnvironmentLoopRemote = ray.remote(
    num_gpus=0, runtime_env=runtime_env_actor
)(EnvironmentLoop)

n_actors = dmpo_config.num_actors


def create_actors(n_actors: int):
    """Return list of requested number of actor instances."""
    actors = []
    for _ in range(n_actors):
        actor_remote = EnvironmentLoopRemote.remote(
            replay_server_address=replay_server.get_server_address.remote(),
            variable_source=learner,
            counter=counter,
            network_factory=network_factory,
            environment_factory=environment_factory,
            dmpo_config=dmpo_config,
            actor_or_evaluator="actor",
        )
        actor = RemoteAsLocal(actor_remote)
        actors.append(actor)
        time.sleep(0.2)
    return actors


# Get actors.
actors = create_actors(n_actors)

# Get evaluator.
evaluator_remote = EnvironmentLoopRemote.remote(
    replay_server_address=replay_server.get_server_address.remote(),
    variable_source=learner,
    counter=counter,
    network_factory=network_factory,
    environment_factory=environment_factory,
    dmpo_config=dmpo_config,
    actor_or_evaluator="evaluator",
)
evaluator = RemoteAsLocal(evaluator_remote)

print("Waiting until actors are ready...")
# Block until all actors and evaluator are ready and have called `get_variables`
# in learner with variable_client.update_and_wait() from _make_actor. Otherwise
# they will be blocked and won't be inserting data to replay table, which in
# turn will cause learner to be blocked.
for actor in actors:
    actor.isready(block=True)
evaluator.isready(block=True)

print("Actors ready, issuing run command to all")

# === Run all.
if hasattr(counter, "run"):
    counter.run(block=False)
for actor in actors:
    actor.run(block=False)
evaluator.run(block=False)

if max_iters is None:
    # 旧行为：无限训练，直到你 Ctrl+C
    while True:
        learner.run(block=True)
else:
    # 新行为：训练 max_iters 次，然后自动结束
    for i in range(max_iters):
        print(f"=== Learner iteration {i + 1}/{max_iters} ===")
        learner.run(block=True)

    print("Training finished, exiting.")
    # 脚本结束时，Ray 子进程也会跟着退出
