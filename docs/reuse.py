# Reuse of the trained flight controller
# Imports
import numpy as np
from dm_control import viewer
import tensorflow as tf
from acme import specs, wrappers
import sonnet as snt

import matplotlib.pyplot as plt
import mediapy
from tqdm import tqdm

from flybody.fly_envs import (
    flight_imitation,
    vision_guided_flight,
)
from flybody.download_data import figshare_download
from flybody.agents.network_factory import make_network_factory_dmpo
from flybody.agents.utils_tf import TestPolicyWrapper
from flybody.agents.network_factory_vis import make_vis_network_factory_two_level_controller


# Helpful functions
def wrap_env(env):
    """Wrap task environment with Acme wrappers."""
    return wrappers.CanonicalSpecWrapper(
        wrappers.SinglePrecisionWrapper(env),
        clip=True)

def time_lapse(frames, num_steps=7):
    """Plot a simple time-lapse from a sequences of frames."""
    plt.figure(figsize=(14, 3))
    for i in range(num_steps):
        frame = frames[i * len(frames) // num_steps]
        plt.subplot(1, num_steps, i+1, xticks=[], yticks=[], frameon=False)
        plt.imshow(frame)
    plt.tight_layout()

render_kwargs = {'width': 640, 'height': 480}
# Prevent tensorflow from stealing all the GPU memory.
physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

    
## Download policy checkpoints and the WPG base pattern
wpg_pattern_path = 'flybody-data/datasets_flight-imitation/wing_pattern_fmech.npy'
low_level_ckpt_path = 'flybody-data/flight-controller-reuse-checkpoints/checkpoints/low-level-controller/ckpt-11'
high_level_ckpt_path = 'flybody-data/flight-controller-reuse-checkpoints/checkpoints/high-level-controllers/trench-task/ckpt-48'
# Create environment for the vision-guided flight task

# For this notebook, we don't need an `environment_factory` function but we will
# use it just as an example that can be directly plugged into a training script later.
def environment_factory(training: bool = True):
    """Create environment replicas."""
    del training  # Unused.
    env = vision_guided_flight(
        wpg_pattern_path=wpg_pattern_path,
        bumps_or_trench='trench',
        joint_filter=0.0002,
    )
    return wrap_env(env)

env = environment_factory()
# Only in this notebook, move the "hero" camera slightly farther away.
camera = env.task.root_entity.mjcf_model.find('camera', 'walker/hero')
camera.pos *= 1.5
## Visualize the initial state of the task
_ = env.reset()


 
plt.figure(figsize=(10, 8))
# Loop over all cameras.
for i in range(9):
    plt.subplot(3, 3, i+1, xticks=[], yticks=[], frameon=False)
   
    pixels = env.physics.render(camera_id=i, **render_kwargs)
    plt.imshow(pixels)
plt.tight_layout()
# Initialize the high-level and the low-level controller networks and the visual module network
### Create networks and load the weights of the pre-trained low-level controller

# Create the same network architecture as in the flight_imitation task used to
# pre-train the low-level flight controller.
ll_network_factory = make_network_factory_dmpo()

# Also (temporarily) create the flight imitation environment to get its specs.
# Important: the flight_imitation environment configuration must be the same as the
# one used to train the low-level controller in the first place.

future_steps = 5  # Number of future steering command timesteps provided as observable.
steering_command_dim = (future_steps + 1) * (3 + 4)  # 3: xyz, 4: quaternion.

ll_env = flight_imitation(
    future_steps=future_steps,
    joint_filter=0.0002)
ll_env = wrap_env(ll_env)
ll_environment_spec = specs.make_environment_spec(ll_env)
del ll_env  # Not needed anymore.

# Create networks for the vision flight task from their network factory.
network_factory = make_vis_network_factory_two_level_controller(
    ll_network_ckpt_path=low_level_ckpt_path,
    ll_network_factory=ll_network_factory,
    ll_environment_spec=ll_environment_spec,
    hl_network_layer_sizes=(256, 256, 128),
    steering_command_dim=steering_command_dim,
    task_input_dim=2,
    vis_output_dim=8,
    critic_layer_sizes=(512, 512, 256),
)
networks = network_factory(env.action_spec())

networks.keys()
# Test the networks

# For testing, stack the visual observation network and the two-level policy.
net_stack = snt.Sequential([
    networks['observation'],
    networks['policy'],
])
net_stack = TestPolicyWrapper(net_stack)


timestep = env.reset()



frames = []
for _ in tqdm(range(200)):
    actions = net_stack(timestep.observation)
    timestep = env.step(np.zeros(12))
    frames.append(env.physics.render(camera_id=6, **render_kwargs))
# Show flight video before training the high-level controller.
mediapy.show_video(frames)
time_lapse(frames)
# Load the trained weights of the high-level and the visual observation networks


# This will load the weight to our `net_stack` from before.
checkpoint = tf.train.Checkpoint(
    target_policy=networks['policy'],
    target_observation=networks['observation'],
)
# For inference or reuse, we only need partial.
status = checkpoint.restore(high_level_ckpt_path).expect_partial()
status.assert_existing_objects_matched()

timestep = env.reset()
def net_stack_dm_control(timestep):
    """A dm_control compatible policy from the stacked networks."""
    action = net_stack(timestep.observation)
    return action

viewer.launch(env,policy=net_stack_dm_control)
frames = []
for _ in tqdm(range(500)):

    action = net_stack(timestep.observation)

    timestep = env.step(action)
    frames.append(env.physics.render(camera_id=6, **render_kwargs))
## Visualize the learned flight navigation
# Show flight video after training the high-level controller.
mediapy.show_video(frames)
time_lapse(frames)
