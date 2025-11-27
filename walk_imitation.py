import os
import numpy as np
import tensorflow as tf
import mediapy

from acme import wrappers
from flybody.fly_envs import walk_on_ball
from flybody.agents.network_factory import make_network_factory_dmpo

CKPT_DIR = "/root/ray-ckpts/9e846300-c5d0-11f0-bccc-0242ac110008/checkpoints/dmpo_learner"
OUT_VIDEO = "walk_on_ball_eval.mp4"

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")


def make_env():
    env = walk_on_ball()
    env = wrappers.SinglePrecisionWrapper(env)
    env = wrappers.CanonicalSpecWrapper(env, clip=True)
    return env


def load_policy(env):
    network_factory = make_network_factory_dmpo()
    networks = network_factory(env.action_spec())
    policy_net = networks["policy"]

    ckpt = tf.train.Checkpoint(policy=policy_net)
    latest = tf.train.latest_checkpoint(CKPT_DIR)
    if latest is None:
        raise RuntimeError(f"在 {CKPT_DIR} 下找不到 checkpoint")
    print("Restoring from:", latest)
    ckpt.restore(latest).expect_partial()
    return policy_net


def run_rollout(num_steps=300, fps=30):
    env = make_env()
    policy_net = load_policy(env)

    timestep = env.reset()
    frames = []

    for t in range(num_steps):
        obs = timestep.observation

        # 把每个观测叶子变成 [1, dim] 的 tf.Tensor，交给 policy_net，
        # 让它内部自己 batch_concat。
        obs_tf = tf.nest.map_structure(
            lambda x: tf.convert_to_tensor(x[None, ...], dtype=tf.float32),
            obs,
        )

        action_dist = policy_net(obs_tf)
        action = action_dist.sample()[0].numpy()

        timestep = env.step(action)

        frame = env.physics.render(camera_id=1, height=480, width=640)
        frames.append(frame)

        if timestep.last():
            print("Episode ended, resetting environment.")
            timestep = env.reset()

    print(f"保存视频到 {OUT_VIDEO} ...")
    mediapy.write_video(OUT_VIDEO, frames, fps=fps)
    print("完成。")


if __name__ == "__main__":
    run_rollout(num_steps=300, fps=30)
