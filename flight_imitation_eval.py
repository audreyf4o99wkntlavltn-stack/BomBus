import os
import numpy as np
import tensorflow as tf
import mediapy

from acme import wrappers
from flybody.fly_envs import flight_imitation
from flybody.agents.network_factory import make_network_factory_dmpo

# 换成你这次“飞行训练”的 checkpoint 目录
CKPT_DIR = "/root/ray-ckpts/7cd4c71a-c9cf-11f0-a1be-0242ac11000d/checkpoints/dmpo_learner"
OUT_VIDEO = "flight_imitation_eval.mp4"

# 远程服务器上用 EGL 渲染（无显示器）
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")


def make_env():
    env = flight_imitation()  # 和训练时 --env flight_imitation 对应
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


def run_rollout(num_steps=600, fps=30):
    """跑一段飞行并存成视频。"""
    env = make_env()
    policy_net = load_policy(env)

    timestep = env.reset()
    frames = []

    for t in range(num_steps):
        obs = timestep.observation

        # 把每个观测叶子变成 [1, ...] 的 tf.Tensor，交给 policy_net
        obs_tf = tf.nest.map_structure(
            lambda x: tf.convert_to_tensor(x[None, ...], dtype=tf.float32),
            obs,
        )

        action_dist = policy_net(obs_tf)
        action = action_dist.sample()[0].numpy()

        timestep = env.step(action)

        # camera_id 可以试 0 / 1 / 2，看哪个视角好
        frame = env.physics.render(camera_id=1, height=480, width=640)
        frames.append(frame)

        if timestep.last():
            print(f"Episode ended at step {t}, resetting environment.")
            timestep = env.reset()

    print(f"保存视频到 {OUT_VIDEO} ...")
    mediapy.write_video(OUT_VIDEO, frames, fps=fps)
    print("完成。")


if __name__ == "__main__":
    run_rollout(num_steps=600, fps=30)
