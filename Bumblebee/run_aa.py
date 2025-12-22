import os
import cv2
import random
import numpy as np
from dm_control import composer, mjcf
from flybody.fruitfly import fruitfly
from flybody.tasks.base import Flying

# === 1. 全局配置 ===
CONFIG = {
    "env_xml": "fly_env.xml",
    "flower_xml": "flower_1.xml",
    "cam_res": (320, 240),
    "safe_area": {"x": 4.8, "y": 2.3},  # 安全生成范围
    "alt": {"search": 3.5, "hover": 2.4},
    "control": {"kp": 0.08, "kd": 0.15, "max_speed": 0.08},
    "hsv_default": (100, 140, 100, 255, 50, 255) # H_min, H_max, S_min, S_max, V_min, V_max
}

# === 2. 视觉子系统 ===
class VisualSystem:
    def __init__(self, win_name='Fly_Vision'):
        self.win_name = win_name
        cv2.namedWindow(win_name)
        labels = ['H Min', 'H Max', 'S Min', 'S Max', 'V Min', 'V Max']
        for label, val in zip(labels, CONFIG["hsv_default"]):
            cv2.createTrackbar(label, win_name, val, 255, lambda x: None)

    def detect(self, physics, cam_id, is_left_eye):
        if cam_id == -1: return None, None # 相机无效直接返回

        w, h = CONFIG["cam_res"]
        rgb = physics.render(w, h, camera_id=cam_id)
        depth = physics.render(w, h, camera_id=cam_id, depth=True)
        bgr = cv2.convertScaleAbs(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), alpha=1.5, beta=30)
        
        # HSV 阈值获取与过滤
        vals = [cv2.getTrackbarPos(l, self.win_name) for l in ['H Min', 'H Max', 'S Min', 'S Max', 'V Min', 'V Max']]
        # 注意: inRange 参数顺序是 (Min_Array, Max_Array)
        lower = np.array([vals[0], vals[2], vals[4]]) # H_min, S_min, V_min
        upper = np.array([vals[1], vals[3], vals[5]]) # H_max, S_max, V_max
        
        mask = cv2.inRange(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV), lower, upper)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        target = None
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(c) > 30:
                cv2.drawContours(bgr, [c], -1, (0, 255, 0), 2)
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
                    rel_x = (cx - w/2) / (w/2)
                    # 偏差计算
                    bias = (-0.5 + rel_x * 0.5) if is_left_eye else (0.5 + rel_x * 0.5)
                    dist = float(depth[np.clip(cy, 0, h-1), np.clip(cx, 0, w-1)])
                    target = {"dist": dist, "bias": bias}

        return bgr, target

# === 3. 环境定义 ===
class SingleFlowerArena(composer.Arena):
    def _build(self, name='arena'):
        super()._build(name=name)
        self._mjcf_root = mjcf.from_path(CONFIG["env_xml"])
        flower = mjcf.from_path(CONFIG["flower_xml"])
        self._mjcf_root.worldbody.add('site', name='flower_site', pos=[3, 0, 0]).attach(flower)
        self._mjcf_root.worldbody.add('light', pos=[0, 0, 10], dir=[0, 0, -1])
        self._mjcf_root.worldbody.add('camera', name='god_cam', pos=[0, 0, 18], xyaxes=[1, 0, 0, 0, 1, 0])

    @property
    def ground_geoms(self): return (self._mjcf_root.find('geom', 'floor'),)

# === 4. 自动驾驶任务 ===
class AutoPilotTask(Flying):
    def __init__(self, walker, arena, time_limit=float('inf'), **kwargs):
        super().__init__(walker=walker, arena=arena, time_limit=time_limit, joint_filter=0.0002, **kwargs)
        
        self._walker.observables.left_eye.enabled = True
        self._walker.observables.right_eye.enabled = True
        self.vision = VisualSystem()
        
        # 相机 ID 缓存
        self.cam_ids = {'left': -1, 'right': -1}
        
        # 飞行状态
        self.target_pos = np.array([0.0, 0.0, 0.1])
        self.target_yaw = 0.0
        self.last_err = 0.0
        self.closest_target = None

    def get_reward_factors(self, physics): return (0.0,)

    def initialize_episode(self, physics, random_state):
        super().initialize_episode(physics, random_state)
        
        # === 1. 稳健的相机 ID 查找逻辑 ===
        print("\n[System] searching for cameras...")
        all_cams = [physics.model.id2name(i, 'camera') for i in range(physics.model.ncam)]
        
        for side in ['left', 'right']:
            # 只要名字包含 side (如'left') 和 'eye' 即可，不强制顺序
            found = False
            for i, name in enumerate(all_cams):
                if name and (side in name) and ('eye' in name):
                    self.cam_ids[side] = i
                    physics.model.cam_fovy[i] = 130 # 设置广角
                    print(f"  -> Found {side} camera: '{name}' (ID: {i})")
                    found = True
                    break
            if not found:
                print(f"  [WARNING] Could not find {side} camera! Available: {all_cams}")

        # === 2. 随机生成花朵 ===
        fid = next(i for i in range(physics.model.nbody) if 'flower_base' in physics.model.id2name(i, 'body'))
        while True:
            pos = [random.uniform(-CONFIG["safe_area"]["x"], CONFIG["safe_area"]["x"]),
                   random.uniform(-CONFIG["safe_area"]["y"], CONFIG["safe_area"]["y"])]
            if np.linalg.norm(pos) > 2.0: break 
        physics.model.body_pos[fid][:2] = pos

    def before_step(self, physics, action, random_state):
        # 强制同步位姿
        physics.data.qpos[:3] = self.target_pos
        physics.data.qpos[3:7] = [np.cos(self.target_yaw/2), 0, 0, np.sin(self.target_yaw/2)]
        physics.data.qvel[:] = 0

        # 低频更新视觉
        if self._step_counter % 3 == 0:
            self._update_vision(physics)
            self._update_control()
        self._step_counter += 1

    def _update_vision(self, physics):
        imgs, targets = [], []
        # 使用缓存的 ID，避免每帧重复查找
        for side in ['left', 'right']:
            cid = self.cam_ids[side]
            if cid != -1:
                img, tgt = self.vision.detect(physics, cid, side == 'left')
                if img is not None: imgs.append(img)
                if tgt: targets.append(tgt)
        
        if imgs:
            cv2.imshow('Fly_Vision', np.hstack(imgs))
        self.closest_target = min(targets, key=lambda x: x['dist']) if targets else None

    def _update_control(self):
        if not self.closest_target:
            self._mode_search()
        else:
            self._mode_track(self.closest_target)

    def _mode_search(self):
        if self.target_pos[2] < CONFIG["alt"]["search"]: self.target_pos[2] += 0.02
        else: self.target_yaw += 0.015
        self.last_err = 0.0

    def _mode_track(self, target):
        dist, bias = target['dist'], target['bias']
        cfg = CONFIG["control"]
        
        # PD 转向
        p_term = cfg["kp"] * (2.0 if abs(bias) > 0.5 else 1.0) * bias
        d_term = cfg["kd"] * (bias - self.last_err)
        turn = p_term + d_term
        self.last_err = bias

        # 高度与速度
        target_z = CONFIG["alt"]["hover"] if dist < 1.5 else CONFIG["alt"]["search"]
        self.target_pos[2] += 0.02 if self.target_pos[2] < target_z else -0.02

        if dist > 0.15: # 接近中
            self.target_yaw -= turn
            speed = cfg["max_speed"] * max(0.3, 1-abs(bias)) * np.clip((dist-0.15)/1.5, 0.2, 1.0)
            self.target_pos[0] += np.cos(self.target_yaw) * speed
            self.target_pos[1] += np.sin(self.target_yaw) * speed
        elif abs(bias) > 0.05: # 悬停微调
            self.target_yaw -= turn * 0.5

# === 5. 主程序 ===
if __name__ == '__main__':
    os.environ['MUJOCO_GL'] = 'glfw'
    try:
        arena = SingleFlowerArena()
        task = AutoPilotTask(walker=fruitfly.FruitFly, arena=arena, time_limit=float('inf'))
        env = composer.Environment(task, random_state=np.random.RandomState(42))
        
        print("仿真运行中... 按 'Q' 退出")
        env.reset()
        while True:
            env.step(np.zeros(env.action_spec().shape))
            if cv2.waitKey(1) & 0xFF == ord('q'): break
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()