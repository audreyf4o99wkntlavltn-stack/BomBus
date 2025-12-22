import os
import sys
import time
import random
import cv2
import numpy as np
import traceback
from dm_control import composer, mjcf
from flybody.fruitfly import fruitfly
from flybody.tasks.base import Flying

# === 1. 渲染后端配置 ===
os.environ['MUJOCO_GL'] = 'glfw'        

# === 路径配置 ===
XML_ENV_PATH = 'fly_env.xml'
XML_FLOWER_PATH = 'flower_1.xml' 

# 路径检查
if not os.path.exists(XML_ENV_PATH): XML_ENV_PATH = 'Bumblebee/fly_env.xml'
if not os.path.exists(XML_FLOWER_PATH): XML_FLOWER_PATH = 'Bumblebee/flower_1.xml'

def nothing(x): pass

# ==========================================
# 1. 环境构建 (双花环境：自动修改材质颜色)
# ==========================================
class TwoFlowerArena(composer.Arena):
    def _build(self, name='two_flower_arena'):
        super()._build(name=name)
        
        # 1. 加载环境主体
        if os.path.exists(XML_ENV_PATH):
            self._mjcf_root = mjcf.from_path(XML_ENV_PATH)
        else:
            raise FileNotFoundError(f"找不到环境文件: {XML_ENV_PATH}")
        
        # 2. 加载并配置花朵
        if os.path.exists(XML_FLOWER_PATH):
            # --- 加载蓝色花朵 (默认) ---
            flower_blue = mjcf.from_path(XML_FLOWER_PATH)
            
            # --- 加载绿色花朵 (修改材质) ---
            flower_green = mjcf.from_path(XML_FLOWER_PATH)
            # [核心逻辑] 找到名为 mat_lego 的材质，将其颜色改为绿色
            mat = flower_green.find('material', 'mat_lego')
            if mat:
                mat.rgba = [0, 1, 0, 1] # RGBA: 绿
                print(">>> ✅ 已修改 flower_green 的材质为绿色")
            
            # --- 挂载到环境 ---
            # 初始位置，会被随机逻辑覆盖
            POS_BLUE = [2.0, 0.0, 1.0]
            POS_GREEN = [-2.0, 0.0, 1.0]

            # 添加 site 并 attach
            site_blue = self._mjcf_root.worldbody.add('site', name='site_blue', pos=POS_BLUE)
            site_blue.attach(flower_blue)
            
            site_green = self._mjcf_root.worldbody.add('site', name='site_green', pos=POS_GREEN)
            site_green.attach(flower_green)
            
        # 添加光源和上帝视角相机
        self._mjcf_root.worldbody.add('light', pos=[0, 0, 10], dir=[0, 0, -1], diffuse=[0.8, 0.8, 0.8])
        self._mjcf_root.worldbody.add('camera', name='god_cam', mode='fixed', 
                                      pos=[0.0, 0, 18], xyaxes=[1, 0, 0, 0, 1, 0])

    @property
    def ground_geoms(self): 
        floor = self._mjcf_root.find('geom', 'floor')
        return tuple(g for g in [floor] if g is not None)

# ==========================================
# 2. 任务类 (支持双色识别与随机位置)
# ==========================================
class AutoPilotTask(Flying):
    def __init__(self, walker, arena, time_limit=float('inf'), **kwargs):
        super().__init__(
            walker=walker, arena=arena, time_limit=time_limit, joint_filter=0.0002, **kwargs
        )
        # 启用复眼
        self._walker.observables.right_eye.enabled = True
        self._walker.observables.left_eye.enabled = True
        self._step_counter = 0
        
        # 相机与物体 ID 缓存
        self._cam_left_id = None 
        self._cam_right_id = None
        self._god_cam_id = None
        self._blue_id = None
        self._green_id = None
        
        # === 核心修正：初始高度设为 1.0 米，防止穿模 ===
        self.target_pos = np.array([0.0, 0.0, 1.0]) 
        self.current_pos = np.array([0.0, 0.0, 1.0]) 
        
        self.target_yaw = 0.0
        self.current_yaw = 0.0
        self.SMOOTH_FACTOR = 0.05 
        self.last_heading_error = 0.0
        self.vision_data = { "targets": [], "closest": None }
        
        # 初始化显示窗口
        cv2.namedWindow('Fly_Binocular_View')
        cv2.namedWindow('Depth_View')
        cv2.namedWindow('God_View')

    def initialize_episode(self, physics, random_state):
        super().initialize_episode(physics, random_state)
        
        # 1. 调整视野 FOV
        for i in range(physics.model.ncam):
            name = physics.model.id2name(i, 'camera')
            if name and 'eye' in name: physics.model.cam_fovy[i] = 130 

        # 2. 查找物体 ID
        if self._blue_id is None or self._green_id is None:
            for i in range(physics.model.nbody):
                name = physics.model.id2name(i, 'body')
                if name:
                    if 'site_blue' in name and 'flower_base' in name: self._blue_id = i
                    if 'site_green' in name and 'flower_base' in name: self._green_id = i

        # 3. 随机位置逻辑 (范围: X[-6,6], Y[-4,4])
        def get_random_pos():
            return random.uniform(-6.0, 6.0), random.uniform(-4.0, 4.0)

        # 设置蓝花位置
        bx, by = 2.0, 0.0 
        if self._blue_id is not None:
            bx, by = get_random_pos()
            physics.model.body_pos[self._blue_id] = [bx, by, 0.0]

        # 设置绿花位置 (避免重叠)
        gx, gy = -2.0, 0.0
        if self._green_id is not None:
            while True:
                gx, gy = get_random_pos()
                if np.sqrt((gx-bx)**2 + (gy-by)**2) > 2.0: 
                    break
            physics.model.body_pos[self._green_id] = [gx, gy, 0.0]
            
        print(f">>> [随机生成] 蓝:({bx:.1f}, {by:.1f}) 绿:({gx:.1f}, {gy:.1f})")

        self.last_heading_error = 0.0
        
        # === 核心修正：重置时也强制设置到 1.0 米 ===
        self.current_pos = np.array([0.0, 0.0, 1.0])
        self.target_pos = np.array([0.0, 0.0, 1.0])
        self._force_teleport(physics)

    def _force_teleport(self, physics):
        yaw = self.current_yaw
        quat = [np.cos(yaw/2), 0, 0, np.sin(yaw/2)]
        physics.data.qpos[:3] = self.current_pos
        physics.data.qpos[3:7] = quat
        physics.data.qvel[:] = 0

    def before_step(self, physics, action, random_state):
        self.current_pos += (self.target_pos - self.current_pos) * self.SMOOTH_FACTOR
        self.current_yaw += (self.target_yaw - self.current_yaw) * self.SMOOTH_FACTOR
        self._force_teleport(physics)
        if self._step_counter % 3 == 0: self._analyze_vision(physics)
        self._step_counter += 1

    def _process_eye(self, physics, cam_id, label):
        RES_W, RES_H = 320, 240
        img_rgb = physics.render(width=RES_W, height=RES_H, camera_id=cam_id)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        
        # 深度图 (灰度)
        try: 
            img_depth = physics.render(width=RES_W, height=RES_H, camera_id=cam_id, depth=True)
            depth_norm = np.clip(img_depth, 0, 8.0) / 8.0 
            depth_gray = (depth_norm * 255).astype(np.uint8)
        except: 
            depth_gray = np.zeros((RES_H, RES_W), dtype=np.uint8)
            img_depth = np.zeros((RES_H, RES_W))

        # HSV 颜色识别
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        
        # 蓝色掩码
        mask_blue = cv2.inRange(img_hsv, np.array([100, 100, 50]), np.array([140, 255, 255]))
        # 绿色掩码
        mask_green = cv2.inRange(img_hsv, np.array([40, 100, 50]), np.array([90, 255, 255]))
        
        combined_mask = cv2.bitwise_or(mask_blue, mask_green)
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        if contours:
            for c in contours:
                if cv2.contourArea(c) > 30: 
                    M = cv2.moments(c)
                    cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else RES_W // 2
                    cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else RES_H // 2
                    cx, cy = np.clip(cx, 0, RES_W-1), np.clip(cy, 0, RES_H-1)
                    
                    dist = float(img_depth[cy, cx])
                    c_type = "BLUE" if mask_blue[cy, cx] > 0 else "GREEN"
                    
                    rel_x = (cx - (RES_W/2)) / (RES_W/2)
                    bias = -0.5 + rel_x * 0.5 if 'LEFT' in label else 0.5 + rel_x * 0.5
                    
                    detections.append({'dist': dist, 'bias': bias, 'type': c_type})
                    
                    color_rgb = (255, 0, 0) if c_type == "BLUE" else (0, 255, 0)
                    cv2.rectangle(img_bgr, cv2.boundingRect(c), color_rgb, 2)
                    cv2.putText(img_bgr, f"{c_type}", (cx, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_rgb, 2)
                    cv2.rectangle(depth_gray, cv2.boundingRect(c), 255, 1)

        return img_bgr, detections, depth_gray

    def _analyze_vision(self, physics):
        try:
            if self._cam_left_id is None:
                for i in range(physics.model.ncam):
                    name = physics.model.id2name(i, 'camera')
                    if name:
                        if 'left' in name and 'eye' in name: self._cam_left_id = i
                        if 'right' in name and 'eye' in name: self._cam_right_id = i
                        if 'god_cam' in name: self._god_cam_id = i

            img_l, dets_l, dep_l = self._process_eye(physics, self._cam_left_id, "LEFT")
            img_r, dets_r, dep_r = self._process_eye(physics, self._cam_right_id, "RIGHT")
            
            if self._god_cam_id is not None:
                img_god = physics.render(width=400, height=300, camera_id=self._god_cam_id)
                cv2.imshow('God_View', cv2.cvtColor(img_god, cv2.COLOR_RGB2BGR))
            
            cv2.imshow('Fly_Binocular_View', np.hstack((img_l, img_r)))
            cv2.imshow('Depth_View', np.hstack((dep_l, dep_r)))

            all_dets = dets_l + dets_r
            all_dets.sort(key=lambda x: x['dist'])
            self.vision_data['closest'] = all_dets[0] if all_dets else None
            self.vision_data['targets'] = all_dets
        except Exception:
            pass

    def get_reward_factors(self, physics):
        return np.array([0.0])

# ==========================================
# 3. 主程序
# ==========================================
if __name__ == '__main__':
    try:
        arena = TwoFlowerArena()
        task = AutoPilotTask(fruitfly.FruitFly, arena)
        env = composer.Environment(task, random_state=np.random.RandomState(42))
        
        print("="*40)
        print("🤖 双色随机花朵环境 (修正穿模版)")
        print("   初始高度: 1.0m (防止卡地)")
        print("   操作: [Q]退出  [R]重置位置")
        print("="*40)

        timestep = env.reset()
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            if key == ord('r'): 
                timestep = env.reset()
                continue

            vdata = task.vision_data
            closest = vdata['closest']
            
            if closest is None:
                # 巡逻
                task.target_pos[2] = 4.0
                task.target_yaw += 0.02
            else:
                dist, bias, c_type = closest['dist'], closest['bias'], closest['type']
                
                # 追踪控制
                error_deriv = bias - task.last_heading_error
                task.target_yaw -= (0.1 * bias + 0.2 * error_deriv)
                task.last_heading_error = bias
                
                speed = 0.1 * np.clip(dist/2.0, 0.2, 1.0)
                task.target_pos[0] += np.cos(task.target_yaw) * speed
                task.target_pos[1] += np.sin(task.target_yaw) * speed
                task.target_pos[2] = 2.0 if dist < 1.5 else 4.0

            # 限制边界
            task.target_pos[0] = np.clip(task.target_pos[0], -7.0, 7.0)
            task.target_pos[1] = np.clip(task.target_pos[1], -4.5, 4.5)
            # 限制最低高度，防止飞行中撞地
            task.target_pos[2] = np.clip(task.target_pos[2], 0.5, 5.0)

            env.step(np.zeros(env.action_spec().shape))

    except Exception:
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()