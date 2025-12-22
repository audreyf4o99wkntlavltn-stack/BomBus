import os
import sys
import time
import random

# === 1. 渲染后端配置 ===
os.environ['MUJOCO_GL'] = 'glfw'        

import cv2
import numpy as np
import traceback
from dm_control import composer, mjcf
from flybody.fruitfly import fruitfly
from flybody.tasks.base import Flying

# ==========================================
# 1. 环境构建 (单花环境)
# ==========================================
XML_ENV_PATH = 'fly_env.xml'
XML_FLOWER_PATH = 'flower_1.xml'
# 路径兼容性检查
if not os.path.exists(XML_ENV_PATH): XML_ENV_PATH = 'Bumblebee/fly_env.xml'
if not os.path.exists(XML_FLOWER_PATH): XML_FLOWER_PATH = 'Bumblebee/flower_1.xml'

def nothing(x): pass

class SingleFlowerArena(composer.Arena):
    def _build(self, name='single_flower_arena'):
        super()._build(name=name)
        
        if os.path.exists(XML_ENV_PATH):
            self._mjcf_root = mjcf.from_path(XML_ENV_PATH)
        else:
            raise FileNotFoundError(f"找不到环境文件: {XML_ENV_PATH}")
        
        POS_1 = [4.0, 0.0, 0.0]
        if os.path.exists(XML_FLOWER_PATH):
            flower_1 = mjcf.from_path(XML_FLOWER_PATH)
            site1 = self._mjcf_root.worldbody.add('site', name='flower_site_1', pos=POS_1)
            site1.attach(flower_1)
            print(f">>> 单花环境: 初始加载于 {POS_1}")
        else:
            self._mjcf_root.worldbody.add('geom', name='flower_1', type='sphere', pos=POS_1, size=[0.5], rgba=[1,0,0,1])

        self._mjcf_root.worldbody.add('light', pos=[0, 0, 10], dir=[0, 0, -1], diffuse=[0.8, 0.8, 0.8])
        self._mjcf_root.worldbody.add('camera', name='god_cam', mode='fixed', 
                                      pos=[0.0, 0, 18], xyaxes=[1, 0, 0, 0, 1, 0]) # 俯视上帝视角

    @property
    def ground_geoms(self): 
        floor = self._mjcf_root.find('geom', 'floor')
        return tuple(g for g in [floor] if g is not None)

# ==========================================
# 2. 任务类 (最终修正版：方向正确 + PD控制 + 激进追踪 + 安全随机)
# ==========================================
class AutoPilotTask(Flying):
    def __init__(self, walker, arena, time_limit=float('inf'), **kwargs):
        super().__init__(
            walker=walker, arena=arena, time_limit=time_limit, joint_filter=0.0002, **kwargs
        )
        self._walker.observables.right_eye.enabled = True
        self._walker.observables.left_eye.enabled = True
        self._step_counter = 0
        
        self._cam_left_id = None 
        self._cam_right_id = None
        self._god_cam_id = None
        self._flower_1_body_id = None
        
        self.target_pos = np.array([0.0, 0.0, 0.1]) 
        self.current_pos = np.array([0.0, 0.0, 0.1]) 
        self.target_yaw = 0.0
        self.current_yaw = 0.0
        
        self.SMOOTH_FACTOR = 0.05 
        
        # PD 控制器状态
        self.last_heading_error = 0.0
        
        self.vision_data = { "targets": [], "closest": None }
        
        cv2.namedWindow('Fly_Binocular_View')
        cv2.namedWindow('God_View')
        # HSV 阈值 (蓝色)
        cv2.createTrackbar('H Min', 'Fly_Binocular_View', 100, 179, nothing)
        cv2.createTrackbar('H Max', 'Fly_Binocular_View', 140, 179, nothing)
        cv2.createTrackbar('S Min', 'Fly_Binocular_View', 100, 255, nothing)
        cv2.createTrackbar('S Max', 'Fly_Binocular_View', 255, 255, nothing)
        cv2.createTrackbar('V Min', 'Fly_Binocular_View', 50, 255, nothing)
        cv2.createTrackbar('V Max', 'Fly_Binocular_View', 255, 255, nothing)
        cv2.namedWindow('Depth_View')
    def initialize_episode(self, physics, random_state):
        super().initialize_episode(physics, random_state)
        for i in range(physics.model.ncam):
            name = physics.model.id2name(i, 'camera')
            if name and 'eye' in name:
                physics.model.cam_fovy[i] = 130 
        
        if self._flower_1_body_id is None:
            found_bodies = []
            for i in range(physics.model.nbody):
                name = physics.model.id2name(i, 'body')
                if name and 'flower_base' in name: found_bodies.append(i)
            if len(found_bodies) >= 1: self._flower_1_body_id = found_bodies[0]

        # === 🎲 安全随机生成 ===
        # 墙壁位置: X=±7.375, Y=±4.875
        # 预留花朵半径缓冲: 2.5m
        SAFE_X = 4.8
        SAFE_Y = 2.3

        if self._flower_1_body_id is not None:
            while True:
                ax = random.uniform(-SAFE_X, SAFE_X)
                ay = random.uniform(-SAFE_Y, SAFE_Y)
                # 避开出生点 (半径 > 2.0m)
                if (ax**2 + ay**2) > 4.0: 
                    break
            
            physics.model.body_pos[self._flower_1_body_id] = [ax, ay, 0.0]
            print(f">>> [安全生成] ({ax:.1f}, {ay:.1f})")

        self.last_heading_error = 0.0
        self._force_teleport(physics)

    def before_step(self, physics, action, random_state):
        self.current_pos += (self.target_pos - self.current_pos) * self.SMOOTH_FACTOR
        self.current_yaw += (self.target_yaw - self.current_yaw) * self.SMOOTH_FACTOR
        self._force_teleport(physics)
        if self._step_counter % 3 == 0: self._analyze_vision(physics)
        self._step_counter += 1
        pass 

    def _force_teleport(self, physics):
        yaw = self.current_yaw
        quat = [np.cos(yaw/2), 0, 0, np.sin(yaw/2)]
        physics.data.qpos[:3] = self.current_pos
        physics.data.qpos[3:7] = quat
        physics.data.qvel[:] = 0

    def _process_eye(self, physics, cam_id, label):
            RES_W, RES_H = 320, 240
            img_rgb = physics.render(width=RES_W, height=RES_H, camera_id=cam_id)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            
            # [新增 1] 高斯模糊：平滑噪点，让颜色块更连续
            img_bgr_blurred = cv2.GaussianBlur(img_bgr, (5, 5), 0)

            # 处理深度图
            try: 
                img_depth = physics.render(width=RES_W, height=RES_H, camera_id=cam_id, depth=True)
                depth_norm = np.clip(img_depth, 0, 5.0) / 5.0
                depth_gray = (depth_norm * 255).astype(np.uint8)
            except: 
                depth_gray = np.zeros((RES_H, RES_W), dtype=np.uint8)
                img_depth = np.zeros((RES_H, RES_W))

            # HSV 转换 (使用模糊后的图像)
            img_hsv = cv2.cvtColor(img_bgr_blurred, cv2.COLOR_BGR2HSV)
            
            h_min = cv2.getTrackbarPos('H Min', 'Fly_Binocular_View')
            h_max = cv2.getTrackbarPos('H Max', 'Fly_Binocular_View')
            s_min = cv2.getTrackbarPos('S Min', 'Fly_Binocular_View')
            s_max = cv2.getTrackbarPos('S Max', 'Fly_Binocular_View')
            v_min = cv2.getTrackbarPos('V Min', 'Fly_Binocular_View')
            v_max = cv2.getTrackbarPos('V Max', 'Fly_Binocular_View')

            mask = cv2.inRange(img_hsv, np.array([h_min, s_min, v_min]), np.array([h_max, s_max, v_max]))
            
            # [新增 2] 形态学操作：核心修复步骤！
            # 定义一个 5x5 的核
            kernel = np.ones((5, 5), np.uint8)
            # "闭运算" (Close)：先膨胀后腐蚀，用于填充物体内部的小黑洞，连接断开的区域
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            # "开运算" (Open)：先腐蚀后膨胀，用于消除背景中的细小噪点
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            detections = []
            if contours:
                for c in contours:
                    # [新增 3] 增大面积过滤阈值：忽略太小的碎片
                    # 如果你的积木在近处很大，建议把 30 改成 100 或 200
                    if cv2.contourArea(c) > 100: 
                        M = cv2.moments(c)
                        cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else RES_W // 2
                        cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else RES_H // 2
                        cx, cy = np.clip(cx, 0, RES_W-1), np.clip(cy, 0, RES_H-1)
                        
                        dist = float(img_depth[cy, cx])
                        rel_x = (cx - (RES_W/2)) / (RES_W/2)
                        bias = -0.5 + rel_x * 0.5 if 'LEFT' in label else 0.5 + rel_x * 0.5
                        
                        detections.append({'dist': dist, 'bias': bias})
                        
                        x, y, w, h = cv2.boundingRect(c)
                        cv2.rectangle(img_bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(img_bgr, f"{dist:.2f}m", (x, y - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.rectangle(depth_gray, (x, y), (x+w, y+h), 0, 2)

            cv2.putText(img_bgr, label, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            return img_bgr, detections, depth_gray
    def _analyze_vision(self, physics):
        try:
            if self._cam_left_id is None:
                # 重新搜索 Camera ID 逻辑... (省略冗余代码)
                for i in range(physics.model.ncam):
                    name = physics.model.id2name(i, 'camera')
                    if name and ('left' in name) and ('eye' in name): self._cam_left_id = i
                    if name and ('right' in name) and ('eye' in name): self._cam_right_id = i
                    if name and 'god_cam' in name: self._god_cam_id = i
                if self._cam_left_id is None: self._cam_left_id = 0
                if self._cam_right_id is None: self._cam_right_id = 0
                if self._god_cam_id is None: self._god_cam_id = 0

            img_god = physics.render(width=800, height=600, camera_id=self._god_cam_id)
            img_god_bgr = cv2.cvtColor(img_god, cv2.COLOR_RGB2BGR)
            cv2.imshow('God_View', img_god_bgr)

            img_l, dets_l, dep_l = self._process_eye(physics, self._cam_left_id, "LEFT")
            img_r, dets_r, dep_r = self._process_eye(physics, self._cam_right_id, "RIGHT")
            combined_eyes = np.hstack((img_l, img_r))
            cv2.imshow('Fly_Binocular_View', combined_eyes)
            
            combined_depth = np.hstack((dep_l, dep_r))
            cv2.imshow('Depth_View', combined_depth)
        
            all_dets = dets_l + dets_r
            all_dets.sort(key=lambda x: x['dist'])
            self.vision_data['targets'] = all_dets
            self.vision_data['closest'] = all_dets[0] if len(all_dets) > 0 else None

        except Exception:
            pass

    def get_reward_factors(self, physics):
        return (0.0,)

# ==========================================
# 3. 主程序
# ==========================================
if __name__ == '__main__':
    try:
        arena = SingleFlowerArena()
        task = AutoPilotTask(fruitfly.FruitFly, arena)
        env = composer.Environment(task, random_state=np.random.RandomState(42))
        
        print("="*40)
        print("🤖 状态同步点: 单花+安全随机+激进追踪")
        print("   [R] 重置布局  [T] 传送  [Q] 退出")
        print("="*40)

        timestep = env.reset()
        
        SEARCH_ALTITUDE = 3.5 
        SIP_HOVER_ALTITUDE = 2.40 
        CLIMB_SPEED = 0.02          
        SEARCH_SPIN_SPEED = 0.015
        
        # 控制参数
        KP_TURN = 0.08
        KD_TURN = 0.15
        MAX_SPEED = 0.08  

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            if key == ord('r'): 
                task.target_pos = np.array([0.0, 0.0, 0.1])
                task.target_yaw = 0.0
                print("\n>>> 重置并随机化位置...")
                timestep = env.reset() 
                continue 
            if key == ord('t'):
                task.target_pos = np.array([3.0, 0.0, 3.5])
                task.target_yaw = 0.0

            vdata = task.vision_data
            status_msg = "IDLE"
            closest = vdata['closest']
            heading_bias = 0.0
            
            if closest is None:
                if task.target_pos[2] < SEARCH_ALTITUDE:
                    status_msg = "🛫 CLIMBING..."
                    task.target_pos[2] += CLIMB_SPEED
                else:
                    status_msg = "🔄 SEARCHING..."
                    task.target_yaw += SEARCH_SPIN_SPEED
                    task.target_pos[2] = SEARCH_ALTITUDE
                task.last_heading_error = 0.0
            else:
                dist = closest['dist']
                heading_bias = closest['bias']
                
                # PD 转向
                error_deriv = heading_bias - task.last_heading_error
                # 增强转向力度
                turn_gain = KP_TURN
                if abs(heading_bias) > 0.5: turn_gain *= 2.0 
                turn_adjustment = (turn_gain * heading_bias) + (KD_TURN * error_deriv)
                task.last_heading_error = heading_bias
                
                # 高度逻辑
                target_z = SEARCH_ALTITUDE
                if dist < 1.5: target_z = SIP_HOVER_ALTITUDE
                if task.target_pos[2] > target_z: task.target_pos[2] -= 0.02 
                elif task.target_pos[2] < target_z: task.target_pos[2] += 0.02 

                if dist > 0.15: 
                    status_msg = f"🚀 APPROACH ({dist:.1f}m)"
                    
                    # === 核心修正: Yaw 减去调整量 ===
                    task.target_yaw -= turn_adjustment
                    
                    # 速度混合控制
                    alignment_score = max(0.3, 1.0 - abs(heading_bias))
                    dist_factor = np.clip((dist - 0.15) / 1.5, 0.2, 1.0)
                    current_speed = MAX_SPEED * alignment_score * dist_factor
                    
                    task.target_pos[0] += np.cos(task.target_yaw) * current_speed
                    task.target_pos[1] += np.sin(task.target_yaw) * current_speed
                    
                    if alignment_score < 0.5: status_msg += " [HARD TURN]"
                    else: status_msg += " [FORWARD]"
                else:
                    # 死区冻结
                    if abs(heading_bias) < 0.05:
                        status_msg = "🛑 LOCKED (HOVER)"
                        task.last_heading_error = 0.0 
                    else:
                        status_msg = "⚠️ FINE TUNING"
                        task.target_yaw -= turn_adjustment * 0.5

            task.target_pos[0] = np.clip(task.target_pos[0], -7, 7)
            task.target_pos[1] = np.clip(task.target_pos[1], -4.5, 4.5)
            task.target_pos[2] = np.clip(task.target_pos[2], 0.1, 5.0)

            timestep = env.step(np.zeros(env.action_spec().shape))
            
            if task._step_counter % 10 == 0:
                print(f"\r[{status_msg}] Alt:{task.target_pos[2]:.2f}m Err:{heading_bias:.2f}         ", end="")

            if timestep.last(): timestep = env.reset()

    except Exception:
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()