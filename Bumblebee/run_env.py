import os
# 关键修复：必须在所有导入前设置渲染后端
os.environ['MUJOCO_GL'] = 'glfw'

import cv2
import numpy as np
import traceback
from dm_control import composer, mjcf
from dm_control.utils import rewards
from flybody.fruitfly import fruitfly
from flybody.tasks.base import Flying

# ==========================================
# 0. 全局配置
# ==========================================
MESH_DIR = 'meshes'
XML_PATH = 'fly_env.xml'
SCALE_FACTOR = 0.25

class Flower(composer.Entity):
    def _build(self, color=(0, 0, 1), name='flower'):
        self._mjcf_root = mjcf.RootElement(model=name)
        self._mjcf_root.compiler.meshdir = MESH_DIR
        self._mjcf_root.asset.add('mesh', name='bottle_mesh', file='bottle_link.STL', scale='25 25 25')
        self._mjcf_root.asset.add('mesh', name='plate_mesh', file='plate_Link.STL', scale='25 25 25')
        self._mjcf_root.asset.add('mesh', name='lego_mesh', file='lego_Link.STL', scale='25 25 25')
        self._mjcf_root.asset.add('material', name='mat_bottle', rgba=[0.8, 0.8, 1, 0.3])
        self._mjcf_root.asset.add('material', name='mat_plate', rgba=[1, 1, 1, 1])
        self._mjcf_root.asset.add('material', name='mat_lego', rgba=[*color, 1])
        self._base_body = self._mjcf_root.worldbody.add('body', name='base_link')
        self._base_body.add('geom', type='mesh', mesh='bottle_mesh', material='mat_bottle')
        plate_body = self._base_body.add('body', name='plate_link', pos=[0, 0, 7.9 * SCALE_FACTOR])
        plate_body.add('geom', type='mesh', mesh='plate_mesh', material='mat_plate')
        lego_body = plate_body.add('body', name='lego_link', pos=[0, 0, 0])
        lego_body.add('joint', name='lego_hinge', type='hinge', axis=[0.63, -0.77, 0])
        lego_body.add('geom', type='mesh', mesh='lego_mesh', material='mat_lego')
        self.nectar_site = lego_body.add('site', name='nectar', pos=[0, 0, 0.69 * SCALE_FACTOR],  
                                         size=[0.4 * SCALE_FACTOR, 0.35 * SCALE_FACTOR], type='cylinder', rgba=[1, 1, 0.6, 0.4])
    @property
    def mjcf_model(self): return self._mjcf_root

class TwoBoxArena(composer.Arena):
    def _build(self, name='two_box_arena'):
        super()._build(name=name)
        # 加载外部 XML 环境
        self._mjcf_root = mjcf.from_path(XML_PATH)
        # 增加环境光照强度，确保相机能看清
        self._mjcf_root.worldbody.add('light', pos=[0, 0, 10], dir=[0, 0, -1], diffuse=[0.9, 0.9, 0.9])
    @property
    def ground_geoms(self): return tuple(g for g in [self._mjcf_root.find('geom', 'floor')] if g is not None)

class FlowerTask(Flying):
    FLY_HEIGHT = 1.2  # 出生高度

    def __init__(self, walker, arena, time_limit=float('inf'), **kwargs):
        super().__init__(walker=walker, arena=arena, time_limit=time_limit, joint_filter=0.0002, **kwargs)
        self._flower = Flower(color=(0, 0, 1), name='blue_flower')
        self._flower_frame = self._arena.attach(self._flower)
        self._flower_joint = self._flower_frame.add('freejoint', name='flower_float')
        # 开启右眼相机观测
        self._walker.observables.right_eye.enabled = True
        self._last_vision_info = {"bearing": 0.0, "found": False}

    def initialize_episode(self, physics, random_state):
        super().initialize_episode(physics, random_state)
        # 果蝇固定在原点上方
        self._walker.set_pose(physics, position=[0, 0, self.FLY_HEIGHT * SCALE_FACTOR], quaternion=[1, 0, 0, 0])
        # 花朵放在果蝇正前方 2.5 距离处
        flower_x = 2.5 * SCALE_FACTOR 
        flower_binding = physics.bind(self._flower_joint)
        if flower_binding is not None:
            flower_binding.qpos[:3] = [flower_x, 0, 0]
            flower_binding.qpos[3:] = [1, 0, 0, 0]
            flower_binding.qvel[:] = 0
        physics.forward()

    def _process_vision(self, physics):
        """实时渲染与视觉处理函数"""
        try:
            # 1. 上帝视角渲染 (camera_id=0 为默认全局相机)
            global_img = physics.render(width=400, height=300, camera_id=0)
            if global_img is not None:
                cv2.imshow('GOD_VIEW_DEBUG', cv2.cvtColor(global_img, cv2.COLOR_RGB2BGR))

            # 2. 果蝇视角渲染
            target_cam = next((c for c in self._walker.mjcf_model.find_all('camera') if 'right_eye' in c.name), None)
            if not target_cam: return {"found": False, "bearing": 0.0}

            img_rgb = physics.render(width=128, height=128, camera_id=target_cam.full_identifier)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            
            # HSV 过滤 (蓝色范围)
            hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
            mask = cv2.inRange(hsv, np.array([100, 80, 20]), np.array([140, 255, 255]))
            
            moments = cv2.moments(mask)
            found = False
            bearing = 0.0
            if moments["m00"] > 2:
                found = True
                cx = moments["m10"] / moments["m00"]
                bearing = (cx - 64) / 64
                cv2.circle(img_bgr, (int(cx), int(moments["m01"]/moments["m00"])), 5, (0, 0, 255), -1)

            # 显示果蝇视角
            show_pov = cv2.resize(img_bgr, (256, 256), interpolation=cv2.INTER_LINEAR)
            text = "FOUND" if found else "NOT FOUND"
            cv2.putText(show_pov, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if found else (0, 0, 255), 2)
            cv2.imshow('FLY_EYE_POV', show_pov)
            
            cv2.waitKey(1) # 刷新 OpenCV 绘图
            return {"found": found, "bearing": bearing}
        except Exception as e:
            print(f"视觉处理异常: {e}")
            return {"found": False, "bearing": 0.0}

    def before_step(self, physics, action, random_state):
        # 频率控制：每 10 步采样一次视觉，兼顾流畅度与实时性
        if self._step_counter % 10 == 0:
            self._last_vision_info = self._process_vision(physics)
        super().before_step(physics, action, random_state)

    def get_reward_factors(self, physics): return (0.0,)

# ==========================================
# 4. 环境主程序
# ==========================================
if __name__ == '__main__':
    try:
        env = composer.Environment(FlowerTask(fruitfly.FruitFly, TwoBoxArena()), 
                                    random_state=np.random.RandomState(42))
        print("仿真初始化成功。正在渲染窗口...")
        timestep = env.reset()
        
        while True:
            # 维持仿真运行，带极小偏航模拟真实状态
            action = np.zeros(env.action_spec().shape)
            action[10] = 0.01  # 微量旋转
            timestep = env.step(action)
            
            # 每隔 100 步在控制台打印状态
            if env.task._step_counter % 100 == 0:
                v = env.task._last_vision_info
                print(f"Step: {env.task._step_counter} | Found: {v['found']} | Bearing: {v['bearing']:.2f}")

            # 额外的 GUI 刷新保护
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception:
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()