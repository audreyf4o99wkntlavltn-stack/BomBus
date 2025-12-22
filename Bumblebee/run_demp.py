import os
import sys
import numpy as np
import cv2
from dm_control import composer, mjcf
from flybody.fruitfly import fruitfly
from flybody.tasks.base import Flying

# === 导入你的视觉模块 ===
from vision_module import VisionModule

# ==========================================
# 0. 环境配置
# ==========================================
os.environ['MUJOCO_GL'] = 'glfw'
XML_ENV_PATH = 'fly_env.xml'
XML_FLOWER_PATH = 'flower_1.xml'

# ==========================================
# 1. 定义双色花竞技场 (Arena)
# ==========================================
class TwoFlowerArena(composer.Arena):
    def _build(self, name='two_flower_arena'):
        super()._build(name=name)
        
        # 加载主环境
        self._mjcf_root = mjcf.from_path(XML_ENV_PATH)
        
        # 灯光与相机
        self._mjcf_root.worldbody.add('light', pos=[0, 0, 10], dir=[0, 0, -1], diffuse=[0.8, 0.8, 0.8])
        self._mjcf_root.worldbody.add('camera', name='god_cam', mode='fixed', pos=[0, 0, 18], xyaxes=[1, 0, 0, 0, 1, 0])

        # -------------------------------------------------
        # 🔵 添加花朵 1 (保持默认蓝色)
        # -------------------------------------------------
        flower_1 = mjcf.from_path(XML_FLOWER_PATH)
        # 给根 Body 起名，方便 Task 查找并修改位置
        flower_1.worldbody.body[0].name = 'flower_body_blue'
        
        site1 = self._mjcf_root.worldbody.add('site', name='site_1', pos=[3.0, 0, 0])
        site1.attach(flower_1)

        # -------------------------------------------------
        # 🟢 添加花朵 2 (修改为绿色)
        # -------------------------------------------------
        flower_2 = mjcf.from_path(XML_FLOWER_PATH)
        flower_2.worldbody.body[0].name = 'flower_body_green'

        # === 核心修复：在挂载前修改材质 ===
        # 找到 mat_lego 材质并修改其颜色
        lego_mat = flower_2.find('material', 'mat_lego')
        if lego_mat is not None:
            lego_mat.rgba = [0.0, 1.0, 0.0, 1.0] # 改为绿色
        
        site2 = self._mjcf_root.worldbody.add('site', name='site_2', pos=[-3.0, 0, 0])
        site2.attach(flower_2)

    @property
    def ground_geoms(self): 
        return (self._mjcf_root.find('geom', 'floor'),)

# ==========================================
# 2. 任务逻辑 (Task)
# ==========================================
class TestTask(Flying):
    def __init__(self, walker, arena, **kwargs):
        super().__init__(
            walker=walker, 
            arena=arena, 
            time_limit=float('inf'), 
            joint_filter=0.0002, 
            **kwargs
        )
        self._walker.observables.right_eye.enabled = True
        self._walker.observables.left_eye.enabled = True
        
        self._blue_id = None
        self._green_id = None

        # 初始让果蝇在地面
        self.target_pos = np.array([0.0, 0.0, 0.2]) 
        self.target_yaw = 0.0
        self.current_pos = np.array([0.0, 0.0, 0.2])
        self.current_yaw = 0.0

    def initialize_episode(self, physics, random_state):
        super().initialize_episode(physics, random_state)
        
        # --- A. 查找 ID ---
        if self._blue_id is None:
            for i in range(physics.model.nbody):
                name = physics.model.id2name(i, 'body')
                if name:
                    if 'flower_body_blue' in name: self._blue_id = i
                    elif 'flower_body_green' in name: self._green_id = i
        
        # --- B. 随机位置生成 (防重叠算法) ---
        # 定义安全区域
        SAFE_X, SAFE_Y = 4.5, 2.5
        MIN_DIST_BETWEEN_FLOWERS = 2.0  # 两个花之间至少相隔 2米
        MIN_DIST_FROM_ORIGIN = 1.0      # 离果蝇出生点(0,0)至少 1米
        
        def get_valid_pos():
            """尝试生成一个合法的随机坐标"""
            for _ in range(100): # 尝试 100 次
                rx = random_state.uniform(-SAFE_X, SAFE_X)
                ry = random_state.uniform(-SAFE_Y, SAFE_Y)
                # 检查是否离原点太近
                if (rx**2 + ry**2) > MIN_DIST_FROM_ORIGIN**2:
                    return np.array([rx, ry])
            return np.array([3.0, 0.0]) # 失败时的保底位置

        # 生成蓝色花位置
        pos_blue = get_valid_pos()
        
        # 生成绿色花位置 (确保和蓝色花不重叠)
        pos_green =np.array([-1.0, 0.0])
        for _ in range(100):
            p = get_valid_pos()
            # 检查距离
            dist = np.linalg.norm(p - pos_blue)
            if dist > MIN_DIST_BETWEEN_FLOWERS:
                pos_green = p
                break
        
        # 应用位置
        if self._blue_id is not None:
            physics.model.body_pos[self._blue_id][:2] = pos_blue
            print(f">>> 🔵 蓝花位置: ({pos_blue[0]:.2f}, {pos_blue[1]:.2f})")
            
        if self._green_id is not None:
            physics.model.body_pos[self._green_id][:2] = pos_green
            print(f">>> 🟢 绿花位置: ({pos_green[0]:.2f}, {pos_green[1]:.2f})")

        # --- C. 重置果蝇 ---
        self.current_pos = np.array([0.0, 0.0, 0.2])
        self.current_yaw = 0.0
        self.target_pos = np.copy(self.current_pos)
        self.target_yaw = self.current_yaw
        self._teleport(physics)

    def before_step(self, physics, action, random_state):
        self.current_pos += (self.target_pos - self.current_pos) * 0.1
        self.current_yaw += (self.target_yaw - self.current_yaw) * 0.1
        self._teleport(physics)

    def _teleport(self, physics):
        yaw = self.current_yaw
        quat = [np.cos(yaw/2), 0, 0, np.sin(yaw/2)]
        physics.data.qpos[:3] = self.current_pos
        physics.data.qpos[3:7] = quat
        physics.data.qvel[:] = 0

    def get_reward_factors(self, physics): return (0.0,)

def render_god_view(physics, cam_id):
    if cam_id != -1:
        img = physics.render(width=640, height=480, camera_id=cam_id)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow('God_View', img)

# ==========================================
# 主程序
# ==========================================
if __name__ == '__main__':
    arena = TwoFlowerArena()
    task = TestTask(fruitfly.FruitFly, arena)
    env = composer.Environment(task, random_state=np.random.RandomState(42))
    
    vision_system = VisionModule(env.physics)
    
    god_cam_id = -1
    for i in range(env.physics.model.ncam):
        if 'god_cam' in env.physics.model.id2name(i, 'camera'): god_cam_id = i

    print(">>> 双色花随机位置测试启动。")
    print(">>> 按 [R] 重置环境并随机生成新位置。")
    timestep = env.reset()

    TARGET_ALTITUDE = 2.40 
    DT = 0.02

    while True:
        sensor_data = vision_system.get_control_signals(env.physics)
        render_god_view(env.physics, god_cam_id)

        # 简单的状态机
        if sensor_data['is_found']:
            body_vx = sensor_data['vx']
            body_vy = sensor_data['vy']
            body_omega = sensor_data['omega']
            err_z = TARGET_ALTITUDE - sensor_data['altitude']
            global_vz = err_z * 0.05 
        else:
            # 搜索模式
            body_vx = 0.0
            body_vy = 0.0
            body_omega = 0.15 
            err_z = TARGET_ALTITUDE - sensor_data['altitude']
            global_vz = err_z * 0.05

        # 坐标变换
        yaw = task.current_yaw
        c = np.cos(yaw)
        s = np.sin(yaw)
        world_dx = (body_vx * c - body_vy * s) * DT
        world_dy = (body_vx * s + body_vy * c) * DT
        
        task.target_pos[0] += world_dx * 5.0
        task.target_pos[1] += world_dy * 5.0
        task.target_pos[2] += global_vz
        task.target_yaw += body_omega * DT * 5.0
        task.target_pos[2] = np.clip(task.target_pos[2], 0.2, 5.0)

        # 打印状态
        if task._step_counter % 10 == 0:
            color_str = str(sensor_data['colors']) if sensor_data['colors'] else "[]"
            target_info = "NONE"
            if sensor_data['raw_targets']:
                best = min(sensor_data['raw_targets'], key=lambda x: x['dist'])
                target_info = f"{best['color'].upper()} ({best['dist']:.1f}m)"

            print(f"\r[Target: {target_info}] "
                  f"Vx:{body_vx:.2f} | "
                  f"Colors:{color_str}      ", end="")
        
        task._step_counter += 1
        env.step(np.zeros(env.action_spec().shape))

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        if key == ord('r'): timestep = env.reset()

    cv2.destroyAllWindows()