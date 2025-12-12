import os
import numpy as np
from dm_control import composer
from dm_control import mjcf
from dm_control import viewer

from flybody.fruitfly import fruitfly
from flybody.tasks.base import Flying

# ==========================================
# 0. 全局配置
# ==========================================
MESH_DIR = 'meshes'
XML_PATH = 'fly_env.xml'
# 缩放因子 S = 0.25 (缩小 4 倍)
SCALE_FACTOR = 0.25

# ==========================================
# 1. 定义花朵组件 (Flower)
# ... (保持不变) ...
# ==========================================
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
        
        self.nectar_site = lego_body.add('site', 
                                         name='nectar', 
                                         pos=[0, 0, 0.69 * SCALE_FACTOR],  
                                         size=[0.4 * SCALE_FACTOR, 0.35 * SCALE_FACTOR],       
                                         type='cylinder', 
                                         rgba=[1, 1, 0.6, 0.4])

    @property
    def mjcf_model(self):
        return self._mjcf_root

# ==========================================
# 2. 定义竞技场 (TwoBoxArena)
# ... (保持不变) ...
# ==========================================
class TwoBoxArena(composer.Arena):
    def _build(self, name='two_box_arena'):
        super()._build(name=name)
        
        if not os.path.exists(XML_PATH):
            raise FileNotFoundError(f"找不到 XML 文件: {os.path.abspath(XML_PATH)}")
        
        self._mjcf_root = mjcf.from_path(XML_PATH)
        
        self._floor_geom = self._mjcf_root.find('geom', 'floor')
        self._box_floor_geom = self._mjcf_root.find('geom', 'big_floor')
        
        self._mjcf_root.worldbody.add('light', pos=[0, 0, 200 * SCALE_FACTOR], dir=[0, 0, -1], diffuse=[0.6, 0.6, 0.6])

    @property
    def ground_geoms(self):
        geoms = [self._floor_geom, self._box_floor_geom]
        return tuple(g for g in geoms if g is not None)

    @property
    def spawn_position(self):
        return np.array([0, 0, 20 * SCALE_FACTOR]) 

# ==========================================
# 3. 定义任务逻辑 (FlowerTask)
# ==========================================
class FlowerTask(Flying):
    
    FLY_HEIGHT = 5.0 

    def __init__(self, walker, arena, time_limit=float('inf'), **kwargs):
        super().__init__(
            walker=walker,
            arena=arena,
            time_limit=time_limit,
            joint_filter=0., 
            future_steps=0,
            floor_contacts=True, 
            **kwargs
        )
        
        self._flower = Flower(color=(0, 0, 1), name='blue_flower')
        self._flower_frame = self._arena.attach(self._flower)
        self._flower_joint = self._flower_frame.add('freejoint', name='flower_float')
        
        self._fly_head_site = self._walker.mjcf_model.find('site', 'head')


    def initialize_episode_mjcf(self, random_state):
        super().initialize_episode_mjcf(random_state)
        # [核心修正] zfar 扩大到 20cm (50 * 0.25 = 12.5 -> 20)
        self.root_entity.mjcf_model.visual.map.znear = 0.01 * SCALE_FACTOR 
        self.root_entity.mjcf_model.visual.map.zfar = 200.0 * SCALE_FACTOR # 80 * 0.25 = 20cm
        self.root_entity.mjcf_model.statistic.extent = 10.0 * SCALE_FACTOR 

    def initialize_episode(self, physics, random_state):
        # 强制设置果蝇悬空位置 (5cm高)
        super().initialize_episode(physics, random_state)

        initial_pos = np.array([0.0, 0.0, self.FLY_HEIGHT])
        initial_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        qpos = physics.data.qpos.copy()
        qpos[:3] = initial_pos
        qpos[3:7] = initial_quat
        
        physics.data.qpos[:] = qpos
        physics.forward() 

        # 刷新花朵位置 (范围 x0.25)
        flower_x = random_state.uniform(-10 * SCALE_FACTOR, 25 * SCALE_FACTOR) 
        flower_y = random_state.uniform(-15 * SCALE_FACTOR, 15 * SCALE_FACTOR) 
        flower_z = 0.0
        
        angle = random_state.uniform(0, 2 * np.pi)
        quat = [np.cos(angle/2), 0, 0, np.sin(angle/2)]

        flower_binding = physics.bind(self._flower_joint)
        if flower_binding is not None:
            flower_binding.qpos[:3] = [flower_x, flower_y, flower_z]
            flower_binding.qpos[3:] = quat
            flower_binding.qvel[:] = 0

    def get_reward(self, physics):
        head_pos = physics.bind(self._fly_head_site).xpos
        nectar_pos = physics.bind(self._flower.nectar_site).xpos
        distance = np.linalg.norm(head_pos - nectar_pos)
        
        touch_threshold = 1.5 * SCALE_FACTOR
        reward = 0.0
        if distance < touch_threshold:
            reward = 10.0
            print(f"🎉 喙触碰到了花蜜! 距离: {distance:.2f} cm (缩小后)")
        
        reward += np.exp(-0.4 * distance)
        return reward

    def get_reward_factors(self, physics):
        return (self.get_reward(physics),)

# ==========================================
# 4. 运行环境
# ==========================================
def build_env(random_seed=42):
    arena = TwoBoxArena()
    task = FlowerTask(walker=fruitfly.FruitFly, arena=arena)
    return composer.Environment(task, random_state=np.random.RandomState(random_seed))

if __name__ == '__main__':
    print("正在构建环境...")
    
    MESH_DIR_USED = MESH_DIR
    required_meshes = ['bottle_link.STL', 'plate_Link.STL', 'lego_Link.STL']
    missing = []
    for m in required_meshes:
        if not os.path.exists(os.path.join(MESH_DIR_USED, m)):
            missing.append(m)
    
    if missing:
        print(f"警告: 找不到以下文件: {missing}。请确保它们在 '{MESH_DIR_USED}' 目录下。")
    else:
        try:
            env = build_env(random_seed=42)
            if env:
                print("环境构建成功！")
                print("【观察】: zfar 已调整到 20cm，箱子应该完整可见。")
                
                def policy(time_step):
                    spec = env.action_spec()
                    return np.zeros(spec.shape)

                viewer.launch(
                    env, 
                    policy=policy
                )
        except Exception as e:
            import traceback
            traceback.print_exc()