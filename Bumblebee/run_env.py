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

# ==========================================
# 1. 定义花朵组件 (Flower)
# ==========================================
class Flower(composer.Entity):

    def _build(self, color=(0, 0, 1), name='flower'):
        self._mjcf_root = mjcf.RootElement(model=name)
        
        # 1. 设置资源
        self._mjcf_root.compiler.meshdir = MESH_DIR
        self._mjcf_root.asset.add('mesh', name='bottle_mesh', file='bottle_link.STL', scale='100 100 100')
        self._mjcf_root.asset.add('mesh', name='plate_mesh', file='plate_Link.STL', scale='100 100 100')
        self._mjcf_root.asset.add('mesh', name='lego_mesh', file='lego_Link.STL', scale='100 100 100')
        
        self._mjcf_root.asset.add('material', name='mat_bottle', rgba=[0.8, 0.8, 1, 0.3])
        self._mjcf_root.asset.add('material', name='mat_plate', rgba=[1, 1, 1, 1])
        self._mjcf_root.asset.add('material', name='mat_lego', rgba=[*color, 1])

        # === 构建物体结构 ===
        self._base_body = self._mjcf_root.worldbody.add('body', name='base_link')
        self._base_body.add('geom', type='mesh', mesh='bottle_mesh', material='mat_bottle')

        # 板子
        plate_body = self._base_body.add('body', name='plate_link', pos=[0, 0, 7.9])
        plate_body.add('geom', type='mesh', mesh='plate_mesh', material='mat_plate')

        # 积木
        lego_body = plate_body.add('body', name='lego_link', pos=[0, 0, 0])
        lego_body.add('joint', name='lego_hinge', type='hinge', axis=[0.63, -0.77, 0])
        lego_body.add('geom', type='mesh', mesh='lego_mesh', material='mat_lego')
        
        self.nectar_site = lego_body.add('site', name='nectar', pos=[0, 0, 1], size=[0.5], rgba=[1, 1, 0, 1])

    @property
    def mjcf_model(self):
        return self._mjcf_root

# ==========================================
# 2. 定义box
# ==========================================
class TwoBoxArena(composer.Arena):
    def _build(self, name='two_box_arena'):
        super()._build(name=name)
        
        if not os.path.exists(XML_PATH):
            raise FileNotFoundError(f"找不到 XML 文件: {os.path.abspath(XML_PATH)}")
        
        self._mjcf_root = mjcf.from_path(XML_PATH)
        
        # 定义果蝇出生点
        self._spawn_site = self._mjcf_root.worldbody.add(
            'site', name='spawn_site', pos=[-20, 0, 15], rgba=[1, 0, 0, 0]
        )
        
        self._mjcf_root.worldbody.add('light', pos=[0, 0, 200], dir=[0, 0, -1], diffuse=[0.6, 0.6, 0.6])

    @property
    def spawn_position(self):
        return self._spawn_site.pos

# ==========================================
# 3. 定义任务逻辑 (FlowerTask)
# ==========================================
class FlowerTask(Flying):
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
        
        # 1. 创建花朵
        self._flower = Flower(color=(0, 0, 1), name='blue_flower')
        
        self._flower_frame = self._arena.attach(self._flower)
        self._flower_joint = self._flower_frame.add('freejoint', name='flower_float')

    def initialize_episode_mjcf(self, random_state):
        super().initialize_episode_mjcf(random_state)
        # 视觉修正
        self.root_entity.mjcf_model.visual.map.znear = 0.1
        self.root_entity.mjcf_model.visual.map.zfar = 5000.0
        self.root_entity.mjcf_model.statistic.extent = 200.0

    def initialize_episode(self, physics, random_state):
        super().initialize_episode(physics, random_state)

        # 1. 随机位置
        flower_x = random_state.uniform(-10, 25)
        flower_y = random_state.uniform(-15, 15)
        flower_z = 0.0 
        
        angle = random_state.uniform(0, 2 * np.pi)
        quat = [np.cos(angle/2), 0, 0, np.sin(angle/2)]

        flower_binding = physics.bind(self._flower_joint)
        
        if flower_binding is not None:
            flower_binding.qpos[:3] = [flower_x, flower_y, flower_z]
            flower_binding.qpos[3:] = quat
            flower_binding.qvel[:] = 0

    def get_reward_factors(self, physics):
        return (0,)

# ==========================================
# 4. 运行环境
# ==========================================
def build_env(random_seed=42):
    arena = TwoBoxArena()
    task = FlowerTask(walker=fruitfly.FruitFly, arena=arena)
    return composer.Environment(task, random_state=np.random.RandomState(random_seed))

if __name__ == '__main__':

    required_meshes = ['bottle_link.STL', 'plate_Link.STL', 'lego_Link.STL']
    missing = []
    for m in required_meshes:
        if not os.path.exists(os.path.join(MESH_DIR, m)):
            missing.append(m)
    
    if missing:
        print(f"警告: 找不到以下文件: {missing}")
    else:
        try:
            env = build_env(random_seed=42)
            if env:
                print("环境构建成功！")
                print("花朵位置将根据 Seed=42 固定随机生成。")
                viewer.launch(env)
        except Exception as e:
            import traceback
            traceback.print_exc()