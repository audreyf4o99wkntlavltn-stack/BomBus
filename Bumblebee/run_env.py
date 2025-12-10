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
# 假设 STL 文件位于您项目目录下的 'swurdf_out/fly_env.SLDASM/meshes'
MESH_DIR = 'meshes'
XML_PATH = 'fly_env.xml'

# ==========================================
# 1. 定义花朵组件 (Flower)
# ==========================================
class Flower(composer.Entity):

    def _build(self, color=(0, 0, 1), name='flower'):
        self._mjcf_root = mjcf.RootElement(model=name)
        
        # 资源加载
        self._mjcf_root.compiler.meshdir = MESH_DIR
        self._mjcf_root.asset.add('mesh', name='bottle_mesh', file='bottle_link.STL', scale='100 100 100')
        self._mjcf_root.asset.add('mesh', name='plate_mesh', file='plate_Link.STL', scale='100 100 100')
        self._mjcf_root.asset.add('mesh', name='lego_mesh', file='lego_Link.STL', scale='100 100 100')
        
        self._mjcf_root.asset.add('material', name='mat_bottle', rgba=[0.8, 0.8, 1, 0.3])
        self._mjcf_root.asset.add('material', name='mat_plate', rgba=[1, 1, 1, 1])
        self._mjcf_root.asset.add('material', name='mat_lego', rgba=[*color, 1])

        # === 构建物体 ===
        self._base_body = self._mjcf_root.worldbody.add('body', name='base_link')
        self._base_body.add('geom', type='mesh', mesh='bottle_mesh', material='mat_bottle')

        # 板子 (高度 7.9cm)
        plate_body = self._base_body.add('body', name='plate_link', pos=[0, 0, 7.9])
        plate_body.add('geom', type='mesh', mesh='plate_mesh', material='mat_plate')

        # 积木
        lego_body = plate_body.add('body', name='lego_link', pos=[0, 0, 0])
        lego_body.add('joint', name='lego_hinge', type='hinge', axis=[0.63, -0.77, 0])
        lego_body.add('geom', type='mesh', mesh='lego_mesh', material='mat_lego')
        
        # === [核心修改] 花蜜虚拟圆柱体 ===
        # 几何类型：cylinder (圆柱)
        # 尺寸 size=[半径, 半高]：0.4cm 半径 (8mm直径), 0.35cm 半高 (总高 7mm)
        # 位置 pos=[0, 0, 0.69]：中心位于凹槽内，0.69cm 是根据积木高度 1.14cm 和凹槽深度 0.8cm 计算得出的中心点
        self.nectar_site = lego_body.add('site', 
                                         name='nectar', 
                                         pos=[0, 0, 0.69],  
                                         size=[0.4, 0.35],       
                                         type='cylinder', 
                                         rgba=[1, 1, 0.6, 0.4]) # 淡黄色，半透明

    @property
    def mjcf_model(self):
        return self._mjcf_root

# ==========================================
# 2. 定义竞技场 (TwoBoxArena)
# ==========================================
class TwoBoxArena(composer.Arena):
    def _build(self, name='two_box_arena'):
        super()._build(name=name)
        
        if not os.path.exists(XML_PATH):
            raise FileNotFoundError(f"找不到 XML 文件: {os.path.abspath(XML_PATH)}")
        
        self._mjcf_root = mjcf.from_path(XML_PATH)
        
        # 果蝇出生点 (靠近管道的左侧区域)
        self._spawn_site = self._mjcf_root.worldbody.add(
            'site', name='spawn_site', pos=[-20, 0, 15], rgba=[1, 0, 0, 0]
        )
        
        # 修复 ground_geoms 缺失问题
        self._floor_geom = self._mjcf_root.find('geom', 'floor')
        self._box_floor_geom = self._mjcf_root.find('geom', 'big_floor')
        
        self._mjcf_root.worldbody.add('light', pos=[0, 0, 200], dir=[0, 0, -1], diffuse=[0.6, 0.6, 0.6])

    @property
    def ground_geoms(self):
        geoms = [self._floor_geom, self._box_floor_geom]
        return tuple(g for g in geoms if g is not None)

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
        
        self._flower = Flower(color=(0, 0, 1), name='blue_flower')
        self._flower_frame = self._arena.attach(self._flower)
        self._flower_joint = self._flower_frame.add('freejoint', name='flower_float')
        
        # 获取果蝇头部 site 的引用
        self._fly_head_site = self._walker.mjcf_model.find('site', 'head')


    def initialize_episode_mjcf(self, random_state):
        super().initialize_episode_mjcf(random_state)
        # 视觉修正
        # 强制设置近裁剪面（znear）和远裁剪面（zfar）
        self.root_entity.mjcf_model.visual.map.znear = 0.1  # 最小可视距离 0.1cm
        self.root_entity.mjcf_model.visual.map.zfar = 5000.0 # 最大可视距离 50米 (5000cm)
        # 调整场景范围基准，改善滚轮缩放灵敏度
        self.root_entity.mjcf_model.statistic.extent = 200.0
    def initialize_episode(self, physics, random_state):
        super().initialize_episode(physics, random_state)

        # 随机刷新花朵位置 (固定种子下可复现)
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

    def get_reward(self, physics):
        # 1. 获取位置
        head_pos = physics.bind(self._fly_head_site).xpos
        nectar_pos = physics.bind(self._flower.nectar_site).xpos
        
        # 2. 计算距离 (厘米)
        distance = np.linalg.norm(head_pos - nectar_pos)
        
        # 3. 触碰反馈逻辑
        # 阈值设定为 1.5 厘米 (稍微大于花蜜的半径 0.4cm + 果蝇头部半径)
        touch_threshold = 1.5 
        
        reward = 0.0
        if distance < touch_threshold:
            reward = 10.0
            print(f"🎉 喙触碰到了花蜜! 距离: {distance:.2f} cm")
        
        reward += np.exp(-0.1 * distance)
        
        return reward

    def get_reward_factors(self, physics):
        return (self.get_reward(physics),)

# ==========================================
# 4. 运行环境
# ==========================================
def build_env(random_seed=88):
    arena = TwoBoxArena()
    task = FlowerTask(walker=fruitfly.FruitFly, arena=arena)
    return composer.Environment(task, random_state=np.random.RandomState(random_seed))

if __name__ == '__main__':
    print("正在构建环境...")
    
    # 简单的网格文件检查
    required_meshes = ['bottle_link.STL', 'plate_Link.STL', 'lego_Link.STL']
    missing = []
    for m in required_meshes:
        if not os.path.exists(os.path.join(MESH_DIR, m)):
            missing.append(m)
    
    if missing:
        print(f"警告: 找不到以下文件: {missing}")
    else:
        try:
            env = build_env(random_seed=88)
            if env:
                viewer.launch(env)
        except Exception as e:
            import traceback
            traceback.print_exc()