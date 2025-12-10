import os
import numpy as np
from dm_control import composer
from dm_control import mjcf 

# 导入静态结构构建函数
from flight_box_structure import build_flight_box_structure 

class BumblebeeExperimentArena(composer.Arena):
    def _build(self):
        super()._build(name='bumblebee_arena')
        
        # 1. 内存设置 (Memory Settings)
        self._mjcf_root.size.memory = "100M"
        self._mjcf_root.size.njmax = None
        self._mjcf_root.size.nconmax = None
        
        # 2. 全局缩放 (Global Scaling)
        self.GLOBAL_SCALE = 10.0 

        # --- A. 环境结构高度 (米) ---
        self.FLIGHT_BASE_Z = 1.25
        self.NEST_BASE_Z   = 1.25
        self.FLIGHT_COVER_Z = 0.125 * self.GLOBAL_SCALE
        self.NEST_COVER_Z   = 0.125 * self.GLOBAL_SCALE
        self.TUNNEL_Z       = 0.05 * self.GLOBAL_SCALE

        # --- B. 花朵组件高度 (米) ---
        self.BOTTLE_Z   = 0.0022 * self.GLOBAL_SCALE
        self.PLATFORM_Z = (0.0022 + 0.079) * self.GLOBAL_SCALE
        self.BLOCK_Z    = self.PLATFORM_Z + (0.002 * self.GLOBAL_SCALE) + (0.005 * self.GLOBAL_SCALE)
        self.LID_Z_ON   = self.BLOCK_Z + (0.0114 * self.GLOBAL_SCALE)

        # 位置设置 (Position Settings)
        self.FLIGHT_POS = [0, 0, 0]
        self.TUNNEL_POS = [-0.003 * self.GLOBAL_SCALE, 0, 3] 
        self.NEST_POS   = [0 * self.GLOBAL_SCALE, 0, 0]

        # 3. 光照 (Lighting) - 仅需在 Arena 中设置，无需传入结构函数
        self._mjcf_root.visual.headlight.set_attributes(
            ambient=[0.4, 0.4, 0.4], diffuse=[0.8, 0.8, 0.8], specular=[0.1, 0.1, 0.1]
        )

        # 4. 资源加载 & 材质定义 (Assets & Materials)
        assets_dir = os.path.join(os.path.dirname(__file__), 'assets')
        self._mjcf_root.compiler.meshdir = assets_dir
        self._mjcf_root.compiler.texturedir = assets_dir
        
        # -------------------------------------------------------------
        # ★★★ 关键步骤：调用结构构建函数来添加箱体和通道 ★★★
        # -------------------------------------------------------------
        self._ground_geom = build_flight_box_structure(
            self._mjcf_root, 
            self.GLOBAL_SCALE, 
            self.FLIGHT_POS, 
            self.NEST_POS, 
            self.TUNNEL_POS,
            self.FLIGHT_BASE_Z,
            self.NEST_BASE_Z,
            self.FLIGHT_COVER_Z,
            self.NEST_COVER_Z,
            self.TUNNEL_Z,
            assets_dir
        )
        
        # -------------------------------------------------------------
        # ★★★ 仅在此处定义花朵的材质和 Mesh ★★★
        # -------------------------------------------------------------
        # 5. 花朵相关材质 (定义在 Arena 中，因为 Arena 拥有 asset 标签)
        self._mjcf_root.asset.add('material', name='bottle_mat',
                                 rgba=[0.3, 0.15, 0.05, 0.95], 
                                 reflectance=0.3, specular=0.5, shininess=0.5)
        self._mjcf_root.asset.add('material', name='white_platform_mat',
                                 rgba=[1, 1, 1, 1],
                                 specular=0.3, shininess=0.1)
        self._mjcf_root.asset.add('material', name='flower_lid_mat',
                                 rgba=[0.9, 0.95, 1.0, 0.25],
                                 reflectance=0.6, shininess=0.8)
        
        # 6. 花朵 Mesh
        s = [self.GLOBAL_SCALE] * 3
        self._mjcf_root.asset.add('mesh', name='f_bottle', file='flower_bottle.stl', scale=s)
        self._mjcf_root.asset.add('mesh', name='f_platform', file='flower_platform.stl', scale=s)
        self._mjcf_root.asset.add('mesh', name='f_block', file='flower_block.stl', scale=s)
        self._mjcf_root.asset.add('mesh', name='f_lid', file='flower_lid.stl', scale=s)

        # 7. 初始化花朵 (Initializing Flowers)
        self._flowers = []
        for i in range(6): 
            self._flowers.append(self._create_flower_body(i))

    # ... (_create_flower_body, regenerate, initialize_episode, ground_geoms 方法保持不变) ...
    def _create_flower_body(self, index):
        # ... (与原代码相同) ...
        flower_body = self._mjcf_root.worldbody.add('body', name=f'flower_{index}', pos=[0,0,0])
        
        # 1. 瓶子
        flower_body.add('geom', type='mesh', mesh='f_bottle', 
                         material='bottle_mat', 
                         pos=[0, 0, self.BOTTLE_Z])
        
        # 2. 平台
        flower_body.add('geom', type='mesh', mesh='f_platform', 
                         material='white_platform_mat', 
                         pos=[0, 0, self.PLATFORM_Z]) 
        
        # 3. 积木
        mat_name = f'block_mat_{index}'
        self._mjcf_root.asset.add('material', name=mat_name, rgba=[0.5, 0.5, 0.5, 1])
        flower_body.add('geom', type='mesh', mesh='f_block', 
                         material=mat_name, pos=[0, 0, self.BLOCK_Z])
        
        # 4. 盖子 
        lid_body = flower_body.add('body', name=f'lid_{index}', pos=[0,0,0])
        lid_body.add('geom', type='mesh', mesh='f_lid', 
                      material='flower_lid_mat', 
                      pos=[0,0,0])
        
        return {'body': flower_body, 'lid': lid_body, 'mat_name': mat_name, 'z_lid_on': self.LID_Z_ON}

    def regenerate(self, random_state):
        # ... (与原代码相同) ...
        # --- 1. 颜色分配：3蓝 3绿 ---
        colors = {'blue': [0, 0, 1, 1], 'green': [0, 1, 0, 1]}
        color_pool = [colors['blue']] * 3 + [colors['green']] * 3
        random_state.shuffle(color_pool) 
        
        is_covered = random_state.choice([True, False])
        
        # --- 2. 随机均匀分布逻辑 ---
        box_half_width = 1.5 
        min_dist = 0.8 
        
        flower_positions = []
        
        for i, flower in enumerate(self._flowers):
            valid_position = False
            attempts = 0
            
            while not valid_position and attempts < 100:
                rel_x = random_state.uniform(-box_half_width, box_half_width)
                rel_y = random_state.uniform(-box_half_width, box_half_width)
                
                abs_x = self.FLIGHT_POS[0] + rel_x
                abs_y = self.FLIGHT_POS[1] + rel_y
                
                too_close = False
                for existing_pos in flower_positions:
                    dist = np.sqrt((abs_x - existing_pos[0])**2 + (abs_y - existing_pos[1])**2)
                    if dist < min_dist:
                        too_close = True
                        break
                
                if not too_close:
                    valid_position = True
                    flower_positions.append([abs_x, abs_y])
                    
                    # 更新位置
                    flower['body'].pos = [abs_x, abs_y, 0]
                    
                    # 更新颜色
                    self._mjcf_root.find('material', flower['mat_name']).rgba = color_pool[i]
                    
                    # 更新盖子
                    if is_covered:
                        flower['lid'].pos = [0, 0, flower['z_lid_on']]
                        flower['lid'].quat = [1, 0, 0, 0]
                    else:
                        shift = 0.04 * self.GLOBAL_SCALE
                        flower['lid'].pos = [shift, 0, flower['z_lid_on']] 
                        flower['lid'].quat = [0.92, 0.38, 0, 0]
                
                attempts += 1
            
            if not valid_position:
                flower['body'].pos = [self.FLIGHT_POS[0], self.FLIGHT_POS[1], 0]

    def initialize_episode(self, physics, random_state):
        # ... (与原代码相同) ...
        super().initialize_episode(physics, random_state)
        # 强制出生在巢箱地板上
        spawn_z = self.NEST_BASE_Z + 0.05 * self.GLOBAL_SCALE
        start_pos = [self.NEST_POS[0], self.NEST_POS[1], spawn_z]
        try:
            physics.named.data.qpos['walker/root'][:3] = start_pos
        except Exception as e:
            print(f"警告: 无法设置果蝇初始位置: {e}")

    @property
    def ground_geoms(self):
        # ... (与原代码相同) ...
        return (self._ground_geom,)