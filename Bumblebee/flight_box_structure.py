import os
from dm_control import mjcf

def build_flight_box_structure(mjcf_root: mjcf.RootElement, 
                               GLOBAL_SCALE: float, 
                               FLIGHT_POS: list, 
                               NEST_POS: list, 
                               TUNNEL_POS: list,
                               FLIGHT_BASE_Z: float,
                               NEST_BASE_Z: float,
                               FLIGHT_COVER_Z: float,
                               NEST_COVER_Z: float,
                               TUNNEL_Z: float,
                               assets_dir: str):
    """
    在给定的 MJCF 根模型中构建飞行箱、巢箱和通道的静态几何结构。

    参数:
        mjcf_root: Arena 的 MJCF 根模型。
        GLOBAL_SCALE: 全局缩放因子。
        ... (其他用于定位和尺寸的参数)
    """

    # --- 1. 加载所有静态结构所需的 Mesh ---
    s = [GLOBAL_SCALE] * 3
    mjcf_root.asset.add('mesh', name='nest_base', file='nest_base.stl', scale=s)
    mjcf_root.asset.add('mesh', name='flight_base', file='flight_box_base.stl', scale=s)
    mjcf_root.asset.add('mesh', name='nest_cover', file='nest_cover.stl', scale=s)
    mjcf_root.asset.add('mesh', name='flight_cover', file='flight_box_cover.stl', scale=s)
    mjcf_root.asset.add('mesh', name='tunnel', file='tunnel.stl', scale=s)
    
    # --- 2. 加载所有结构所需的 Material ---
    mjcf_root.asset.add('texture', name='wood_tex', file='wood.png', type='2d')
    mjcf_root.asset.add('material', name='wood_mat', texture='wood_tex', 
                         specular=0.2, shininess=0.3)
    mjcf_root.asset.add('material', name='glass_mat', 
                         rgba=[0.95, 0.95, 1.0, 0.3], reflectance=0.5, shininess=0.9)
    mjcf_root.asset.add('material', name='tunnel_mat', 
                         rgba=[0.9, 0.9, 0.9, 0.2], reflectance=0.3, shininess=0.8)


    # --- 3. 搭建环境 ---
    
    # A. 飞行箱 (Flight Box)
    flight_group = mjcf_root.worldbody.add('body', name='flight_group', pos=FLIGHT_POS)
    flight_group.add('geom', type='mesh', mesh='flight_base', 
                     material='wood_mat', pos=[0, 0, FLIGHT_BASE_Z])
    flight_group.add('geom', type='mesh', mesh='flight_cover', 
                     material='glass_mat', pos=[0, 0, FLIGHT_COVER_Z])
    
    # B. 通道 (Tunnel)
    tunnel_pos_z = TUNNEL_Z 
    mjcf_root.worldbody.add('geom', type='mesh', mesh='tunnel', 
                            material='tunnel_mat', pos=[TUNNEL_POS[0], TUNNEL_POS[1], tunnel_pos_z])
    
    # C. 巢箱 (Nest)
    nest_group = mjcf_root.worldbody.add('body', name='nest_group', pos=NEST_POS)
    nest_group.add('geom', type='mesh', mesh='nest_base', 
                    material='wood_mat', pos=[0, 0, NEST_BASE_Z])
    nest_group.add('geom', type='mesh', mesh='nest_cover', 
                    material='glass_mat', pos=[0, 0, NEST_COVER_Z])

    # D. 地面 (Ground)
    ground_geom = mjcf_root.worldbody.add('geom', type='plane', size=[10, 10, 0.1], 
                                          rgba=[0.2, 0.2, 0.2, 1], pos=[0, 0, -0.001])
    
    return ground_geom 