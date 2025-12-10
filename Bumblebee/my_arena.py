import os
from dm_control import composer
from dm_control import mjcf

# 引用您的 XML 文件路径
XML_PATH = 'swurdf_out/fly_env.SLDASM/fly_env.xml'

class TwoBoxArena(composer.Arena):
    """
    双箱环境竞技场。
    参考 turagalab/flybody/flybody/tasks/arenas/hills.py 构建。
    """

    def _build(self, name='two_box_arena'):
        # 调用父类构造函数
        super()._build(name=name)
        
        # 1. 加载您的 SolidWorks 导出 XML
        # 确保路径正确，如果不正确会报错
        if not os.path.exists(XML_PATH):
            raise FileNotFoundError(f"找不到文件: {XML_PATH}")
            
        self._mjcf_root = mjcf.from_path(XML_PATH)
        
        # 2. 获取关键部件的引用 (可选)
        # 类似于 hills.py 中获取 terrain_geom
        # 这样做的好处是以后如果要在 "part1_big_box" 里加东西，可以直接用 self._flight_box
        self._flight_box = self._mjcf_root.find('body', 'part1_big_box')
        
        # 3. 添加重生点 (Spawn Site)
        # 如果您的 XML 里没有定义 spawn site，我们可以在这里手动加一个
        # 这样果蝇出生时就会自动寻找这个点，而不是默认的 (0,0,0)
        # 假设我们想让果蝇出生在第一个盒子的中心上方
        self._spawn_site = self._mjcf_root.worldbody.add(
            'site', 
            name='spawn_site', 
            pos=[0, 0, 0.2],  # 高度 0.2，避免卡在地板里
            rgba=[1, 0, 0, 0] # 不可见
        )

        # 4. 优化视觉效果 (可选)
        # 如果 XML 里的光照不够，可以在这里通过代码补光
        self._mjcf_root.worldbody.add('light', pos=[0, 0, 2], dir=[0, 0, -1], diffuse=[0.6, 0.6, 0.6])

    @property
    def spawn_position(self):
        """返回建议的果蝇出生点坐标。"""
        return self._spawn_site.pos

    def regenerate(self, random_state):
        """
        如果环境需要随机化（比如移动箱子位置），可以在这里写。
        目前保持为空，表示静态环境。
        """
        pass