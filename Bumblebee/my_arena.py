import os
from dm_control import composer
from dm_control import mjcf

XML_PATH = 'fly_env.xml'

class TwoBoxArena(composer.Arena):

    def _build(self, name='two_box_arena'):

        super()._build(name=name)
        # 1. 加载您的 SolidWorks 导出 XML
        if not os.path.exists(XML_PATH):
            raise FileNotFoundError(f"找不到文件: {XML_PATH}")
            
        self._mjcf_root = mjcf.from_path(XML_PATH)
        
        # 2. 获取关键部件的引用
        self._flight_box = self._mjcf_root.find('body', 'part1_big_box')
        
        # 3. 添加重生点 (Spawn Site)
        self._spawn_site = self._mjcf_root.worldbody.add(
            'site', 
            name='spawn_site', 
            pos=[0, 0, 0.2],  # 高度 0.2，避免卡在地板里
            rgba=[1, 0, 0, 0] # 透明不可见
        )

        # 4. 视觉效果
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