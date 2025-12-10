import numpy as np
from dm_control import composer
from dm_control import viewer
from dm_control import mjcf 

# 假设您的 Walker 模型 (果蝇/熊蜂) 位于 flybody.fruitfly
# 请根据您的实际项目路径调整此行
from flybody.fruitfly import fruitfly 

# 导入自定义的 Arena 类
# 确保 bumblebee_arena.py 文件在正确的路径下
from bumblebee_arena import BumblebeeExperimentArena 

# --- 任务定义 ---
class BumblebeeForagingTask(composer.Task):
    
    def __init__(self, walker, arena, time_limit):
        self._walker = walker
        self._arena = arena
        self._time_limit = time_limit
        
        # 1. 移除 Walker 模型中可能与 Arena 冲突的全局属性
        walker_model = walker.mjcf_model
        if hasattr(walker_model.visual, 'headlight') and hasattr(walker_model.visual.headlight, 'ambient'):
              del walker_model.visual.headlight.ambient
        
        # 2. 将 Walker 的模型附加到 Arena 的 worldbody 上
        self._arena.mjcf_model.worldbody.attach(walker_model) 
        
        self._root_entity = arena 

    @property
    def root_entity(self):
        return self._arena
    
    # 奖励判定逻辑 (测试时返回 0.0)
    def get_reward(self, physics):
        return 0.0

# --- 环境创建函数 ---
def custom_bumblebee_env(random_state=None):
    """创建并初始化熊蜂实验环境。"""
    # 实例化 Walker 对象 (熊蜂)
    walker = fruitfly.FruitFly
    
    # 实例化 Arena (这会触发环境结构搭建和花朵随机化)
    arena = BumblebeeExperimentArena()

    # 创建 Task
    task = BumblebeeForagingTask(walker=walker, arena=arena, time_limit=10.0)

    # 返回 composer.Environment 实例
    return composer.Environment(task=task, random_state=random_state)


# --- 启动器 ---
if __name__ == '__main__':
    
    SEED = 42 
    random_state = np.random.RandomState(SEED)
    
    # 1. 创建环境
    env = custom_bumblebee_env(random_state=random_state)
    
    # 2. 随机策略 (用于测试，让熊蜂乱动)
    def random_policy(time_step):
        return np.random.uniform(-0.1, 0.1, size=env.action_spec().shape)

    print("环境加载成功，正在启动 MuJoCo 查看器...")
    # 3. 启动查看器 
    viewer.launch(env, policy=random_policy)