"""SwanLab 实验跟踪管理器 - 单例模式"""
from typing import Optional, Dict, Any


class SwanLabManager:
    """SwanLab 实验跟踪管理器"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.available = False
        self.swanlab = None
        self._initialized = True
    
    def init(self, project: str, experiment_name: str, config: dict, 
             resume: bool = False, exp_id: Optional[str] = None,
             logdir: Optional[str] = None, disable: bool = False):
        """初始化 SwanLab 实验"""
        if disable:
            print("SwanLab 已禁用")
            return
        
        try:
            import swanlab
            self.swanlab = swanlab
            self.available = True
            
            swanlab.login()
            swanlab.init(
                project=project,
                experiment_name=experiment_name,
                resume=resume,
                id=exp_id,
                config=config,
                logdir=logdir
            )
            print(f"✅ SwanLab 初始化成功: {experiment_name}")
        except ImportError:
            print("⚠️  SwanLab 未安装，实验跟踪将跳过")
            self.available = False
        except Exception as e:
            print(f"❌ SwanLab 初始化失败: {e}")
            self.available = False
    
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """记录指标"""
        if self.available and self.swanlab:
            try:
                self.swanlab.log(metrics, step=step)
            except Exception as e:
                print(f"SwanLab 记录失败: {e}")
    
    def finish(self):
        """结束实验"""
        if self.available and self.swanlab:
            try:
                self.swanlab.finish()
                print("✅ SwanLab 实验已结束")
            except Exception as e:
                print(f"SwanLab 结束失败: {e}")


# 全局单例
swanlab_manager = SwanLabManager()
