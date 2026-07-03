"""训练装饰器 - 完全解耦 SwanLab 日志记录"""
import functools
from typing import Dict, Any, Optional
from .swanlab_manager import swanlab_manager


def auto_log_training(step_key: str = "epoch", 
                      metrics_prefix: str = "train",
                      log_interval: int = 1):
    """
    自动记录训练指标的装饰器
    
    【完全解耦】训练函数只需返回/yield 包含指标的字典，无需任何 SwanLab 代码
    
    支持两种模式：
    1. 普通函数：返回最终指标字典
    2. 生成器函数：每次 yield 都会自动记录
    
    Args:
        step_key: 步数字段名（如 'epoch'）
        metrics_prefix: 指标前缀（如 'train' -> 'train/loss'）
        log_interval: 记录间隔（每 N 步记录一次）
    
    Example:
        @auto_log_training(step_key="epoch", metrics_prefix="train")
        def train_model(config, data, labels):
            for epoch in range(num_epochs):
                loss = train_one_epoch()
                acc = validate()
                
                # 只需 yield 指标，装饰器自动记录
                yield {
                    "epoch": epoch,
                    "loss": loss,
                    "accuracy": acc
                }
            
            return {"final_loss": loss, "final_accuracy": acc}
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            
            # 检测是否为生成器
            if hasattr(result, '__iter__') and hasattr(result, '__next__'):
                # 生成器模式：每次 yield 自动记录
                for metrics in result:
                    if isinstance(metrics, dict):
                        step = metrics.get(step_key)
                        
                        # 添加前缀
                        logged_metrics = {}
                        for key, value in metrics.items():
                            if key != step_key:  # 排除 step_key 本身
                                logged_key = f"{metrics_prefix}/{key}" if metrics_prefix else key
                                logged_metrics[logged_key] = value
                        
                        # 按间隔记录
                        if step is None or step % log_interval == 0:
                            swanlab_manager.log(logged_metrics, step=step)
                    
                    yield metrics  # 继续传递 yield 的值
            else:
                # 普通函数模式：记录最终结果
                if isinstance(result, dict):
                    step = result.get(step_key)
                    
                    logged_metrics = {}
                    for key, value in result.items():
                        if key != step_key:
                            logged_key = f"{metrics_prefix}/{key}" if metrics_prefix else key
                            logged_metrics[logged_key] = value
                    
                    swanlab_manager.log(logged_metrics, step=step)
            
            return result
        return wrapper
    return decorator


def track_experiment(project: str = "default",
                     experiment_name_key: str = "info.name",
                     config_key: str = "config",
                     disable_key: str = "disable_swanlab"):
    """
    实验生命周期管理装饰器
    
    自动处理 SwanLab 的 init 和 finish
    
    Args:
        project: 项目名称
        experiment_name_key: 从 config 中提取实验名称的路径（如 'info.name'）
        config_key: 配置参数名
        disable_key: 禁用标志键名
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 提取 config
            config = kwargs.get(config_key) or (args[0] if args else None)
            
            if config:
                # 提取实验名称（支持嵌套属性）
                exp_name = config
                for key in experiment_name_key.split('.'):
                    exp_name = getattr(exp_name, key, None)
                    if exp_name is None:
                        break
                
                # 构建配置字典
                swanlab_config = {}
                if hasattr(config, 'net_type'):
                    swanlab_config['network_type'] = config.net_type.value
                if hasattr(config, 'preprocess_type'):
                    swanlab_config['preprocess_type'] = config.preprocess_type.value
                if hasattr(config, 'HP'):
                    swanlab_config.update({
                        'batch_size': config.HP.get('batch_size'),
                        'num_epochs': config.HP.get('num_epochs'),
                        'learning_rate': config.HP.get('learning_rate'),
                    })
                
                # 初始化
                swanlab_manager.init(
                    project=project,
                    experiment_name=str(exp_name),
                    config=swanlab_config,
                    disable=getattr(config, disable_key, False)
                )
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                # 确保结束时关闭
                swanlab_manager.finish()
        
        return wrapper
    return decorator


def combined_training_decorator(project: str = "Lightweight_LoRa_RFFI",
                                 metrics_prefix: str = "train",
                                 step_key: str = "epoch"):
    """
    组合装饰器：同时处理实验管理和指标记录
    
    这是最推荐的使用方式，一行装饰器搞定所有日志
    """
    def decorator(func):
        @track_experiment(project=project)
        @auto_log_training(step_key=step_key, metrics_prefix=metrics_prefix)
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator
