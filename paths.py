import re
from pathlib import Path
from typing import Optional, Dict, List, Tuple


class ExperimentInfo:
    """实验信息模型，负责实验元数据的逻辑处理"""

    def __init__(self, exp_number: int, net_type: str, description: str,
                 base_version: Optional[str] = None):
        self.exp_number = exp_number
        self.net_type = net_type
        self.description = description
        self.base_version = base_version  # 如 "01", "02"
        self.is_extension = base_version is not None

    @classmethod
    def from_name(cls, experiment_name: str) -> 'ExperimentInfo':
        """从实验名称字符串（或日志记录）解析并重建对象"""
        # 匹配 EXP_05_LIGHTNET_v02_KD_PCA16 或 EXP_01_RESNET_Base
        pattern = re.compile(r'^EXP_(\d+)_([A-Za-z0-9]+)_(?:v(\d+)_)?(.+)$')
        match = pattern.match(experiment_name)

        if not match:
            raise ValueError(f"无效的实验名称格式: {experiment_name}")

        exp_num = int(match.group(1))
        net_type = match.group(2)
        base_ver = match.group(3)  # 如果没有 vXX，则为 None
        desc = match.group(4)

        return cls(exp_num, net_type, desc, base_ver)

    @property
    def name(self) -> str:
        """生成标准化的实验文件夹名称"""
        if self.is_extension:
            return f"EXP_{self.exp_number:02d}_{self.net_type}_v{self.base_version}_{self.description}"
        return f"EXP_{self.exp_number:02d}_{self.net_type}_{self.description}"

    def get_base_exp_number(self) -> Optional[int]:
        """获取父实验的数字编号"""
        return int(self.base_version) if self.base_version else None

    def __str__(self):
        rel = f" <- Base: {self.base_version}" if self.is_extension else " (Base)"
        return f"[{self.exp_number:02d}] {self.net_type}{rel} | {self.description}"


class PathManager:
    """路径管理类，负责物理路径的生成与管理"""

    def __init__(self, project_root: Optional[str] = None):
        # 基础根目录设置
        self.root = Path(project_root).absolute() if project_root else Path(__file__).parent.absolute()
        self.checkpoints_dir = self.root / "checkpoints"

        # 论文相关路径
        self.paper_dir = self.root / "paper"
        self.paper_results = self.paper_dir / "results"
        self.paper_figures = self.paper_dir / "figures"

        self._ensure_dirs()

    # 预定义论文输出文件映射
    def get_paper_output_files(self):
        paper_files = {
            'model_statistics': self.paper_results / "model_statistics.csv",
            'channel_robustness': self.paper_results / "channel_robustness_comparison.pdf",
            'mobility_robustness': self.paper_results / "mobility_robustness.pdf",
            'accuracy_heatmap': self.paper_results / "accuracy_heatmap_conditions.pdf",
            'detailed_accuracy_heatmap': self.paper_results / "detailed_accuracy_heatmap.pdf",
            'pca_ablation': self.paper_results / "pca_ablation_scenario_based.pdf",
            'pca_origin_pca': self.paper_results / "comparison_origin_pca.pdf",
        }
        return paper_files

    def _ensure_dirs(self):
        """初始化时自动创建核心目录"""
        for d in [self.checkpoints_dir, self.paper_results, self.paper_figures]:
            d.mkdir(parents=True, exist_ok=True)

    # --- 实验管理功能 ---

    def create_experiment(self, net_type_val: str, description: str,
                          base_exp_num: Optional[int] = None) -> ExperimentInfo:
        """创建一个全新的实验对象并分配下一个可用序号"""
        next_num = self._get_next_num()
        base_ver = f"{base_exp_num:02d}" if base_exp_num is not None else None

        return ExperimentInfo(next_num, net_type_val, description, base_ver)

    def _get_next_num(self) -> int:
        """扫描目录获取下一个实验 ID"""
        existing_nums = [
            int(m.group(1)) for d in self.checkpoints_dir.iterdir()
            if d.is_dir() and (m := re.match(r'^EXP_(\d+)_', d.name))
        ]
        return max(existing_nums, default=0) + 1

    def get_exp_paths(self, exp: ExperimentInfo) -> Dict[str, Path]:
        """为给定实验生成完整的路径束"""
        exp_root = self.checkpoints_dir / exp.name
        weights_dir = exp_root / "weights"

        # 自动生成 PCA 路径
        pca_dir = weights_dir / "pca_results"

        paths = {
            "root": exp_root,
            "weights": weights_dir,
            "eval": exp_root / "eval",
            "pca_dir": pca_dir,
            "pca_input": pca_dir / "teacher_feats.npz",
            "pca_output": lambda dim: pca_dir / f"pca_{dim}.npz",  # 支持动态维度
            "Extractor_best": weights_dir / "Extractor_best.pth",
        }
        return paths

    def find_base_exp_path(self, exp: ExperimentInfo) -> Optional[Path]:
        """根据当前实验信息，抓取其 Base 实验的名称字符串"""
        base_num = exp.get_base_exp_number()
        if base_num is None:
            return None

        pattern = f"EXP_{base_num:02d}_*"
        matches = list(self.checkpoints_dir.glob(pattern))
        return matches[0] if matches else None

    def get_lineage_infos(self, exp_name: str) -> List[ExperimentInfo]:
        """从日志中的实验名开始，向上回溯完整的继承链对象"""
        lineage = []
        curr_name = exp_name

        while curr_name:
            info = ExperimentInfo.from_name(curr_name)
            lineage.insert(0, info)

            base_dir = self.find_base_exp_dir(info)
            curr_name = base_dir.name if base_dir else None

        return lineage

if __name__ == '__main__':
    pm = PathManager()

    # 情况 A：从日志中恢复实验并查找其父实验
    log_exp_name = "EXP_05_LIGHTNET_v02_KD_PCA16"
    info = ExperimentInfo.from_name(log_exp_name)
    print(f"当前实验: {info}")
    # 输出: [05] LIGHTNET <- Base: 02 | KD_PCA16

    base_path = pm.find_base_exp_path(info)
    print(f"父实验路径: {base_path}")

    # 情况 B：创建一个新的继承实验
    new_exp = pm.create_experiment("ResNet", "KD_PCA8", base_exp_num=1)
    paths = pm.get_exp_paths(new_exp)

    print(f"新实验目录: {paths['root']}")
    print(f"PCA 输出路径: {paths['pca_output'](16)}")

    # 情况 C：访问静态论文路径
    print(f"论文图表目录: {pm.paper_figures}")