## GeoSTGAN 项目结构

本仓库按最佳实践重组为四大模块：

- 预处理（preprocess）
  - `preprocess/data`: 数据加载与工具（原 data/*）
  - `preprocess/preprocessor`: 影像/地理预处理工具（原 preprocessor/*）
- 模型结构（models）
  - 现有模型代码保持不变（GAU/GRU/GAN 等）
- 训练与推理（train_infer）
  - `train_infer/losses.py`: 训练损失
  - `train_infer/sharpening_losses.py`: 锐化损失
  - `train_infer/ewc.py`: EWC 接口（转发到原实现）
  - 训练/分布式流程脚本（`step_train.py`, `distributed_constraint.py`, `constraint_*.py`, `compute_pretrain_ewc.py` 等引用已适配）
- 可视化（visualization）
  - 统一可视化工具与脚本（当前保留 `visual.py`, `model_visual.py`，后续可迁移到该目录）

### 全局配置

- `config/config.py`: 全局路径与运行参数（PATHS、RUNTIME）。
- `config/model_config.py`: 模型规模（通道/维度等，统一入口）。

示例：

```python
from config.config import PATHS, ensure_dirs
from config.model_config import MODEL_SIZE

ensure_dirs()
print(PATHS.checkpoints_dir)
print(MODEL_SIZE.hidden_channels)
```

### 常见入口

- 预训练/训练：`python step_train.py`
- 约束训练：`python constraint_train.py`
- 约束推理：`python constraint_infer.py`
- Fisher/EWC 计算：`python compute_pretrain_ewc.py`

请先准备特征/标签目录并在 `config/config.py` 中设置 `PATHS.factors_data_dir` 与 `PATHS.labels_data_dir`，或通过环境变量覆盖。

### 依赖

安装：

```bash
pip install -r requirements.txt
```

### 备注

- 模型与算法核心实现保持不变；仅重构工具与导入路径。
- 代码中硬编码路径已逐步替换为 `config.PATHS`，如有遗漏，请在本地再行调整。
