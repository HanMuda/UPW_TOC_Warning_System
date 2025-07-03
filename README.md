# 芯片产线超纯水TOC浓度实时监控与预警系统

**TOC Real-time Monitoring and Early Warning System for Ultrapure Water in Semiconductor Fabrication**

---

## 1. 项目简介

本项目旨在为半导体（芯片）制造过程中的**电子级超纯水（UPW）制备系统**提供一个智能的**总有机碳（TOC）浓度监控与预警解决方案**。

在芯片生产中，超纯水中的TOC含量是衡量水质的关键指标之一，即使是微量的有机物污染也可能导致晶圆表面缺陷，从而严重影响芯片的良率和性能。本系统通过分析生产过程中的多个相关传感器数据，利用深度学习时间序列模型，实现以下核心目标：

* **实时预测**：提前预测未来一段时间内的TOC浓度，为产线工程师提供决策支持。
* **早期预警**：当预测到TOC浓度有超标风险时，系统能够发出预警，帮助工程师提前介入，避免重大生产事故。

## 2. 主要功能与亮点

本系统集成了多种先进技术，构成了一个功能完备、灵活且可扩展的时间序列预测框架。

* **多模型支持**：内置多种强大的深度学习模型，可通过配置文件一键切换：
    * **LSTM**: 经典且强大的循环神经网络。
    * **TCN**: 善于捕捉长程时间依赖的时间卷积网络。
    * **TCN-LSTM-Attention**: 结合TCN的特征提取能力、LSTM的时序记忆能力和Attention机制的聚焦能力，构成的混合模型。

* **自动化超参数寻优**：集成 **Optuna** 框架，可对选定模型的所有关键超参数（如网络层数、学习率、批大小等）进行自动化、高效的搜索，找到最佳模型配置。

* **先进的特征工程**：支持灵活、可配置的特征工程流水线，包括：
    * **时间特征**: 自动提取小时、星期、月份等周期性特征，并进行 `sin/cos` 编码。
    * **衍生特征**: 自动计算各输入特征与目标值的**差值**和**变化率**，为模型提供更丰富的关系信息。

* **针对性的不平衡数据处理**：
    * **Jittering 数据增强**: 对TOC浓度超标的“少数类”样本进行加噪增强，扩充高价值训练数据。
    * **加权损失函数 (WeightedMSELoss)**: 在计算损失时，对预测错高浓度样本施加更大的惩罚，使模型更关注预警任务。

* **全面的模型评估与可视化**：
    * **性能指标**: 自动计算回归指标 (RMSE, MAE, R-squared) 和针对预警任务的分类指标 (TP, FP, FN, F1-Score)。
    * **可视化报告**: 自动生成多种图表，包括：
        * 训练/验证损失曲线
        * 预测值 vs. 真实值对比图（以日期为横轴，精确标示训练/验证/测试分割点）
        * **预测误差时间序列图**
        * **混淆矩阵**

* **高度可配置与灵活性**：
    * 所有实验参数（模型选择、特征列表、超参数、路径等）均在 `config.yaml` 中统一管理，无需修改任何Python代码即可调整实验。
    * 支持“一键”切换**“调优模式”**与**“直接训练模式”**。
    * 输出路径（模型权重、结果文件）可根据模型名称**动态生成**，完美隔离不同模型的实验结果。

* **可复现的训练过程**：内置随机种子设置功能，确保在相同配置下，每次实验的结果都可以精确复现。

## 3. 项目结构

```
TOC_Warning_System/
├── data/                  # (被.gitignore忽略) 存放原始数据集
├── outputs/               # 存放所有输出结果
│   ├── checkpoints/       # 存放训练好的模型权重 (.pth)
│   └── results/           # 存放CSV预测结果, 训练图表, 混淆矩阵等
├── src/                   # 核心源代码
│   ├── config/
│   │   └── config.yaml    # **项目核心配置文件**
│   ├── data_loader/
│   │   └── data_processor.py
│   ├── model/
│   │   └── model.py
│   ├── train/
│   │   └── trainer.py
│   ├── utils/
│   │   └── helpers.py
│   └── visualization/
│       └── plot.py
├── .gitignore             # Git忽略文件配置
├── main.py                # **项目主入口**
└── README.md              # 项目说明文档
```

## 4. 环境配置与安装

1.  **克隆项目**
    ```bash
    git clone <your-repository-url>
    cd TOC_Warning_System
    ```

2.  **创建并激活Conda环境** (推荐)
    ```bash
    conda create -n TOC_env python=3.8
    conda activate TOC_env
    ```

3.  **安装依赖包**
    ```bash
    pip install torch torchvision torchaudio
    pip install pandas numpy scikit-learn pyyaml optuna matplotlib seaborn
    ```
    *（注意：请根据您的CUDA版本安装对应的PyTorch版本）*

## 5. 使用方法

### 步骤1：准备数据
将您的数据集CSV文件（确保包含 `timestamp` 列和所有特征列）放入 `data/` 文件夹中。

### 步骤2：核心配置
打开 `src/config/config.yaml` 文件，这是控制整个项目的“大脑”。根据您的需求修改以下关键参数：

* `model_name`: 选择您要运行的模型 (`'lstm'`, `'tcn'`, `'tcn_lstm_attention'`)。
* `perform_tuning`: 设置为 `true` 进行Optuna超参数调优；设置为 `false` 则跳过调优，直接使用下方 `model_params` 中定义的参数进行训练。
* `base_features`: 定义用于创建衍生特征的原始输入特征列表。
* `final_model_inputs`: 定义最终要喂给模型的完整特征列表（可包含原始特征和衍生特征）。
* `HRT_values`: 为每个基础特征配置其对应的HRT值。
* 以及其他训练参数，如 `sequence_length`, `num_epochs`, `batch_size` 等。

### 步骤3：运行程序
在项目根目录下，执行以下命令：
```bash
python main.py
```

### 步骤4：查看结果
程序运行结束后，所有产出物都会保存在 `outputs/` 文件夹中：
* **模型权重**: 位于 `outputs/checkpoints/`，文件名根据 `model_name` 动态生成。
* **结果文件**: 位于 `outputs/results/`，包含：
    * Optuna调优报告 (`.csv`)
    * 测试集和全量数据的预测结果 (`.csv`)
    * 损失曲线、预测对比图、误差图、混淆矩阵等 (`.png`)

## 6. 未来工作

* [ ] **模型扩展**: 引入基于Transformer的更前沿的时间序列预测模型（如 Informer, PatchTST）。
* [ ] **概率预测**: 将模型从点预测升级为概率预测，输出预测区间，为风险评估提供不确定性度量。
* [ ] **实验跟踪集成**: 引入 MLflow 或 Weights & Biases，对实验进行更系统化的跟踪、比较和管理。
* [ ] **模型服务化**: 使用 FastAPI 或 Flask 将模型预测逻辑封装成REST API，方便系统集成和实际部署。

## 7. 许可证

本项目采用 [MIT License](LICENSE) 开源许可证。