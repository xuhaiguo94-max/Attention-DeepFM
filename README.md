# CTR Prediction: DeepFM vs AttentionDeepFM

本项目实现并对比了两种用于广告点击率（CTR）预测的深度学习模型：经典的 **DeepFM** 与在原基础上进行创新升级的 **AttentionDeepFM**。模型均基于 TensorFlow 2 构建，并在真实的 Criteo 数据集上进行训练、评估与对比验证。

---

## 项目结构

* `main.py`
    主程序入口。负责 GPU 显存防爆设置、加载 Criteo 数据、构建高效的 TensorFlow 数据管道、实例化模型并执行训练流程，最终进行结果（Loss 和 AUC 曲线）的可视化与保存。
* `AttentionDeepFM.py`
    增强版模型架构。在 DeepFM 的基础上引入了多头自注意力机制（Multi-Head Self-Attention），使得模型能够显式地捕捉不同特征空间之间的高阶交互关系。
* `DeepFM.py`
    原版模型架构。实现了经典的 DeepFM 算法，包含一阶线性特征交叉、二阶 FM 特征交叉以及用于隐式高阶特征学习的深度神经网络（DNN）。可作为 Baseline 模型进行对比参考。
* `dataset.py`
    数据预处理模块。包含 `CriteoDataset` 类，负责从 CSV 文件加载数据、全量读取与内存回收、执行连续特征归一化（Min-Max）以及类别特征极速编码，并将其转化为模型需要的索引和数值张量格式。

---

## 模型对比与核心创新

本项目提供了两种网络拓扑架构的完整实现，主要区别在于特征送入深度神经网络（DNN）之前的处理方式：

### 1. 经典 DeepFM (Baseline)
在 `DeepFM.py` 中，模型主要由两部分并行组成：
* **FM 组件**：负责一阶特征的线性组合与二阶特征交叉。
* **Deep 组件**：将二阶 Embedding 特征直接拼接（Concat），送入多层全连接网络，以隐式方式学习高阶非线性特征交互。

### 2. 增强版 AttentionDeepFM
在 `AttentionDeepFM.py` 中，为了解决传统 DNN 提取高阶特征时缺乏针对性交互的问题，引入了**多头自注意力网络拓扑**：
* **特征堆叠**：将所有特征的 Embedding 向量堆叠为三维张量。
* **多头自注意力计算**：在送入 DNN 之前，先让不同领域的特征（Fields）通过自注意力机制相互“关注”，显式计算并赋予不同特征组合不同的权重关联。
* **残差连接与归一化**：加入 Dropout、Layer Normalization 以及残差连接，保证梯度稳定并加速模型收敛。
* **深度前向传播**：将加权融合后的特征展平，再送入下游 DNN 进行最终的点击率预测。

通过这种对比设计，可以直观地验证引入自注意力机制后，模型在特征提取效率和最终 AUC 指标上的性能差异。

---

## 环境依赖

运行本项目需要以下核心环境：

* `tensorflow` (建议版本 >= 2.0)
* `numpy`
* `pandas`
* `matplotlib`

---

## 数据集准备

本项目采用 **Criteo** 广告点击预测数据集。请在项目根目录下建立 `data` 文件夹，并放入训练与测试数据：

```text
/项目根目录
├── data/
│   ├── train.csv
│   └── test.csv
├── main.py
├── AttentionDeepFM.py
├── DeepFM.py
└── dataset.py
