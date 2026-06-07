# A-DeepFM: Attention-Enhanced Deep Factorization Machine

![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue)
![Deep Learning Framework](https://img.shields.io/badge/Framework-PyTorch%20%7C%20TensorFlow-orange)
![License](https://img.shields.io/badge/License-MIT-green)

本仓库提供了一种改进的点击率（CTR）预测模型：**A-DeepFM (Attention-DeepFM)**。
该模型在传统 DeepFM 的基础上，创新性地引入了**多头自注意力机制（Multi-Head Self-Attention）**以及**残差与归一化模块（Add & Norm）**，有效提升了模型在极度稀疏场景下的高阶特征交叉挖掘能力与深层网络收敛稳定性。

---

## 核心创新点 (Key Contributions)

1. **动态路由过滤 (Dynamic Attention Routing)**
   放弃了传统 DeepFM 中对隐向量的机械式等权拼接，在右路 DNN 入口处引入 Multi-Head Attention。通过动态评估上下文语境，放大核心业务特征，并对无效的冗余噪声特征进行“静音”过滤。
2. **训练稳定器 (Add & Norm 机制)**
   遵循 $Z = \text{LayerNorm}(E + \text{Dropout}(\text{MultiHead}(E, E, E)))$ 的范式：
   * **Add (残差连接)**：构建“梯度高速公路”，保证底层特征无损穿透，解决深层网络的梯度弥散问题。
   * **Norm (层归一化)**：消除内部协变量偏移（Internal Covariate Shift），平滑优化地形，极大提升面对长尾冷启动数据时的收敛鲁棒性。
3. **二阶张量计算降维**
   将繁冗的二阶嵌套求和等价转化为“和的平方”与“平方的和”之差，将计算复杂度由 $O(k \cdot d^2)$ 压缩至 $O(k \cdot d)$。

---

## 模型全局架构 (Architecture)

> 

- **FM 一阶部分**：提取特征的线性加权重要性。
- **FM 二阶部分**：通过隐向量两两内积提取显式特征交叉。
- **A-DNN 高阶部分**：经注意力机制提纯后的张量，馈入多层前馈网络挖掘隐式非线性拓扑。
  三路信号最终在顶层被等权累加，并通过 Sigmoid 映射为预测概率。

---

## 实验环境与依赖 (Environment Setup)

本项目在以下软硬件环境中经过严谨测试：

- **OS**: Ubuntu 20.04 / Windows 10
- **CPU**: Intel Core i7 / AMD Ryzen 7
- **GPU**: NVIDIA RTX 4060
- **RAM**: 16 GB+

**快速安装依赖：**

```bash
# 克隆仓库
git clone [https://github.com/[xuhaiguo94-max](https://github.com/xuhaiguo94-max)/Attention-DeepFM.git](https://github.com/YourUsername/Attention-DeepFM.git)
cd Attention-DeepFM

# 安装依赖
pip install -r requirements.txt
```

