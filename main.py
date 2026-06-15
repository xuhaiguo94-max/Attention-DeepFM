import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from model.AttentionDeepFM import AttentionDeepFM
from data.dataset import CriteoDataset

# ============ 显存防爆设置 ============
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# ============ 数据加载 ============
batch_size = 2048 
dataset = CriteoDataset('./data', train=True)

Xi, Xv, y = dataset.Xi, dataset.Xv, dataset.target

total_samples = len(y)

num_train = int(total_samples * 0.8)

# 划分训练集和验证集 
Xi_train, Xv_train, y_train = Xi[:num_train], Xv[:num_train], y[:num_train]
Xi_val, Xv_val, y_val = Xi[num_train:], Xv[num_train:], y[num_train:]

# 构建 TensorFlow 数据管道 
loader_train = tf.data.Dataset.from_tensor_slices((Xi_train, Xv_train, y_train)).shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
loader_val = tf.data.Dataset.from_tensor_slices((Xi_val, Xv_val, y_val)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# ============ 计算特征维度 ============

feature_sizes = [int(np.max(Xi[:, i, 0])) + 1 for i in range(Xi.shape[1])]

# ============ 实例化与训练 ============
model = AttentionDeepFM(feature_sizes)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

print("开始正式训练")
history = model.fit_model(loader_train, loader_val, optimizer, epochs=20)

# ============  结果可视化 ============
#print("正在绘制并保存 Loss 和 AUC 曲线...")
plt.figure(figsize=(12, 5))

# 绘制 Loss 曲线
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss', marker='o')
plt.plot(history['val_loss'], label='Val Loss', marker='o')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# 绘制 AUC 曲线
plt.subplot(1, 2, 2)
plt.plot(history['train_auc'], label='Train AUC', marker='o')
plt.plot(history['val_auc'], label='Val AUC', marker='o')
plt.title('Model AUC')
plt.xlabel('Epochs')
plt.ylabel('AUC')
plt.legend()
plt.grid(True)

plt.tight_layout()
#plt.savefig('AttentionDeepFM_training_curves.png') # 图片会保存在你代码运行的根目录下
#print("训练完成！曲线图已保存为 AttentionDeepFM_training_curves.png，同时显示如下：")
plt.show()
