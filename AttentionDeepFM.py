# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 17:11:13 2026

@author: 徐海果
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
import time

class AttentionDeepFM(Model):
    def __init__(self, feature_sizes, embedding_size=4, hidden_dims=[32, 32], dropout=[0.5, 0.5], num_heads=2, **kwargs):
        super(AttentionDeepFM, self).__init__(**kwargs)
        self.feature_sizes = feature_sizes
        self.embedding_size = embedding_size
        
        self.bias_weight = tf.Variable(tf.random.normal([1]), trainable=True)
        
        self.fm_first_order_embeddings = [layers.Embedding(input_dim=feat_size, output_dim=1) for feat_size in self.feature_sizes]
        self.fm_second_order_embeddings = [layers.Embedding(input_dim=feat_size, output_dim=self.embedding_size) for feat_size in self.feature_sizes]
        
        # ========== 引入多头自注意力层 ==========
        self.attention_layer = layers.MultiHeadAttention(num_heads=num_heads, key_dim=self.embedding_size)
        self.attention_norm = layers.LayerNormalization()
        self.attention_dropout = layers.Dropout(0.5)

        self.deep_layers = []
        for i, dim in enumerate(hidden_dims):
            self.deep_layers.append({
                'linear': layers.Dense(dim),
                'batch_norm': layers.BatchNormalization(),
                'dropout': layers.Dropout(dropout[i]),
                'activation': layers.ReLU() # 修复了原代码中缺失的非线性激活
            })

    def call(self, inputs, training=False):
        Xi, Xv = inputs

        # 第一阶 
        fm_first_order_emb_arr = []
        for i, emb in enumerate(self.fm_first_order_embeddings):
            xi_slice = Xi[:, i, 0] 
            emb_out = tf.reduce_sum(emb(xi_slice), axis=1) if len(emb(xi_slice).shape) > 2 else emb(xi_slice)
            fm_first_order_emb_arr.append(emb_out * Xv[:, i, :])
        fm_first_order = tf.concat(fm_first_order_emb_arr, axis=1)

        # 第二阶 Embedding 提取
        fm_second_order_emb_arr = []
        for i, emb in enumerate(self.fm_second_order_embeddings):
            xi_slice = Xi[:, i, 0]
            emb_out = tf.reduce_sum(emb(xi_slice), axis=1) if len(emb(xi_slice).shape) > 2 else emb(xi_slice)
            fm_second_order_emb_arr.append(emb_out * Xv[:, i, :])

        # FM 组件交叉
        fm_sum_second_order_emb = tf.add_n(fm_second_order_emb_arr)
        fm_sum_second_order_emb_square = tf.square(fm_sum_second_order_emb)
        fm_second_order_emb_square_sum = tf.add_n([tf.square(item) for item in fm_second_order_emb_arr])
        fm_second_order = (fm_sum_second_order_emb_square - fm_second_order_emb_square_sum) * 0.5
        
        # ========== 注意力网络拓扑 ==========
        # 1. 将 list 堆叠为三维张量: (Batch, Num_Fields, Embedding_Size)
        emb_stack = tf.stack(fm_second_order_emb_arr, axis=1)
        
        # 2. 自注意力计算，让特征之间相互“关注”
        attn_out = self.attention_layer(emb_stack, emb_stack, training=training)
        attn_out = self.attention_dropout(attn_out, training=training)
        
        # 3. 残差连接 (Residual Connection) 与归一化
        deep_in = self.attention_norm(emb_stack + attn_out)
        
        # 4. 展平张量，送入下游 DNN
        deep_out = tf.keras.layers.Flatten()(deep_in)
        # ====================================================

        # Deep 部分前向传播
        for layer_dict in self.deep_layers:
            deep_out = layer_dict['linear'](deep_out)
            deep_out = layer_dict['batch_norm'](deep_out, training=training)
            deep_out = layer_dict['activation'](deep_out) # 应用非线性
            deep_out = layer_dict['dropout'](deep_out, training=training)
            
        total_sum = tf.reduce_sum(fm_first_order, axis=1) + tf.reduce_sum(fm_second_order, axis=1) + tf.reduce_sum(deep_out, axis=1) + self.bias_weight[0]
        return total_sum

    # fit_model 函数
    def fit_model(self, loader_train, loader_val, optimizer, epochs=5):
        criterion = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        history = {'train_loss': [], 'val_loss': [], 'train_auc': [], 'val_auc': [], 'train_acc': [], 'val_acc': []}

        for epoch in range(epochs):
            start_time = time.time()
            train_loss_sum, train_steps = 0.0, 0
            train_auc_metric = tf.keras.metrics.AUC(from_logits=True)
            # 因为模型输出的是 Logits，大于0代表大于50%概率，所以 threshold 设为 0.0
            train_acc_metric = tf.keras.metrics.BinaryAccuracy(threshold=0.0)

            # 训练阶段
            for xi, xv, y in loader_train:
                with tf.GradientTape() as tape:
                    total = self((xi, xv), training=True)
                    loss = criterion(y, total)
                
                gradients = tape.gradient(loss, self.trainable_variables)
                optimizer.apply_gradients(zip(gradients, self.trainable_variables))

                train_loss_sum += loss.numpy()
                train_steps += 1
                train_auc_metric.update_state(y, total)
                train_acc_metric.update_state(y, total) # 记录准确率

            train_loss = train_loss_sum / train_steps
            train_auc = train_auc_metric.result().numpy()
            train_acc = train_acc_metric.result().numpy()

            # 验证阶段
            val_loss_sum, val_steps = 0.0, 0
            val_auc_metric = tf.keras.metrics.AUC(from_logits=True)
            val_acc_metric = tf.keras.metrics.BinaryAccuracy(threshold=0.0)
            
            for xi, xv, y in loader_val:
                total = self((xi, xv), training=False)
                loss = criterion(y, total)
                val_loss_sum += loss.numpy()
                val_steps += 1
                val_auc_metric.update_state(y, total)
                val_acc_metric.update_state(y, total)

            val_loss = val_loss_sum / val_steps
            val_auc = val_auc_metric.result().numpy()
            val_acc = val_acc_metric.result().numpy()

            history['train_loss'].append(train_loss)
            history['train_auc'].append(train_auc)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_auc'].append(val_auc)
            history['val_acc'].append(val_acc)
            time_taken = time.time() - start_time
            print(f"Epoch {epoch+1}/{epochs} [{time_taken:.1f}s] - "
                  f"loss: {train_loss:.4f} - auc: {train_auc:.4f} - acc: {train_acc:.4f} | "
                  f"val_loss: {val_loss:.4f} - val_auc: {val_auc:.4f} - val_acc: {val_acc:.4f}")
            
        return history
