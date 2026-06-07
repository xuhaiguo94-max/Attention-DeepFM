import tensorflow as tf
from tensorflow.keras import layers, Model
import time

class DeepFM(Model):
    def __init__(self, feature_sizes, embedding_size=4, hidden_dims=[32, 32], dropout=[0.5, 0.5], **kwargs):
        super(DeepFM, self).__init__(**kwargs)
        self.feature_sizes = feature_sizes
        self.embedding_size = embedding_size
        
        self.bias_weight = tf.Variable(tf.random.normal([1]), trainable=True)
        
        self.fm_first_order_embeddings = [layers.Embedding(input_dim=feat_size, output_dim=1) for feat_size in self.feature_sizes]
        self.fm_second_order_embeddings = [layers.Embedding(input_dim=feat_size, output_dim=self.embedding_size) for feat_size in self.feature_sizes]
        
        self.deep_layers = []
        for i, dim in enumerate(hidden_dims):
            self.deep_layers.append({
                'linear': layers.Dense(dim),
                'batch_norm': layers.BatchNormalization(),
                'dropout': layers.Dropout(dropout[i])
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

        # 第二阶
        fm_second_order_emb_arr = []
        for i, emb in enumerate(self.fm_second_order_embeddings):
            xi_slice = Xi[:, i, 0]
            emb_out = tf.reduce_sum(emb(xi_slice), axis=1) if len(emb(xi_slice).shape) > 2 else emb(xi_slice)
            fm_second_order_emb_arr.append(emb_out * Xv[:, i, :])

        fm_sum_second_order_emb = tf.add_n(fm_second_order_emb_arr)
        fm_sum_second_order_emb_square = tf.square(fm_sum_second_order_emb)
        fm_second_order_emb_square_sum = tf.add_n([tf.square(item) for item in fm_second_order_emb_arr])
        fm_second_order = (fm_sum_second_order_emb_square - fm_second_order_emb_square_sum) * 0.5
        
        # Deep 部分
        deep_out = tf.concat(fm_second_order_emb_arr, axis=1)
        for layer_dict in self.deep_layers:
            deep_out = layer_dict['linear'](deep_out)
            deep_out = layer_dict['batch_norm'](deep_out, training=training)
            deep_out = layer_dict['dropout'](deep_out, training=training)
            
        total_sum = tf.reduce_sum(fm_first_order, axis=1) + tf.reduce_sum(fm_second_order, axis=1) + tf.reduce_sum(deep_out, axis=1) + self.bias_weight[0]
        return total_sum

    def fit_model(self, loader_train, loader_val, optimizer, epochs=5):
        criterion = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        # 新增 train_acc 和 val_acc 的记录
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

            # 打印中加入准确率 acc
            time_taken = time.time() - start_time
            print(f"Epoch {epoch+1}/{epochs} [{time_taken:.1f}s] - "
                  f"loss: {train_loss:.4f} - auc: {train_auc:.4f} - acc: {train_acc:.4f} | "
                  f"val_loss: {val_loss:.4f} - val_auc: {val_auc:.4f} - val_acc: {val_acc:.4f}")
            
        return history