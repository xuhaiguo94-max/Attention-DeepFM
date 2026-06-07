import os
import pandas as pd
import numpy as np
import gc

class CriteoDataset:
    def __init__(self, root, train=True):
        self.root = root
        self.train = train

        if not self._check_exists():
            raise RuntimeError('Dataset not found.')

        filename = 'train.csv' if self.train else 'test.csv'
        print(f"正在加载 {filename}...")
        
        # 全量读取数据
        data = pd.read_csv(os.path.join(self.root, filename))

        # 第 0 列是 target，第 1 列往后是特征
        self.target = data.iloc[:, 0].values.astype(np.float32)
        features = data.iloc[:, 1:].copy()
        
        # 及时清理原始 data 释放几百 MB 内存
        del data
        gc.collect()

        print("正在进行特征向量化处理...")
        # 归一化连续特征
        features.iloc[:, :13] = features.iloc[:, :13].fillna(0.0)
        cont_vals = features.iloc[:, :13].values.astype(np.float32)
        
        min_vals = np.min(cont_vals, axis=0)
        max_vals = np.max(cont_vals, axis=0)
        diff = max_vals - min_vals
        diff[diff == 0] = 1.0 
        cont_vals = (cont_vals - min_vals) / diff
        
        # 类别特征极速编码
        cat_cols = features.columns[13:]
        for col in cat_cols:
            features[col] = features[col].fillna('-1').astype('category').cat.codes.astype(np.int32)

        # ================= 核心优化：直接在内存中构建好 Xi 和 Xv =================
        # 连续特征的 Index 全部补 0
        Xi_cont = np.zeros(cont_vals.shape, dtype=np.int32)
        Xi_cat = features[cat_cols].values.astype(np.int32)
        # 拼接并增加最后一个维度，形状变为 (N, 39, 1)
        self.Xi = np.concatenate([Xi_cont, Xi_cat], axis=1)[..., np.newaxis]

        # 类别特征的 Value 全部补 1
        Xv_cont = cont_vals
        Xv_cat = np.ones(Xi_cat.shape, dtype=np.float32)
        # 拼接并增加最后一个维度，形状变为 (N, 39, 1)
        self.Xv = np.concatenate([Xv_cont, Xv_cat], axis=1)[..., np.newaxis]
        
        # 清理 features 释放内存
        del features
        gc.collect()
        print("数据处理完毕...")

    def _check_exists(self):
        return os.path.exists(self.root)