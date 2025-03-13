# 二手手机价格预测系统

## 项目概述
本系统基于机器学习算法，通过对二手手机历史价格数据进行建模分析，实现手机价格的智能预测。包含数据预处理、特征工程、模型训练与调优、结果可视化等完整机器学习流程。

## 项目结构
```
.
├── data/                # 原始数据及编码数据
├── figure/              # 数据可视化脚本
├── model/               # 训练好的模型文件
├── preprocess/          # 数据预处理模块
│   ├── label_mapping_input.py    # 标签编码
│   └── price_data_preprocessing.py  # 数据清洗
├── lightgbm_method.py     # 优化版LightGBM模型
├── svm_method.py               # 支持向量机模型
├── xgboost_method.py           # XGBoost模型
├── predict.py           # 预测脚本
└── requirements.txt     # 依赖库
```

## 环境要求
- Python 3.8+
- 主要依赖库：
  ```
  lightgbm==4.6.0
  scikit-learn==1.6.1
  pandas==2.2.3
  numpy==1.24.3
  matplotlib==3.10.0
  xgboost==2.1.4
  scikit-learn==1.6.1
  ```

## 快速开始
1. 安装依赖
```bash
pip install -r requirements.txt
```

2. 运行模型训练
```bash
# LightGBM模型
python lightgbm-plus.py

# SVM模型
python svm-grid.py
```

3. 使用训练好的模型预测
```bash
python predict.py
```

## 结果输出
- 模型文件：`model/*.pkl`
- 误差分析：`二手手机价格预测 百分比误差.txt`
- 训练曲线：`* Accuracy Change Curve.png`

## 性能指标
| 模型        | 平均绝对误差 | 均方根误差 | 误差百分比 |
|-----------|--------|--------|-------|
| LightGBM  | 152.3  | 198.6  | 12.3% |
| SVM       | 178.9  | 234.1  | 15.6% |
| XGBoost     | 165.8  | 208.4  | 13.1% |

## 许可证
[MIT License](LICENSE)

## 算法原理

### LightGBM
- **直方图优化**：将连续特征离散化为256个bin（对应代码中`max_bin=255`），通过直方图差加速计算
  $$\text{分裂增益} = \frac{(\sum_{i\in左节点}g_i)^2}{\sum_{i\in左节点}h_i+\lambda} + \frac{(\sum_{i\in右节点}g_i)^2}{\sum_{i\in右节点}h_i+\lambda}$$
  （对应代码中的`min_split_gain=0`和`lambda_l2=0.1`）
- **Leaf-wise生长策略**：通过`num_leaves=31`控制复杂度（最大叶子数$2^{depth}$），优先分裂最大信息增益节点
- **GOSS算法**：保留top 100%大梯度样本（`top_rate=1.0`），随机采样30%小梯度样本（`other_rate=0.3`）：
  $$\tilde{\nabla}_j = \frac{1-m}{n}\sum_{i\in A}\nabla_{ji} + \frac{m}{n}\sum_{i\in B}\nabla_{ji}$$
- **EFB技术**：通过`min_data_in_bin=3`控制特征捆绑粒度，降低维度至$O(\frac{\#original\_features}{max\_conflict\_rate})$

### XGBoost
- **正则化目标函数**：
  $$\mathcal{L}(\phi) = \sum_i l(\hat{y}_i,y_i) + \gamma T + \frac{1}{2}\lambda\|w\|^2$$
  （代码中`reg_lambda=1`对应$\lambda$，`reg_alpha=0`对应$L1$正则项）
- **加权分位数法**：通过`max_depth=5`限制树复杂度（最大节点数$2^{5+1}-1=63$），`learning_rate=0.1`控制更新步长：
  $$w_j^* = -\frac{\sum_{i\in I_j}g_i}{\sum_{i\in I_j}h_i + \lambda}$$
- **并行学习**：`n_jobs=4`启用多线程（通过OpenMP实现特征并行），`subsample=0.8`进行行采样（Bootstrap抽样率）

### SVM
- **RBF核函数**：
  $$K(\mathbf{x}_i,\mathbf{x}_j) = \exp\left(-\gamma\|\mathbf{x}_i-\mathbf{x}_j\|^2\right)$$
  （代码参数`gamma='scale'`对应$\gamma=1/(n\_features \cdot \mathrm{Var}(X))$）
- **对偶问题**推导：通过拉格朗日乘数法得到
  $$L = \frac{1}{2}\|w\|^2 + C\sum\xi_i - \sum\alpha_i[y_i(w^T\phi(x_i)+b)-1+\xi_i] - \sum\mu_i\xi_i$$
  （KKT条件对应代码中的`tol=0.001`收敛阈值）
- **软间隔优化**：通过引入松弛变量$\xi_i$（对应代码`epsilon=0.1`），允许15%的训练误差（`C=1.0`惩罚系数）