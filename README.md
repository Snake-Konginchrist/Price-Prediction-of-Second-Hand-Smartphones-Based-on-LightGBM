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


## 算法原理

### LightGBM
LightGBM（Light Gradient Boosting Machine）是一种基于决策树的梯度提升框架，专为分布式和高效处理大数据集而设计。以下是LightGBM的一些关键特性：

- **梯度提升框架**：LightGBM基于梯度提升决策树（GBDT）算法，其目标函数可以表示为：
  $$\mathcal{L}(\Theta) = \sum_{i=1}^n l(y_i, \hat{y}_i) + \sum_{k=1}^K \Omega(f_k)$$
  其中，$l(y_i, \hat{y}_i)$ 是损失函数，$\Omega(f_k)$ 是第 $k$ 棵树的复杂度惩罚项，$\hat{y}_i = \sum_{k=1}^K f_k(x_i)$ 是模型预测值。

- **直方图算法**：LightGBM使用直方图算法来加速决策树的构建。它通过将连续特征值离散化为有限个bin来减少计算量。每个特征的值被分配到一个bin中，然后在这些bin上进行分裂点的选择。这样可以显著减少计算量和内存使用。具体来说，直方图算法通过将特征值分桶，计算每个bin的梯度和Hessian和，从而快速找到最佳分裂点。分裂增益计算公式为：
  $$\text{Gain} = \frac{1}{2} \left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda} \right] - \gamma$$
  其中，$G_L$ 和 $G_R$ 分别是左右子节点的梯度和，$H_L$ 和 $H_R$ 分别是左右子节点的Hessian和，$\lambda$ 是L2正则化项，$\gamma$ 是复杂度惩罚项。

- **Leaf-wise生长策略**：与其他基于深度的生长策略不同，LightGBM采用leaf-wise的生长策略。它选择具有最大增益的叶子节点进行分裂，而不是层次生长。这种方法可以更好地减少误差，但也可能导致过拟合，因此需要通过设置`num_leaves`和`max_depth`等参数来控制树的复杂度。Leaf-wise生长策略的优点是可以更好地拟合数据，特别是在数据集较大时表现出色。叶子节点的分裂选择基于以下公式：
  $$\text{leaf}^* = \arg\max_{\text{leaf}} \text{Gain}(\text{leaf})$$

- **GOSS算法**：Gradient-based One-Side Sampling (GOSS) 是LightGBM的一种采样方法。GOSS通过保留大梯度的样本和随机采样小梯度的样本来加速训练过程。通过这种方法，LightGBM可以在不显著降低精度的情况下减少计算量。GOSS的数学表达为：
  $$\tilde{\nabla} = \sum_{i \in A} \nabla_i + \frac{1-a}{b} \sum_{i \in B} \nabla_i$$
  其中，$A$ 是大梯度样本集合，$B$ 是随机采样的小梯度样本集合，$a$ 是保留大梯度样本的比例，$b$ 是随机采样小梯度样本的比例。

- **EFB技术**：Exclusive Feature Bundling (EFB) 是一种特征捆绑技术，用于减少特征的维度，特别适用于稀疏特征。EFB通过将互斥的特征捆绑在一起，减少了特征的数量，从而提高了计算效率。两个特征的冲突度量可以表示为：
  $$\text{conflict}(i, j) = \frac{|\{k | x_{ki} \neq 0 \text{ and } x_{kj} \neq 0\}|}{|\{k | x_{ki} \neq 0 \text{ or } x_{kj} \neq 0\}|}$$
  其中，$x_{ki}$ 表示第 $k$ 个样本的第 $i$ 个特征值。

### XGBoost
XGBoost（eXtreme Gradient Boosting）是一个高效且灵活的梯度提升框架，广泛应用于机器学习竞赛和实际应用中。以下是XGBoost的一些关键特性：

- **Boosting算法**：XGBoost使用梯度提升算法，通过逐步构建一系列弱学习器（通常是决策树）来提高模型的预测性能。每个新树的构建是为了纠正之前所有树的错误。XGBoost通过加权的方式来组合多个弱学习器的预测结果，从而提高整体模型的准确性。模型的预测值可以表示为：
  $$\hat{y}_i = \sum_{k=1}^K f_k(x_i)$$
  其中，$f_k$ 是第 $k$ 棵树，$K$ 是树的总数。

- **目标函数**：XGBoost的目标函数包含损失函数和正则化项：
  $$\mathcal{L}(\phi) = \sum_{i=1}^n l(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)) + \Omega(f_t)$$
  其中，$l$ 是损失函数，$\hat{y}_i^{(t-1)}$ 是前 $t-1$ 棵树的预测值，$f_t$ 是第 $t$ 棵树，$\Omega$ 是正则化项。

- **正则化**：XGBoost在目标函数中引入了L1和L2正则化项，以防止过拟合。正则化项通过惩罚复杂模型来提高模型的泛化能力。L1正则化可以产生稀疏模型，而L2正则化则有助于防止过拟合。正则化项的表达式为：
  $$\Omega(f) = \gamma T + \frac{1}{2} \lambda \sum_{j=1}^T w_j^2 + \alpha \sum_{j=1}^T |w_j|$$
  其中，$T$ 是叶子节点的数量，$w_j$ 是第 $j$ 个叶子节点的权重，$\gamma$、$\lambda$ 和 $\alpha$ 分别是复杂度惩罚系数、L2正则化系数和L1正则化系数。

- **二阶泰勒展开**：XGBoost使用二阶泰勒展开来近似目标函数，从而加速优化过程：
  $$\mathcal{L}^{(t)} \approx \sum_{i=1}^n [l(y_i, \hat{y}_i^{(t-1)}) + g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i)] + \Omega(f_t)$$
  其中，$g_i = \partial_{\hat{y}_i^{(t-1)}} l(y_i, \hat{y}_i^{(t-1)})$ 是一阶导数（梯度），$h_i = \partial_{\hat{y}_i^{(t-1)}}^2 l(y_i, \hat{y}_i^{(t-1)})$ 是二阶导数（Hessian）。

- **分裂增益**：XGBoost在选择分裂点时，使用以下公式计算增益：
  $$\text{Gain} = \frac{1}{2} \left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda} \right] - \gamma$$
  其中，$G_L$ 和 $G_R$ 分别是左右子节点的梯度和，$H_L$ 和 $H_R$ 分别是左右子节点的Hessian和。

- **缺失值处理**：XGBoost能够自动处理缺失值，通过学习缺失值应该被分到左子树还是右子树来最大化增益。对于每个分裂点，XGBoost会尝试将缺失值分到左子树和右子树，然后选择增益最大的方向。

### SVM
支持向量机（Support Vector Machine, SVM）是一种用于分类和回归的监督学习模型，特别适用于高维数据。以下是SVM的一些关键特性：

- **基本原理**：SVM的目标是找到一个最佳的超平面，将不同类别的样本分开。对于线性可分的数据，SVM通过最大化边界（即支持向量到超平面的距离）来找到这个超平面。

- **线性SVM**：对于线性可分的数据，SVM通过以下优化问题来找到超平面：
  $$\min_{w,b} \frac{1}{2} \|w\|^2$$
  $$\text{subject to } y_i(w^T x_i + b) \geq 1, \forall i$$
  其中，$w$是超平面的法向量，$b$是偏置项。

- **软间隔SVM**：对于线性不可分的数据，SVM引入松弛变量 \(\xi_i\) 来允许一些误分类：
  $$\min_{w,b,\xi} \frac{1}{2} \|w\|^2 + C \sum \xi_i$$
  $$\text{subject to } y_i(w^T x_i + b) \geq 1 - \xi_i, \xi_i \geq 0, \forall i$$
  其中，\(C\) 是惩罚系数，控制误分类的惩罚程度。

- **核方法**：SVM通过核函数将数据映射到高维空间，以处理非线性可分的数据。常用的核函数包括：
  - **线性核**：$K(x_i, x_j) = x_i^T x_j$
  - **多项式核**：$K(x_i, x_j) = (\gamma x_i^T x_j + r)^d$
  - **RBF核（径向基核）**：$K(x_i, x_j) = \exp(-\gamma \|x_i - x_j\|^2)$
  - **Sigmoid核**：$K(x_i, x_j) = \tanh(\gamma x_i^T x_j + r)$

- **对偶问题**：通过拉格朗日乘数法，SVM的优化问题可以转化为对偶问题：
  $$\max_{\alpha} \sum \alpha_i - \frac{1}{2} \sum \sum \alpha_i \alpha_j y_i y_j K(x_i, x_j)$$
  $$\text{subject to } \sum \alpha_i y_i = 0, 0 \leq \alpha_i \leq C, \forall i$$
  其中，$\alpha_i$是拉格朗日乘子。

- **支持向量**：在对偶问题的解中，$\alpha_i > 0$的样本称为支持向量，这些样本对超平面的确定起关键作用。

## 交流群
欢迎加入我们的开源交流QQ群：1022820973，与更多开发者交流学习！

## 许可证
[MIT License](LICENSE)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Snake-Konginchrist/Price-Prediction-of-Second-Hand-Smartphones-Based-on-LightGBM&type=Date)](https://www.star-history.com/#Snake-Konginchrist/Price-Prediction-of-Second-Hand-Smartphones-Based-on-LightGBM&Date)