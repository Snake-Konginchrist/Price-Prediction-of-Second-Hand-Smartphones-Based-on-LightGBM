import pandas as pd
import numpy as np
import pickle
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 数据加载模块
# 使用预处理后的中文数据集
DATA_PATH = 'data/processed_phone_prices_cn.xlsx'

# 读取Excel数据
def load_data():
    """
    加载预处理后的手机价格数据集
    返回特征矩阵X和目标变量Y
    """
    data = pd.read_excel(DATA_PATH)
    X = data.iloc[:, :-1]  # 最后一列是目标变量
    Y = data.iloc[:, -1]   # 当前价格作为预测目标
    return X, Y

# 参数调优模块
def tune_parameters(X_train, y_train):
    """
    使用网格搜索进行超参数优化
    返回最佳参数的XGBoost模型
    """
    # 基础参数设置（控制模型复杂度的核心参数）
    base_params = {
        'n_estimators': 100,     # 决策树的数量
        'learning_rate': 0.1,   # 学习率（步长收缩）
        'max_depth': 5,         # 单棵树最大深度
        'subsample': 0.8,       # 样本采样比例
        'random_state': 42      # 随机种子保证可复现性
    }

    # 网格搜索参数空间（探索不同参数组合）
    param_grid = {
        'max_depth': [3, 5, 7],            # 树深度调节（防止过拟合）
        'learning_rate': [0.01, 0.1, 0.2], # 学习率调节（平衡收敛速度与精度）
        'subsample': [0.6, 0.8, 1.0],      # 行采样比例（增强泛化能力）
        'colsample_bytree': [0.6, 0.8, 1.0] # 列采样比例（特征随机性）
    }

    # 创建网格搜索对象
    grid_search = GridSearchCV(
        estimator=XGBRegressor(**base_params),
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=5,
        n_jobs=-1
    )

    # 执行参数搜索
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

# 模型评估模块
def evaluate_model(model, X_test, y_test):
    """
    评估模型性能并输出关键指标
    包含MAE、MSE、RMSE和百分比误差
    """
    # 生成预测结果
    y_pred = model.predict(X_test)

    # 计算绝对百分比误差（衡量预测值与实际值的相对偏差）
    percent_error = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    # 计算回归指标（绝对值误差、均方误差及其平方根）
    print(f"平均百分比误差: {percent_error:.2f}%")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

# 主流程控制
if __name__ == '__main__':
    # 数据准备
    X, Y = load_data()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # 参数调优
    print("正在进行参数优化...")
    best_model = tune_parameters(X_train, Y_train)
    print(f"最优参数模型: {best_model}")

    # 交叉验证评估（5折交叉验证评估模型稳定性）
    print("\n交叉验证结果:")
    cv_scores = cross_val_score(best_model, X_train, Y_train, 
                              cv=5, scoring='neg_mean_squared_error')
    print(f"MSE平均值: {-cv_scores.mean():.2f} (±{cv_scores.std():.2f})")

    # 最终模型训练
    print("\n训练最终模型...")
    best_model.fit(X_train, Y_train)

    # 测试集评估
    print("\n测试集评估结果:")
    evaluate_model(best_model, X_test, Y_test)

    # 模型持久化
    MODEL_PATH = 'model/xgboost_phone_price-predict.pkl'
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(best_model, f)
    print(f"\n模型已保存至: {MODEL_PATH}")