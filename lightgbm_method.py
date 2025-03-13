import pickle
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 记录程序开始时间
start_time = time.time()

# 加载数据
data = pd.read_excel("data/processed_phone_prices_zh.xlsx", header=0)
X = data.iloc[:, :-1]
Y = data.iloc[:, -1]

# 划分数据集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 定义参数空间
param_space = {
    'boosting_type': ['gbdt', 'dart'],  # 创建一个LightGBM回归模型对象
    'objective': ['regression'],  # 设置回归任务
    'num_leaves': np.arange(10, 200, 5),
    'learning_rate': [0.01],  # np.arange(0.01, 0.1, 0.01),
    'max_depth': np.arange(3, 100),
    'n_estimators': np.arange(50, 5000, 50)
}

# # 创建模型
clf = lgb.LGBMRegressor()

# 使用RandomizedSearchCV进行随机搜索
rand_search = RandomizedSearchCV(estimator=clf,  # 定义使用的模型算法
                                 param_distributions=param_space,  # 定义超参数的空间范围
                                 n_iter=100,  # 定义随机搜索时搜索的超参数组合个数
                                 cv=10,  # 定义交叉验证的折数
                                 random_state=42,  # 定义随机数种子，确保每次随机搜索得到相同的结果
                                 scoring='neg_mean_squared_error',  # 定义评估指标，这里是使用负平均平方误差
                                 verbose=2,  # 控制输出的详细程度，1表示每个候选模型输出一次信息
                                 n_jobs=-1)  # 定义并行运行的进程数，使用所有可用的CPU核心进行训练

# 训练模型并输出最优参数
rand_search.fit(X_train, Y_train)
print(rand_search.best_params_)

# 使用最佳参数拟合模型
best_clf = lgb.LGBMRegressor(**rand_search.best_params_)
best_clf.fit(X_train, Y_train)

# 预测
Y_pred = best_clf.predict(X_test)

# 计算百分比误差
# 确保 Y_test 和 Y_pred 是 numpy 数组并且形状一致
Y_test = np.array(Y_test)
Y_pred = np.array(Y_pred)
percent_error = (Y_pred - Y_test) / Y_test * 100
total_percent_error = np.mean(np.abs(percent_error))
print("Total percent error of the test set:", total_percent_error, "%")

with open('model/LightGBM_phone_price-predict.pkl', 'wb') as f:
    pickle.dump(best_clf, f)

# 将误差写入文件
with open('data/price_prediction_percent_error.txt', 'w') as f:
    result_df = pd.DataFrame({'Original Price': Y_test, 'Predicted Price': Y_pred,
                              'Percentage Error': percent_error})
    f.write(result_df.to_string(col_space=12, justify='left'))

# 计算MAE、MSE和RMSE
mae = mean_absolute_error(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)
# 没有名为 “squared” 的参数，使用 np.sqrt 计算均方误差的平方根
rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
print("MAE(Mean Absolute Error):", mae)
print("MSE(Mean Squared Error):", mse)
print("RMSE(Root Mean Squared Error):", rmse)

# 获取每次迭代的平均测试分数和参数组合
scores = rand_search.cv_results_['mean_test_score']
params = rand_search.cv_results_['params']

# 创建图形窗口并绘制准确率变化曲线
plt.figure(figsize=(12, 6))
plt.plot(range(1, len(scores) + 1), scores, 'ro-')  # 绘制平滑的准确率变化曲线，使用红色圆点
plt.xlabel('Iterations')  # x轴标签，表示迭代次数plt.xlabel('迭代次数')  # x轴标签，表示迭代次数
plt.ylabel('Negative Mean Squared Error')  # y轴标签，表示负均方误差
plt.title('LightGBM Accuracy Change Curve')  # 图形标题，表示准确率随迭代次数的变化曲线
plt.tight_layout()  # 调整图形布局，避免标签重叠
# plt.show()  # 显示图形
plt.savefig('LightGBM Accuracy Change Curve.png')

# 记录程序结束时间
end_time = time.time()

# 计算程序运行时间
run_time = end_time - start_time
print("Program running time: %.2f seconds" % run_time)
