import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# 记录程序开始时间
start_time = time.time()

# 加载数据
data = pd.read_excel("processed_phone_prices_cn.xlsx", header=0)
X = data.iloc[:, :-1]
Y = data.iloc[:, -1]

# 划分数据集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 定义参数空间
param_space = {
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],  # SVM模型的核函数类型
    'C': np.logspace(-1, 1, 3),  # SVM模型的正则化参数C的取值范围，以对数刻度生成
    'gamma': ['scale', 'auto'],  # 核函数的gamma参数取值
    'epsilon': np.logspace(-1, 1, 3),  # SVM模型的容忍度参数epsilon的取值范围，以对数刻度生成
}

# 创建模型
clf = SVR()

# 使用RandomizedSearchCV进行随机搜索
rand_search = RandomizedSearchCV(
    estimator=clf,  # 要进行参数搜索的模型估计器（例如SVR）
    param_distributions=param_space,  # 参数空间的分布或列表，用于进行随机搜索
    n_iter=50,  # 搜索的参数组合数量（随机搜索的迭代次数）
    cv=10,  # 交叉验证的折数（将数据集划分为多少个部分进行训练和评估）
    random_state=42,  # 随机数种子，确保每次随机搜索得到相同的结果
    scoring='r2',  # 评估指标，用于衡量模型的性能（这里使用R2得分）
    verbose=2,  # 控制输出的详细程度，值为2表示每个候选模型输出一次信息
    n_jobs=-1  # 并行运行的进程数，-1表示使用所有可用的CPU核心进行训练
)

# 训练模型并输出最优参数
rand_search.fit(X_train, Y_train)
print(rand_search.best_params_)

# 使用最佳参数拟合模型
best_clf = SVR(**rand_search.best_params_)
best_clf.fit(X_train, Y_train)

# 预测
Y_pred = best_clf.predict(X_test)

# 重定向标准输出流到文件
# output_file = 'SVM_Predict_output.txt'
# sys.stdout = open(output_file, 'w')

# 计算百分比误差
percent_error = (Y_pred - Y_test) / Y_test * 100
total_percent_error = np.mean(np.abs(percent_error))
print("Total percent error of the test set:", total_percent_error, "%")

# 计算MAE、MSE和RMSE
mae = mean_absolute_error(Y_test, Y_pred)  # 计算平均绝对误差
mse = mean_squared_error(Y_test, Y_pred)  # 计算均方误差
rmse = np.sqrt(mse)

print("MAE (Mean Absolute Error):", mae)
print("MSE (Mean Squared Error):", mse)
print("RMSE (Root Mean Squared Error):", rmse)

# 计算R2得分
r2 = r2_score(Y_test, Y_pred)
print("R2 score of the test set:", r2)

# 绘制准确率变化曲线
scores = rand_search.cv_results_['mean_test_score']
params = rand_search.cv_results_['params']
n_iterations = range(1, len(scores) + 1)
params_str = [str(param) for param in params]

# 输出每次迭代的参数组合和得分
for iteration, param, score in zip(n_iterations, params, scores):
    print(f"Iteration {iteration}:")
    print("Parameters:", param)
    print("R2 Score:", score)

plt.figure(figsize=(12, 6))  # 创建一个图形窗口，指定图形的尺寸为宽度12，高度6
# plt.plot(params_str, scores, 'bo-')  # 绘制准确率变化曲线，使用蓝色圆点连接的线条
# plt.xticks(rotation=45)  # 旋转x轴刻度标签，使其更好地展示
plt.plot(n_iterations, scores, 'bo-')
plt.xlabel('Iterations')
# plt.xlabel('Parameter Combinations')  # x轴标签，表示参数组合
plt.ylabel('R2 Score')  # y轴标签，表示R2得分
plt.title('SVM Accuracy Change Curve')  # 图形标题，表示准确率变化曲线
plt.tight_layout()  # 调整图形布局，避免标签重叠
# plt.show()  # 显示图形
plt.savefig('SVM Accuracy Change Curve.png')

# 保存模型
with open('SVM_phone_price-predict.pkl', 'wb') as f:
    pickle.dump(best_clf, f)

# 记录程序结束时间
end_time = time.time()

# 计算程序运行时间
run_time = end_time - start_time
print("程序运行时间：%.2f秒" % run_time)
