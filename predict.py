import pickle
import numpy as np

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 使用 Pandas 库的 ExcelFile 函数打开一个 Excel 文件
xls = pd.ExcelFile('data/feature_mapping_table.xlsx')
label_map = {}

for sheet_name in xls.sheet_names:
    # 读取编码-标签映射关系
    label_mapping = pd.read_excel('data/feature_mapping_table.xlsx', sheet_name=sheet_name, header=0)

    # 获取编码列和标签列
    code_col = label_mapping.iloc[:, 0]
    label_col = label_mapping.iloc[:, 1]

    # 将编码-标签映射关系保存到label_map中
    label_map[sheet_name] = dict(zip(code_col, label_col))

# 读取保存的模型
with open('model/LightGBM_phone_price-predict.pkl', 'rb') as f:
    model = pickle.load(f)

data = pd.read_excel('data/processed_phone_prices_zh.xlsx')
X = data.iloc[:, :-1]
Y = data.iloc[:, -1]
Y_pred = model.predict(X)

# 计算百分比误差
percent_error = (Y_pred - Y) / Y * 100
total_percent_error = np.mean(np.abs(percent_error))
print("Total percent error of the test set:", total_percent_error, "%")

# 计算MAE、MSE和RMSE
mae = mean_absolute_error(Y, Y_pred)
mse = mean_squared_error(Y, Y_pred)
# 修改为手动计算RMSE
rmse = np.sqrt(mean_squared_error(Y, Y_pred))
print("MAE(Mean Absolute Error):", mae)
print("MSE(Mean Squared Error):", mse)
print("RMSE(Root Mean Squared Error):", rmse)
