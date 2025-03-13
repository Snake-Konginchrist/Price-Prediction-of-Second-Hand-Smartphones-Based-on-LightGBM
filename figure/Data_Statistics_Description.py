import pandas as pd

# 读取Excel文件
data = pd.read_excel("同城帮 二手手机价格数据（编码后）_英文.xlsx")

# 提取RAM、ROM、Original_Price和Current_Price这四列数据
columns_of_interest = ['RAM', 'ROM', 'Original_Price', 'Current_Price']
selected_data = data[columns_of_interest]

# 统计数据
statistics = selected_data.describe()

# 打印统计结果
print(statistics)
print(statistics.transpose())
