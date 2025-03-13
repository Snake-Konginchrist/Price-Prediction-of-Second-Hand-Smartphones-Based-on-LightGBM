import pandas as pd
import matplotlib.pyplot as plt

# 读取Excel文件
data = pd.read_excel("同城帮 二手手机价格数据（编码后）_英文.xlsx")

# 计算相关性矩阵
correlation_matrix = data.corr()

# 绘制相关性热力图
plt.figure(figsize=(12, 8))
plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='nearest')
plt.colorbar()
plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
plt.title("Correlation Matrix")
# 保存图片
# plt.savefig('correlation_matrix.png')

plt.show()
