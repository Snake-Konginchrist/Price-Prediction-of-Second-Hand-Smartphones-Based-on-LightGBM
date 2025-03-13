import pandas as pd
import matplotlib.pyplot as plt

# 读取Excel文件
df = pd.read_excel('同城帮 二手手机价格数据（编码后）_英文.xlsx')

# 创建2x2的子图
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# 绘制RAM的分布图
axes[0, 0].hist(df['RAM'])
axes[0, 0].set_title('RAM Distribution')

# 绘制ROM的分布图
axes[0, 1].hist(df['ROM'])
axes[0, 1].set_title('ROM Distribution')

# 绘制OriginPrice的分布图
axes[1, 0].hist(df['Original_Price'])
axes[1, 0].set_title('Original Price Distribution')

# 绘制CurrentPrice的分布图
axes[1, 1].hist(df['Current_Price'])
axes[1, 1].set_title('Current Price Distribution')

# 调整子图之间的间距
plt.tight_layout()

# 显示图形
plt.show()
plt.savefig("Data Histogram.png")
