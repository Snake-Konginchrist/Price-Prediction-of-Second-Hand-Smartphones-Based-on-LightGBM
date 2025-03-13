import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# 读取Excel表格
df = pd.read_excel('同城帮 二手手机价格数据.xlsx')

# 将品牌、型号、颜色这三列选取出来，保存到一个新的DataFrame中
brand_df = df['品牌'].to_frame()
model_df = df['型号'].to_frame()
color_df = df['颜色'].to_frame()

# 使用标签编码对品牌、型号、颜色这三列进行转换
label_encoder = LabelEncoder()
brand_encoded = label_encoder.fit_transform(brand_df['品牌'])
model_encoded = label_encoder.fit_transform(model_df['型号'])
color_encoded = label_encoder.fit_transform(color_df['颜色'])

# 将编码结果添加到原来的DataFrame中
df_encoded = df.copy()
df_encoded['品牌编码'] = brand_encoded
df_encoded['型号编码'] = model_encoded
df_encoded['颜色编码'] = color_encoded

# 将网络和成色字段选取出来，保存到一个新的DataFrame中
network_df = df_encoded['网络'].to_frame()
condition_df = df_encoded['成色'].to_frame()

# 使用独热编码对网络和成色字段进行转换
network_encoder = OneHotEncoder(categories='auto', sparse=False)
condition_encoder = OneHotEncoder(categories='auto', sparse=False)
network_encoded = network_encoder.fit_transform(network_df)
condition_encoded = condition_encoder.fit_transform(condition_df)
print(network_encoded.shape, condition_encoded.shape)
# 将编码结果添加到原来的DataFrame中
network_columns = network_encoder.get_feature_names_out()
condition_columns = condition_encoder.get_feature_names_out()
print(network_columns, condition_columns)
network_df_encoded = pd.DataFrame(network_encoded, columns=network_columns)
condition_df_encoded = pd.DataFrame(condition_encoded, columns=condition_columns)
df_encoded = pd.concat([df_encoded, network_df_encoded, condition_df_encoded], axis=1)

# 将原价和现价两列添加到新的DataFrame中
df_encoded['内存（GB）'] = df['内存（GB）']
df_encoded['存储（GB）'] = df['存储（GB）']
df_encoded['原价（元）'] = df['原价（元）']
df_encoded['现价（元）'] = df['现价（元）']
# df_encoded = df_encoded.drop(columns=['品牌', '型号', '网络', '成色', '颜色'])

# 将编码结果和原价现价两列保存到新的Excel表格中
df_encoded.to_excel('二手手机编码表（总表）.xlsx', index=False)
