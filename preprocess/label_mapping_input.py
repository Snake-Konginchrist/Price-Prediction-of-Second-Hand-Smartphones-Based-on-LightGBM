import pandas as pd

file_path = '二手手机编码表（总表）.xlsx'

# 读取 Excel 表格
df = pd.read_excel(file_path)

# 构建品牌、型号、颜色的标签和编码列表
brand_list = list(zip(df.iloc[:, 0], df.iloc[:, 1]))
model_list = list(zip(df.iloc[:, 2], df.iloc[:, 3]))
color_list = list(zip(df.iloc[:, 4], df.iloc[:, 5]))

# 去除重复的元素
brand_list = list(set(brand_list))
model_list = list(set(model_list))
color_list = list(set(color_list))

# 构建品牌、型号、颜色的标签和编码字典
brand_dict = dict(brand_list)
model_dict = dict(model_list)
color_dict = dict(color_list)

# 构建总的 label_map
label_map = {
    '品牌': brand_dict,
    '型号': model_dict,
    '颜色': color_dict
}

# print(label_map)

code_list = []
# selected_brand = None
for key, value in label_map.items():
    print("请选择" + key + "编号：")
    # if key == "型号" and selected_brand:
    #     # 如果用户已经选择了品牌，只显示该品牌对应的型号
    #     value = {k: v for k, v in value.items() if v.startswith(selected_brand)}
    codes = list(value.keys())
    for i in range(0, len(codes), 5):
        for j in range(i, min(i + 5, len(codes))):
            label = value[codes[j]]
            print(f"{codes[j]}: {str(label).ljust(20)}", end="")
        print()
    code = int(input())
    code_list.append(code)
    # if key == "品牌":
    #     # 如果用户选择了品牌，记录下选择的品牌
    #     selected_brand = value[code]
    print()
