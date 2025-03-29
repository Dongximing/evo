# import pandas as pd
# import json
#
# # 读取TSV文件
# file_path = '/Users/ximing/Desktop/liar_dataset/train.tsv'
# data = pd.read_csv(file_path, sep='\t', header=None)  # 如果文件没有列标题，添加 header=None
#
# # 假设第二列是索引1（Python索引从0开始）
# target_column = 1
# input_column = 2
#
# # 处理target列，只保留'true'或'false'
# data[target_column] = data[target_column].apply(lambda x: 'true' if 'true' in str(x).lower() else 'false')
#
# # 选取符合条件的样本，这里我们先筛选出符合true或false的，如果超过400条，就取前400条
# filtered_data = data[(data[target_column] == 'true') | (data[target_column] == 'false')].head(400)
#
# # 构造JSON结构
# examples = []
# for index, row in filtered_data.iterrows():
#     example = {
#         "input": row[input_column],
#         "target": row[target_column]
#     }
#     examples.append(example)
#
# json_output = {
#     "examples": examples
# }
#
# # 保存为JSON文件
# json_file_path = '/Users/ximing/Desktop/processed_data.json'
# with open(json_file_path, 'w') as outfile:
#     json.dump(json_output, outfile, indent=4)
#
# print(f'JSON file saved to {json_file_path}')

# import pandas as pd
# from openai import OpenAI
# import  os
# # 初始化OpenAI客户端
#
# client = OpenAI()
#
# # 定义获取嵌入向量的函数
# def get_embedding(text, model="text-embedding-3-small"):
#     try:
#         # 处理文本，替换换行符
#         text = text.replace("\n", " ")
#         # 发起API调用，获取嵌入向量
#         response = client.embeddings.create(input=[text], model=model)
#         return response.data[0].embedding
#     except Exception as e:
#         print(f"Error processing text: {text}. Error: {e}")
#         return None  # 返回None或适当的默认值
#
# # 读取JSON文件，假设'combined'是你需要处理的列名
# json_file_path = 'processed_data.json'  # 替换为你的文件路径
# df = pd.read_json(json_file_path)
# df = pd.DataFrame(df['examples'].tolist())
#
# # 应用函数到DataFrame的'input'列
# df['embedding'] = df['input'].apply(lambda x: get_embedding(x, model='text-embedding-3-small'))
#
# # 保存处理后的数据到CSV文件
# df.to_json(json_file_path, orient='records', lines=True)
# import pandas as pd
# import json
#
# # 指定输入JSON文件路径，其中每行是一个独立的JSON对象
# input_json_path = 'processed_data.json'
#
# # 读取这个文件，注意 lines=True 参数是必需的，因为每行是一个独立的JSON对象
# df = pd.read_json(input_json_path, lines=True)
#
# # 将DataFrame转换回字典列表
# data_list = df.to_dict(orient='records')
#
# # 指定输出JSON文件路径
# output_json_path = 'processed_data.json'
#
# # 使用json库将列表写入文件，包括在一个大数组中
# with open(output_json_path, 'w') as file:
#     json.dump(data_list, file, indent=4)  # indent=4 用于美化输出，使JSON文件易于阅读
#
# print(f'Data with embeddings saved as an array to {output_json_path}')
import pandas as pd
# from sklearn.cluster import KMeans
# import json
#
# # 加载JSON数据
# input_json_path ='processed_data.json'
# with open(input_json_path, 'r') as file:
#     data = json.load(file)
#
# # 提取嵌入向量列表
# embeddings = [item['embedding'] for item in data]
#
# # 应用K-means聚类
# kmeans = KMeans(n_clusters=5, random_state=42)  # 设置5个聚类中心
# kmeans.fit(embeddings)  # 训练模型
# labels = kmeans.labels_  # 获取聚类标签
#
# # 将聚类标签添加到原始数据中
# for i, item in enumerate(data):
#     item['cluster_label'] = int(labels[i])  # 转换为整数，更美观
#
# # 保存更新后的数据到JSON文件
# output_json_path = 'processed_data.json'
# with open(output_json_path, 'w') as file:
#     json.dump(data, file, indent=4)  # 美化输出
#
# print(f'Data with cluster labels saved to {output_json_path}')
import json

# 读取之前保存的带有聚类标签的数据
input_json_path = 'processed_data.json'
with open(input_json_path, 'r') as file:
    data = json.load(file)

# 确定分割点
train_data = data[:300]  # 前300个样本为训练数据
test_data = data[300:400]  # 接下来的100个样本为测试数据

# 保存训练数据到JSON文件
train_output_path = 'fake_train_data.json'
with open(train_output_path, 'w') as file:
    json.dump(train_data, file, indent=4)
print(f'Training data saved to {train_output_path}')

# 保存测试数据到JSON文件
test_output_path = 'fake_test_data.json'
with open(test_output_path, 'w') as file:
    json.dump(test_data, file, indent=4)
print(f'Test data saved to {test_output_path}')





