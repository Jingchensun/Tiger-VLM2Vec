# import json
# from datasets import load_dataset

# # 指定数据集名称和子集名称
# data_args = {
#     "dataset_name": "TIGER-Lab/MMEB-eval",
#     "subset_names": ["VisualNews_t2i", "VisDial", "MSCOCO_t2i", "WebQA"],
#     "dataset_split": "test"
# }

# # 定义要替换的文本映射
# replacement_map = {
#     "Find a Wikipedia image that answers this question:": "Find an image that matches the given text:",
#     "Retrieve an image of this news caption.": "Find an image that matches the given text.",
#     "Represent the given dialogue about an image, which is used for image retrieval:": "Find an image that matches the given text.",
#     "Find me an everyday image that matches the given caption:": "Find an image that matches the given text."
# }

# # 初始化数据列表
# data_list = []

# # 加载并合并数据集
# for subset in data_args["subset_names"]:
#     print(f"Loading dataset {data_args['dataset_name']} - {subset}")
#     dataset = load_dataset(
#         data_args["dataset_name"],
#         subset,
#         split=data_args["dataset_split"]
#     )
#     for sample in dataset:
#         if "qry_text" in sample:
#             for old_text, new_text in replacement_map.items():
#                 if sample["qry_text"].startswith(old_text):
#                     sample["qry_text"] = sample["qry_text"].replace(old_text, new_text)
#         data_list.append(sample)

# # 保存到 JSON 文件
# output_file = "eval.json"
# with open(output_file, "w", encoding="utf-8") as f:
#     json.dump(data_list, f, ensure_ascii=False, indent=4)

# print(f"Extracted {len(data_list)} samples from {data_args['subset_names']} and saved to {output_file}")

import json
from datasets import load_dataset

# 指定数据集名称和子集名称
data_args = {
    "dataset_name": "TIGER-Lab/MMEB-eval",
    "subset_names": ["VisualNews_t2i", "VisDial", "MSCOCO_t2i", "WebQA"],
    "dataset_split": "test"
}

# 定义系统 prompt 替换规则
replacement_map = {
    "Find a Wikipedia image that answers this question:": "",
    "Retrieve an image of this news caption.": "",
    "Represent the given dialogue about an image, which is used for image retrieval:": "",
    "Find me an everyday image that matches the given caption:": "",
}

# 初始化数据列表
data_list = []

# 加载并处理数据集
for subset in data_args["subset_names"]:
    print(f"Loading dataset {data_args['dataset_name']} - {subset}")
    dataset = load_dataset(
        data_args["dataset_name"],
        subset,
        split=data_args["dataset_split"]
    )
    
    for sample in dataset:
        # 移除系统 prompt
        qry_text = sample["qry_text"]
        for prompt, replacement in replacement_map.items():
            if qry_text.startswith(prompt):
                qry_text = qry_text[len(prompt):].strip()  # 去掉 prompt 并移除前后空格
        
        # 处理数据
        filtered_sample = {
            "qry_text": qry_text,
            "tgt_img_path": sample["tgt_img_path"][0] if sample["tgt_img_path"] else "",
            "subset_name": subset
        }
        data_list.append(filtered_sample)

# 保存到 JSON 文件
output_file = "filtered_eval.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(data_list, f, ensure_ascii=False, indent=4)

print(f"Filtered {len(data_list)} samples and saved to {output_file}")

