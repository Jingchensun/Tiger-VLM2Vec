import json
import torch
import numpy as np
import os
import argparse
from tqdm import tqdm
from gme_inference import GmeQwen2VL

# 关闭 HuggingFace Tokenizer 并行化
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 确保 image_embedding 目录存在
IMAGE_EMBEDDING_DIR = "image_embedding"
os.makedirs(IMAGE_EMBEDDING_DIR, exist_ok=True)

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate GmeQwen2VL on a specific subset")
    parser.add_argument("--subset", type=str, required=True, help="Subset name to evaluate (e.g., VisualNews_t2i, VisDial, MSCOCO_t2i, WebQA)")
    return parser.parse_args()

# 加载数据
def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# 计算相似性矩阵
def compute_similarity_matrix(text_embeddings, image_embeddings):
    return torch.matmul(text_embeddings, image_embeddings.T)

# 计算Top-1 Recall
def compute_top1_recall(similarity_matrix):
    predictions = similarity_matrix.argmax(dim=1)
    ground_truth = torch.arange(similarity_matrix.size(0))
    correct = (predictions == ground_truth).sum().item()
    return correct / similarity_matrix.size(0)

# 处理批量数据，减少显存占用，并显示进度
def batched_image_embeddings(model, images, batch_size=4):
    image_embeds = []
    print("Encoding image embeddings...")
    for i in tqdm(range(0, len(images), batch_size), desc="Processing Images", unit="batch"):
        batch = images[i:i + batch_size]
        image_embeds.append(model.get_image_embeddings(images=batch, is_query=False))
    return torch.cat(image_embeds)

# 评估模型
def evaluate_model(data, model, subset_name):
    print(f"Evaluating subset: {subset_name}")
    subset_data = [sample for sample in data if sample["subset_name"] == subset_name]
    
    qry_texts = [sample["qry_text"] for sample in subset_data]
    tgt_img_paths = ["/home/onsi/jsun/VLM2Vec/eval_images/" + sample["tgt_img_path"] for sample in subset_data]
    
    # 获取文本嵌入
    text_embeddings = model.get_text_embeddings(texts=qry_texts, instruction='Find an image that matches the given text.')
    
    # 读取或计算图像嵌入
    embedding_file = os.path.join(IMAGE_EMBEDDING_DIR, f"{subset_name}_image_embeddings.pt")
    if os.path.exists(embedding_file):
        print(f"Loading precomputed image embeddings from {embedding_file}")
        image_embeddings = torch.load(embedding_file)
    else:
        print(f"Computing image embeddings for {subset_name}")
        image_embeddings = batched_image_embeddings(model, tgt_img_paths, batch_size=20)
        torch.save(image_embeddings, embedding_file)
        print(f"Saved image embeddings to {embedding_file}")
    
    # 计算相似性矩阵
    similarity_matrix = compute_similarity_matrix(text_embeddings, image_embeddings)
    
    # 计算Top-1 Recall
    top1_recall = compute_top1_recall(similarity_matrix)
    print(f"Subset: {subset_name}, Top-1 Recall: {top1_recall:.4f}")
    
    return top1_recall

if __name__ == "__main__":
    args = parse_args()
    selected_subset = args.subset
    
    data_file = "filtered_eval.json"
    data = load_data(data_file)
    
    gme = GmeQwen2VL("Alibaba-NLP/gme-Qwen2-VL-2B-Instruct")
    
    result = evaluate_model(data, gme, selected_subset)
    
    # 保存评估结果
    score_file = os.path.join(IMAGE_EMBEDDING_DIR, f"{selected_subset}_score.json")
    with open(score_file, "w", encoding="utf-8") as f:
        json.dump({"subset": selected_subset, "top1_recall": result}, f, ensure_ascii=False, indent=4)
    print(f"Saved evaluation results to {score_file}")
