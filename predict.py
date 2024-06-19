# coding=gbk
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd

# 加载标签映射
labels_df = pd.read_csv('./data/labels_and_ids.csv')
labels_map = dict(zip(labels_df['label_ids'], labels_df['labels']))

# 加载分词器和模型
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=len(labels_map))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 加载训练好的检查点
checkpoint_path = './checkpoints/latest_checkpoint.pt'
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])

def infer(text):
    model.eval()  # 设置模型为评估模式
    inputs = tokenizer(text, padding="max_length", truncation=True, return_tensors="pt", max_length=128)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=1).item()
    return labels_map[predicted_class_id]

# 测试推理函数

while True:
    test_text = input("请输入句子（输入 'exit' 退出）：")
    if test_text.lower() == 'exit':
        break
    predicted_label = infer(test_text)
    print(f"输入文本: {test_text}")
    print(f"预测标签: {predicted_label}")