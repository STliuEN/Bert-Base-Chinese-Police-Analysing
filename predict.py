# coding=gbk
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd

# ���ر�ǩӳ��
labels_df = pd.read_csv('./data/labels_and_ids.csv')
labels_map = dict(zip(labels_df['label_ids'], labels_df['labels']))

# ���طִ�����ģ��
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=len(labels_map))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ����ѵ���õļ���
checkpoint_path = './checkpoints/latest_checkpoint.pt'
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])

def infer(text):
    model.eval()  # ����ģ��Ϊ����ģʽ
    inputs = tokenizer(text, padding="max_length", truncation=True, return_tensors="pt", max_length=128)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=1).item()
    return labels_map[predicted_class_id]

# ����������

while True:
    test_text = input("��������ӣ����� 'exit' �˳�����")
    if test_text.lower() == 'exit':
        break
    predicted_label = infer(test_text)
    print(f"�����ı�: {test_text}")
    print(f"Ԥ���ǩ: {predicted_label}")