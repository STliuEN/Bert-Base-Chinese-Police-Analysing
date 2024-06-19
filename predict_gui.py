# coding=gbk

import re
import tkinter as tk
from tkinter import messagebox
from tkinter import scrolledtext
from tkinter import BooleanVar
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
import os

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

# Excel�ļ�·��
excel_file_path = '������¼.xlsx'

# ����ʱ����ȡ��������ʽ
time_pattern = re.compile(r'(\d{4})��(\d{1,2})��(\d{1,2})��\s*(\d{1,2})ʱ(\d{1,2})��(\d{1,2})��')

def extract_time_from_text(text):
    match = time_pattern.search(text)
    if match:
        return f"{match.group(1)}��{match.group(2)}��{match.group(3)}�� {match.group(4)}ʱ{match.group(5)}��{match.group(6)}��"
    else:
        return ""

def infer(text):
    model.eval()  # ����ģ��Ϊ����ģʽ
    inputs = tokenizer(text, padding="max_length", truncation=True, return_tensors="pt", max_length=128)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=1).item()
    return labels_map[predicted_class_id]

def append_to_excel(timestamp, content, rough_category, fine_category):
    if os.path.exists(excel_file_path):
        df = pd.read_excel(excel_file_path)
    else:
        df = pd.DataFrame(columns=['ʱ��', '��������', '����������', '����ϸ�����'])
    
    new_row = {'ʱ��': timestamp, '��������': content, '����������': rough_category, '����ϸ�����': fine_category}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    
    # ���� '����������' �н�������
    df.sort_values(by=['����������'], inplace=True)
    
    df.to_excel(excel_file_path, index=False)

# GUIӦ�ó���
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("�����ǩԤ��")
        self.root.geometry("600x400")
        self.root.configure(bg='#f0f0f0')

        self.label = tk.Label(root, text="���������:", bg='#f0f0f0', font=('Helvetica', 12))
        self.label.pack(pady=10)

        self.text = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=70, height=10, font=('Helvetica', 10))
        self.text.pack(pady=10, padx=10)

        self.save_to_excel = BooleanVar()
        self.checkbox = tk.Checkbutton(root, text="��������Excel", variable=self.save_to_excel, bg='#f0f0f0', font=('Helvetica', 12))
        self.checkbox.pack(pady=10)

        self.button = tk.Button(root, text="Ԥ���ǩ", command=self.predict, bg='#4CAF50', fg='white', font=('Helvetica', 12), padx=20, pady=10)
        self.button.pack(pady=10)

        self.result_label = tk.Label(root, text="", fg="blue", bg='#f0f0f0', font=('Helvetica', 12))
        self.result_label.pack(pady=10)

    def predict(self):
        text = self.text.get("1.0", tk.END).strip()
        if text == "":
            messagebox.showwarning("�������", "������һ������")
            return
        predicted_label = infer(text)
        self.result_label.config(text=f"Ԥ���ǩ: {predicted_label}")
        
        # �ӱ�����������ȡʱ��
        timestamp = extract_time_from_text(text)
        
        # �����ȡ����ʱ�䣬������Ϊ��ֵ
        if not timestamp:
            timestamp = ""
        
        # �����������ϸ������ڱ�ǩ���� ' - ' �ָ�
        rough_category, fine_category = predicted_label.split(' - ', 1)
        
        # �����ѡ�˱�������Excel
        if self.save_to_excel.get():
            append_to_excel(timestamp, text, rough_category, fine_category)
            messagebox.showinfo("�ɹ�", "��Ϣ����ӵ�Excel���в�����ǩ����")

# ����Ӧ�ó���
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
