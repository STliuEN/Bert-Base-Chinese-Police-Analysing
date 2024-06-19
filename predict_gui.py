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

# Excel文件路径
excel_file_path = '报警记录.xlsx'

# 定义时间提取的正则表达式
time_pattern = re.compile(r'(\d{4})年(\d{1,2})月(\d{1,2})日\s*(\d{1,2})时(\d{1,2})分(\d{1,2})秒')

def extract_time_from_text(text):
    match = time_pattern.search(text)
    if match:
        return f"{match.group(1)}年{match.group(2)}月{match.group(3)}日 {match.group(4)}时{match.group(5)}分{match.group(6)}秒"
    else:
        return ""

def infer(text):
    model.eval()  # 设置模型为评估模式
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
        df = pd.DataFrame(columns=['时间', '报警内容', '警情粗略类别', '警情细致类别'])
    
    new_row = {'时间': timestamp, '报警内容': content, '警情粗略类别': rough_category, '警情细致类别': fine_category}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    
    # 根据 '警情粗略类别' 列进行排序
    df.sort_values(by=['警情粗略类别'], inplace=True)
    
    df.to_excel(excel_file_path, index=False)

# GUI应用程序
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("警情标签预测")
        self.root.geometry("600x400")
        self.root.configure(bg='#f0f0f0')

        self.label = tk.Label(root, text="请输入句子:", bg='#f0f0f0', font=('Helvetica', 12))
        self.label.pack(pady=10)

        self.text = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=70, height=10, font=('Helvetica', 10))
        self.text.pack(pady=10, padx=10)

        self.save_to_excel = BooleanVar()
        self.checkbox = tk.Checkbutton(root, text="保存结果到Excel", variable=self.save_to_excel, bg='#f0f0f0', font=('Helvetica', 12))
        self.checkbox.pack(pady=10)

        self.button = tk.Button(root, text="预测标签", command=self.predict, bg='#4CAF50', fg='white', font=('Helvetica', 12), padx=20, pady=10)
        self.button.pack(pady=10)

        self.result_label = tk.Label(root, text="", fg="blue", bg='#f0f0f0', font=('Helvetica', 12))
        self.result_label.pack(pady=10)

    def predict(self):
        text = self.text.get("1.0", tk.END).strip()
        if text == "":
            messagebox.showwarning("输入错误", "请输入一个句子")
            return
        predicted_label = infer(text)
        self.result_label.config(text=f"预测标签: {predicted_label}")
        
        # 从报警内容中提取时间
        timestamp = extract_time_from_text(text)
        
        # 如果提取不到时间，则设置为空值
        if not timestamp:
            timestamp = ""
        
        # 假设粗略类别和细致类别在标签中用 ' - ' 分隔
        rough_category, fine_category = predicted_label.split(' - ', 1)
        
        # 如果勾选了保存结果到Excel
        if self.save_to_excel.get():
            append_to_excel(timestamp, text, rough_category, fine_category)
            messagebox.showinfo("成功", "信息已添加到Excel表中并按标签排序")

# 运行应用程序
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
