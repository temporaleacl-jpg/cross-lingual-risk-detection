# -*- coding: utf-8 -*-
"""
01_bert_training.py
BERT-based L2/L3 Risk Classifier Training
EACL 2027 Submission (ARR August 2026)

Requirements: pip install transformers torch scikit-learn pandas
Usage: python 01_bert_training.py --data bert_training_data.csv
"""
import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer)
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import os

# ============================================================
# 설정
# ============================================================
MODEL_NAME = 'klue/roberta-base'
MAX_LEN = 256
BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-5
SEED = 42

# ============================================================
# Dataset
# ============================================================
class RiskDataset(Dataset):
    def __init__(self, texts, labels,
                 tokenizer, max_len=MAX_LEN):
        self.texts    = texts
        self.labels   = labels
        self.tokenizer = tokenizer
        self.max_len  = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt')
        return {
            'input_ids':
                enc['input_ids'].squeeze(),
            'attention_mask':
                enc['attention_mask'].squeeze(),
            'labels': torch.tensor(
                self.labels[idx],
                dtype=torch.long)
        }

# ============================================================
# Train
# ============================================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {'f1': f1_score(
        labels, preds, average='binary')}

def train_classifier(df, label_col,
                     tokenizer, output_dir):
    texts  = df['text'].tolist()
    labels = df[label_col].fillna(0)\
                          .astype(int).tolist()

    X_tr, X_val, y_tr, y_val = \
        train_test_split(
            texts, labels,
            test_size=0.2,
            stratify=labels,
            random_state=SEED)

    train_ds = RiskDataset(X_tr, y_tr, tokenizer)
    val_ds   = RiskDataset(X_val, y_val, tokenizer)

    model = AutoModelForSequenceClassification\
        .from_pretrained(
            MODEL_NAME, num_labels=2)

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=32,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        learning_rate=LR,
        weight_decay=0.01,
        logging_steps=50,
        seed=SEED)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics)

    trainer.train()
    return trainer

# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data',
        default='bert_training_data.csv')
    parser.add_argument(
        '--output_dir',
        default='./models')
    args = parser.parse_args()

    print(f"Loading data: {args.data}")
    df = pd.read_csv(
        args.data, encoding='utf-8-sig')
    ko_df = df[df['lang']=='ko'].copy()
    print(f"KO samples: {len(ko_df)}")
    print(f"L2 positive: {ko_df['L2'].sum()}")
    print(f"L3 positive: {ko_df['L3'].sum()}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME)

    os.makedirs(args.output_dir, exist_ok=True)

    # L2 훈련
    print("\n[1/2] Training L2 classifier...")
    train_classifier(
        ko_df, 'L2', tokenizer,
        os.path.join(args.output_dir, 'L2'))

    # L3 훈련
    print("\n[2/2] Training L3 classifier...")
    train_classifier(
        ko_df, 'L3', tokenizer,
        os.path.join(args.output_dir, 'L3'))

    print("\n✅ Training complete!")
    print(f"Models saved to: {args.output_dir}")

if __name__ == '__main__':
    main()
