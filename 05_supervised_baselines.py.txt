# -*- coding: utf-8 -*-
"""
05_supervised_baselines.py
Supervised Single-turn Classification Baselines
EACL 2027 Submission (ARR August 2026)

Baselines:
  1. Random
  2. Majority class
  3. TF-IDF + LR
  4. mBERT + LR
  5. XLM-R + LR
  6. LaBSE + LR
  7. GPT-4 zero-shot (optional)
  8. klue/RoBERTa (Korean-only)
  9. Majority Vote Window (Korean-only)

Requirements:
    pip install scikit-learn sentence-transformers
    pip install transformers torch openai pandas numpy
Usage:
    python 05_supervised_baselines.py --data bert_training_data.csv
    python 05_supervised_baselines.py --data bert_training_data.csv \
        --openai_key YOUR_KEY
"""

import argparse
import time
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from scipy.stats import mode as scipy_mode
from sentence_transformers import SentenceTransformer

SEED     = 42
N_FOLDS  = 5
AVERAGE  = 'macro'   # consistent with paper


# ── Cross-validation helper ────────────────────────────────────────────────────

def cross_val(X, y, clf, vectorizer=None):
    skf = StratifiedKFold(
        n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    f1s = []
    for tr_idx, val_idx in skf.split(
            X if isinstance(X, np.ndarray)
            else list(range(len(y))), y):
        if isinstance(X, np.ndarray):
            X_tr, X_val = X[tr_idx], X[val_idx]
        else:
            X_tr  = [X[i] for i in tr_idx]
            X_val = [X[i] for i in val_idx]
        y_tr  = y[tr_idx]
        y_val = y[val_idx]

        if vectorizer is not None:
            X_tr  = vectorizer.fit_transform(X_tr)
            X_val = vectorizer.transform(X_val)

        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_val)
        f1s.append(f1_score(y_val, preds, average=AVERAGE))

    return np.mean(f1s), np.std(f1s)


# ── GPT-4 zero-shot ────────────────────────────────────────────────────────────

def gpt4_classify(texts, labels, api_key,
                  n_sample=200, seed=SEED):
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
    except ImportError:
        print("  openai not installed. Skipping GPT-4.")
        return None

    np.random.seed(seed)
    idx           = np.random.choice(len(texts), n_sample, replace=False)
    sample_texts  = [texts[i] for i in idx]
    sample_labels = [labels[i] for i in idx]

    preds = []
    for i, text in enumerate(sample_texts):
        try:
            resp = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system",
                     "content": (
                         "Classify the counseling utterance as "
                         "'risk' (moderate-to-severe distress, "
                         "L2-L3) or 'no-risk' (none-to-mild, "
                         "L0-L1). Reply with one word only.")},
                    {"role": "user",
                     "content": str(text)[:500]}
                ],
                max_tokens=5,
                temperature=0)
            r = resp.choices[0].message.content.strip().lower()
            preds.append(0 if 'no' in r else 1)
        except Exception as e:
            print(f"  Error at {i}: {e}")
            preds.append(-1)
        time.sleep(0.3)

        if (i + 1) % 50 == 0:
            valid_p = [p for p in preds if p != -1]
            valid_l = [sample_labels[j]
                       for j, p in enumerate(preds) if p != -1]
            if valid_p:
                print(f"  {i+1}/{n_sample}: "
                      f"F1={f1_score(valid_l, valid_p, average=AVERAGE):.3f}")

    valid_idx = [i for i, p in enumerate(preds) if p != -1]
    vt = [sample_labels[i] for i in valid_idx]
    vp = [preds[i]         for i in valid_idx]
    return f1_score(vt, vp, average=AVERAGE), np.std([])


# ── klue/RoBERTa (Korean-only) ─────────────────────────────────────────────────

def klue_roberta_baseline(df):
    """Fine-tune klue/roberta-base on Korean-only data (5-fold CV)."""
    try:
        import torch
        from transformers import (AutoTokenizer,
                                  AutoModelForSequenceClassification)
        from torch.utils.data import Dataset, DataLoader
        from torch.optim import AdamW
    except ImportError:
        print("  transformers/torch not installed. Skipping klue/RoBERTa.")
        return None

    df_ko = df[df['lang'] == 'ko'].reset_index(drop=True)
    print(f"  Korean samples: {len(df_ko)} "
          f"(risk={df_ko['label'].sum()})")

    MODEL    = 'klue/roberta-base'
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    device    = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    class RiskDS(Dataset):
        def __init__(self, texts, labels):
            self.enc = tokenizer(
                list(texts), truncation=True,
                max_length=128, padding='max_length',
                return_tensors='pt')
            self.labels = torch.tensor(
                list(labels), dtype=torch.long)
        def __len__(self):
            return len(self.labels)
        def __getitem__(self, i):
            return ({k: v[i] for k, v in self.enc.items()},
                    self.labels[i])

    skf  = StratifiedKFold(
        n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    f1s  = []

    for fold, (tr_idx, val_idx) in enumerate(
            skf.split(df_ko['text'], df_ko['label'])):
        tr  = df_ko.iloc[tr_idx]
        val = df_ko.iloc[val_idx]
        tr_dl  = DataLoader(
            RiskDS(tr['text'], tr['label']),
            batch_size=16, shuffle=True)
        val_dl = DataLoader(
            RiskDS(val['text'], val['label']),
            batch_size=32)

        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL, num_labels=2).to(device)
        opt   = AdamW(model.parameters(), lr=2e-5)

        for _ in range(3):
            model.train()
            for batch, labels in tr_dl:
                batch  = {k: v.to(device) for k, v in batch.items()}
                labels = labels.to(device)
                loss   = model(**batch, labels=labels).loss
                loss.backward()
                opt.step()
                opt.zero_grad()

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch, labels in val_dl:
                batch  = {k: v.to(device) for k, v in batch.items()}
                logits = model(**batch).logits
                preds.extend(logits.argmax(-1).cpu().tolist())
                trues.extend(labels.tolist())

        f = f1_score(trues, preds, average=AVERAGE)
        f1s.append(f)
        print(f"  Fold {fold+1}: F1={f:.3f}")

    return np.mean(f1s), np.std(f1s)


# ── Majority Vote Window (Korean-only) ─────────────────────────────────────────

def majority_vote_window(df):
    """
    TF-IDF + LR predictions aggregated by majority vote
    over K consecutive 8-turn chunks (Korean-only).
    """
    df_ko = df[df['lang'] == 'ko'].copy()
    df_ko['session_id'] = df_ko['dialogue_id'].str.replace(
        r'_chunk_\d+', '', regex=True)
    df_ko['chunk_num']  = df_ko['dialogue_id'].str.extract(
        r'_chunk_(\d+)').astype(float)
    df_ko = df_ko.sort_values(
        ['session_id', 'chunk_num']).reset_index(drop=True)

    texts  = df_ko['text'].fillna('').tolist()
    labels = df_ko['label'].tolist()

    results = {}
    skf = StratifiedKFold(
        n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    for K in [1, 2, 4]:
        f1s = []
        for tr_idx, val_idx in skf.split(texts, labels):
            X_tr  = [texts[i]  for i in tr_idx]
            X_val = [texts[i]  for i in val_idx]
            y_tr  = [labels[i] for i in tr_idx]
            y_val = [labels[i] for i in val_idx]

            vec   = TfidfVectorizer(
                max_features=5000)
            clf   = LogisticRegression(
                max_iter=1000, random_state=SEED)
            clf.fit(vec.fit_transform(X_tr), y_tr)
            raw_preds = clf.predict(vec.transform(X_val))

            if K == 1:
                voted = raw_preds
            else:
                voted = []
                for i in range(0, len(raw_preds), K):
                    window = raw_preds[i:i+K]
                    mv = scipy_mode(window, keepdims=True).mode[0]
                    voted.extend([mv] * len(window))
                voted = np.array(voted[:len(y_val)])

            f1s.append(
                f1_score(y_val, voted[:len(y_val)],
                         average=AVERAGE))

        results[K] = (np.mean(f1s), np.std(f1s))

    return results


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',
        default='bert_training_data.csv')
    parser.add_argument('--openai_key',
        default=None)
    parser.add_argument('--skip_klue',
        action='store_true',
        help='Skip klue/RoBERTa (slow, needs GPU)')
    args = parser.parse_args()

    print(f"Loading: {args.data}")
    df = pd.read_csv(args.data, encoding='utf-8-sig')
    X_text = df['text'].fillna('').tolist()
    y      = df['label'].values
    print(f"Total: {len(df)}, "
          f"Risk(L2-L3)={y.sum()}, "
          f"No-risk(L0-L1)={(y==0).sum()}")

    results = {}

    # 1. Random
    print("\n[1/8] Random baseline...")
    X_dummy = np.zeros((len(y), 1))
    f1, std = cross_val(
        X_dummy, y,
        DummyClassifier(strategy='uniform',
                        random_state=SEED))
    results['Random'] = (f1, std)
    print(f"  F1={f1:.3f} ±{std:.3f}")

    # 2. Majority class
    print("\n[2/8] Majority class...")
    f1, std = cross_val(
        X_dummy, y,
        DummyClassifier(strategy='most_frequent'))
    results['Majority class'] = (f1, std)
    print(f"  F1={f1:.3f} ±{std:.3f}")

    # 3. TF-IDF + LR
    print("\n[3/8] TF-IDF + LR...")
    f1, std = cross_val(
        X_text, y,
        LogisticRegression(max_iter=1000,
                           class_weight='balanced',
                           random_state=SEED),
        TfidfVectorizer(max_features=10000,
                        ngram_range=(1, 2)))
    results['TF-IDF + LR'] = (f1, std)
    print(f"  F1={f1:.3f} ±{std:.3f}")

    # 4. mBERT + LR
    print("\n[4/8] mBERT + LR...")
    mbert  = SentenceTransformer(
        'bert-base-multilingual-cased')
    X_mbert = mbert.encode(
        X_text, batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True)
    f1, std = cross_val(
        X_mbert, y,
        LogisticRegression(max_iter=1000,
                           class_weight='balanced',
                           random_state=SEED))
    results['mBERT + LR'] = (f1, std)
    print(f"  F1={f1:.3f} ±{std:.3f}")

    # 5. XLM-R + LR
    print("\n[5/8] XLM-R + LR...")
    xlmr   = SentenceTransformer('xlm-roberta-base')
    X_xlmr = xlmr.encode(
        X_text, batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True)
    f1, std = cross_val(
        X_xlmr, y,
        LogisticRegression(max_iter=1000,
                           class_weight='balanced',
                           random_state=SEED))
    results['XLM-R + LR'] = (f1, std)
    print(f"  F1={f1:.3f} ±{std:.3f}")

    # 6. LaBSE + LR
    print("\n[6/8] LaBSE + LR...")
    labse   = SentenceTransformer(
        'sentence-transformers/LaBSE')
    X_labse = labse.encode(
        X_text, batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True)
    f1, std = cross_val(
        X_labse, y,
        LogisticRegression(max_iter=1000,
                           class_weight='balanced',
                           random_state=SEED))
    results['LaBSE + LR'] = (f1, std)
    print(f"  F1={f1:.3f} ±{std:.3f}")

    # 7. GPT-4 zero-shot (optional)
    if args.openai_key:
        print("\n[7/8] GPT-4 zero-shot "
              f"(n=200, model=gpt-4)...")
        res = gpt4_classify(
            X_text, y.tolist(), args.openai_key)
        if res:
            f1, std = res
            results['GPT-4 zero-shot'] = (f1, std)
            print(f"  F1={f1:.3f}")
    else:
        print("\n[7/8] GPT-4 skipped "
              "(no --openai_key provided)")
        print("  Expected: F1=0.699 "
              "(KO=0.614, EN=0.776, n=200)")

    # 8. klue/RoBERTa (Korean-only)
    if not args.skip_klue:
        print("\n[8/8] klue/RoBERTa "
              "(Korean-only, n=2833)...")
        res = klue_roberta_baseline(df)
        if res:
            f1, std = res
            results['klue/RoBERTa†'] = (f1, std)
            print(f"  Mean F1={f1:.3f} ±{std:.3f}")
    else:
        print("\n[8/8] klue/RoBERTa skipped "
              "(--skip_klue flag set)")
        print("  Expected: F1=0.684 ±0.023")

    # 9. Majority Vote Window (Korean-only)
    print("\n[9/9] Majority Vote Window "
          "(Korean-only)...")
    mv_results = majority_vote_window(df)
    for K, (f1, std) in mv_results.items():
        results[f'MajVote-{K*8}t‡'] = (f1, std)
        print(f"  Window-{K*8}t: "
              f"F1={f1:.3f} ±{std:.3f}")

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("FINAL RESULTS (Macro F1)")
    print("=" * 50)
    print(f"{'Method':<26} {'F1':>6}  {'±std':>6}")
    print("-" * 42)
    for method, (f1, std) in results.items():
        print(f"{method:<26} {f1:>6.3f}  {std:>6.3f}")
    print("-" * 42)
    print("† Korean-only fine-tuning (n=2,833; 5-fold CV)")
    print("‡ KO-only chunk-level majority vote; 8-turn windows")


if __name__ == '__main__':
    main()
