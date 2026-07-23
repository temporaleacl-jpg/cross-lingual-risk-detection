"""
06_detection_coverage.py
MSW-Informed Detection Coverage
Any-Risk-Until-t aggregation using klue/RoBERTa L2 classifier
EACL 2027 Submission (ARR August 2026)

Usage:
    python 06_detection_coverage.py \
        --model_dir ./models/L2 \
        --jsonl ko_raw_all.jsonl \
        --min_client_turns 84 \
        --output detection_coverage.csv
Requirements:
    pip install transformers torch pandas tqdm matplotlib
"""

import argparse
import json
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm


def get_risk_prob(text, model, tokenizer, device, max_len=256):
    """Return P(L2 risk) for a single utterance."""
    inputs = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        max_length=max_len,
        padding=True
    ).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)
    return probs[0][1].item()  # index 1 = positive (L2 risk)


def run_detection_coverage(
    model_dir,
    jsonl_path,
    min_client_turns=84,
    threshold=0.5,
    turn_budgets=None,
    output_path='detection_coverage.csv',
    figure_path='fig_msw_detection.png'
):
    if turn_budgets is None:
        turn_budgets = [4, 8, 12, 16, 24, 32, 48, 64, 80]

    # ── Load model ────────────────────────────────────────
    print(f"Loading model from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    print(f"✅ Model loaded ({device})")

    # ── Load sessions ──────────────────────────────────────
    sessions = []
    with open(jsonl_path, encoding='utf-8') as f:
        for line in f:
            sessions.append(json.loads(line))
    print(f"✅ Sessions loaded: {len(sessions)}")

    # ── Per-session inference ──────────────────────────────
    session_results = []

    for sess in tqdm(sessions, desc="Processing sessions"):
        # Extract client turns only (순서 유지)
        client_turns = [
            t['text'] for t in sess['turns']
            if t.get('spk', '') == '내담자'
        ]
        total_client_turns = len(client_turns)

        # Quality filter
        if total_client_turns < min_client_turns:
            continue

        # Per-turn risk probability
        risk_probs = []
        for text in client_turns:
            prob = get_risk_prob(text, model, tokenizer, device)
            risk_probs.append(prob)

        # Session-level ground truth
        has_risk = any(p >= threshold for p in risk_probs)

        # First risk detection turn
        first_risk_turn = next(
            (i + 1 for i, p in enumerate(risk_probs) if p >= threshold),
            None
        )

        session_results.append({
            'total_client_turns': total_client_turns,
            'has_risk': has_risk,
            'first_risk_turn': first_risk_turn,
        })

    df = pd.DataFrame(session_results)
    n_total = len(df)
    n_risk = df['has_risk'].sum()
    n_nonrisk = (~df['has_risk']).sum()

    print(f"\n✅ Sessions processed: {n_total}")
    print(f"   Risk sessions:    {n_risk}")
    print(f"   Non-risk sessions: {n_nonrisk}")

    # ── Detection Curve ────────────────────────────────────
    risk_df = df[df['has_risk']]
    recalls, f1s, tps = [], [], []

    print(f"\n{'Turns':>6} {'Recall':>8} {'F1':>8} {'TP':>10}")
    print("-" * 38)

    for t in turn_budgets:
        TP = (risk_df['first_risk_turn'] <= t).sum()
        FN = n_risk - TP
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        prec = 1.0 if TP > 0 else 0.0
        f1 = (2 * prec * recall / (prec + recall)
              if (prec + recall) > 0 else 0.0)
        recalls.append(recall)
        f1s.append(f1)
        tps.append(int(TP))
        print(f"{t:>6} {recall:>8.3f} {f1:>8.3f} "
              f"{int(TP):>4}/{int(n_risk)}")

    # ── Save CSV ───────────────────────────────────────────
    results_df = pd.DataFrame({
        'turn_budget': turn_budgets,
        'recall': recalls,
        'f1': f1s,
        'tp': tps,
        'n_sessions': [n_total] * len(turn_budgets)
    })
    results_df.to_csv(output_path, index=False)
    print(f"\n✅ Saved: {output_path}")

    # ── Plot ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.plot(turn_budgets, recalls, 'b-o',
            lw=2, ms=7, label='Recall (session-level)')
    ax.plot(turn_budgets, f1s, 'r-s',
            lw=2, ms=7, label='F1')

    ax.axvline(x=12, color='#e67e22', ls='--', lw=1.8, alpha=0.9,
               label=r'$\mathrm{MSW}_{0.5}=12$')
    ax.axvline(x=32, color='#2c3e50', ls='--', lw=1.8, alpha=0.9,
               label=r'$\mathrm{MSW}_{0.9}=32$')

    ax.fill_betweenx([0, 1.05], 10, 14,
                     alpha=0.08, color='orange')
    ax.fill_betweenx([0, 1.05], 28, 36,
                     alpha=0.08, color='gray')

    ax.set_xlabel('Turn Budget (client turns)', fontsize=12)
    ax.set_ylabel('Session-level Score', fontsize=12)
    ax.set_title(
        'MSW-Informed Detection Coverage\n'
        r'Any-Risk-Until-$t$, klue/RoBERTa, '
        f'N={n_total}',
        fontsize=12
    )
    ax.legend(fontsize=10, loc='lower right')
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, max(turn_budgets) + 8)
    ax.grid(alpha=0.3, ls='--')
    plt.tight_layout()
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {figure_path}")
    plt.show()

    return results_df


def main():
    parser = argparse.ArgumentParser(
        description='MSW-Informed Detection Coverage'
    )
    parser.add_argument('--model_dir', type=str,
                        default='./models/L2',
                        help='Path to klue/RoBERTa L2 model')
    parser.add_argument('--jsonl', type=str,
                        default='ko_raw_all.jsonl',
                        help='Path to ko_raw_all.jsonl')
    parser.add_argument('--min_client_turns', type=int,
                        default=84,
                        help='Minimum client turns (quality filter)')
    parser.add_argument('--threshold', type=float,
                        default=0.5,
                        help='Risk probability threshold')
    parser.add_argument('--output', type=str,
                        default='detection_coverage.csv',
                        help='Output CSV path')
    parser.add_argument('--figure', type=str,
                        default='fig_msw_detection.png',
                        help='Output figure path')
    args = parser.parse_args()

    run_detection_coverage(
        model_dir=args.model_dir,
        jsonl_path=args.jsonl,
        min_client_turns=args.min_client_turns,
        threshold=args.threshold,
        output_path=args.output,
        figure_path=args.figure
    )


if __name__ == '__main__':
    main()
