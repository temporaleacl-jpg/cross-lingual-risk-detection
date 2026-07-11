"""
=============================================================
Step 7: Language-Controlled Non-Linear Clustering (HDBSCAN)

Tests whether the K-Means null clustering result (Wasserstein
alignment without categorical separability; see Steps 1-3 and
the main paper, Section 5.1) is specific to linear clustering,
and isolates a language-identity confound in pooled (KO+EN)
density-based clustering.

Pipeline:
  1. Extract CLS-pooled LaBSE embeddings (NOT SentenceTransformer's
     default mean pooling -- see note below).
  2. Pooled (KO+EN) HDBSCAN across a coarse grid; check whether
     discovered clusters are confounded by language identity.
  3. Monolingual (KO-only, EN-only) HDBSCAN across a fine
     integer grid, isolating risk-level structure from language.
  4. Permutation test at a fixed grid point (min_cluster_size=10)
     to assess statistical significance against label-shuffled
     null distributions.
  5. Cluster-composition cross-tabulation against risk level at
     the same grid point.

IMPORTANT: This script uses CLS-token pooling
(`last_hidden_state[:, 0, :]`, L2-normalized), matching the
embedding extraction used in 01_bert_training.py and throughout
the paper. Using SentenceTransformer's `.encode()` default
(mean pooling) instead will NOT reproduce the reported numbers;
the two pooling strategies define materially different
embedding geometries for this corpus.
=============================================================
"""

import argparse
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score
import hdbscan
import warnings
warnings.filterwarnings('ignore')

SEED = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def parse_args():
    parser = argparse.ArgumentParser(
        description="Language-controlled HDBSCAN clustering (pooled + monolingual)."
    )
    parser.add_argument('--data', type=str, required=True,
                         help="Path to bert_training_data.csv (must contain "
                              "columns: text, lang, level_primary).")
    parser.add_argument('--embedding_mode', type=str, default='cls_pooling',
                         choices=['cls_pooling'],
                         help="Embedding extraction method. Only CLS-pooling "
                              "is supported/validated for reproducing paper numbers.")
    parser.add_argument('--min_cluster_sizes', type=str,
                         default='6,7,8,9,10,11,12,13,14,15,16,17,18,19,20',
                         help="Comma-separated list of min_cluster_size values "
                              "for the monolingual (fine) grid.")
    parser.add_argument('--pooled_grid', type=str, default='5,10,20,50,100',
                         help="Comma-separated list of min_cluster_size values "
                              "for the pooled (KO+EN) coarse grid.")
    parser.add_argument('--permutation_min_size', type=int, default=10,
                         help="min_cluster_size at which the permutation test "
                              "is run (should match a value in --min_cluster_sizes).")
    parser.add_argument('--n_permutations', type=int, default=1000)
    parser.add_argument('--output', type=str, default='hdbscan_en_grid.csv')
    return parser.parse_args()


def encode_cls_pooling(texts, tokenizer, model, batch_size=64):
    """LaBSE embeddings via CLS-token pooling + L2 normalization."""
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(
            batch, padding=True, truncation=True,
            max_length=128, return_tensors='pt'
        ).to(DEVICE)
        with torch.no_grad():
            out = model(**enc)
            emb = out.last_hidden_state[:, 0, :]
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        all_embs.append(emb.cpu().numpy())
    return np.vstack(all_embs)


def run_hdbscan(X, labels, min_cluster_size):
    """Fit HDBSCAN and return (ari, n_clusters, noise_pct, pred)."""
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric='euclidean',
        cluster_selection_method='eom'
    )
    pred = clusterer.fit_predict(X)
    mask = pred != -1
    n_clusters = len(set(pred[mask]))
    noise_pct = (pred == -1).sum() / len(pred) * 100
    if mask.sum() > 50 and n_clusters > 1:
        ari = adjusted_rand_score(labels[mask], pred[mask])
    else:
        ari = np.nan
    return ari, n_clusters, noise_pct, pred, mask


def main():
    args = parse_args()
    min_sizes_fine = [int(x) for x in args.min_cluster_sizes.split(',')]
    min_sizes_pooled = [int(x) for x in args.pooled_grid.split(',')]

    print("=" * 60)
    print("Step 7: Language-Controlled HDBSCAN Clustering")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    df = pd.read_csv(args.data, encoding='utf-8-sig')
    label_map_reverse = {0: 'L0', 1: 'L3', 2: 'L2', 3: 'L1'}
    df['label'] = df['level_primary'].map(label_map_reverse)

    print(f"\nTotal utterances: {len(df)}")
    print(f"  Korean (ko): {(df['lang'] == 'ko').sum()}")
    print(f"  English (en): {(df['lang'] == 'en').sum()}")

    # ------------------------------------------------------------------
    # 2. CLS-pooled LaBSE embeddings (full corpus)
    # ------------------------------------------------------------------
    print("\nExtracting CLS-pooled LaBSE embeddings...")
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/LaBSE')
    model = AutoModel.from_pretrained('sentence-transformers/LaBSE').to(DEVICE)
    model.eval()

    embeddings = encode_cls_pooling(df['text'].tolist(), tokenizer, model)
    print(f"Embeddings shape: {embeddings.shape}")

    # ==================================================================
    # 3. Pooled (KO+EN) HDBSCAN: language-confound diagnostic
    # ==================================================================
    print("\n" + "=" * 60)
    print("Pooled (KO+EN) HDBSCAN -- language purity diagnostic")
    print("=" * 60)

    X_pooled = StandardScaler().fit_transform(embeddings)
    y_pooled = df['level_primary'].values
    lang_arr = df['lang'].values

    pooled_results = []
    for ms in min_sizes_pooled:
        ari, n_cls, noise_pct, pred, mask = run_hdbscan(X_pooled, y_pooled, ms)

        # language purity per cluster
        if n_cls > 0:
            purities = []
            for c in set(pred[mask]):
                c_mask = pred == c
                lang_counts = pd.Series(lang_arr[c_mask]).value_counts(normalize=True)
                purities.append(lang_counts.max())
            mean_purity = np.mean(purities)
        else:
            mean_purity = np.nan

        print(f"  min_size={ms:>4}: clusters={n_cls:>3}, noise%={noise_pct:5.1f}%, "
              f"ARI={ari if not np.isnan(ari) else float('nan'):.4f}, "
              f"mean_lang_purity={mean_purity:.3f}" if not np.isnan(mean_purity)
              else f"  min_size={ms:>4}: clusters={n_cls}, noise%={noise_pct:.1f}%, ARI=n/a")

        pooled_results.append({
            'condition': 'pooled', 'min_cluster_size': ms,
            'n_clusters': n_cls, 'noise_pct': noise_pct,
            'ARI': ari, 'mean_lang_purity': mean_purity
        })

    # ==================================================================
    # 4. Monolingual re-analysis (language-controlled)
    # ==================================================================
    print("\n" + "=" * 60)
    print("Monolingual HDBSCAN (language-controlled)")
    print("=" * 60)

    mono_results = []
    en_mask_full = df['lang'].values == 'en'
    ko_mask_full = df['lang'].values == 'ko'

    for lang_name, lang_mask in [('ko', ko_mask_full), ('en', en_mask_full)]:
        print(f"\n--- {lang_name.upper()} (N={lang_mask.sum()}) ---")
        X_lang = StandardScaler().fit_transform(embeddings[lang_mask])
        y_lang = df['level_primary'].values[lang_mask]

        for ms in min_sizes_fine:
            ari, n_cls, noise_pct, pred, mask = run_hdbscan(X_lang, y_lang, ms)
            ari_str = f"{ari:.4f}" if not np.isnan(ari) else "n/a"
            print(f"  min_size={ms:>4}: clusters={n_cls:>3}, "
                  f"noise%={noise_pct:5.1f}%, ARI={ari_str}")
            mono_results.append({
                'condition': lang_name, 'min_cluster_size': ms,
                'n_clusters': n_cls, 'noise_pct': noise_pct, 'ARI': ari
            })

    # ==================================================================
    # 5. Permutation test at fixed grid point (English, min_size=10 default)
    # ==================================================================
    print("\n" + "=" * 60)
    print(f"Permutation test (EN-only, min_size={args.permutation_min_size})")
    print("=" * 60)

    X_en = StandardScaler().fit_transform(embeddings[en_mask_full])
    y_en = df['level_primary'].values[en_mask_full]

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=args.permutation_min_size,
        metric='euclidean', cluster_selection_method='eom'
    )
    pred_fixed = clusterer.fit_predict(X_en)
    mask_fixed = pred_fixed != -1
    observed_ari = adjusted_rand_score(y_en[mask_fixed], pred_fixed[mask_fixed])

    rng = np.random.RandomState(SEED)
    null_aris = np.array([
        adjusted_rand_score(rng.permutation(y_en[mask_fixed]), pred_fixed[mask_fixed])
        for _ in range(args.n_permutations)
    ])
    p_value = (null_aris >= observed_ari).mean()

    print(f"Observed ARI: {observed_ari:.4f}")
    print(f"Null distribution: mean={null_aris.mean():.4f}, std={null_aris.std():.4f}")
    print(f"Permutation p-value (n={args.n_permutations}): {p_value:.4f}")

    # ==================================================================
    # 6. Cluster composition cross-tabulation at fixed grid point
    # ==================================================================
    print("\n" + "=" * 60)
    print(f"Cluster composition (EN-only, min_size={args.permutation_min_size})")
    print("=" * 60)

    label_map_reverse_local = {0: 'L0', 1: 'L3', 2: 'L2', 3: 'L1'}
    y_en_named = pd.Series(y_en[mask_fixed]).map(label_map_reverse_local)
    composition = pd.crosstab(pred_fixed[mask_fixed], y_en_named)
    print(composition)

    for level in ['L0', 'L1', 'L2', 'L3']:
        total = (df.loc[en_mask_full, 'label'] == level).sum()
        print(f"  {level} total in EN subset: {total} "
              f"({100 * total / en_mask_full.sum():.1f}%)")

    # ==================================================================
    # 7. Save results
    # ==================================================================
    all_results = pd.DataFrame(pooled_results + mono_results)
    all_results.to_csv(args.output, index=False)
    print(f"\nSaved grid results to: {args.output}")

    composition.to_csv(args.output.replace('.csv', '_composition.csv'))
    print(f"Saved cluster composition to: "
          f"{args.output.replace('.csv', '_composition.csv')}")

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("Pooled (KO+EN): near-zero ARI across all tested min_cluster_size,")
    print("  but clusters are overwhelmingly monolingual (see mean_lang_purity)")
    print("  -- the null result reflects a language-identity axis, not evidence")
    print("  against risk-level structure.")
    print("Korean-only: expect near-total noise (inconclusive) at")
    print("  min_cluster_size >= 10 due to smaller sample size and class imbalance.")
    print("English-only: ARI should rise monotonically to a grid maximum around")
    print("  min_cluster_size=10, then collapse to a near-null regime, with a")
    print("  possible isolated exception at specific grid points (see paper")
    print("  Appendix for diagnostics on any such anomaly).")


if __name__ == '__main__':
    main()
