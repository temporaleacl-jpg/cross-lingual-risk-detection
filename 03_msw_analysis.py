# -*- coding: utf-8 -*-
"""
03_msw_analysis.py
MSW Calculation with Cross-Corpus Validation (ESConv, AnnoMI)
EACL 2027 Submission (ARR August 2026)
Requirements:
    pip install lifelines pandas numpy scipy
"""

import argparse
import json
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, kruskal
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test, proportional_hazard_test

# ── Constants ─────────────────────────────────────────────────────────────────

THRESHOLD = 0.5   # P(L2) >= 0.5 → risk onset

# English risk cue keywords (ESConv / AnnoMI)
RISK_CUES_EN = [
    'suicide', 'suicidal', 'kill myself', 'end my life', 'want to die',
    'wish i was dead', 'better off dead', 'not worth living',
    'no reason to live', 'self-harm', 'hurt myself', 'cut myself',
    'hopeless', 'worthless', 'meaningless', 'empty', 'numb', 'no hope',
    'give up', "can't go on", 'no point', 'depressed', 'depression',
    'anxiety', 'anxious', 'overwhelmed', 'exhausted', 'hate myself',
    'hate my life', 'lonely', 'alone', 'isolated', 'nobody cares',
    'no one cares', 'crying', 'cry all the time', 'cannot cope',
    "can't cope", 'too much', 'falling apart', 'breaking down', 'lost',
    'scared', 'terrified', 'tired of living', 'tired of life',
    'done with life', 'want to disappear',
]

# ── Korean MSW utilities ───────────────────────────────────────────────────────

def find_first_risk_turn(dlg_turns):
    dlg_turns = dlg_turns.sort_values('turn_id')
    for _, turn in dlg_turns.iterrows():
        if turn['L2_score'] >= THRESHOLD:
            return turn['turn_id']
    return None


def fit_km(df, dur_col='first_risk_turn',
           total_col='total_turns', risk_col='has_risk'):
    dur, ev = [], []
    for _, row in df.iterrows():
        if row[risk_col]:
            dur.append(float(row[dur_col]))
            ev.append(True)
        else:
            dur.append(float(row[total_col]))
            ev.append(False)
    kmf = KaplanMeierFitter()
    kmf.fit(durations=dur, event_observed=ev)
    return kmf


def get_msw(kmf, alpha):
    sf = kmf.survival_function_
    below = sf[sf.iloc[:, 0] <= 1 - alpha]
    return int(below.index[0]) if len(below) > 0 else None


def bootstrap_ci(df, alpha, n=1000, seed=42):
    np.random.seed(seed)
    boots = []
    for _ in range(n):
        s = df.sample(len(df), replace=True)
        m = get_msw(fit_km(s), alpha)
        if m is not None:
            boots.append(m)
    if len(boots) < 10:
        return None, None
    return (int(np.percentile(boots, 2.5)),
            int(np.percentile(boots, 97.5)))


def km_arrays(df, dur_col='first_risk_turn',
              total_col='total_turns', risk_col='has_risk'):
    d = np.array([
        float(r[dur_col]) if r[risk_col] else float(r[total_col])
        for _, r in df.iterrows()])
    e = np.array([int(r[risk_col]) for _, r in df.iterrows()])
    return d, e


# ── English corpus utilities ───────────────────────────────────────────────────

def find_first_risk_en(dialog):
    """Find first turn index where seeker expresses a risk cue."""
    for i, turn in enumerate(dialog):
        if turn.get('speaker') == 'seeker':
            text = str(turn.get('content', '')).lower()
            if any(c in text for c in RISK_CUES_EN):
                return i + 1
    return None


def load_esconv(path):
    """Load ESConv and compute first_risk_turn per dialogue."""
    with open(path, 'r', encoding='utf-8') as f:
        esconv = json.load(f)

    records = []
    for d in esconv:
        dialog  = d.get('dialog', [])
        total   = len(dialog)
        first   = find_first_risk_en(dialog)
        emotion = d.get('emotion_type', '')
        problem = d.get('problem_type', '')
        is_dep  = (emotion == 'depression' or
                   'depression' in str(problem).lower())
        records.append({
            'total_turns':     total,
            'first_risk_turn': first,
            'has_risk':        first is not None,
            'is_depression':   is_dep,
        })
    return pd.DataFrame(records)


def load_annomi(path):
    """Load AnnoMI-simple and compute first_risk_turn per transcript."""
    df = pd.read_csv(path)
    records = []
    for tid, grp in df.groupby('transcript_id'):
        grp = grp.sort_values('utterance_id')
        total = len(grp)
        first = None
        for _, row in grp.iterrows():
            text = str(row.get('utterance_text', '')).lower()
            if any(c in text for c in RISK_CUES_EN):
                first = int(row['utterance_id'])
                break
        records.append({
            'total_turns':     total,
            'first_risk_turn': first,
            'has_risk':        first is not None,
        })
    return pd.DataFrame(records)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='MSW analysis with cross-corpus validation')
    parser.add_argument('--scores',
        default='ko_turn_scores.csv',
        help='Korean turn-level L2 scores')
    parser.add_argument('--manifest',
        default='ko_www_manifest.csv',
        help='Korean session manifest')
    parser.add_argument('--esconv',
        default='ESConv.json',
        help='ESConv JSON file')
    parser.add_argument('--annomi',
        default='AnnoMI-simple.csv',
        help='AnnoMI CSV file')
    parser.add_argument('--output',
        default='ko_dialogue_stats_final.csv')
    args = parser.parse_args()

    # ── 1. Korean MSW ──────────────────────────────────────────────────────────
    print('=' * 60)
    print('1. KOREAN MSW ANALYSIS')
    print('=' * 60)

    turn_df  = pd.read_csv(args.scores)
    manifest = pd.read_csv(args.manifest)

    records = []
    for _, row in manifest.iterrows():
        did   = row['dialogue_id']
        cat   = row['stratum']
        total = row['turn_count']
        dlg   = turn_df[turn_df['dialogue_id'] == did]
        first = find_first_risk_turn(dlg) if len(dlg) > 0 else None
        records.append({
            'dialogue_id':    did,
            'category':       cat,
            'total_turns':    total,
            'first_risk_turn': first,
            'has_risk':       first is not None,
        })

    ko_df = pd.DataFrame(records)
    print(f"Risk detected: {ko_df['has_risk'].sum()}/{len(ko_df)}")

    # Overall MSW
    print('\n--- Overall MSW ---')
    kmf_all = fit_km(ko_df)
    for alpha in [0.5, 0.8, 0.9, 0.95]:
        t      = get_msw(kmf_all, alpha)
        lo, hi = bootstrap_ci(ko_df, alpha)
        print(f"  MSW_{alpha}: {t} [{lo}–{hi}]")

    # Category MSW
    print('\n--- Category MSW ---')
    cats = ['depression', 'anxiety', 'addiction', 'normative']
    for cat in cats:
        cat_df = ko_df[ko_df['category'] == cat]
        kmf_c  = fit_km(cat_df)
        t5, t9 = get_msw(kmf_c, 0.5), get_msw(kmf_c, 0.9)
        lo5, hi5 = bootstrap_ci(cat_df, 0.5)
        lo9, hi9 = bootstrap_ci(cat_df, 0.9)
        print(f"\n  {cat} (n={len(cat_df)}):")
        print(f"    MSW_0.5: {t5} [{lo5}–{hi5}]")
        print(f"    MSW_0.9: {t9} [{lo9}–{hi9}]")

    # Log-rank: depression vs others
    print('\n--- Log-rank: Depression vs Others ---')
    dep_ko = ko_df[ko_df['category'] == 'depression']
    oth_ko = ko_df[ko_df['category'] != 'depression']
    res    = logrank_test(*km_arrays(dep_ko), *km_arrays(oth_ko))
    print(f"  p = {res.p_value:.4f}")

    # Cox PH + Schoenfeld
    print('\n--- Cox PH + Schoenfeld ---')
    cox = ko_df.copy()
    cox['is_dep']   = (cox['category'] == 'depression').astype(int)
    cox['duration'] = cox.apply(
        lambda r: float(r['first_risk_turn'])
        if r['has_risk'] else float(r['total_turns']), axis=1)
    cox['event'] = cox['has_risk'].astype(int)
    cph = CoxPHFitter()
    cph.fit(cox[['duration', 'event', 'is_dep']],
            duration_col='duration', event_col='event')
    print(cph.summary[['coef', 'exp(coef)', 'p']].round(4))
    try:
        ph = proportional_hazard_test(
            cph, cox[['duration', 'event', 'is_dep']],
            time_transform='rank')
        print(f"  Schoenfeld p = "
              f"{ph.summary['p'].values[0]:.4f}")
    except Exception as e:
        print(f"  Schoenfeld test skipped: {e}")

    ko_df.to_csv(args.output, index=False, encoding='utf-8-sig')
    print(f"\n✅ Korean stats saved: {args.output}")

    # ── 2. ESConv Cross-Corpus Validation ─────────────────────────────────────
    print('\n' + '=' * 60)
    print('2. ESCONV CROSS-CORPUS VALIDATION')
    print('=' * 60)

    try:
        es_df = load_esconv(args.esconv)
        es_dep = es_df[es_df['is_depression']].copy()
        es_all = es_df.copy()

        print(f"ESConv total:      {len(es_all)} dialogues")
        print(f"ESConv depression: {len(es_dep)} dialogues")

        # MSW: ESConv overall
        kmf_es_all = fit_km(es_all)
        t_es_all   = get_msw(kmf_es_all, 0.5)
        lo, hi     = bootstrap_ci(es_all, 0.5)
        print(f"\n  ESConv All MSW_0.5: {t_es_all} [{lo}–{hi}]")

        # MSW: ESConv depression
        kmf_es_dep = fit_km(es_dep)
        t_es_dep   = get_msw(kmf_es_dep, 0.5)
        lo, hi     = bootstrap_ci(es_dep, 0.5)
        print(f"  ESConv Dep MSW_0.5: {t_es_dep} [{lo}–{hi}]")

        # Log-rank: KO depression vs ESConv depression
        print('\n--- Log-rank: KO-Dep vs ESConv-Dep ---')
        res_dep = logrank_test(
            *km_arrays(dep_ko),
            *km_arrays(es_dep))
        print(f"  p = {res_dep.p_value:.4f} "
              f"({'ns' if res_dep.p_value > 0.05 else '*'})")

        # Mann-Whitney: raw first_risk_turn
        print('\n--- Mann-Whitney: first_risk_turn ---')
        ko_frt = dep_ko[dep_ko['has_risk']]['first_risk_turn'].dropna()
        es_frt = es_dep[es_dep['has_risk']]['first_risk_turn'].dropna()
        stat, p_mw = mannwhitneyu(ko_frt, es_frt, alternative='two-sided')
        print(f"  KO-Dep: n={len(ko_frt)}, "
              f"median={ko_frt.median():.1f}, "
              f"mean={ko_frt.mean():.1f}")
        print(f"  ES-Dep: n={len(es_frt)}, "
              f"median={es_frt.median():.1f}, "
              f"mean={es_frt.mean():.1f}")
        print(f"  U={stat:.1f}, p={p_mw:.4f} "
              f"({'*' if p_mw < 0.05 else 'ns'})")

        # Kruskal-Wallis: KO vs ESConv vs AnnoMI (loaded below)
        print('\n--- Kruskal-Wallis: KO vs ESConv vs AnnoMI ---')
        ko_all_frt = ko_df[ko_df['has_risk']]['first_risk_turn'].dropna()
        es_all_frt = es_all[es_all['has_risk']]['first_risk_turn'].dropna()

        try:
            an_df      = load_annomi(args.annomi)
            an_frt     = an_df[an_df['has_risk']]['first_risk_turn'].dropna()
            stat_kw, p_kw = kruskal(ko_all_frt, es_all_frt, an_frt)
            print(f"  stat={stat_kw:.3f}, p={p_kw:.4f} "
                  f"({'***' if p_kw < 0.001 else '*' if p_kw < 0.05 else 'ns'})")
        except Exception as e:
            print(f"  AnnoMI not loaded for KW test: {e}")
            stat_kw, p_kw = kruskal(ko_all_frt, es_all_frt)
            print(f"  (KO vs ESConv only) "
                  f"stat={stat_kw:.3f}, p={p_kw:.4f}")

    except FileNotFoundError:
        print(f"  ESConv file not found: {args.esconv}")
        print("  Skipping cross-corpus validation.")

    # ── 3. AnnoMI Validation ──────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print('3. ANNOMI VALIDATION')
    print('=' * 60)

    try:
        an_df = load_annomi(args.annomi)
        print(f"AnnoMI total: {len(an_df)} transcripts")
        print(f"Risk detected: "
              f"{an_df['has_risk'].sum()}/{len(an_df)}")

        kmf_an  = fit_km(an_df)
        t_an_50 = get_msw(kmf_an, 0.5)
        t_an_90 = get_msw(kmf_an, 0.9)
        print(f"\n  AnnoMI MSW_0.5: "
              f"{t_an_50 if t_an_50 else '>max'}")
        print(f"  AnnoMI MSW_0.9: "
              f"{t_an_90 if t_an_90 else '>max'}")

        an_frt = an_df[an_df['has_risk']]['first_risk_turn'].dropna()
        print(f"\n  first_risk_turn stats (events only):")
        print(f"    n={len(an_frt)}, "
              f"median={an_frt.median():.1f}, "
              f"mean={an_frt.mean():.1f}")

    except FileNotFoundError:
        print(f"  AnnoMI file not found: {args.annomi}")
        print("  Skipping AnnoMI validation.")

    print('\n' + '=' * 60)
    print('DONE')
    print('=' * 60)


if __name__ == '__main__':
    main()
