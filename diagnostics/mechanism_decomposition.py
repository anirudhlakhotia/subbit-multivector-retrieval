"""c6 argmax-floor decomposition.

For each MS MARCO Passage dev query and each query token, compute:

  - fp32 per-q-token MaxSim contribution s_i = max_(j, t) q_i . d_t^(j)
    over the top-100 fp32 candidate docs (matching argmax_stability_rsweep.py).
  - per-q-token margin m_i = s_i - s_i^(2) where s_i^(2) is the
    second-best (cand-doc, doc-token) similarity for query token i.
  - preservation flag preserved_i^(r) at r=64 random orthogonal (seed 42),
    asymmetric scoring: q stays fp32, doc becomes sign(R d).

Aggregate per query:
  S_high(q)  = sum over q-tokens with m_i >= 0.2 of s_i
  S_low(q)   = sum over q-tokens with m_i <  0.2 of s_i
  S_total(q) = S_high(q) + S_low(q)
  f_high(q)  = S_high(q) / S_total(q)
  n_high(q) / m_q

Verdict rule is recorded in the paper diagnostic artifacts.
Output: outputs/c6_decomposition.json.
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path('.').resolve()))
from src.subbit.data import EmbeddingStore, resolve_embedding_cache_path, load_qrels


R_DIM = 64
MARGIN_THRESHOLD = 0.2
SEED = 42
TOP_K_CANDIDATES = 100


def build_random_R(r_dim, seed):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(r_dim, 128, dtype=torch.float64, generator=g)
    Q, _ = torch.linalg.qr(A.T)
    return Q.T[:r_dim, :].to(torch.float32)


def _stats(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return {
            'n': 0,
            'mean': None,
            'median': None,
            'p01': None,
            'p05': None,
            'p25': None,
            'p75': None,
            'p95': None,
            'p99': None,
            'min': None,
            'max': None,
            'frac_ge_minus_0_01': None,
            'frac_ge_minus_0_02': None,
            'frac_ge_minus_0_05': None,
            'frac_ge_minus_0_10': None,
        }
    return {
        'n': int(x.size),
        'mean': float(np.mean(x)),
        'median': float(np.median(x)),
        'p01': float(np.quantile(x, 0.01)),
        'p05': float(np.quantile(x, 0.05)),
        'p25': float(np.quantile(x, 0.25)),
        'p75': float(np.quantile(x, 0.75)),
        'p95': float(np.quantile(x, 0.95)),
        'p99': float(np.quantile(x, 0.99)),
        'min': float(np.min(x)),
        'max': float(np.max(x)),
        'frac_ge_minus_0_01': float(np.mean(x >= -0.01)),
        'frac_ge_minus_0_02': float(np.mean(x >= -0.02)),
        'frac_ge_minus_0_05': float(np.mean(x >= -0.05)),
        'frac_ge_minus_0_10': float(np.mean(x >= -0.10)),
    }


def summarize_query_residuals(arrays):
    qids = arrays['qid'].astype(str)
    unique, inv = np.unique(qids, return_inverse=True)
    query_delta = np.zeros(len(unique), dtype=np.float64)
    np.add.at(query_delta, inv, arrays['delta_fp32'].astype(np.float64))
    out = {
        'n_queries': int(len(unique)),
        'sum_delta_fp32': _stats(query_delta),
    }
    if 'query_fp32_sum' in arrays:
        total_accum = np.zeros(len(unique), dtype=np.float64)
        counts = np.zeros(len(unique), dtype=np.float64)
        np.add.at(total_accum, inv, arrays['query_fp32_sum'].astype(np.float64))
        np.add.at(counts, inv, 1.0)
        query_total = total_accum / np.maximum(counts, 1.0)
        rel = query_delta / np.maximum(query_total, 1e-12)
        out['relative_to_fp32_sum'] = _stats(rel)
    return out


def summarize_residuals(arrays, margin_threshold):
    delta = arrays['delta_fp32']
    margins = arrays['margin']
    preserved = arrays['preserved'].astype(bool)
    low = margins < margin_threshold
    high = ~low
    flipped = ~preserved
    return {
        'definition': (
            'delta_fp32 = fp32_score(sign_coded_argmax_token) - '
            'fp32_score(fp32_argmax_token); values are <= 0 by construction.'
        ),
        'all_q_tokens': _stats(delta),
        'low_margin_flipped': _stats(delta[low & flipped]),
        'low_margin_preserved': _stats(delta[low & preserved]),
        'high_margin_flipped': _stats(delta[high & flipped]),
        'high_margin_preserved': _stats(delta[high & preserved]),
        'per_query_aggregate': summarize_query_residuals(arrays),
    }


def plot_residual_density(arrays, stats, output_path, margin_threshold):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    delta = arrays['delta_fp32']
    margins = arrays['margin']
    preserved = arrays['preserved'].astype(bool)
    mask = (margins < margin_threshold) & (~preserved)
    x = delta[mask]
    if x.size == 0:
        raise ValueError('no flipped low-margin q-token residuals to plot')

    x_min = max(float(np.quantile(x, 0.005)), -0.30)
    bins = np.linspace(x_min, 0.0, 80)

    fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.15))
    ax = axes[0]
    ax.hist(x, bins=bins, density=True, color='#4C72B0',
            alpha=0.72, edgecolor='white', linewidth=0.25)
    ax.axvline(0.0, color='black', linewidth=0.9)
    ax.axvline(stats['low_margin_flipped']['median'], color='#B03A2E',
               linestyle='--', linewidth=0.9)
    ax.set_xlabel(r'fp32 residual $\delta_i$')
    ax.set_ylabel('density')
    ax.set_title('flipped low-margin q-tokens')
    ax.text(
        0.03, 0.96,
        (
            f"n={stats['low_margin_flipped']['n']:,}\n"
            f"median={stats['low_margin_flipped']['median']:.4f}\n"
            f"5th pct={stats['low_margin_flipped']['p05']:.4f}"
        ),
        transform=ax.transAxes,
        va='top',
        ha='left',
        fontsize=7,
        bbox={'boxstyle': 'round,pad=0.25', 'facecolor': 'white',
              'edgecolor': '#cccccc', 'alpha': 0.92},
    )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    qids = arrays['qid'].astype(str)
    unique, inv = np.unique(qids, return_inverse=True)
    query_delta = np.zeros(len(unique), dtype=np.float64)
    np.add.at(query_delta, inv, delta.astype(np.float64))

    ax = axes[1]
    if 'query_fp32_sum' in arrays:
        total_accum = np.zeros(len(unique), dtype=np.float64)
        counts = np.zeros(len(unique), dtype=np.float64)
        np.add.at(total_accum, inv, arrays['query_fp32_sum'].astype(np.float64))
        np.add.at(counts, inv, 1.0)
        query_total = total_accum / np.maximum(counts, 1.0)
        y = query_delta / np.maximum(query_total, 1e-12)
        y_min = max(float(np.quantile(y, 0.005)), -0.22)
        y_bins = np.linspace(y_min, 0.0, 70)
        ax.hist(y, bins=y_bins, density=True, color='#55A868',
                alpha=0.72, edgecolor='white', linewidth=0.25)
        ax.axvline(0.0, color='black', linewidth=0.9)
        ax.axvline(np.median(y), color='#B03A2E',
                   linestyle='--', linewidth=0.9)
        ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.set_xlabel(r'query residual / fp32 sum')
        ax.text(
            0.03, 0.96,
            (
                f"n={len(unique):,}\n"
                f"median={np.median(y):.1%}\n"
                f"{np.mean(y >= -0.10):.1%} >= -10%"
            ),
            transform=ax.transAxes,
            va='top',
            ha='left',
            fontsize=7,
            bbox={'boxstyle': 'round,pad=0.25', 'facecolor': 'white',
                  'edgecolor': '#cccccc', 'alpha': 0.92},
        )
    else:
        y_min = max(float(np.quantile(query_delta, 0.005)), -1.5)
        y_bins = np.linspace(y_min, 0.0, 70)
        ax.hist(query_delta, bins=y_bins, density=True, color='#55A868',
                alpha=0.72, edgecolor='white', linewidth=0.25)
        ax.axvline(0.0, color='black', linewidth=0.9)
        ax.axvline(np.median(query_delta), color='#B03A2E',
                   linestyle='--', linewidth=0.9)
        ax.set_xlabel(r'query residual $\sum_i \delta_i$')
    ax.set_ylabel('density')
    ax.set_title('per-query aggregate')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout(pad=0.35)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)


def main():
    global R_DIM   # allow --r-dim to override the module default within main
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--emb-dir', type=Path, default=Path('data/embeddings/msmarco/100k'))
    ap.add_argument('--output', type=Path, default=Path('outputs/c6_decomposition.json'))
    ap.add_argument('--n-queries', default='all',
                    help='"all" or integer count of judged queries to sample.')
    ap.add_argument('--margin-threshold', type=float, default=MARGIN_THRESHOLD)
    ap.add_argument('--r-dim', type=int, default=R_DIM,
                    help='Sign-code dim: 64 (down-projected, default) or 128 (full-rotation sign).')
    ap.add_argument('--save-top-k', type=int, default=512,
                    help='Per-q-token top-K fp32 candidate sims persisted in the residual npz '
                         '(reusable substrate: margin/plateau/sum-decomp/gaps at any threshold, offline).')
    ap.add_argument('--checkpoint', type=Path, default=None,
                    help='Optional trained SubBit checkpoint: use its R (R.weight) for the sign code '
                         'instead of a random orthogonal. Scale head is irrelevant here (per-q-token '
                         'argmax is invariant to a positive per-token scalar), so only R is loaded.')
    ap.add_argument('--residual-output', type=Path, default=None,
                    help='Optional .npz path for per-q-token residual arrays.')
    ap.add_argument('--residual-summary-output', type=Path, default=None,
                    help='Optional JSON path for residual summary statistics.')
    ap.add_argument('--residual-fig-output', type=Path, default=None,
                    help='Optional PDF/PNG path for the low-margin residual density figure.')
    args = ap.parse_args()

    EMB_DIR = args.emb_dir
    OUT_JSON = args.output
    M_THRESH = float(args.margin_threshold)
    R_DIM = args.r_dim   # local override of the module default (e.g. 128)
    SAVE_TOP_K = max(10, int(args.save_top_k))   # >=10 for the gap features
    if isinstance(args.n_queries, str) and args.n_queries.lower() == 'all':
        N_SAMPLED = None
    else:
        N_SAMPLED = int(args.n_queries)

    print('=== c6 argmax-floor decomposition ===', flush=True)
    print(f'  emb_dir          = {EMB_DIR}', flush=True)
    print(f'  output           = {OUT_JSON}', flush=True)
    print(f'  margin threshold = {M_THRESH}', flush=True)
    print(f'  r (preservation) = {R_DIM}, seed = {SEED}', flush=True)
    print(f'  top_k_candidates = {TOP_K_CANDIDATES}', flush=True)
    if args.residual_output:
        print(f'  residual_output  = {args.residual_output}', flush=True)
    if args.residual_summary_output:
        print(f'  residual_summary = {args.residual_summary_output}', flush=True)
    if args.residual_fig_output:
        print(f'  residual_figure  = {args.residual_fig_output}', flush=True)
    print()

    print('Loading ColBERTv2 stores...', flush=True)
    doc_store = EmbeddingStore(resolve_embedding_cache_path(EMB_DIR, 'doc'), mode='dict'); doc_store.load()
    qry_store = EmbeddingStore(resolve_embedding_cache_path(EMB_DIR, 'query'), mode='dict'); qry_store.load()
    doc_ids = list(doc_store.get_all_ids())
    N = len(doc_ids)
    qrels = load_qrels(EMB_DIR / 'qrels.tsv')

    print('Building padded fp32 doc tensor...', flush=True)
    t0 = time.perf_counter()
    lengths = np.array([doc_store.get(d).shape[0] for d in doc_ids], dtype=np.int32)
    max_len = int(lengths.max())
    doc_padded = torch.zeros(N, max_len, 128, dtype=torch.float32)
    doc_mask = torch.zeros(N, max_len, dtype=torch.bool)
    for i, did in enumerate(doc_ids):
        d_fp = doc_store.get(did).float()
        n_t = d_fp.shape[0]
        doc_padded[i, :n_t] = d_fp
        doc_mask[i, :n_t] = True
    print(f'  built in {time.perf_counter()-t0:.1f}s', flush=True)

    if args.checkpoint is not None:
        ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        sd = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        R = sd['R.weight'].float()
        assert R.shape == (R_DIM, 128), f'checkpoint R is {tuple(R.shape)}, expected ({R_DIM}, 128) — pass matching --r-dim'
        R_SOURCE = f'trained:{args.checkpoint}'
        print(f'R: TRAINED from {args.checkpoint}  ||R||_F={R.norm():.4f}', flush=True)
    else:
        R = build_random_R(R_DIM, SEED)
        R_SOURCE = f'random_orthogonal_seed_{SEED}'
    print('Pre-projecting doc corpus under sign(R d)...', flush=True)
    t0 = time.perf_counter()
    doc_signs = torch.sign(doc_padded.reshape(-1, 128) @ R.T).reshape(N, max_len, R_DIM)
    doc_signs = doc_signs.masked_fill(~doc_mask.unsqueeze(-1), 0.0)
    print(f'  projected in {time.perf_counter()-t0:.1f}s', flush=True)

    judged_qids = sorted(qrels.keys())
    rng = np.random.default_rng(SEED)
    if N_SAMPLED is not None and N_SAMPLED < len(judged_qids):
        idx = rng.permutation(len(judged_qids))[:N_SAMPLED]
        sample_qids = [judged_qids[i] for i in idx]
    else:
        sample_qids = judged_qids
    print(f'Running over {len(sample_qids)} judged queries', flush=True)

    per_query = []  # one dict per query
    save_residuals = any([
        args.residual_output,
        args.residual_summary_output,
        args.residual_fig_output,
    ])
    residual_chunks = {
        'qid': [],
        'qtok_idx': [],
        'top_k_sims': [],         # (n_qtok, save_top_k) top-K fp32 candidate sims — reusable substrate
        'margin': [],
        'preserved': [],
        'fp32_top': [],
        'fp32_at_sign_winner': [],
        'delta_fp32': [],
        'query_fp32_sum': [],
        'sign_winner_rank': [],   # rank of sign-coded winner in fp32 order (1 = fp32 argmax)
        'gap_top2': [],           # fp32_top - fp32_2nd  (near-tie cluster gaps)
        'gap_top3': [],
        'gap_top5': [],
        'gap_top10': [],
        'plateau_eps02': [],   # #{doc-tokens with fp32 score > s_max - 0.02}  (>=1, incl. winner)
        'plateau_eps05': [],   # near-tie plateau size at eps=0.05
        'plateau_eps10': [],   # near-tie plateau size at eps=0.10
    }

    t0 = time.perf_counter()
    for qid in tqdm(sample_qids, desc='queries'):
        try:
            q_fp32 = qry_store.get(qid).float()  # (m_q, 128)
        except (KeyError, FileNotFoundError):
            continue
        m_q = q_fp32.shape[0]

        with torch.no_grad():
            # Brute fp32 MaxSim over corpus to pick top-100 candidates
            sim_full = torch.einsum('md,knd->kmn', q_fp32, doc_padded)  # (N, m_q, max_len)
            sim_full = sim_full.masked_fill(~doc_mask[:, None, :], float('-inf'))
            ms_full = sim_full.max(dim=-1).values   # (N, m_q)
            scores_full = ms_full.sum(dim=-1)       # (N,)

            topk_idx = torch.topk(scores_full, TOP_K_CANDIDATES).indices  # (K,)

            cand_docs = doc_padded[topk_idx]        # (K, max_len, 128)
            cand_mask = doc_mask[topk_idx]          # (K, max_len)
            cand_signs = doc_signs[topk_idx]        # (K, max_len, R_DIM)

            # fp32 per-q-token argmax over (cand, doc_token)
            sim_cand = torch.einsum('md,knd->mkn', q_fp32, cand_docs)  # (m_q, K, max_len)
            sim_cand = sim_cand.masked_fill(~cand_mask[None, :, :], float('-inf'))
            flat = sim_cand.reshape(m_q, TOP_K_CANDIDATES * max_len)
            fp32_winner = flat.argmax(dim=-1)                 # (m_q,)
            fp32_top    = flat.max(dim=-1).values              # (m_q,)
            flat_clone  = flat.clone()
            flat_clone.scatter_(1, fp32_winner.unsqueeze(-1), float('-inf'))
            fp32_second = flat_clone.max(dim=-1).values        # (m_q,)
            fp32_margin = (fp32_top - fp32_second)             # (m_q,)

            # Sign-coded per-q-token argmax at r=R_DIM, asymmetric
            q_proj = q_fp32 @ R.T                                            # (m_q, r)
            sim_r = torch.einsum('md,knd->mkn', q_proj, cand_signs)          # (m_q, K, max_len)
            sim_r = sim_r.masked_fill(~cand_mask[None, :, :], float('-inf'))
            r_flat = sim_r.reshape(m_q, TOP_K_CANDIDATES * max_len)
            r_winner = r_flat.argmax(dim=-1)                                 # (m_q,)
            if save_residuals:
                fp32_at_r_winner = flat.gather(1, r_winner.unsqueeze(-1)).squeeze(-1)
                delta_fp32 = fp32_at_r_winner - fp32_top
                # Rank of the sign-coded winner in the fp32 doc-token ordering
                # (1 = the flip preserved the fp32 argmax). Counts candidate
                # doc-tokens scoring strictly above where the flip landed;
                # masked (-inf) positions never count.
                sign_winner_rank = (flat > fp32_at_r_winner.unsqueeze(-1)).sum(dim=-1) + 1
                # fp32 gaps top-1 -> k-th best doc-token: the local near-tie cluster.
                Keff = min(SAVE_TOP_K, flat.shape[1])
                topvals = torch.topk(flat, Keff, dim=-1).values          # (m_q, Keff) sorted desc
                gap_top2  = fp32_top - topvals[:, 1]
                gap_top3  = fp32_top - topvals[:, 2]
                gap_top5  = fp32_top - topvals[:, 4]
                gap_top10 = fp32_top - topvals[:, 9]
                # full top-K fp32 candidate sims = the reusable substrate (any-threshold
                # margin / plateau / sum-decomposition / top1-topk gaps, all offline).
                tksims = topvals.detach().cpu().numpy().astype(np.float32)
                if Keff < SAVE_TOP_K:
                    tksims = np.concatenate(
                        [tksims, np.full((tksims.shape[0], SAVE_TOP_K - Keff), -np.inf, np.float32)], axis=1)
                # Exact near-tie plateau size: # candidate doc-tokens within eps of the
                # winner's fp32 score (masked -inf positions never count; winner included).
                ft = fp32_top.unsqueeze(-1)
                plateau_eps02 = (flat > (ft - 0.02)).sum(dim=-1)
                plateau_eps05 = (flat > (ft - 0.05)).sum(dim=-1)
                plateau_eps10 = (flat > (ft - 0.10)).sum(dim=-1)

        preserved = (r_winner == fp32_winner)

        if save_residuals:
            residual_chunks['qid'].append(np.array([qid] * m_q))
            residual_chunks['qtok_idx'].append(np.arange(m_q, dtype=np.int16))
            residual_chunks['top_k_sims'].append(tksims)
            residual_chunks['margin'].append(fp32_margin.numpy().astype(np.float32))
            residual_chunks['preserved'].append(preserved.numpy().astype(bool))
            residual_chunks['fp32_top'].append(fp32_top.numpy().astype(np.float32))
            residual_chunks['fp32_at_sign_winner'].append(
                fp32_at_r_winner.numpy().astype(np.float32)
            )
            residual_chunks['delta_fp32'].append(delta_fp32.numpy().astype(np.float32))
            residual_chunks['query_fp32_sum'].append(
                np.full(m_q, float(fp32_top.sum().item()), dtype=np.float32)
            )
            residual_chunks['sign_winner_rank'].append(sign_winner_rank.numpy().astype(np.int32))
            residual_chunks['gap_top2'].append(gap_top2.numpy().astype(np.float32))
            residual_chunks['gap_top3'].append(gap_top3.numpy().astype(np.float32))
            residual_chunks['gap_top5'].append(gap_top5.numpy().astype(np.float32))
            residual_chunks['gap_top10'].append(gap_top10.numpy().astype(np.float32))
            residual_chunks['plateau_eps02'].append(plateau_eps02.numpy().astype(np.int32))
            residual_chunks['plateau_eps05'].append(plateau_eps05.numpy().astype(np.int32))
            residual_chunks['plateau_eps10'].append(plateau_eps10.numpy().astype(np.int32))

        # Bucketize
        fp32_top_np = fp32_top.numpy().astype(np.float64)
        fp32_margin_np = fp32_margin.numpy().astype(np.float64)
        preserved_np = preserved.numpy().astype(bool)

        is_high = fp32_margin_np >= M_THRESH
        n_high = int(is_high.sum())
        S_high = float(fp32_top_np[is_high].sum())
        S_low  = float(fp32_top_np[~is_high].sum())
        S_total = S_high + S_low
        S_high_preserved = float(fp32_top_np[is_high & preserved_np].sum())
        S_low_preserved  = float(fp32_top_np[~is_high & preserved_np].sum())
        n_high_preserved = int((is_high & preserved_np).sum())
        n_low_preserved  = int((~is_high & preserved_np).sum())

        per_query.append({
            'qid': qid,
            'm_q': int(m_q),
            'n_high': n_high,
            'n_low': int(m_q - n_high),
            'S_high': S_high,
            'S_low': S_low,
            'S_total': S_total,
            'f_high': (S_high / S_total) if S_total > 0 else 0.0,
            'n_high_frac': n_high / m_q,
            'S_high_preserved': S_high_preserved,
            'S_low_preserved': S_low_preserved,
            'n_high_preserved': n_high_preserved,
            'n_low_preserved': n_low_preserved,
        })

    elapsed = time.perf_counter() - t0
    print(f'\n  total: {elapsed:.1f}s for {len(per_query)} queries', flush=True)

    # Aggregate
    f_high = np.array([q['f_high'] for q in per_query])
    n_high_frac = np.array([q['n_high_frac'] for q in per_query])

    total_high = sum(q['S_high'] for q in per_query)
    total_low  = sum(q['S_low']  for q in per_query)
    total_all  = total_high + total_low
    total_n_high = sum(q['n_high'] for q in per_query)
    total_m_q    = sum(q['m_q']    for q in per_query)
    total_S_high_preserved = sum(q['S_high_preserved'] for q in per_query)
    total_S_low_preserved  = sum(q['S_low_preserved']  for q in per_query)
    total_n_high_preserved = sum(q['n_high_preserved'] for q in per_query)
    total_n_low_preserved  = sum(q['n_low_preserved']  for q in per_query)

    summary = {
        'config': {
            'n_queries': len(per_query),
            'n_judged_total': len(judged_qids),
            'margin_threshold': M_THRESH,
            'r_dim': R_DIM,
            'seed': SEED,
            'r_source': R_SOURCE,
            'top_k_candidates': TOP_K_CANDIDATES,
            'corpus_n': N,
            'emb_dir': str(EMB_DIR),
        },
        'population': {
            'median_n_high_frac': float(np.median(n_high_frac)),
            'mean_n_high_frac':   float(np.mean(n_high_frac)),
            'pooled_n_high_frac': total_n_high / total_m_q,
            'total_q_tokens':     total_m_q,
            'total_high_margin_q_tokens': total_n_high,
        },
        'sum_decomposition': {
            'median_f_high': float(np.median(f_high)),
            'mean_f_high':   float(np.mean(f_high)),
            'pooled_f_high': total_high / total_all,
            'pooled_S_high': total_high,
            'pooled_S_low':  total_low,
            'pooled_S_total': total_all,
        },
        'sign_preservation_at_r64': {
            'high_margin_preservation_rate':
                total_n_high_preserved / max(total_n_high, 1),
            'low_margin_preservation_rate':
                total_n_low_preserved / max(total_m_q - total_n_high, 1),
            'high_margin_sum_preserved_frac':
                total_S_high_preserved / max(total_high, 1e-12),
            'low_margin_sum_preserved_frac':
                total_S_low_preserved / max(total_low, 1e-12),
        },
        'per_query': per_query,
    }

    residual_arrays = None
    residual_stats = None
    if save_residuals:
        residual_arrays = {
            k: np.concatenate(v) if v else np.array([])
            for k, v in residual_chunks.items()
        }
        residual_stats = summarize_residuals(residual_arrays, M_THRESH)
        summary['flip_residuals'] = residual_stats

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open('w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nwrote {OUT_JSON}', flush=True)

    if save_residuals and args.residual_output:
        args.residual_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.residual_output, **residual_arrays)
        print(f'wrote {args.residual_output}', flush=True)

    if save_residuals and args.residual_summary_output:
        args.residual_summary_output.parent.mkdir(parents=True, exist_ok=True)
        with args.residual_summary_output.open('w') as f:
            json.dump({
                'config': summary['config'],
                'flip_residuals': residual_stats,
            }, f, indent=2)
        print(f'wrote {args.residual_summary_output}', flush=True)

    if save_residuals and args.residual_fig_output:
        plot_residual_density(residual_arrays, residual_stats,
                              args.residual_fig_output, M_THRESH)
        print(f'wrote {args.residual_fig_output}', flush=True)

    # Print the verdict-rule answer
    print('\n=== Verdict-rule application ===\n')
    print(f'median n_high / m_q       = {summary["population"]["median_n_high_frac"]:.4f}')
    print(f'pooled n_high / m_q       = {summary["population"]["pooled_n_high_frac"]:.4f}')
    print(f'median f_high             = {summary["sum_decomposition"]["median_f_high"]:.4f}')
    print(f'pooled f_high             = {summary["sum_decomposition"]["pooled_f_high"]:.4f}')
    print(f'high-margin q-tok preserv = {summary["sign_preservation_at_r64"]["high_margin_preservation_rate"]:.4f}')
    print(f'high-margin SUM preserved = {summary["sign_preservation_at_r64"]["high_margin_sum_preserved_frac"]:.4f}')
    print(f'low-margin q-tok preserv  = {summary["sign_preservation_at_r64"]["low_margin_preservation_rate"]:.4f}')
    print()
    cond_pop = summary['population']['median_n_high_frac'] <= 0.05
    cond_sum = summary['sum_decomposition']['median_f_high'] >= 0.50
    print(f'CONDITION 1 (population imbalance <= 5%): {"PASS" if cond_pop else "FAIL"}')
    print(f'CONDITION 2 (sum dominance     >= 50%):    {"PASS" if cond_sum else "FAIL"}')
    if cond_pop and cond_sum:
        print('VERDICT: CLEAN PASS — argmax-floor reading is measurement-discriminable.')
    elif cond_pop and not cond_sum:
        print('VERDICT: CLEAN FAIL — low-margin cancellation is the load-bearing reading.')
    elif not cond_pop and cond_sum:
        print('VERDICT: PATHOLOGICAL — high-margin q-tokens are NOT a small minority.')
    else:
        print('VERDICT: BORDERLINE — keep two-readings hedge.')


if __name__ == '__main__':
    main()
