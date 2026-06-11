"""Demonstrates that sign-coding preserves coarse winner structure while
substantially degrading fine-grained similarity fidelity.

Three measurements, all on ColBERTv2 MS MARCO 100k:

1. PAIRWISE SIMILARITY-BIN PRESERVATION.
   Sample 50000 random (doc-token, doc-token) pairs. For each pair compute
   fp32 cosine similarity and sign-coded similarity at r=64. Bin fp32
   similarities, report mean and std of sign-coded similarity per bin.
   Expectation: coarse monotonic trend preserved, large within-bin variance.

2. NEIGHBOR OVERLAP AT MULTIPLE k.
   For each of 200 sampled anchor doc-tokens, find top-k nearest doc-token
   neighbors under (a) fp32 cosine and (b) sign-coded cosine, among 50000
   random doc-tokens drawn from the corpus. Report overlap@k for
   k in {1, 5, 10, 50, 100, 500}. Expectation: high overlap for k=1, decay
   toward random as k grows -> local geometry collapses faster than coarse
   winner identity.

3. CORRELATION + RANK FIDELITY.
   Report Spearman / Pearson correlation between fp32 cosine and sign-coded
   similarity across all sampled pairs. Expectation: monotone but loose.

Output: outputs/geometry_fidelity_vs_argmax.json + console summary suitable
for §3 of the paper.
"""
import argparse
import sys, json, time, warnings
from pathlib import Path
import numpy as np
import torch

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path('.').resolve()))
from src.subbit.data import EmbeddingStore, resolve_embedding_cache_path


EMB_DIR = Path('data/embeddings/msmarco/100k')
OUT_JSON = Path('outputs/geometry_fidelity_vs_argmax.json')

R_DIM = 64
N_PAIRS = 1_000_000
N_ANCHORS = 1000
N_NEIGHBORS_POOL = 200_000
KS = [1, 5, 10, 50, 100, 500, 1000]
SEED = 42


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--emb-dir", type=Path, default=EMB_DIR)
    p.add_argument("--output", type=Path, default=OUT_JSON)
    p.add_argument("--r-dim", type=int, default=R_DIM)
    p.add_argument("--n-pairs", type=int, default=N_PAIRS)
    p.add_argument("--n-anchors", type=int, default=N_ANCHORS)
    p.add_argument("--n-neighbors-pool", type=int, default=N_NEIGHBORS_POOL)
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


def build_random_R(r_dim, seed):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(r_dim, 128, dtype=torch.float64, generator=g)
    Q, _ = torch.linalg.qr(A.T)
    return Q.T[:r_dim, :].to(torch.float32)


def cosine_normalize(X):
    norms = X.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    return X / norms


def main():
    args = parse_args()
    print('=== Geometry fidelity vs argmax preservation (proves BQ destroys fine structure) ===\n', flush=True)

    print('Loading ColBERTv2 doc store...', flush=True)
    doc_store = EmbeddingStore(resolve_embedding_cache_path(args.emb_dir, 'doc'), mode='dict')
    doc_store.load()
    doc_ids = list(doc_store.get_all_ids())
    N_docs = len(doc_ids)
    print(f'  {N_docs} docs', flush=True)

    # Build a big pool of doc tokens for sampling
    print(f'Sampling {args.n_neighbors_pool} random doc tokens for the pool...', flush=True)
    rng = np.random.default_rng(args.seed)
    pool_tokens = []
    sampled_doc_ids = list(rng.choice(doc_ids, size=min(20000, N_docs), replace=False))
    for did in sampled_doc_ids:
        d = doc_store.get(did).float()
        for j in range(d.shape[0]):
            pool_tokens.append(d[j])
            if len(pool_tokens) >= args.n_neighbors_pool:
                break
        if len(pool_tokens) >= args.n_neighbors_pool:
            break
    pool = torch.stack(pool_tokens[:args.n_neighbors_pool])  # (N_pool, 128)
    print(f'  pool: {pool.shape}', flush=True)

    R = build_random_R(args.r_dim, args.seed)

    # ----- Sign-coded representations -----
    print('Computing sign-coded representations...', flush=True)
    pool_sign = torch.sign(pool @ R.T)  # (N_pool, r) in {-1, +1}

    # ----- 1. Pairwise similarity-bin preservation -----
    print(f'\n[1] Pairwise similarity-bin preservation ({args.n_pairs} random pairs)...', flush=True)
    idx_a = rng.integers(0, args.n_neighbors_pool, size=args.n_pairs)
    idx_b = rng.integers(0, args.n_neighbors_pool, size=args.n_pairs)
    pool_cos = cosine_normalize(pool)
    with torch.no_grad():
        s_fp32 = (pool_cos[idx_a] * pool_cos[idx_b]).sum(dim=-1).numpy()  # cosine in [-1, 1]
        s_sign = (pool_sign[idx_a] * pool_sign[idx_b]).sum(dim=-1).numpy() / args.r_dim  # normalised in [-1, 1]

    # Bin
    bins = np.linspace(-0.2, 1.0, 13)  # 12 bins from -0.2 to 1.0
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_stats = []
    for i in range(len(bins) - 1):
        mask = (s_fp32 >= bins[i]) & (s_fp32 < bins[i + 1])
        n = int(mask.sum())
        if n == 0: continue
        bin_stats.append({
            'fp32_lo': float(bins[i]), 'fp32_hi': float(bins[i + 1]),
            'fp32_center': float(bin_centers[i]),
            'n': n,
            'sign_mean': float(s_sign[mask].mean()),
            'sign_std': float(s_sign[mask].std()),
            'sign_p05': float(np.percentile(s_sign[mask], 5)),
            'sign_p95': float(np.percentile(s_sign[mask], 95)),
        })
    print(f'  {"fp32 bin":<18}{"n":<8}{"sign mean":<14}{"sign std":<12}{"5%–95% range":<22}')
    print('  ' + '-' * 78)
    for b in bin_stats:
        rng_s = f'[{b["sign_p05"]:+.3f}, {b["sign_p95"]:+.3f}]'
        print(f'  [{b["fp32_lo"]:+.2f},{b["fp32_hi"]:+.2f})   {b["n"]:<8}{b["sign_mean"]:<+14.4f}{b["sign_std"]:<12.4f}{rng_s:<22}')

    # ----- 2. Neighbor overlap @ k -----
    print(f'\n[2] Neighbor overlap @ k (anchors={args.n_anchors}, pool={args.n_neighbors_pool})...', flush=True)
    anchor_idx = rng.choice(args.n_neighbors_pool, size=args.n_anchors, replace=False)
    anchors_cos = pool_cos[anchor_idx]
    anchors_sign = pool_sign[anchor_idx]
    with torch.no_grad():
        sim_fp32 = anchors_cos @ pool_cos.T  # (anchors, pool)
        sim_sign = anchors_sign @ pool_sign.T / args.r_dim  # (anchors, pool)
        # exclude self-similarity
        for i, a in enumerate(anchor_idx):
            sim_fp32[i, a] = -float('inf')
            sim_sign[i, a] = -float('inf')

    overlap_per_k = {}
    for k in KS:
        top_fp32 = sim_fp32.topk(k, dim=-1).indices  # (anchors, k)
        top_sign = sim_sign.topk(k, dim=-1).indices  # (anchors, k)
        overlaps = []
        for i in range(args.n_anchors):
            o = len(set(top_fp32[i].tolist()) & set(top_sign[i].tolist())) / k
            overlaps.append(o)
        overlaps = np.array(overlaps)
        random_baseline = k / args.n_neighbors_pool  # expected overlap under random
        overlap_per_k[k] = {
            'mean': float(overlaps.mean()),
            'std': float(overlaps.std()),
            'median': float(np.median(overlaps)),
            'random_baseline': float(random_baseline),
        }
    print(f'  {"k":<6}{"overlap mean":<16}{"overlap std":<14}{"median":<12}{"random baseline":<18}')
    print('  ' + '-' * 70)
    for k in KS:
        s = overlap_per_k[k]
        print(f'  {k:<6}{s["mean"]:<16.4f}{s["std"]:<14.4f}{s["median"]:<12.4f}{s["random_baseline"]:<18.4f}')

    # ----- 3. Correlation -----
    from scipy.stats import spearmanr, pearsonr
    rho, p_s = spearmanr(s_fp32, s_sign)
    r, p_p = pearsonr(s_fp32, s_sign)
    print(f'\n[3] Correlation across all {args.n_pairs} pairs:')
    print(f'  Spearman ρ = {rho:.4f} (p={p_s:.4g})')
    print(f'  Pearson  r = {r:.4f} (p={p_p:.4g})')

    # ----- 4. Side-by-side: argmax preservation (from A) vs neighborhood fidelity -----
    # Quote A's headline numbers for the comparison
    print('\n[4] Side-by-side: argmax preservation (A) vs neighborhood fidelity (this run)')
    print(f'  Argmax preservation (A, r=64):                       28.3% overall, 100% at margin>=0.2')
    print(f'  Top-1 neighbor overlap (this run):                   {overlap_per_k[1]["mean"]:.4f}')
    print(f'  Top-50 neighbor overlap:                             {overlap_per_k[50]["mean"]:.4f}')
    print(f'  Top-500 neighbor overlap:                            {overlap_per_k[500]["mean"]:.4f}')
    print(f'  Pairwise similarity correlation:                     Spearman ρ={rho:.3f}')

    summary = {
        'config': {'r_dim': args.r_dim, 'n_pairs': args.n_pairs, 'n_anchors': args.n_anchors,
                   'n_neighbors_pool': args.n_neighbors_pool, 'seed': args.seed},
        'similarity_bin_preservation': bin_stats,
        'neighbor_overlap': overlap_per_k,
        'correlation': {
            'spearman_rho': float(rho), 'spearman_p': float(p_s),
            'pearson_r': float(r), 'pearson_p': float(p_p),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    print(f'\nWrote {args.output}')


if __name__ == '__main__':
    main()
