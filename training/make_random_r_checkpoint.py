"""Save a random-orthogonal R (no training, no scale head) as a SubBitModel
checkpoint, so eval scripts that take --checkpoint (e.g. evaluate_two_stage_rerank)
can run the random-projection arm.

The R matrix is byte-identical to the baseline table's random_proj row:
torch.manual_seed(seed); Q, _ = qr(randn(128, r)); R = Q.T — the same
construction as run_baseline_comparison.build_random_projection and
baselines.random_projection_binary at the default seed 42, so stage-1 binary
numbers replicate the published random row (0.8393 MRR@10 on 100k_aug).

Output: outputs/random_r64_seed42/checkpoints/best.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.subbit.model import SubBitModel


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--r", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=Path,
                    default=Path("outputs/random_r64_seed42/checkpoints/best.pt"))
    args = ap.parse_args()

    d = 128
    torch.manual_seed(args.seed)
    Q, _ = torch.linalg.qr(torch.randn(d, args.r))
    R = Q.T.contiguous()  # (r, d)

    # Cross-check against the baseline-table construction.
    from src.subbit.baselines import random_projection_binary
    probe = torch.randn(5, d)
    _, R_baseline = random_projection_binary(probe, projected_dim=args.r, seed=args.seed)
    assert torch.allclose(R, R_baseline), "R does not match baselines.random_projection_binary"

    model = SubBitModel(input_dim=d, projected_dim=args.r, use_scale=False)
    model.R.weight.data = R
    model.save(args.output)

    # Round-trip check: loaded model must reproduce sign(R d) and R q exactly.
    loaded = SubBitModel.load(str(args.output))
    x = torch.randn(7, d)
    assert torch.allclose(loaded.R.weight, R)
    doc = loaded.encode_document(x, use_ste=False)
    expected = torch.sign(x @ R.T)
    expected[expected == 0] = 1.0
    assert torch.equal(doc, expected), "encode_document must be sign(R·d)"
    enc_q = loaded.encode_query(x, symmetric=False, use_ste=False)
    assert torch.allclose(enc_q, x @ R.T), "encode_query must be plain R·q with use_scale=False"
    print(f"wrote {args.output}  R={tuple(R.shape)}  use_scale=False  seed={args.seed}")


if __name__ == "__main__":
    main()
