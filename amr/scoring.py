"""CLI to compute SMATCH F1 between predicted and gold AMR files."""
import argparse

from .data_processing import read_amr_direct
from .rewards import compute_smatch_f1


def main(args: argparse.Namespace) -> None:
    predict_df = read_amr_direct(args.predict_file)
    gold_df = read_amr_direct(args.gold_file)
    predicts = predict_df["amr"].tolist()
    golds = gold_df["amr"].tolist()
    print(f"Number of predictions: {len(predicts)}, Number of golds: {len(golds)}")

    n = min(len(predicts), len(golds))
    f1s, ps, rs = [], [], []
    for i in range(n):
        f1, p, r = compute_smatch_f1(predicts[i], golds[i])
        f1s.append(f1)
        ps.append(p)
        rs.append(r)

    f1_avg = sum(f1s) / len(f1s) if f1s else 0.0
    p_avg = sum(ps) / len(ps) if ps else 0.0
    r_avg = sum(rs) / len(rs) if rs else 0.0
    print(f"F1 Score: {f1_avg:.4f}, Precision: {p_avg:.4f}, Recall: {r_avg:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute AMR scores.")
    parser.add_argument("--predict_file", type=str, required=True)
    parser.add_argument("--gold_file", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
