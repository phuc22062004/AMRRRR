"""Split raw AMR files into train/test blocks."""
import argparse
import random


def read_amr_blocks(file_path: str) -> list[str]:
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    return [b.strip() for b in content.split("\n\n") if b.strip()]


def split_blocks(blocks: list[str], test_ratio: float = 0.2) -> tuple[list[str], list[str]]:
    random.shuffle(blocks)
    test_size = int(len(blocks) * test_ratio)
    return blocks[test_size:], blocks[:test_size]


def write_blocks(file_path: str, blocks: list[str]) -> None:
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write("\n\n".join(blocks) + "\n")


def main(args: argparse.Namespace) -> None:
    train_all, test_all = [], []
    for src in args.inputs:
        train, test = split_blocks(read_amr_blocks(src), args.test_ratio)
        train_all.extend(train)
        test_all.extend(test)
    write_blocks(args.train_out, train_all)
    write_blocks(args.test_out, test_all)
    print(f"Train size: {len(train_all)}, Test size: {len(test_all)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split AMR files into train/test.")
    parser.add_argument("--inputs", nargs="+", required=True, help="One or more input AMR files")
    parser.add_argument("--train_out", type=str, required=True)
    parser.add_argument("--test_out", type=str, required=True)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
