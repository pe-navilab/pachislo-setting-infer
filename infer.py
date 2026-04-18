import json
import math

def load_machine(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def log_likelihood(n_games: int, n_hit: int, p: float) -> float:
    if p <= 0 or p >= 1:
        return float("-inf")
    return n_hit * math.log(p) + (n_games - n_hit) * math.log(1 - p)

def infer_setting(machine, n_games: int, n_big: int, n_reg: int):
    log_ls = {}

    for s, probs in machine["settings"].items():
        p_big = 1 / probs["big"]
        p_reg = 1 / probs["reg"]

        log_big = log_likelihood(n_games, n_big, p_big)
        log_reg = log_likelihood(n_games, n_reg, p_reg)

        log_ls[s] = log_big + log_reg

    max_log_l = max(log_ls.values())
    weights = {s: math.exp(l - max_log_l) for s, l in log_ls.items()}

    total = sum(weights.values())
    return {s: w / total for s, w in weights.items()}

def main():
    print("=== パチスロ設定推測ツール（拡張版） ===")
    machine_path = input("機種データ(JSON)のパスを入力してください: ")

    machine = load_machine(machine_path)

    print(f"\n対象機種：{machine['name']}")
    n_games = int(input("総ゲーム数: "))
    n_big = int(input("BIG回数: "))
    n_reg = int(input("REG回数: "))

    probs = infer_setting(machine, n_games, n_big, n_reg)

    print("\n--- 推測結果 ---")
    for s, p in probs.items():
        print(f"設定{s}: {p*100:.2f}%")

    best = max(probs, key=probs.get)
    print(f"\n最も可能性が高いのは『設定{best}』です。")

if __name__ == "__main__":
    main()

