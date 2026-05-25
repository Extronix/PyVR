"""
Анализ логов step7: строит графики и сводную таблицу из всех CSV в logs/.

Запуск:
    python analyze_logs.py
    python analyze_logs.py --file logs/buffer_20260523_164443.csv

Что делает:
  1) По каждому CSV: графики traffic(t), miss(t), сводная таблица
  2) По всем CSV вместе: сравнительная таблица "латентность vs miss/traffic"
  3) Парето-график: traffic vs miss для всех стратегий

Зависимости: pip install pandas matplotlib
"""

import os
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt

LOG_DIR = "logs"
OUT_DIR = "reports"

STRATEGIES = ["tight", "ring1", "ring2", "predict"]
STRATEGY_LABELS = {
    "tight":   "TILED-tight",
    "ring1":   "TILED-1ring",
    "ring2":   "TILED-2ring",
    "predict": "TILED-predict",
}
STRATEGY_COLORS = {
    "tight":   "#d62728",
    "ring1":   "#2ca02c",
    "ring2":   "#1f77b4",
    "predict": "#ff7f0e",
}


def load_csv(path):
    df = pd.read_csv(path)
    df["t_sec"] = pd.to_numeric(df["t_sec"], errors="coerce")
    for s in STRATEGIES:
        df[f"{s}_traffic"] = pd.to_numeric(df[f"{s}_traffic"], errors="coerce") * 100
        df[f"{s}_miss"]    = pd.to_numeric(df[f"{s}_miss"],    errors="coerce") * 100
    df["latency"] = pd.to_numeric(df["latency"], errors="coerce")
    return df


def per_file_report(df, csv_name, out_dir):
    """Графики и таблица по одному файлу."""
    base = os.path.splitext(csv_name)[0]
    os.makedirs(out_dir, exist_ok=True)

    # === 1. Сводная таблица по стратегиям ===
    summary = []
    for s in STRATEGIES:
        summary.append({
            "strategy":     STRATEGY_LABELS[s],
            "traffic_mean": df[f"{s}_traffic"].mean(),
            "traffic_std":  df[f"{s}_traffic"].std(),
            "miss_mean":    df[f"{s}_miss"].mean(),
            "miss_p95":     df[f"{s}_miss"].quantile(0.95),
            "miss_max":     df[f"{s}_miss"].max(),
        })
    tbl = pd.DataFrame(summary)
    tbl_path = os.path.join(out_dir, f"{base}_summary.csv")
    tbl.to_csv(tbl_path, index=False, float_format="%.3f")
    print(f"  📄 {tbl_path}")
    print(tbl.to_string(index=False, float_format=lambda x: f"{x:7.3f}"))

    # === 2. График traffic(t) ===
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for s in STRATEGIES:
        ax.plot(df["t_sec"], df[f"{s}_traffic"],
                label=STRATEGY_LABELS[s], color=STRATEGY_COLORS[s], lw=1.2)
    ax.set_xlabel("Время, с")
    ax.set_ylabel("Трафик, % от full-frame")
    ax.set_title(f"Трафик по стратегиям  ({csv_name})")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    ax.set_ylim(0, 105)
    fig.tight_layout()
    p = os.path.join(out_dir, f"{base}_traffic.png")
    fig.savefig(p, dpi=120)
    plt.close(fig)
    print(f"  🖼  {p}")

    # === 3. График miss(t) ===
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for s in STRATEGIES:
        ax.plot(df["t_sec"], df[f"{s}_miss"],
                label=STRATEGY_LABELS[s], color=STRATEGY_COLORS[s], lw=1.2)
    ax.set_xlabel("Время, с")
    ax.set_ylabel("Miss, % площади экрана")
    ax.set_title(f"Промахи (чёрные зоны) по стратегиям  ({csv_name})")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    p = os.path.join(out_dir, f"{base}_miss.png")
    fig.savefig(p, dpi=120)
    plt.close(fig)
    print(f"  🖼  {p}")

    # === 4. Парето-точка: traffic vs miss ===
    fig, ax = plt.subplots(figsize=(7, 6))
    for s in STRATEGIES:
        ax.scatter(df[f"{s}_traffic"].mean(), df[f"{s}_miss"].mean(),
                   s=180, color=STRATEGY_COLORS[s],
                   label=STRATEGY_LABELS[s], edgecolor="black", zorder=3)
        ax.annotate(STRATEGY_LABELS[s],
                    (df[f"{s}_traffic"].mean(), df[f"{s}_miss"].mean()),
                    xytext=(8, 6), textcoords="offset points", fontsize=9)
    ax.set_xlabel("Средний трафик, %")
    ax.set_ylabel("Средний miss, %")
    ax.set_title(f"Парето: трафик ↔ качество  ({csv_name})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(out_dir, f"{base}_pareto.png")
    fig.savefig(p, dpi=120)
    plt.close(fig)
    print(f"  🖼  {p}")

    return tbl.assign(source=csv_name,
                      latency=int(df["latency"].mode().iloc[0]))


def combined_report(all_tables, out_dir):
    """Сводный отчёт по всем CSV — сравнение по латентности."""
    if not all_tables:
        return
    big = pd.concat(all_tables, ignore_index=True)
    big = big.sort_values(["latency", "strategy"])
    p = os.path.join(out_dir, "ALL_runs_summary.csv")
    big.to_csv(p, index=False, float_format="%.3f")
    print(f"\n📊 Сводно: {p}")
    print(big.to_string(index=False, float_format=lambda x: f"{x:7.3f}"))

    latencies = sorted(big["latency"].unique())
    if len(latencies) < 2:
        print("\n(для графика 'miss vs latency' нужно ≥2 запусков с разной задержкой)")
        return

    # miss vs latency для каждой стратегии
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for s in STRATEGIES:
        label = STRATEGY_LABELS[s]
        sub = big[big["strategy"] == label].sort_values("latency")
        ax.plot(sub["latency"], sub["miss_mean"], "o-",
                label=label, color=STRATEGY_COLORS[s], lw=2, ms=8)
    ax.set_xlabel("Сетевая задержка, кадров")
    ax.set_ylabel("Средний miss, %")
    ax.set_title("Деградация качества при росте задержки")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    p = os.path.join(out_dir, "ALL_miss_vs_latency.png")
    fig.savefig(p, dpi=120)
    plt.close(fig)
    print(f"🖼  {p}")

    # traffic vs latency
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for s in STRATEGIES:
        label = STRATEGY_LABELS[s]
        sub = big[big["strategy"] == label].sort_values("latency")
        ax.plot(sub["latency"], sub["traffic_mean"], "o-",
                label=label, color=STRATEGY_COLORS[s], lw=2, ms=8)
    ax.set_xlabel("Сетевая задержка, кадров")
    ax.set_ylabel("Средний трафик, %")
    ax.set_title("Трафик при росте задержки")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    p = os.path.join(out_dir, "ALL_traffic_vs_latency.png")
    fig.savefig(p, dpi=120)
    plt.close(fig)
    print(f"🖼  {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", help="конкретный CSV (иначе все из logs/)")
    ap.add_argument("--outdir", default=OUT_DIR)
    args = ap.parse_args()

    if args.file:
        files = [args.file]
    else:
        files = sorted(glob.glob(os.path.join(LOG_DIR, "buffer_*.csv")))

    if not files:
        print(f"❌ Не нашёл CSV в {LOG_DIR}/")
        return

    print(f"🔍 Найдено файлов: {len(files)}")
    os.makedirs(args.outdir, exist_ok=True)

    all_tables = []
    for f in files:
        name = os.path.basename(f)
        print(f"\n=== {name} ===")
        df = load_csv(f)
        if df.empty:
            print("  (пусто)")
            continue
        tbl = per_file_report(df, name, args.outdir)
        all_tables.append(tbl)

    combined_report(all_tables, args.outdir)
    print(f"\n✅ Все отчёты в: {args.outdir}/")


if __name__ == "__main__":
    main()