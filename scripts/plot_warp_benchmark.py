"""Plot Warp GPU benchmark vs CPU baseline."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

# Data from mjwarp-testspeed (GPU) and 30s_stability_test (CPU)
n_hands_gpu = np.array([1, 64, 1024, 4096])
throughput_gpu = np.array([5_631, 277_627, 3_774_070, 10_175_746])
cpu_baseline = 93_500


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.loglog(n_hands_gpu, throughput_gpu, "o-", linewidth=2, markersize=10,
               label="GPU (RTX 5080)", color="#76B900")
    ax1.axhline(cpu_baseline, color="#0072B2", linestyle="--", linewidth=2,
                label=f"CPU 1 hand = {cpu_baseline//1000}k steps/s")
    ax1.set_xlabel("Number of parallel hands")
    ax1.set_ylabel("Total throughput (steps/sec)")
    ax1.set_title("Total throughput")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(loc="lower right")
    for x, y in zip(n_hands_gpu, throughput_gpu):
        ax1.annotate(f"{y/1000:.0f}k", xy=(x, y), xytext=(8, 8),
                     textcoords="offset points", fontsize=9)

    per_hand_gpu = throughput_gpu / n_hands_gpu
    ax2.semilogx(n_hands_gpu, per_hand_gpu, "o-", linewidth=2, markersize=10,
                 label="GPU per-hand", color="#76B900")
    ax2.axhline(cpu_baseline, color="#0072B2", linestyle="--", linewidth=2,
                label="CPU per-hand")
    ax2.set_xlabel("Number of parallel hands")
    ax2.set_ylabel("Per-hand throughput (steps/sec)")
    ax2.set_title("Per-hand throughput")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(loc="upper right")

    plt.suptitle("hand_final.xml: CPU vs GPU on RTX 5080 (sm_120)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    out_dir = "/home/rivery/duke_hand/docs"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "warp_benchmark.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"Saved: {out_path}")

    print("\nSummary:")
    print(f"  CPU 1 hand:    {cpu_baseline:>10,} steps/s")
    for n, t in zip(n_hands_gpu, throughput_gpu):
        print(f"  GPU {n:>5} hands: {t:>10,} total ({t//n:>6,} per-hand, "
              f"{t*0.002:>7.1f}x realtime)")


if __name__ == "__main__":
    main()
