import os
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


# Report the average of numbers from a file
def average_from_file(filename: str) -> float:
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            numbers = [int(line.strip()) for line in file]
        return sum(numbers) / len(numbers) if numbers else 0
    else:
        return 0


def find_min_ratio(input_dir: str) -> float:
    min_ratio = 1

    for agg in range(101):
        for dum in range(100):
            filename = os.path.join(input_dir, f'agg{agg}_dum{dum}_bitflip.txt')
            if average_from_file(filename) > 0:
                ratio = agg / (agg + dum)
                if ratio < min_ratio:
                    min_ratio = ratio

    return min_ratio


def plot_trh(flip_names, trh, plt=plt):
    sorted_pairs = sorted(zip(flip_names, trh), key=lambda x: x[1])
    sorted_flip_names, sorted_trh = zip(*sorted_pairs)

    fig, ax = plt.subplots(figsize=(5.7, 2))
    bars = ax.bar(sorted_flip_names, sorted_trh)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x}K'))
    ax.set_xlabel('Bit-flips', fontsize=14)
    ax.set_ylabel('Rowhammer\nThreshold', fontsize=14)
    ax.set_ylim(top=20)
    
    # Label the bars with their values
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height,
                f'{height:.1f}K', ha='center', va='bottom', fontsize=9)


if __name__ == "__main__":

    flip_names = ['A1', 'B1', 'B2', 'B3', 'C1', 'D1', 'D2', 'D3']
    trh = [0.0] * len(flip_names)

    HAMMER_ROOT = os.environ['HAMMER_ROOT']
    OUTPUT_DIR = os.path.join(HAMMER_ROOT, "results", "fig11")

    act_per_trefi = 16.384       # 2^14 activations per tREFI

    for i in range(len(flip_names)):
        INPUT_DIR = os.path.join(HAMMER_ROOT, "results", "fig11", flip_names[i])

        min_ratio = find_min_ratio(INPUT_DIR)
        trh[i] = min_ratio * act_per_trefi

        print(f"TRH for {flip_names[i]}: {min_ratio * act_per_trefi:.4f} K")

    # Plot and save the figure
    plot_trh(flip_names, trh)
    output_image = os.path.join(OUTPUT_DIR, f"fig11.pdf")
    plt.tight_layout()
    plt.savefig(output_image)
    
