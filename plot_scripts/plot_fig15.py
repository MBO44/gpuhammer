import matplotlib.pyplot as plt
import numpy as np
import sys, os
import math
from matplotlib.widgets import Cursor

HAMMER_ROOT = os.environ['HAMMER_ROOT']
row_set_file = open(f"{HAMMER_ROOT}/results/row_sets/ROW_SET_0.txt")
cmap = plt.get_cmap("tab10")
markers = ['^', "P", "d", "*"]


def plot_distance_cursor(i, distance):
    top = i + 0.9
    bot = i + 0.1
    plt.plot([8, 12], [bot, bot], 'k-', lw=2)
    plt.plot([10, 10], [bot, top], 'k-', lw=2)
    plt.plot([8, 12], [top, top], 'k-', lw=2)
    plt.text(12, i + 0.5, str(distance * 256) + " Bytes", fontsize=16, va='center', ha='left', weight='bold', c="black")

row_count = 4
row_pos = [[] for _ in range(4)]
row_offset = []
for i in range (row_count):
    row = row_set_file.readline().strip().split('\t')
    offset = int(row[0])
    pos = 0
    row_offset.append(offset // 256)
    row_pos[i].append(pos)
    row.pop(0)
    while len(row) != 0:
        while offset != int(row[0]):
            offset += 256
            pos += 1
        row_pos[i].append(pos)
        row.pop(0)

fig,ax = plt.subplots()
fig.set_size_inches(10, 4)
ax.tick_params(axis='both', which='major', labelsize=16, left=False)
ax.tick_params(axis='both', which='minor', labelsize=16, left=False)
for x_v in ax.spines.values():
    x_v.set_alpha(0.5)
    x_v.set_edgecolor('grey') 
plt.grid(axis='y', color='grey', linestyle='-', alpha=0.5,zorder=0)

for i in range (row_count):
    plt.plot(row_pos[i], [i + 1 for _ in range(len(row_pos[i]))], linestyle=':', color=cmap(i), marker=markers[i], markersize=13, label=f"Row {i + 1}", linewidth=3)
    if i < row_count - 1:
        plot_distance_cursor(i + 1, row_offset[i + 1] - row_offset[i])

# plt.ylabel('Time per Round (NS)', fontsize=22)
# plt.yticks([i for i in range(1, row_count + 1)], [f"Row {i}" for i in range(1, row_count + 1)])
plt.yticks([])
plt.xlabel('Row Address Separation (256 Byte)', fontsize=22)
plt.grid(True)
handles, labels = ax.get_legend_handles_labels()
plt.legend(
    loc="upper center",
    fontsize=16,
    ncols=5,
    bbox_to_anchor=(0.5, 1.24),
    numpoints=1,
    handlelength=0)

fig.savefig(f"{HAMMER_ROOT}/results/fig15/fig15.pdf", transparent=True, format="pdf", bbox_inches="tight")