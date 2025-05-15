import matplotlib.pyplot as plt
import numpy as np
import sys, os
import math

HAMMER_ROOT = os.environ['HAMMER_ROOT']
modifiers = []
time = []

with open(f"{HAMMER_ROOT}/results/fig2/LOAD_TIMING.txt", "r") as f:
    for line in f:
        parts = line.strip().split('\t')
        modifiers.append(parts[0])
        time.append(int(parts[1]))

# # Sample data
cmap = plt.get_cmap("tab10")

# # Setting the positions and width for the bars
bar_width = 0.3
r1 = np.arange(len(modifiers))
r1 = [x + bar_width for x in r1]

fig,ax = plt.subplots()
fig.set_size_inches(10, 3)
ax.tick_params(axis='both', which='major', labelsize=16, left=False)
ax.tick_params(axis='both', which='minor', labelsize=16, left=False)
# Creating the bar plot
for x in ax.spines.values():
    x.set_alpha(0.5)
    x.set_edgecolor('grey') 
plt.grid(axis='y', color='grey', linestyle='-', alpha=0.5,zorder=0)
plt.bar(r1, time, width=bar_width, label='Different Bank', color=["grey" for _ in range(len(modifiers) - 1)] + [cmap(0)], edgecolor = "black", linewidth=1,zorder=3)

# # Adding labels
# plt.xlabel('Back-Off Threshold, $N_{BO}$', fontsize=22)
plt.ylabel('Time (ns)', fontsize=22)
plt.xlabel('Load Modifiers', fontsize=22)
plt.xticks([r + bar_width for r in range(len(modifiers))], [f"{r}" for r in modifiers])
# plt.yscale('log')
# plt.yticks([0])


fig.savefig(f"{HAMMER_ROOT}/results/fig2/fig2.pdf", transparent=True, format="pdf", bbox_inches="tight")