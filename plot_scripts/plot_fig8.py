import matplotlib.pyplot as plt
import numpy as np
import sys, os

HAMMER_ROOT = os.environ['HAMMER_ROOT']
delay_files = [f"{HAMMER_ROOT}/results/fig8/st_delays.txt",
               f"{HAMMER_ROOT}/results/fig8/mt_delays.txt",
               f"{HAMMER_ROOT}/results/fig8/mw_delays.txt"
]

# it = float(sys.argv[fileCount+1]) * 23052288
it = 10000 * 32000000
timeList = [[] for i in range(len(delay_files))]
for i in range(len(delay_files)):
    with open(delay_files[i]) as f:
        for line in f:
            timeList[i].append(int(line.strip()))
        timeList[i] = np.divide(it, timeList[i])
        for j in range(2, 33):
            timeList[i][j - 2] = j * timeList[i][j - 2]

fig, ax = plt.subplots()
fig.set_size_inches(10, 3)
x = np.arange(2, len(timeList[0]) + 2, 1, dtype=int)
# ax.set_yticks(np.arange(0, 6, step=0.1))
ax.set_xticks(np.arange(x[0], x[-1] + 1, step=6))

cmap = plt.get_cmap("tab10")
ax.grid(axis="y")
# Creating the bar plot

xticks = ax.get_xticks()
yticks = ax.get_yticks()
ax.tick_params(axis='both', which='major', labelsize=16, left=False)
ax.tick_params(axis='both', which='minor', labelsize=16, left=False)
for ytick in yticks:
    for xtick in xticks:
         ax.vlines(xtick, ytick+800000, ytick, linewidth=0.5,  color='grey', linestyle='-', alpha=0.5)
# marker=['o', 's', '^'][i]
for i in range(len(delay_files)):
    plt.plot(x, timeList[i], marker=['^', 'o', 's'][i], linewidth=3, markersize=8, color=[cmap(2), cmap(0), cmap(1)][i], markevery=[x-2 for x in ax.get_xticks()])
plt.axhline(y=32000000/45, c="red", linewidth=1)
plt.text(len(x) / 2 + 2, 32000000/45 + 50000, 'Theoretical Maximum', fontsize=16, va='center', ha='center', weight='bold', c="black")

markers = ['^', 's', 'o']
colors = [cmap(2), cmap(1), cmap(0)]
f = lambda m,c: plt.plot([],[],marker=m, color=c, ls="none")[0]
handles = [f(markers[i], colors[i]) for i in range(3)]
leg = plt.legend(
    handles,
    ["Single-Thread", "N-Thread", "N-Warp"],
    loc="upper center",
    fontsize=16,
    ncols=3,
    markerscale=2.0,
    bbox_to_anchor=(0.5, 1.25))

plt.xlabel('N-Aggressors', fontsize=22)
plt.ylabel('ACT per tREFW', fontsize=22)
plt.yticks([100000 * i for i in range(1, 9, 2)], [f"{i}k" for i in [j * 100 for j in range(1, 9, 2)]])
# plt.title('ACTs in a tREFw. (16K REF)', fontsize=22)
plt.show()
fig.savefig(f"{HAMMER_ROOT}/results/fig8/fig8.pdf", transparent=True, format="pdf", bbox_inches="tight")