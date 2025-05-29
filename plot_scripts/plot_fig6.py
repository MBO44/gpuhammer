import matplotlib.pyplot as plt
import numpy as np
import sys, os
import math

HAMMER_ROOT = os.environ['HAMMER_ROOT']
all_addresses = []
max_delay = -1
min_delay = math.inf

def roundup_m10(x):
    return math.ceil(x / 10.0) * 10

def initialize_data(all_file):
    global max_delay
    global min_delay
    all_f = open(all_file)
    counter = 0
    for line in all_f:
        val = int(line.strip())
        all_addresses.append(val)

        max_delay = max(max_delay, val)
        min_delay = min(min_delay, val)
        counter += 256

    min_delay = round(min_delay, -1)
    max_delay = roundup_m10(max_delay) + 1

all_file = sys.argv[1]
initialize_data(all_file)
counts_all, bins = np.histogram(all_addresses, bins=range(min_delay, max_delay, 10))

# # Sample data
categories = list(range(min_delay, max_delay, 10))
categories.pop()

# # Setting the positions and width for the bars
bar_width = 0.5
r1 = np.arange(len(categories))
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
plt.bar(r1, counts_all, width=bar_width, label='Different Bank', edgecolor = "black", linewidth=1,zorder=3)

# # Adding labels
# plt.xlabel('Back-Off Threshold, $N_{BO}$', fontsize=22)
plt.ylabel('Addresses\n(256B apart)', fontsize=22)
plt.xlabel('Time (ns)', fontsize=22)
plt.xticks([r + bar_width for r in range(len(categories))], [f"{r}" for r in categories])
plt.yscale('log')

# Set custom y-ticks
plt.yticks([1e4, 1e6, 1e8, 1e9], ['10^4', '10^6', '10^8', ''])

rects = ax.patches
NA=[]
ZERO=[4]
for index, rect in enumerate(rects):
    if index in NA or index in ZERO:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height + 1e8/2, "reference" if index in ZERO else "N/A",
                ha='center', va='bottom', fontsize=14, c='black', weight='bold')   
# Adding legend
# handles, labels = ax.get_legend_handles_labels()
# plt.legend(
#     loc="upper center",
#     fontsize=16,
#     ncols=5,
#     bbox_to_anchor=(0.5, 1.24))

fig.savefig(f"{HAMMER_ROOT}/results/fig5_6/fig6.pdf", transparent=True, format="pdf", bbox_inches="tight")