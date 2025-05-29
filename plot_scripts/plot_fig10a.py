import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
import sys, os
from math import pi

HAMMER_ROOT = os.environ['HAMMER_ROOT']
delay_files = [f"{HAMMER_ROOT}/results/fig10/delay_8w.txt",
               f"{HAMMER_ROOT}/results/fig10/delay_12w.txt",
               f"{HAMMER_ROOT}/results/fig10/delay_16w.txt",
               f"{HAMMER_ROOT}/results/fig10/delay_24w.txt"
]

it = 10000
timeList = []

def read_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    y_values = [float(line.strip()) for line in lines]
    z_values = [ y / float(it) for y in y_values]
    x_values = list(range(len(y_values)))
    return x_values, z_values

for i in range(len(delay_files)):
    timeList.append(read_file(delay_files[i])[1])


fig,ax = plt.subplots()
fig.set_size_inches(10, 3)
ax.tick_params(axis='both', which='major', labelsize=16, left=False)
ax.tick_params(axis='both', which='minor', labelsize=16, left=False)

for x_v in ax.spines.values():
    x_v.set_alpha(0.5)
    x_v.set_edgecolor('grey') 
plt.grid(axis='y', color='grey', linestyle='-', alpha=0.5,zorder=0)

delay_left = 0
delay_right = 65
for i in range(len(timeList)):
    this_list = timeList[i][delay_left : delay_right]
    plt.plot(list(range(len(this_list))), this_list, linewidth=3)
plt.yticks([800, 1400, 2000])
plt.xticks(range(delay_left, delay_right, 4))
plt.axhline(y=1407, c="black", linewidth=1)
plt.text(72, 1420, '1407', fontsize=16, va='center', ha='center', weight='bold', c="white")
plt.ylabel('Time Per\nRound (ns)', fontsize=22)
plt.xlabel('Delay (adds)', fontsize=22)

u=13.     #x-position of the center
v=1407    #y-position of the center
a=5.     #radius on the x-axis
b=70    #radius on the y-axis

t = np.linspace(0, 2*pi, 100)
plt.plot( u+a*np.cos(t) , v+b*np.sin(t) , linewidth=2, color="black", linestyle=':')

u=39.     #x-position of the center
v=1407    #y-position of the center
a=1.5     #radius on the x-axis
b=70    #radius on the y-axis
plt.plot( u+a*np.cos(t) , v+b*np.sin(t) , linewidth=2, color="black", linestyle=':')

u=44.     #x-position of the center
v=1407    #y-position of the center
a=1.     #radius on the x-axis
b=70    #radius on the y-axis
plt.plot( u+a*np.cos(t) , v+b*np.sin(t) , linewidth=2, color="black", linestyle=':')

u=54.     #x-position of the center
v=1407    #y-position of the center
a=2.5     #radius on the x-axis
b=70    #radius on the y-axis
plt.plot( u+a*np.cos(t) , v+b*np.sin(t) , linewidth=2, color="black", linestyle=':')


# Adding legend
# handles, labels = ax.get_legend_handles_labels()
plt.legend(["8-sided (8-warp)", "12-sided (12-warp)", "16-sided (16-warp)", "24-sided (24-warp)"],
    loc="upper center",
    fontsize=16,
    ncol=2,
    bbox_to_anchor=(0.5, 1.4))
# plt.legend(["8-sided", "12-sided", "16-sided", "24-sided"], loc="upper left", fontsize=16)
fig.savefig(f"{HAMMER_ROOT}/results/fig10/fig10a.pdf", transparent=True, format="pdf", bbox_inches="tight")