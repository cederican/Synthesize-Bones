import matplotlib.pyplot as plt
import numpy as np

# top 1
data_dict = {"columns": ["rank_index", "count"], "data": [[0, 1122], [1, 758], [2, 641], [3, 478], [4, 434], [5, 369], [6, 352], [7, 271], [8, 216], [9, 205], [10, 193], [11, 173], [12, 159], [13, 150], [14, 118], [15, 83], [16, 73], [17, 76], [18, 51], [19, 30]]}
# top 3
#data_dict = {"columns": ["rank_index", "count"], "data": [[0, 2521], [1, 2106], [2, 1781], [3, 1481], [4, 1345], [5, 1168], [6, 1099], [7, 911], [8, 793], [9, 717], [10, 638], [11, 561], [12, 538], [13, 497], [14, 422], [15, 376], [16, 301], [17, 234], [18, 223], [19, 144]]}
match_index = np.array([row[0] for row in data_dict["data"]])
count = np.array([row[1] for row in data_dict["data"]])


total = count.sum()
mean = np.average(match_index, weights=count)
variance = np.average((match_index - mean)**2, weights=count)
std = np.sqrt(variance)
sem = std / np.sqrt(total)


# Plot
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(match_index, count, color="paleturquoise", edgecolor="black")

# Axis labels and title
ax.set_xlabel("Rank Index", fontsize=14)
ax.set_ylabel("Count", fontsize=14)
ax.set_title(f"Histogram of Top-{3} Ranking", fontsize=16)

# Aesthetic improvements
ax.set_xticks(match_index)
ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.spines[['top', 'right']].set_visible(False)

# Add vertical lines for mean ± std and mean ± sem
ax.axvline(mean, color='red', linestyle='--', linewidth=3, label=f'Mean = {mean:.2f}')
ax.axvline(mean - std, color='orange', linestyle=':', linewidth=2.5, label=f'Std = {std:.2f}')
ax.axvline(mean + std, color='orange', linestyle=':', linewidth=2.5)
ax.axvline(mean - sem, color='green', linestyle='-.', linewidth=2.5, label=f'SEM = {sem:.2f}')
ax.axvline(mean + sem, color='green', linestyle='-.', linewidth=2.5)

# Add legend
ax.legend(loc='upper right', fontsize=10)

plt.tight_layout()

# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height}', xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4), textcoords="offset points", ha='center', fontsize=10)

# Add description text box in top right corner (relative coordinates)
description = "*This histogram shows how\n the regressors predicted top-3\n fracture reassemblies are distributed\n across the ground truth ranking."
ax.text(0.985, 0.82, description,
        transform=ax.transAxes,
        fontsize=11,
        va='top', ha='right',
        bbox=dict(boxstyle="round,pad=0.4", facecolor='white', edgecolor='gray', alpha=0.8))

plt.show()

#plt.savefig("top3_reg_histogram.png", dpi=300, bbox_inches='tight')
