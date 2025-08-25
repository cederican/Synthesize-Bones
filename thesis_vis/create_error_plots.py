import numpy as np
import matplotlib.pyplot as plt

# Define metrics and configurations
metrics = ["part_acc ↑", "rmse_r (*1e-2) ↓", "rmse_t (*1e1) ↓", "shape_cd (*1e1) ↓", "%matchpoints ↑"]
#configs = ["AE 0 bones", "AE 1k bones", "AE 10k bones"]
configs = ["SOTA; Denoiser Only", "SOTA; wVerifier (6iter)", "Single-Shot; Denoiser Only", "Single-Shot; wVerifier (6iter)", "PercMP Multi-Shot 16; DDPM 20", "PercMP Multi-Shot 32; DDPM 10", "PercMP Multi-Shot 64; DDPM 5", "GTALL Multi-Shot 16; DDPM 20", "GTALL Multi-Shot 32; DDPM 10", "GTALL Multi-Shot 64; DDPM 5"]

means = np.array([
    [0.6730, 0.7060, 0.7266, 0.7525, 0.8159, 0.8246, 0.8339, 0.8327, 0.8491, 0.8603],    # part_acc
    [40.8000, 38.1000, 39.4167, 37.1776, 27.9407, 25.2483, 20.0419, 21.1637, 17.9627, 14.2021],  # rmse_r
    [0.0906, 0.0804, 0.0735, 0.0669, 0.0501, 0.0469, 0.0442, 0.0360, 0.0326, 0.0296],     # rmse_t
    [0.0065, 0.0060, 0.0069, 0.0064, 0.0051, 0.0045, 0.0045, 0.0040, 0.0036, 0.0032],      # shape_cd
    [0, 0, 0, 0, 0.4188, 0.4255, 0.4215, 0, 0, 0]
])
means[1] /= 100  # Scale rmse_r
means[2] *= 10  # Scale rmse_t
means[3] *= 10  # Scale shape_cd

# Standard deviations for each configuration (with scaling applied)
std_devs = np.array([
    [0.3209, 0.3081, 0.3108, 0.2964, 0.2662, 0.2591, 0.2515, 0.2320, 0.2212, 0.2117],     # part_acc
    [27.714, 27.6054, 27.8434, 27.2040, 26.1092, 24.5319, 21.3203, 20.9852, 18.9062, 16.2326],  # rmse_r (scaled)
    [0.0944, 0.0879, 0.0936, 0.0881, 0.0763, 0.0703, 0.0680, 0.0481, 0.0426, 0.0363],     # rmse_t
    [0.0266, 0.0253, 0.0286, 0.0293, 0.0245, 0.0214, 0.0210, 0.0153, 0.0132, 0.0121],      # shape_cd
    [0, 0, 0, 0, 0.3277, 0.3275, 0.3298, 0, 0, 0]
])
std_devs[1] /= 100  # Scale rmse_r
std_devs[2] *= 10  # Scale rmse_t
std_devs[3] *= 10  # Scale shape_cd

# Standard errors (SEM) for each configuration (with scaling applied)
sem_values = np.array([
    [0.0036, 0.0035, 0.0035, 0.0034, 0.0030, 0.0029, 0.0029, 0.0026, 0.0025, 0.0024],     # part_acc
    [0.3162, 0.3150, 0.3177, 0.3104, 0.2979, 0.2799, 0.2433, 0.2395, 0.2157, 0.1852],     # rmse_r (scaled)
    [0.0010, 0.0010, 0.0011, 0.0010, 0.0009, 0.0008, 0.0008, 0.0005, 0.0005, 0.0004],     # rmse_t
    [0.0003, 0.0002, 0.0003, 0.0003, 0.0003, 0.0002, 0.0002, 0.0002, 0.0002, 0.0001],      # shape_cd
    [0, 0, 0, 0, 0.0037, 0.0037, 0.0037, 0, 0, 0]
])
sem_values[1] /= 100  # Scale rmse_r
sem_values[2] *= 10  # Scale rmse_t
sem_values[3] *= 10  # Scale shape_cd

# Example means for each metric in three configurations
# means = np.array([
#     [0.8946, 0.8991, 0.9016],    # part_acc
#     [17.7383, 17.1230, 17.4799],  # rmse_r
#     [0.0273, 0.0261, 0.0264],     # rmse_t
#     [0.0006, 0.0006, 0.0006]      # shape_cd
# ])
# means[1] /= 100  # Scale rmse_r
# means[3] *= 100  # Scale shape_cd

# # Standard deviations for each configuration (with scaling applied)
# std_devs = np.array([
#     [0.1601, 0.1572, 0.1568],     # part_acc
#     [17.8020, 17.6519, 17.8211],  # rmse_r (scaled)
#     [0.0273, 0.0261, 0.0272],     # rmse_t
#     [0.0009, 0.0008, 0.0012]      # shape_cd
# ])
# std_devs[1] /= 100  # Scale rmse_r
# std_devs[3] *= 100  # Scale shape_cd

# # Standard errors (SEM) for each configuration (with scaling applied)
# sem_values = np.array([
#     [0.0051, 0.0050, 0.0050],     # part_acc
#     [0.5629, 0.5582, 0.5636],     # rmse_r (scaled)
#     [0.0009, 0.0008, 0.0009],     # rmse_t
#     [0.0000, 0.0000, 0.0000]      # shape_cd
# ])
# sem_values[1] /= 100  # Scale rmse_r
# sem_values[3] *= 100  # Scale shape_cd

# means = np.array([
#     [0.8120, 0.8097, 0.8085],    # part_acc
#     [26.6439, 26.7396, 26.7280],  # rmse_r
#     [0.0419, 0.0419, 0.0424],     # rmse_t
#     [0.0014, 0.0013, 0.0014]      # shape_cd
# ])
# means[1] /= 100  # Scale rmse_r
# means[3] *= 100  # Scale shape_cd

# # Standard deviations for each configuration (with scaling applied)
# std_devs = np.array([
#     [0.2254, 0.2265, 0.2242],     # part_acc
#     [22.3035, 22.5256, 22.5077],  # rmse_r (scaled)
#     [0.0383, 0.0380, 0.0381],     # rmse_t
#     [0.0032, 0.0030, 0.0030]      # shape_cd
# ])
# std_devs[1] /= 100  # Scale rmse_r
# std_devs[3] *= 100  # Scale shape_cd

# # Standard errors (SEM) for each configuration (with scaling applied)
# sem_values = np.array([
#     [0.0071, 0.0072, 0.0071],     # part_acc
#     [0.7053, 0.7123, 0.7118],     # rmse_r (scaled)
#     [0.0012, 0.0012, 0.0012],     # rmse_t
#     [0.0001, 0.0001, 0.0001]      # shape_cd
# ])
# sem_values[1] /= 100  # Scale rmse_r
# sem_values[3] *= 100  # Scale shape_cd

# 0_denoiser
# means = np.array([
#     [0.3919, 0.3633, 0.3535],    # part_acc
#     [58.4234, 59.9400, 60.4328],  # rmse_r
#     [0.1535, 0.1629, 0.1644],     # rmse_t
#     [0.0147, 0.0160, 0.0167]      # shape_cd
# ])
# means[1] /= 100  # Scale rmse_r
# means[3] *= 10  # Scale shape_cd

# # Standard deviations for each configuration (with scaling applied)
# std_devs = np.array([
#     [0.2368, 0.2159, 0.2102],     # part_acc
#     [19.2883, 18.7213, 17.8344],  # rmse_r (scaled)
#     [0.0755, 0.0740, 0.0704],     # rmse_t
#     [0.0189, 0.0199, 0.0202]      # shape_cd
# ])
# std_devs[1] /= 100  # Scale rmse_r
# std_devs[3] *= 10  # Scale shape_cd

# # Standard errors (SEM) for each configuration (with scaling applied)
# sem_values = np.array([
#     [0.0075, 0.0068, 0.0066],     # part_acc
#     [0.6099, 0.5920, 0.5640],     # rmse_r (scaled)
#     [0.0024, 0.0023, 0.0022],     # rmse_t
#     [0.0006, 0.0006, 0.0006]      # shape_cd
# ])
# sem_values[1] /= 100  # Scale rmse_r
# sem_values[3] *= 10  # Scale shape_cd

# Plot setup
fig, ax = plt.subplots(figsize=(16, 8))

# X-axis positions for grouped bars
bar_width = 1.0  # Width of each bar
x = np.arange(len(metrics)) * (len(configs) + 4)  # Space between metric groups

# Define colors
colors = ['powderblue', 'mistyrose', 'skyblue', 'lightcoral', 'lightgreen', 'plum', 'khaki', 'mediumaquamarine', 'lavender', 'paleturquoise']



# Create bars for each configuration
for i in range(len(configs)):
    if std_devs[:, i][0] is not None:
        ax.bar(x + i * bar_width, means[:, i], width=bar_width, yerr=std_devs[:, i], 
               capsize=5, color=colors[i], edgecolor='black', label=f"{configs[i]}")
    else:
        ax.bar(x + i * bar_width, means[:, i], width=bar_width, color=colors[i], 
               edgecolor='black', label=f"{configs[i]}")
    
    if sem_values[:, i][0] is not None:
        # Add SEM error bars as small dots
        ax.errorbar(x + i * bar_width, means[:, i], yerr=sem_values[:, i], fmt='o', 
                    color='grey', capsize=4, markersize=4)

# Labels and title
ax.set_xticks(x + bar_width + 3.0)  # Center the labels
ax.set_xticklabels(metrics, fontsize=14)  # Increase font size of x-axis labels
ax.set_ylabel("Metric Value", fontsize=14)  # Increase font size of y-axis label
ax.set_title("Evaluation Metrics on SE3 DiT | Denoiser on 34K train data; 8K val data", fontsize=14)  # Increase font size of title

ax.set_ylim([-0.2, 1.05])
ax.set_yticks(np.arange(-0.0, 1.05, 0.05)) 

# Add legend and grid
ax.legend(fontsize=12)  # Increase font size of legend
ax.grid(axis='y', linestyle='--', alpha=0.6)

# Show plot
plt.show()
