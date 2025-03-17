import os
import subprocess

# Define local and remote paths
local_base = "/Users/cedimac/fracture-modes/data/healthy"
remote_user = "cederic"
remote_host = "arp"
remote_base_L = "/home/datasets/Breaking-Bad-Dataset.synthezised.bones/data/breaking_bad/bones/Tibia_L"
remote_base_R = "/home/datasets/Breaking-Bad-Dataset.synthezised.bones/data/breaking_bad/bones/Tibia_R"

data_lst = os.listdir(local_base)
data_lst.sort(key=lambda x: int(x))
print(f"Processing {len(data_lst)} files")

for idx in data_lst:
    folder_path = os.path.join(local_base, idx)
    if not os.path.isdir(folder_path):
        continue

    for subfolder in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder)
        
        if os.path.isdir(subfolder_path) and subfolder.startswith(f"{idx}_tibia"):
            if "_L" in subfolder:
                remote_base = remote_base_L
            elif "_R" in subfolder:
                remote_base = remote_base_R
            else:
                print(f"Skipping {subfolder}, does not match expected format.")
                continue

            remote_path = f"{remote_user}@{remote_host}:{remote_base}/{subfolder}"
            
            command = ["rsync", "-avz", subfolder_path + "/", remote_path]
            print(f"Uploading {subfolder_path} to {remote_path}...")
            subprocess.run(command, check=True)

print("All folders uploaded successfully!")
