from context import fracture_utility as fracture
import os
import trimesh
import numpy as np


if __name__ == "__main__":
    
    overfit = 53
    dataset_name = "advanced_"
    input_path = "/Users/cedimac/fracture-modes/data/healthy/"
    data_lst = os.listdir(input_path)
    data_lst.sort(key=lambda x: int(x))

    if overfit != -1:
        data_lst = data_lst[overfit:]
    print(f"Processing {len(data_lst)} files")
    
    for _, idx in enumerate(data_lst):
            if not os.path.isdir(os.path.join(input_path, idx)):
                continue
            mesh_files = [os.path.join(input_path, idx, f) for f in os.listdir(os.path.join(input_path, idx)) if f.endswith('.obj') and "TibiaHead" in f]

            for mesh_file in mesh_files:
                output_dir = os.path.join(input_path, idx, dataset_name + mesh_file.split('/')[-1].split('.')[0])
                if not os.path.exists(output_dir):
                    os.mkdir(output_dir)
                
                # Normalize mesh
                mesh = trimesh.load(mesh_file)
                mesh.vertices -= mesh.vertices.mean(axis=0)
                max_abs = np.max(np.abs(mesh.vertices))
                mesh.vertices /= (2 * max_abs)
                mesh.export(mesh_file)

                # Generate fractures
                num_modes = 8 #if int(idx) % 2 else 20
                fracture.generate_fractures(mesh_file,
                                            output_dir=output_dir,
                                            num_modes=num_modes,
                                            num_impacts=80,
                                            verbose=True,
                                            compressed=False,
                                            cage_size=4000,
                                            volume_constraint=0.005,
                                            impact_intervall=[-1.0, 1.0], # -0.5, 0.5
                                            threshold_interval=[5, 40]) # 0.0, 1000.0