import trimesh
import numpy as np
import os
import shutil

if __name__ == "__main__":
    
    overfit = -1
    dataset_name = "real_"
    input_path = "/Users/cedimac/fracture-modes/data/PreOP/"
    data_lst = os.listdir(input_path)
    data_lst.sort(key=lambda x: int(x))

    if overfit != -1:
        data_lst = data_lst[:overfit]
    print(f"Processing {len(data_lst)} files")
    
    for _, idx in enumerate(data_lst):
            if idx in ['004', '011', '012']:
                continue
            if not os.path.isdir(os.path.join(input_path, idx)):
                continue
            output_dir = os.path.join(input_path, idx, dataset_name + idx, "fractured_0")
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            os.makedirs(output_dir, exist_ok=True)
            mesh_files = [os.path.join(input_path, idx, f) for f in os.listdir(os.path.join(input_path, idx)) if f.endswith('.obj') and "RealTibia" in f]

            meshes = [trimesh.load(mesh_file) for mesh_file in mesh_files]
            largest_mesh_idx = [i for i, f in enumerate(mesh_files) if 'fragment' not in f][0]
            largest_mesh = meshes[largest_mesh_idx]

            min_coords = np.min(largest_mesh.vertices, axis=0)
            max_coords = np.max(largest_mesh.vertices, axis=0)
            center_point = (min_coords + max_coords) / 2        

            largest_mesh.vertices -= center_point
            max_abs = np.max(np.abs(largest_mesh.vertices))
            largest_mesh.vertices /= (2 * max_abs)

            for i, mesh in enumerate(meshes):
                if i != largest_mesh_idx:
                    mesh.vertices -= center_point
                    mesh.vertices /= (2 * max_abs)

            # Save the adjusted meshes
            largest_mesh.export(os.path.join(output_dir, 'piece_0.obj'))
            piece_idx = 1
            for i, mesh in enumerate(meshes):
                if i != largest_mesh_idx:
                    mesh.export(os.path.join(output_dir, f'piece_{piece_idx}.obj'))
                    piece_idx += 1