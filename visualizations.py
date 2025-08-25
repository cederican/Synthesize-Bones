import os
import trimesh
import numpy as np
from scipy.spatial import cKDTree
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.spatial.transform import Rotation as R
import torch

def apply_random_transform(points, normals=None, rotation_deg=30, translation_range=0.1, seed=None):
    if seed is not None:
        np.random.seed(seed)
    angle = np.deg2rad(np.random.uniform(-rotation_deg, rotation_deg))
    axis = np.random.normal(size=3)
    axis /= np.linalg.norm(axis)
    rot = R.from_rotvec(angle * axis)
    R_mat = rot.as_matrix()

    transformed_points = points @ R_mat.T
    transformed_normals = None
    if normals is not None:
        transformed_normals = normals @ R_mat.T

    translation = np.random.uniform(-translation_range, translation_range, size=(1, 3))
    transformed_points += translation

    return transformed_points, transformed_normals

# ---------------------
# Config
# ---------------------
OBJ_DIR = '/Users/cedimac/fracture-modes/data/mode_5'   # ← put your .obj files here
SAMPLES_PER_FRAGMENT = 1000
INTERFACE_THRESHOLD = 0.05  # distance in same units as mesh scale (e.g., mm)

# ---------------------
# Helper Functions
# ---------------------

def load_meshes_from_directory(directory):
    meshes = []
    labels = []
    for idx, filename in enumerate(sorted(os.listdir(directory))):
        if filename.endswith(".obj"):
            mesh = trimesh.load(os.path.join(directory, filename), force='mesh')
            if not isinstance(mesh, trimesh.Trimesh):
                print(f"Skipping non-mesh file: {filename}")
                continue
            meshes.append(mesh)
            labels.append(idx)
    return meshes, labels

def sample_points_from_mesh(mesh, label, n_samples):
    points, face_indices = trimesh.sample.sample_surface(mesh, n_samples)
    normals = mesh.face_normals[face_indices]
    return {
        'points': points,
        'normals': normals,
        'label': np.full((n_samples,), label),
        'mesh': mesh
    }

def compute_interface_values(points, all_other_points_tree):
    dists, indices = all_other_points_tree.query(points, k=1)
    values = np.clip(1.0 - (dists / INTERFACE_THRESHOLD), 0.0, 1.0)
    return values, indices

meshes, labels = load_meshes_from_directory(OBJ_DIR)
print(f"Loaded {len(meshes)} mesh fragments.")

all_data = []

# Sample points from each mesh
for mesh, label in zip(meshes, labels):
    data = sample_points_from_mesh(mesh, label, SAMPLES_PER_FRAGMENT)
    if label == 0:
        transfromed_points, transformed_normals = apply_random_transform(
            data['points'], data['normals'], rotation_deg=30, translation_range=0.2, seed=None
        )
        data['points'] = transfromed_points
        data['normals'] = transformed_normals if transformed_normals is not None else data['normals']
    all_data.append(data)


alignment_scores = []

# Compute interface values for each fragment's points
output = []
for data in all_data:
    points = data['points']
    normals = data['normals']

    all_other_points = np.vstack([d['points'] for d in all_data if not np.array_equal(d['points'], points)])
    all_other_normals = np.vstack([d['normals'] for d in all_data if not np.array_equal(d['normals'], points)])
    tree = cKDTree(all_other_points)    

    interface_scores, nn_indices = compute_interface_values(points, tree)

    matched_normals = all_other_normals[nn_indices]

    cos_sims = np.einsum('ij,ij->i', normals, matched_normals)
    normal_alignment = 1.0 - np.abs(cos_sims)

    pointwise_facs = interface_scores * normal_alignment
    alignment_scores.append(pointwise_facs)

    for i in range(len(points)):
        output.append({
            'point': points[i],
            'normal': normals[i],
            'matched_normal': matched_normals[i],
            'label': int(data['label'][i]),
            'interface_score': float(interface_scores[i])
        })

facs_score = np.sum(np.concatenate(alignment_scores)) / np.sum([
    np.sum(s > 0.01) for s, _ in [compute_interface_values(d['points'], tree) for d in all_data]
])

print(f"FACS Score: {facs_score:.4f}")




    # # Remove self-points from KDTree query (optional, more precise)
    # other_points = np.vstack([d['points'] for d in all_data if not np.array_equal(d['points'], points)])
    # tree = cKDTree(other_points)
    # interface_vals = compute_interface_values(points, tree)

    # for i in range(len(points)):
    #     output.append({
    #         'point': points[i],
    #         'normal': data['normals'][i],
    #         'label': int(data['label'][i]),
    #         'interface_score': float(interface_vals[i])
    #     })

# ---------------------
# Save or Inspect Output
# ---------------------

print(f"\nTotal sampled points: {len(output)}")

# Example: print first 5 entries
# for i in range(5):
#     p = output[i]
#     print(f"Point {i}: Pos={p['point']}, Label={p['label']}, Interface Score={p['interface_score']:.3f}")

# ---------------------
# Create Colored Point Cloud
# ---------------------

# points_np = np.load('/Users/cedimac/fracture-modes/data/predicted_points_1.npy')
# normals_np = np.load('/Users/cedimac/fracture-modes/data/predicted_normals_1.npy')
# scores = np.load('/Users/cedimac/fracture-modes/data/predicted_locdists_1.npy')

x = 1

points_torch = torch.load('/Users/cedimac/fracture-modes/vis_architecture/2reg_coords_T.pt', map_location='cpu')
scores_torch = torch.load('/Users/cedimac/fracture-modes/vis_architecture/2reg_fracture_surface_T.pt', map_location='cpu')
normals_torch = torch.load('/Users/cedimac/fracture-modes/vis_architecture/normals.pt', map_location='cpu')
latent_torch = torch.load('/Users/cedimac/fracture-modes/vis_architecture/latent_coords.pt', map_location='cpu')

if x == 0:
    label_to_color = {
        4: [0.6, 0.0, 0.5],    # purple
        1: [0.0, 0.8, 0.8],    # turquoise
        3: [1.0, 0.3, 0.0],    # orange
        0: [0.0, 0.6, 0.0],    # dark green
        2: [0.0, 0.0, 0.7],    # blue
        5: [0.5, 0.7, 0.2],
    }

    for i in range(points_torch.shape[0]):

        points_np = np.array(points_torch.tolist())[i].reshape(-1,3)
        normals_np = np.array(normals_torch.tolist())[i].reshape(-1,3)
        scores_np = np.array(scores_torch.tolist())[i].reshape(-1,1)
        latents_np = np.array(latent_torch.tolist())[i].reshape(-1,3)

        color = np.array(label_to_color[i])
        colors = np.tile(color, (latents_np.shape[0], 1))

        #colors = cm.jet(scores_np)[:, 0, :3]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(latents_np)
        #pcd.normals = o3d.utility.Vector3dVector(normals_np)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries(
            [pcd],
            window_name='Test',
            point_show_normal=True
        )
elif x == 1:
    points_np = np.array(points_torch.tolist()).reshape(-1,3)
    normals_np = np.array(normals_torch.tolist()).reshape(-1,3)
    scores_np = np.array(scores_torch.tolist()).reshape(-1,1)
    latents_np = np.array(latent_torch.tolist()).reshape(-1,3)
    #color = np.array(label_to_color[i])
    #colors = np.tile(color, (points_np.shape[0], 1))
    colors = cm.jet(scores_np)[:, 0, :3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)
    #pcd.normals = o3d.utility.Vector3dVector(normals_np)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries(
        [pcd],
        window_name='Test',
        point_show_normal=True
    )


scores = np.array([p['interface_score'] for p in output])
colors = cm.jet(scores)[:, :3]

points_np = np.array([p['point'] for p in output])
normals_np = np.array([p['normal'] for p in output])
matched_normals_np = np.array([p['matched_normal'] for p in output])
labels_np = np.array([p['label'] for p in output])

unique_labels = np.unique(labels_np)
colormap = plt.get_cmap("inferno")
label_to_color = {
    4: [0.6, 0.0, 0.5],    # purple
    1: [0.0, 0.8, 0.8],    # turquoise
    3: [1.0, 0.3, 0.0],    # orange
    0: [0.0, 0.6, 0.0],    # dark green
    2: [0.0, 0.0, 0.7],    # blue
}
colors = np.array([label_to_color[label] for label in labels_np])
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points_np)
pcd.colors = o3d.utility.Vector3dVector(colors)
pcd.normals = o3d.utility.Vector3dVector(normals_np * 0.8)
# o3d.visualization.draw_geometries(
#     [pcd],
#     window_name='Coordinates',
#     point_show_normal=False
# )

label_to_color = {
    4: [1.0, 1.0, 1.0],    # purple
    1: [1.0, 1.0, 1.0],    # turquoise
    3: [1.0, 1.0, 1.0],    # orange
    0: [1.0, 1.0, 1.0],    # dark green
    2: [1.0, 1.0, 1.0],    # blue
}
colors = np.array([label_to_color[label] for label in labels_np])
pcd.colors = o3d.utility.Vector3dVector(colors)
# o3d.visualization.draw_geometries(
#     [pcd],
#     window_name='Normals',
#     point_show_normal=True
# )

colors = cm.jet(scores)[:, :3]
pcd.colors = o3d.utility.Vector3dVector(colors)
o3d.visualization.draw_geometries(
    [pcd],
    window_name='Fracture Surface',
    point_show_normal=True
)


vis = o3d.visualization.Visualizer()
vis.create_window(visible=False)
vis.add_geometry(pcd)
vis.update_geometry(pcd)
vis.poll_events()
vis.update_renderer()

# Save as high-res PNG
vis.capture_screen_image("cloud.png", do_render=True)
vis.destroy_window()


o3d.visualization.gui.Application.instance.initialize()
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points_np)
pcd.colors = o3d.utility.Vector3dVector(colors)
pcd.normals = o3d.utility.Vector3dVector(normals_np)

# Scene rendering with GUI (not headless)
vis = o3d.visualization.O3DVisualizer("Point Cloud Viewer", 1024, 768)
vis.show_settings = True
vis.add_geometry("pcd", pcd)
vis.setup_camera(
    80.0,
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0]
)
#o3d.visualization.draw([pcd])





