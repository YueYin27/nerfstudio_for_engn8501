import torch
import trimesh
import open3d as o3d

# load mesh
mesh = trimesh.load_mesh('water.ply')
# create some rays
ray_origins = torch.tensor([[0.1619, -0.1041,  0.3506]])  # [num_rays, num_samples, coordinates]
ray_directions = torch.tensor([[-0.1439,  0.5516, -0.8216]])  # [num_rays, num_samples, coordinates]

# Convert trimesh vertices and faces to tensors
vertices_tensor = o3d.core.Tensor(mesh.vertices, dtype=o3d.core.Dtype.Float32)
triangles_tensor = o3d.core.Tensor(mesh.faces, dtype=o3d.core.Dtype.UInt32)  # Convert to UInt32

scene = o3d.t.geometry.RaycastingScene()  # Create a RaycastingScene
scene.add_triangles(vertices_tensor, triangles_tensor)  # add the triangles
rays = torch.cat((ray_origins, ray_directions), dim=-1)  # Prepare rays in the required format
rays_o3d = o3d.core.Tensor(rays.numpy(), dtype=o3d.core.Dtype.Float32)
results = scene.cast_rays(rays_o3d)  # Cast rays

intersections = ray_origins.numpy() + results['t_hit'].numpy()[:, None] * ray_directions.numpy()  # intersections
print(intersections)

normals = results["primitive_normals"].numpy()  # unit normals
print(normals)




# Convert trimesh mesh to Open3D TriangleMesh
mesh_o3d = o3d.geometry.TriangleMesh(
    vertices=o3d.utility.Vector3dVector(mesh.vertices),
    triangles=o3d.utility.Vector3iVector(mesh.faces)
)

# Create a PointCloud for the intersections
intersection_point = o3d.geometry.PointCloud()
intersection_point.points = o3d.utility.Vector3dVector(intersections)

# Visualize the normals using a LineSet
lines = []
points = []
for i in range(len(intersections)):
    points.append(intersections[i])
    points.append(intersections[i] + normals[i])
    lines.append([2*i, 2*i + 1])

colors = [[1, 0, 0] for _ in range(len(lines))]  # Red color for the normals
line_set = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(points),
    lines=o3d.utility.Vector2iVector(lines)
)
line_set.colors = o3d.utility.Vector3dVector(colors)

# Visualize everything together
o3d.visualization.draw_geometries([mesh_o3d, intersection_point, line_set])
