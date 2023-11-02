import torch
import trimesh
import open3d as o3d

# load mesh
mesh = trimesh.load_mesh('water.ply')
# create some rays
# ray_origins = torch.tensor([[0.0, 0.0,  0.4]])  # [num_rays, num_samples, coordinates]
# ray_directions = torch.tensor([[0.0,  0.0, -1.0]])  # [num_rays, num_samples, coordinates]
ray_origins = torch.tensor([[[0.0, 0.0,  0.4]]])  # [num_rays, num_samples, coordinates]
ray_directions = torch.tensor([[[0.0,  0.0, -1.0]]])  # [num_rays, num_samples, coordinates]
scale_vector = 0.1

# Convert trimesh vertices and faces to tensors
vertices_tensor = o3d.core.Tensor(mesh.vertices * 0.1, dtype=o3d.core.Dtype.Float32)
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

dot_products = torch.einsum('ijk,ijk->ij', ray_directions, normals)

# Calculate the reflected direction
# The operation is batched over the first two dimensions
reflected_directions = ray_directions - 2 * dot_products.unsqueeze(-1) * normals

# Normalize the reflected directions
reflection_direction = torch.nn.functional.normalize(reflected_directions, dim=2)

r = 1/1.33  # n1/n2
l = ray_directions  # [4096, 48, 3]

# Calculate the cosine of the angle of incidence
cos_theta_i = -torch.einsum('ijk,ijk->ij', l, normals)
print(cos_theta_i)
sin_theta_i_squared = 1 - cos_theta_i ** 2

# Calculate the sine of the refraction angle using Snell's law
sin_theta_t_squared = r ** 2 * sin_theta_i_squared
cos_theta_t = torch.sqrt(1 - sin_theta_t_squared)

# Calculate Rs and Rp using the Fresnel equations
Rs = ((r * cos_theta_i - cos_theta_t) / (r * cos_theta_i + cos_theta_t)) ** 2
Rp = ((r * cos_theta_t - cos_theta_i) / (cos_theta_i + r * cos_theta_t)) ** 2

# Calculate the average reflectance
R = (Rs + Rp) / 2
print(Rs,Rp,R)

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
