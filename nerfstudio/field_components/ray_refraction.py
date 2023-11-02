import numpy as np
import plotly.graph_objects as go
import torch
import trimesh
import open3d as o3d


def visualization(ray_samples, idx_start, idx_end) -> None:
    positions = ray_samples.frustums.get_positions()
    # radius = 0.9
    radius = 0.9 * 0.1
    for i in range(idx_start, idx_end):
        p = positions[i, ...]
        x = p[..., 0].flatten().cpu().numpy()
        y = p[..., 1].flatten().cpu().numpy()
        z = p[..., 2].flatten().cpu().numpy()
        # Create a meshgrid of points on the sphere's surface
        theta = np.linspace(0, 2 * np.pi, 100)
        phi = np.linspace(0, np.pi, 100)
        theta, phi = np.meshgrid(theta, phi)
        xs = radius * np.sin(phi) * np.cos(theta)
        ys = radius * np.sin(phi) * np.sin(theta)
        zs = radius * np.cos(phi)
        scatter_trace = go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=5))
        line_trace = go.Scatter3d(x=x, y=y, z=z, mode='lines', line=dict(color='blue'))
        o = ray_samples.frustums.origins[i].cpu()
        origins = go.Scatter3d(x=o[..., 0], y=o[..., 1], z=o[..., 2], mode='markers', marker=dict(size=5))
        surface_trace = go.Surface(x=xs, y=ys, z=zs, opacity=0.3)
        fig = go.Figure(data=[scatter_trace, line_trace, origins, surface_trace])
        fig.update_layout(scene=dict(aspectmode='data', aspectratio=dict(x=1, y=1, z=1)))
        fig.show()

    import sys
    sys.exit(0)


class RayRefraction:
    """Ray refracting

    Args:
        origins: camera ray origins
        directions: original directions of camera rays
        positions: original positions of sample points
        r: n1/n2
    """

    def __init__(self, origins, directions, positions, r=None, radius=None):
        self.origins = origins
        self.directions = directions / torch.norm(directions, p=2, dim=2, keepdim=True)
        self.positions = positions
        self.r = r
        self.radius = radius

    def get_intersections_and_normals(self, condition):
        raise NotImplementedError

    # @functools.lru_cache(maxsize=128)
    def snell_fn(self, n):
        """Get new ray directions based on Snell's Law

        Args:
            n: surface normals
        """
        r = self.r
        l = self.directions  # [4096, 48, 3]
        c = - torch.einsum('ijk, ijk -> ij', n, l)  # dot product, [4096, 48]
        return r * l + (r * c - torch.sqrt(1 - (r ** 2) * (1 - c ** 2))).unsqueeze(-1) * n

    def update_sample_points(self, intersections, directions_new, condition, mask):
        raise NotImplementedError


class WaterBallRefraction(RayRefraction):

    def __init__(self, origins, directions, positions, r, radius):
        super().__init__(origins, directions, positions, r, radius)

    # @functools.lru_cache(maxsize=128)
    def get_intersections_and_normals(self, condition):
        """Get intersections and surface normals

        Args:
            condition: -
        Returns:
            intersections: intersections
            normals: normals
        """

        # Define the sphere
        sphere_origin = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=self.origins.device)
        sphere_radius = self.radius

        # Reshape the origins and directions tensors to (num_rays * num_samples, 3)
        ray_origins = self.origins.reshape(-1, 3)
        ray_directions = self.directions.reshape(-1, 3)

        # Compute the discriminant for each ray
        oc = ray_origins - sphere_origin
        a = torch.sum(ray_directions * ray_directions, dim=1)
        b = 2.0 * torch.sum(oc * ray_directions, dim=1)
        c = torch.sum(oc * oc, dim=1) - sphere_radius ** 2
        discriminant = b ** 2 - 4 * a * c

        if condition == 'in':
            # Compute the intersections and normals for each ray
            t = (-b - torch.sqrt(discriminant)) / (2 * a)
            intersections = ray_origins + t[:, None] * ray_directions
            normals = (intersections - sphere_origin) / sphere_radius

            # Convert the intersections and normals back to their original shape
            intersections = intersections.view(*self.origins.shape)
            normals = normals.view(*self.origins.shape)
        else:
            # Compute the intersections and normals for each ray
            t = (-b + torch.sqrt(discriminant)) / (2 * a)
            intersections = ray_origins + t[:, None] * ray_directions
            normals = (intersections - sphere_origin) / sphere_radius

            # Convert the intersections and normals back to their original shape
            intersections = intersections.view(*self.origins.shape)
            normals = normals.view(*self.origins.shape)

        return intersections, normals

    # @functools.lru_cache(maxsize=128)
    def update_sample_points(self, intersections, directions_new, condition, mask):
        """Update sample points

        Args:
            intersections: intersections of the camera ray with the surface of the object
            directions_new: refracted directions
            condition: -
            mask: -
        """
        # 1. Get the index of the first point to be updated
        if condition == 'in':
            mask = torch.logical_and((torch.norm(self.positions, dim=-1)) < self.radius, mask)
        else:
            mask = torch.logical_and((torch.norm(self.positions, dim=-1)) > self.radius, mask)

        mask = mask.unsqueeze(-1).expand(-1, -1, 3)
        masked_positions = torch.where(mask, self.positions, intersections)
        distances = torch.norm(intersections - masked_positions, dim=-1)  # Calculate Euclidean distances [4096, 48]
        first_idx = torch.argmin(torch.where(distances != 0, distances, torch.finfo(distances.dtype).max), dim=1)

        # 2. Get the mask of all samples to be updated
        first_idx = first_idx.unsqueeze(1)
        # mask = torch.arange(48).to(device='cuda:0') >= first_idx
        mask = ~torch.isnan(intersections).any(dim=2) & ~torch.isnan(directions_new).any(dim=2)
        mask = mask & (torch.arange(self.positions.shape[1], device=self.positions.device).unsqueeze(0) >= first_idx)

        # 3. Move the original sample points onto the refracted ray
        distances_to_intersection = torch.norm(self.positions - intersections, dim=2)
        distances_to_intersection[~mask] = float('nan')
        updated_positions = intersections + distances_to_intersection.unsqueeze(2) * directions_new
        self.positions[mask] = updated_positions[mask]

        return self.positions, mask


class MeshRefraction(RayRefraction):

    def __init__(self, origins, directions, positions, r):
        super().__init__(origins, directions, positions, r)

    def get_intersections_and_normals(self, mesh):

        """Get intersections and surface normals

        Args:
            mesh: mesh of the 3D object
        Returns:
            intersections: intersections
            normals: normals
        """
        device = self.origins.device
        scale_factor = 0.1

        # Convert trimesh vertices and faces to tensors
        vertices_tensor = o3d.core.Tensor(mesh.vertices * scale_factor, dtype=o3d.core.Dtype.Float32)  # Scale mesh.vertices!
        triangles_tensor = o3d.core.Tensor(mesh.faces, dtype=o3d.core.Dtype.UInt32)  # Convert to UInt32

        scene = o3d.t.geometry.RaycastingScene()  # Create a RaycastingScene
        scene.add_triangles(vertices_tensor, triangles_tensor)  # add the triangles
        rays = torch.cat((self.origins, self.directions), dim=-1).cpu().numpy()  # Prepare rays in the required format
        rays_o3d = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
        results = scene.cast_rays(rays_o3d)  # Cast rays

        # Convert results to PyTorch tensors and move to the correct device
        t_hit = torch.tensor(results['t_hit'].cpu().numpy(), device=device).unsqueeze(-1)
        intersections = self.origins + t_hit * self.directions
        normals = torch.tensor(results["primitive_normals"].cpu().numpy(), device=device)

        # check if the normals are pointing upwards
        normal_up = torch.tensor([0.0, 0.0, 1.0], device=device)
        dot_products = torch.einsum('ijk,k->ij', normals, normal_up)
        mask = (dot_products > 0).unsqueeze(-1)
        intersections = torch.where(mask, intersections, torch.tensor(float('nan'), device=device))
        normals = torch.where(mask, normals, torch.tensor(float('nan'), device=device))

        return intersections, normals

    def update_sample_points(self, intersections, directions_new, condition, mask):
        """Update sample points

                Args:
                    intersections: intersections of the camera ray with the surface of the object
                    directions_new: refracted directions
                    condition: -
                    mask: -
                """
        # 1. Get the index of the first point to be updated
        mask = torch.logical_and((torch.norm(self.positions - self.origins, dim=-1)) >
                                 (torch.norm(intersections - self.origins, dim=-1)), mask)  # samples after intersections
        mask = mask.unsqueeze(-1).expand(-1, -1, 3)
        masked_positions = torch.where(mask, self.positions, intersections)
        distances = torch.norm(intersections - masked_positions, dim=-1)  # Calculate Euclidean distances [4096, 48]
        first_idx = torch.argmin(torch.where(distances != 0, distances, torch.finfo(distances.dtype).max), dim=1)

        # 2. Get the mask of all samples to be updated
        first_idx = first_idx.unsqueeze(1)
        mask = ~torch.isnan(intersections).any(dim=2) & ~torch.isnan(directions_new).any(dim=2)
        mask = mask & (torch.arange(self.positions.shape[1], device=self.positions.device).unsqueeze(0) >= first_idx)

        # 3. Move the original sample points onto the refracted ray
        distances_to_intersection = torch.norm(self.positions - intersections, dim=2)
        distances_to_intersection[~mask] = float('nan')
        updated_positions = intersections + distances_to_intersection.unsqueeze(2) * directions_new
        self.positions[mask] = updated_positions[mask]

        return self.positions, mask, first_idx
