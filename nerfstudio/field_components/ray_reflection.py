import numpy as np
import plotly.graph_objects as go
import torch
import open3d as o3d


class RayReflection:
    """Ray reflection

    Args:
        origins: camera ray origins
        directions: original directions of camera rays
        positions: original positions of sample points
        r: n1/n2
    """

    def __init__(self, origins, directions, positions, r=None):
        self.origins = origins
        self.directions = directions / torch.norm(directions, p=2, dim=2, keepdim=True)
        self.positions = positions
        self.r = r  # n1/n2

    def get_reflected_directions(self, normals):
        """Get new ray directions based on Fresnel Equations

        Args:
            normals: surface normals (unit vector)
        Returns:
            directions: reflected directions (unit vector)
        """
        l = self.directions  # [4096, 48, 3]
        dot_products = torch.einsum('ijk,ijk->ij', l, normals)

        # Calculate the reflected direction
        # The operation is batched over the first two dimensions
        reflected_directions = l - 2 * dot_products.unsqueeze(-1) * normals

        # Normalize the reflected directions
        reflected_directions = torch.nn.functional.normalize(reflected_directions, dim=2)

        return reflected_directions

    def fresnel_fn(self, normals):
        """Get new ray directions based on Fresnel Equations

        Args:
            normals: surface normals
        Returns:
            R: fraction of reflection
        """
        r = self.r  # n1/n2
        l = self.directions  # [4096, 48, 3]

        # Calculate the cosine of the angle of incidence
        cos_theta_i = -torch.einsum('ijk,ijk->ij', l, normals)
        sin_theta_i_squared = 1 - cos_theta_i ** 2

        # Calculate the sine of the refraction angle using Snell's law
        sin_theta_t_squared = r ** 2 * sin_theta_i_squared
        cos_theta_t = torch.sqrt(1 - sin_theta_t_squared)

        # Calculate Rs and Rp using the Fresnel equations
        Rs = ((r * cos_theta_i - cos_theta_t) / (r * cos_theta_i + cos_theta_t)) ** 2
        Rp = ((r * cos_theta_t - cos_theta_i) / (cos_theta_i + r * cos_theta_t)) ** 2

        # Calculate the average reflectance
        R = (Rs + Rp) / 2

        return R

    def update_sample_points(self, intersections, directions_new, mask):
        """Add sample points along reflected rays
        Args:
            intersections
            directions_new
            idx
            mask
        Returns:
            mask
        """
        positions = self.positions.clone()

        # Move the original sample points onto the refracted ray
        distances_to_intersection = torch.norm(positions - intersections, dim=2)
        distances_to_intersection[~mask] = float('nan')
        updated_positions = intersections + distances_to_intersection.unsqueeze(2) * directions_new
        positions[mask] = updated_positions[mask]
        self.positions = positions
