# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Some ray datastructures.
"""
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, Literal, Optional, Tuple, Union, overload, List, Any

import torch
import trimesh
from jaxtyping import Float, Int, Shaped
from torch import Tensor

from nerfstudio.field_components.ray_refraction import WaterBallRefraction, MeshRefraction
from nerfstudio.field_components.ray_reflection import RayReflection
from nerfstudio.utils.math import Gaussians, conical_frustum_to_gaussian
from nerfstudio.utils.tensor_dataclass import TensorDataclass

TORCH_DEVICE = Union[str, torch.device]
mesh = trimesh.load_mesh('water.ply')


@dataclass
class Frustums(TensorDataclass):
    """Describes region of space as a frustum."""

    origins: Float[Tensor, "*bs 3"]
    """xyz coordinate for ray origin."""
    directions: Float[Tensor, "*bs 3"]
    """Direction of ray."""
    starts: Float[Tensor, "*bs 1"]
    """Where the frustum starts along a ray."""
    ends: Float[Tensor, "*bs 1"]
    """Where the frustum ends along a ray."""
    pixel_area: Float[Tensor, "*bs 1"]
    """Projected area of pixel a distance 1 away from origin."""
    offsets: Optional[Float[Tensor, "*bs 3"]] = None
    """Offsets for each sample position"""
    intersections: Optional[Float[Tensor, "*bs 3"]] = None
    """Intersections between each ray and surfaces"""
    normals: Optional[Float[Tensor, "*bs 3"]] = None
    """normals at each intersection"""
    mask: Optional = None
    """mask"""

    def get_positions(self) -> Float[Tensor, "*batch 3"]:
        """Calculates "center" position of frustum. Not weighted by mass.

        Returns:
            xyz positions.
        """
        pos = self.origins + self.directions * (self.starts + self.ends) / 2
        if self.offsets is not None:
            pos = pos + self.offsets
        return pos

    def get_start_positions(self) -> Float[Tensor, "*batch 3"]:
        """Calculates "start" position of frustum.

        Returns:
            xyz positions.
        """
        return self.origins + self.directions * self.starts

    def set_offsets(self, offsets):
        """Sets offsets for this frustum for computing positions"""
        self.offsets = offsets

    def get_gaussian_blob(self) -> Gaussians:
        """Calculates guassian approximation of conical frustum.

        Returns:
            Conical frustums approximated by gaussian distribution.
        """
        # Cone radius is set such that the square pixel_area matches the cone area.
        cone_radius = torch.sqrt(self.pixel_area) / 1.7724538509055159  # r = sqrt(pixel_area / pi)
        if self.offsets is not None:
            raise NotImplementedError()
        return conical_frustum_to_gaussian(
            origins=self.origins,
            directions=self.directions,
            starts=self.starts,
            ends=self.ends,
            radius=cone_radius,
        )

    @classmethod
    def get_mock_frustum(cls, device: Optional[TORCH_DEVICE] = "cpu") -> "Frustums":
        """Helper function to generate a placeholder frustum.

        Returns:
            A size 1 frustum with meaningless values.
        """
        return Frustums(
            origins=torch.ones((1, 3)).to(device),
            directions=torch.ones((1, 3)).to(device),
            starts=torch.ones((1, 1)).to(device),
            ends=torch.ones((1, 1)).to(device),
            pixel_area=torch.ones((1, 1)).to(device),
        )


@dataclass
class RaySamples(TensorDataclass):
    """Samples along a ray"""

    frustums: Frustums
    """Frustums along ray."""
    camera_indices: Optional[Int[Tensor, "*bs 1"]] = None
    """Camera index."""
    deltas: Optional[Float[Tensor, "*bs 1"]] = None
    """"width" of each sample."""
    spacing_starts: Optional[Float[Tensor, "*bs num_samples 1"]] = None
    """Start of normalized bin edges along ray [0,1], before warping is applied, ie. linear in disparity sampling."""
    spacing_ends: Optional[Float[Tensor, "*bs num_samples 1"]] = None
    """Start of normalized bin edges along ray [0,1], before warping is applied, ie. linear in disparity sampling."""
    spacing_to_euclidean_fn: Optional[Callable] = None
    """Function to convert bins to euclidean distance."""
    metadata: Optional[Dict[str, Shaped[Tensor, "*bs latent_dims"]]] = None
    """additional information relevant to generating ray samples"""

    times: Optional[Float[Tensor, "*batch 1"]] = None
    """Times at which rays are sampled"""

    def get_refracted_rays_old(self) -> None:
        # Modify the origins and directions of frustums here
        positions = self.frustums.get_positions()  # [4096, 48, 3] ([num_rays_per_batch, num_samples_per_ray, 3])

        # Modify positions based on known geometry and Snell's law
        # 1. Get origins, directions, r1, r2 directly
        origins = self.frustums.origins  # [4096, 48, 3]
        directions = self.frustums.directions  # [4096, 48, 3]
        r1, r2 = 1.0 / 1.33, 1.33 / 1.0
        # radius = 0.9
        radius = 0.9 * 0.1

        # 2. Get normals from the geometry, calculate new directions, and update positions after the first refraction
        ray_refraction_1 = WaterBallRefraction(origins, directions, positions, r1, radius)
        intersections_1, normals_1 = ray_refraction_1.get_intersections_and_normals('in')  # [4096, 48, 3]
        directions_1 = ray_refraction_1.snell_fn(normals_1)
        positions, mask_1 = ray_refraction_1.update_sample_points(intersections_1, directions_1, 'in',
                                                                  torch.ones([positions.shape[0],
                                                                              positions.shape[1]],
                                                                             dtype=torch.bool,
                                                                             device=positions.device))

        # 3. Get normals from the geometry, calculate new directions, and update positions after the second refraction
        ray_refraction_2 = WaterBallRefraction(intersections_1, directions_1, positions, r2, radius)
        intersections_2, normals_2 = ray_refraction_2.get_intersections_and_normals('out')
        directions_2 = ray_refraction_2.snell_fn(-normals_2)
        positions, mask_2 = ray_refraction_2.update_sample_points(intersections_2, directions_2, 'out', mask_1)

        self.frustums.intersections = [intersections_1, intersections_2]

        # 4. Update ray_samples.frustums.directions
        directions_new = directions.clone()
        directions_new[mask_1] = directions_1[mask_1]
        directions_new[mask_2] = directions_2[mask_2]

        # 5. Update ray_samples.frustums.origins
        origins_new = origins.clone()
        origins_1 = intersections_1 - directions_1 * torch.norm(origins - intersections_1, dim=-1).unsqueeze(2)
        origins_2 = intersections_2 - directions_2 * (
                torch.norm(origins - intersections_1, dim=-1) + torch.norm(intersections_1 - intersections_2,
                                                                           dim=-1)).unsqueeze(2)
        origins_new[mask_1] = origins_1[mask_1]
        origins_new[mask_2] = origins_2[mask_2]

        self.frustums.directions = directions_new
        self.frustums.origins = origins_new

    def get_refracted_rays(self):
        # Modify the origins and directions of frustums here
        positions = self.frustums.get_positions()  # [4096, 48, 3] ([num_rays_per_batch, num_samples_per_ray, 3])

        # Refraction: Modify positions based on known geometry and Snell's law
        # 1. Get origins, directions, r directly
        origins = self.frustums.origins  # [4096, 48, 3]
        directions = self.frustums.directions  # [4096, 48, 3]
        r = 1.0 / 1.33

        # 2. Get normals from the geometry, calculate new directions, and update positions after the first refraction
        ray_refraction = MeshRefraction(origins, directions, positions, r)
        intersections, normals = ray_refraction.get_intersections_and_normals(mesh)  # [4096, 48, 3]
        directions_1 = ray_refraction.snell_fn(normals)

        positions_new, mask, inter_id = ray_refraction.update_sample_points(intersections, directions, 'in',
                                                                            torch.ones([positions.shape[0],
                                                                                        positions.shape[1]],
                                                                                       dtype=torch.bool,
                                                                                       device=positions.device))

        self.frustums.intersections = [intersections]
        self.frustums.normals = [normals]
        self.frustums.mask = mask

        # 3. Update ray_samples.frustums.directions
        directions_new = directions.clone()
        directions_new[mask] = directions_1[mask]

        # 4. Update ray_samples.frustums.origins
        origins_new = origins.clone()
        origins_1 = intersections - directions_1 * torch.norm(origins - intersections, dim=-1).unsqueeze(2)
        origins_new[mask] = origins_1[mask]

        self.frustums.directions = directions_new
        self.frustums.origins = origins_new

        return intersections, normals, mask

    def get_reflected_rays(self, intersections, normals, mask) -> None:
        origins = self.frustums.origins
        directions = self.frustums.directions
        positions = self.frustums.get_positions()

        # 1) Get reflective directions
        ray_reflection = RayReflection(origins, directions, positions)
        directions_1 = ray_reflection.get_reflected_directions(normals)
        ray_reflection.update_sample_points(intersections, directions_1, mask)

        # 2) Update ray_samples.frustums.directions
        directions_reflection_new = directions.clone()
        directions_reflection_new[mask] = directions_1[mask]

        # 3) Update ray_samples.frustums.origins
        origins_reflection_new = origins.clone()
        origins_reflection_1 = intersections - directions_1 * torch.norm(origins - intersections, dim=-1).unsqueeze(2)
        origins_reflection_new[mask] = origins_reflection_1[mask]

        self.frustums.intersections = [intersections]
        self.frustums.normals = [normals]
        self.frustums.mask = mask

    def get_straight_rays(self) -> None:
        # Modify the origins and directions of frustums here
        positions = self.frustums.get_positions()  # [4096, 48, 3] ([num_rays_per_batch, num_samples_per_ray, 3])

        # Modify positions based on known geometry and Snell's law
        # 1. Get origins, directions, r1, r2 directly
        origins = self.frustums.origins  # [4096, 48, 3]
        directions = self.frustums.directions  # [4096, 48, 3]
        r1, r2 = 1.0 / 1.33, 1.33 / 1.0
        # radius = 0.9
        radius = 0.9 * 0.1

        # 2. Get normals from the geometry, calculate new directions, and update positions after the first refraction
        ray_1 = WaterBallRefraction(origins, directions, positions, r1, radius)
        intersections_1, _ = ray_1.get_intersections_and_normals('in')  # [4096, 48, 3]

        # 3. Get normals from the geometry, calculate new directions, and update positions after the second refraction
        ray_2 = WaterBallRefraction(intersections_1, directions, positions, r2, radius)
        intersections_2, _ = ray_2.get_intersections_and_normals('out')

        self.frustums.intersections = [intersections_1, intersections_2]

    def get_weights(self, densities: Float[Tensor, "*batch num_samples 1"]) -> Float[Tensor, "*batch num_samples 1"]:
        """Return weights based on predicted densities

        Args:
            densities: Predicted densities for samples along ray

        Returns:
            Weights for each sample
        """

        delta_density = self.deltas * densities
        alphas = 1 - torch.exp(-delta_density)

        transmittance = torch.cumsum(delta_density[..., :-1, :], dim=-2)
        transmittance = torch.cat(
            [torch.zeros((*transmittance.shape[:1], 1, 1), device=densities.device), transmittance], dim=-2
        )
        transmittance = torch.exp(-transmittance)  # [..., "num_samples"]

        weights = alphas * transmittance  # [..., "num_samples"]
        weights = torch.nan_to_num(weights)

        return weights

    @overload
    @staticmethod
    def get_weights_and_transmittance_from_alphas(
            alphas: Float[Tensor, "*batch num_samples 1"], weights_only: Literal[True]
    ) -> Float[Tensor, "*batch num_samples 1"]:
        ...

    @overload
    @staticmethod
    def get_weights_and_transmittance_from_alphas(
            alphas: Float[Tensor, "*batch num_samples 1"], weights_only: Literal[False] = False
    ) -> Tuple[Float[Tensor, "*batch num_samples 1"], Float[Tensor, "*batch num_samples 1"]]:
        ...

    @staticmethod
    def get_weights_and_transmittance_from_alphas(
            alphas: Float[Tensor, "*batch num_samples 1"], weights_only: bool = False
    ) -> Union[
        Float[Tensor, "*batch num_samples 1"],
        Tuple[Float[Tensor, "*batch num_samples 1"], Float[Tensor, "*batch num_samples 1"]],
    ]:
        """Return weights based on predicted alphas
        Args:
            alphas: Predicted alphas (maybe from sdf) for samples along ray
            weights_only: If function should return only weights
        Returns:
            Tuple of weights and transmittance for each sample
        """

        transmittance = torch.cumprod(
            torch.cat([torch.ones((*alphas.shape[:1], 1, 1), device=alphas.device), 1.0 - alphas + 1e-7], 1), 1
        )

        weights = alphas * transmittance[:, :-1, :]
        if weights_only:
            return weights
        return weights, transmittance


@dataclass
class RayBundle(TensorDataclass):
    """A bundle of ray parameters."""

    # TODO(ethan): make sure the sizes with ... are correct
    origins: Float[Tensor, "*batch 3"]
    """Ray origins (XYZ)"""
    directions: Float[Tensor, "*batch 3"]
    """Unit ray direction vector"""
    pixel_area: Float[Tensor, "*batch 1"]
    """Projected area of pixel a distance 1 away from origin"""
    camera_indices: Optional[Int[Tensor, "*batch 1"]] = None
    """Camera indices"""
    nears: Optional[Float[Tensor, "*batch 1"]] = None
    """Distance along ray to start sampling"""
    fars: Optional[Float[Tensor, "*batch 1"]] = None
    """Rays Distance along ray to stop sampling"""
    metadata: Dict[str, Shaped[Tensor, "num_rays latent_dims"]] = field(default_factory=dict)
    """Additional metadata or data needed for interpolation, will mimic shape of rays"""
    times: Optional[Float[Tensor, "*batch 1"]] = None
    """Times at which rays are sampled"""

    def set_camera_indices(self, camera_index: int) -> None:
        """Sets all the camera indices to a specific camera index.

        Args:
            camera_index: Camera index.
        """
        self.camera_indices = torch.ones_like(self.origins[..., 0:1]).long() * camera_index

    def __len__(self) -> int:
        num_rays = torch.numel(self.origins) // self.origins.shape[-1]
        return num_rays

    def sample(self, num_rays: int) -> "RayBundle":
        """Returns a RayBundle as a subset of rays.

        Args:
            num_rays: Number of rays in output RayBundle

        Returns:
            RayBundle with subset of rays.
        """
        assert num_rays <= len(self)
        indices = random.sample(range(len(self)), k=num_rays)
        return self[indices]

    def get_row_major_sliced_ray_bundle(self, start_idx: int, end_idx: int) -> "RayBundle":
        """Flattens RayBundle and extracts chunk given start and end indices.

        Args:
            start_idx: Start index of RayBundle chunk.
            end_idx: End index of RayBundle chunk.

        Returns:
            Flattened RayBundle with end_idx-start_idx rays.

        """
        return self.flatten()[start_idx:end_idx]

    def get_ray_samples(
            self,
            bin_starts: Float[Tensor, "*bs num_samples 1"],
            bin_ends: Float[Tensor, "*bs num_samples 1"],
            spacing_starts: Optional[Float[Tensor, "*bs num_samples 1"]] = None,
            spacing_ends: Optional[Float[Tensor, "*bs num_samples 1"]] = None,
            spacing_to_euclidean_fn: Optional[Callable] = None,
    ) -> RaySamples:
        """Produces samples for each ray by projection points along the ray direction. Currently samples uniformly.

        Args:
            bin_starts: Distance from origin to start of bin.
            bin_ends: Distance from origin to end of bin.

        Returns:
            Samples projected along ray.
        """
        deltas = bin_ends - bin_starts
        if self.camera_indices is not None:
            camera_indices = self.camera_indices[..., None]
        else:
            camera_indices = None

        shaped_raybundle_fields = self[..., None]

        frustums = Frustums(
            origins=shaped_raybundle_fields.origins,  # [..., 1, 3]
            directions=shaped_raybundle_fields.directions,  # [..., 1, 3]
            starts=bin_starts,  # [..., num_samples, 1], euclidean
            ends=bin_ends,  # [..., num_samples, 1], euclidean
            pixel_area=shaped_raybundle_fields.pixel_area,  # [..., 1, 1]
        )

        ray_samples = RaySamples(
            frustums=frustums,  # the viewing volume of the camera
            camera_indices=camera_indices,  # [..., 1, 1]
            deltas=deltas,  # [..., num_samples, 1], step size between adjacent sample points along the rays
            spacing_starts=spacing_starts,  # [..., num_samples, 1], normalized starting positions
            spacing_ends=spacing_ends,  # [..., num_samples, 1], normalized ending positions
            spacing_to_euclidean_fn=spacing_to_euclidean_fn,  # mapping to euclidean space
            metadata=shaped_raybundle_fields.metadata,
            times=None if self.times is None else self.times[..., None],  # [..., 1, 1]
        )

        # ray_samples.get_refracted_rays()

        return ray_samples
