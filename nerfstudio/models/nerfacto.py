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
NeRF implementation that combines many recent advancements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple, Type

import numpy as np
import torch
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.field_components.ray_refraction import visualization, WaterBallRefraction
from nerfstudio.fields.density_fields import HashMLPDensityField
from nerfstudio.fields.nerfacto_field import NerfactoField
from nerfstudio.model_components.losses import (
    MSELoss,
    distortion_loss,
    interlevel_loss,
    orientation_loss,
    pred_normal_loss,
    scale_gradients_by_distance_squared,
)
from nerfstudio.model_components.ray_samplers import ProposalNetworkSampler, UniformSampler
from nerfstudio.model_components.renderers import AccumulationRenderer, DepthRenderer, NormalsRenderer, RGBRenderer
from nerfstudio.model_components.scene_colliders import NearFarCollider
from nerfstudio.model_components.shaders import NormalsShader
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps

import matplotlib.pyplot as plt


@dataclass
class NerfactoModelConfig(ModelConfig):
    """Nerfacto Model Config"""

    _target: Type = field(default_factory=lambda: NerfactoModel)
    near_plane: float = 0.05
    """How far along the ray to start sampling."""
    far_plane: float = 1000.0
    """How far along the ray to stop sampling."""
    background_color: Literal["random", "last_sample", "black", "white", "grey"] = "last_sample"
    """Whether to randomize the background color."""
    hidden_dim: int = 64
    """Dimension of hidden layers"""
    hidden_dim_color: int = 64
    """Dimension of hidden layers for color network"""
    hidden_dim_transient: int = 64
    """Dimension of hidden layers for transient network"""
    num_levels: int = 16
    """Number of levels of the hashmap for the base mlp."""
    base_res: int = 16
    """Resolution of the base grid for the hasgrid."""
    max_res: int = 2048
    """Maximum resolution of the hashmap for the base mlp."""
    log2_hashmap_size: int = 19
    """Size of the hashmap for the base mlp"""
    features_per_level: int = 2
    """How many hashgrid features per level"""
    num_proposal_samples_per_ray: Tuple[int, ...] = (256, 96)
    """Number of samples per ray for each proposal network."""
    num_nerf_samples_per_ray: int = 48
    """Number of samples per ray for the nerf network."""
    proposal_update_every: int = 5
    """Sample every n steps after the warmup"""
    proposal_warmup: int = 5000
    """Scales n from 1 to proposal_update_every over this many steps"""
    num_proposal_iterations: int = 2
    """Number of proposal network iterations."""
    use_same_proposal_network: bool = False
    """Use the same proposal network. Otherwise use different ones."""
    proposal_net_args_list: List[Dict] = field(
        default_factory=lambda: [
            {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 128, "use_linear": False},
            {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 256, "use_linear": False},
        ]
    )
    """Arguments for the proposal density fields."""
    proposal_initial_sampler: Literal["piecewise", "uniform"] = "piecewise"
    """Initial sampler for the proposal network. Piecewise is preferred for unbounded scenes."""
    interlevel_loss_mult: float = 1.0
    """Proposal loss multiplier."""
    distortion_loss_mult: float = 0.002
    """Distortion loss multiplier."""
    orientation_loss_mult: float = 0.0001
    """Orientation loss multiplier on computed normals."""
    pred_normal_loss_mult: float = 0.001
    """Predicted normal loss multiplier."""
    use_proposal_weight_anneal: bool = True
    """Whether to use proposal weight annealing."""
    use_average_appearance_embedding: bool = True
    """Whether to use average appearance embedding or zeros for inference."""
    proposal_weights_anneal_slope: float = 10.0
    """Slope of the annealing function for the proposal weights."""
    proposal_weights_anneal_max_num_iters: int = 1000
    """Max num iterations for the annealing function."""
    use_single_jitter: bool = True
    """Whether use single jitter or not for the proposal networks."""
    predict_normals: bool = False
    """Whether to predict normals or not."""
    disable_scene_contraction: bool = False
    """Whether to disable scene contraction or not."""
    use_gradient_scaling: bool = False
    """Use gradient scaler where the gradients are lower for points closer to the camera."""
    implementation: Literal["tcnn", "torch"] = "tcnn"
    """Which implementation to use for the model."""
    appearance_embed_dim: int = 32
    """Dimension of the appearance embedding."""


def ray_cube_intersection(origins, directions, cube_size):
    """
    Find the intersection of rays with a cube.

    Args:
    ray_origins (Tensor): The origins of the rays.
    ray_directions (Tensor): The directions of the rays.
    cube_size (float): The size of the cube.

    Returns:
    Tensor: Intersection in world coordinate origin.
    """
    # Define the half size of the cube
    half_size = cube_size / 2.0

    # Calculate the intersection of the ray with each plane of the cube
    t_near = (half_size - origins) / directions
    t_far = (-half_size - origins) / directions

    # Find the intersection points
    t = torch.min(torch.max(t_near, t_far), dim=2)[0]
    intersection = origins + t.unsqueeze(2) * directions  # in world coordinate origin

    return intersection


def plot_weights_and_density(weights, density, z, idx_start, idx_end,
                             intersections=None, inter_cube=None, scale_factor=1.0):
    """
    Plot the weights and density vs. z for a specific ray.

    Args:
        weights (Tensor or array): The computed weights for samples along the ray.
        density (Tensor or array): The computed density for samples along the ray.
        z (Tensor or array): distances along the ray.
        idx_start (int): Start index of the rays you want to visualize.
        idx_end (int): End index of the rays you want to visualize.
        intersections (list): -
        inter_cube (Tensor): -
        scale_factor (float): scale factor
    """

    # Convert tensors to numpy arrays if necessary
    if hasattr(weights, "cpu"):
        weights = weights.cpu().numpy()  # [32768, 128, 1]
    if hasattr(density, "cpu"):
        density = density.cpu().numpy()  # [32768, 128, 1]
    if hasattr(z, "cpu"):
        z = z.cpu().numpy()  # [32768, 128, 1]

    # clip maximum density
    density = torch.clamp(torch.tensor(density), max=1e-10)

    for i in range(idx_start, idx_end):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex='col')

        # Plot intersections if they are not all 'nan'
        if intersections is not None:
            norm_dis1, norm_dis2 = intersections

            valid_indices = np.where((z[i] >= 0.05*scale_factor) & (z[i] <= 14*scale_factor))[0]
            # valid_indices = np.where((z[i] >= 2.0*scale_factor) & (z[i] <= 6.0*scale_factor))[0]
            z_scaled = z[i][valid_indices] / scale_factor
            if not torch.isnan(norm_dis1[i]):
                color_1 = '#1585E1'
                ax1.set_ylabel('Density', color=color_1)
                ax1.plot(z_scaled, density[i][valid_indices], '-o', color=color_1, markersize=3)
                ax1.tick_params(axis='y', labelcolor=color_1)

                color_2 = '#8C30E3'
                ax2.set_xlabel('Distance along the ray')
                ax2.set_ylabel('Weights', color=color_2)
                ax2.plot(z_scaled, weights[i][valid_indices], '-o', color=color_2, markersize=3)
                ax2.tick_params(axis='y', labelcolor=color_2)

                v1 = norm_dis1[i] / scale_factor
                v2 = norm_dis2[i] / scale_factor
                ax1.axvline(x=v1.cpu().numpy(), color='r', linestyle='--', label='Intersection 1')
                ax2.axvline(x=v1.cpu().numpy(), color='r', linestyle='--', label='Intersection 1')
                if not torch.isnan(norm_dis2[i]):
                    ax1.axvline(x=v2.cpu().numpy(), color='r', linestyle='-.', label='Intersection 2')
                    ax2.axvline(x=v2.cpu().numpy(), color='r', linestyle='-.', label='Intersection 2')
                    print(f"Intersection 1: {v1}, Intersection 2: {v2}")
                if inter_cube is not None:
                    v3 = inter_cube[i] / scale_factor
                    ax1.axvline(x=v3.cpu().numpy(), color='r', linestyle=':', label='Intersection 3')
                    ax2.axvline(x=v3.cpu().numpy(), color='r', linestyle=':', label='Intersection 3')

                ax1.legend(loc='upper right')
                ax2.legend(loc='upper right')
                ax1.grid(True)
                ax2.grid(True)
                ax1.set_title(f"Density and Weights along Ray {i}")

                fig.tight_layout()
                plt.savefig(f'data/figures/density_along_ray_{i}.png', bbox_inches='tight')
                # plt.show()

    import sys
    sys.exit(0)


class NerfactoModel(Model):
    """Nerfacto model

    Args:
        config: Nerfacto configuration to instantiate model
    """

    config: NerfactoModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))

        # Fields
        self.field = NerfactoField(
            self.scene_box.aabb,
            hidden_dim=self.config.hidden_dim,
            num_levels=self.config.num_levels,
            max_res=self.config.max_res,
            log2_hashmap_size=self.config.log2_hashmap_size,
            hidden_dim_color=self.config.hidden_dim_color,
            hidden_dim_transient=self.config.hidden_dim_transient,
            spatial_distortion=scene_contraction,
            num_images=self.num_train_data,
            use_pred_normals=self.config.predict_normals,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
            appearance_embedding_dim=self.config.appearance_embed_dim,
            implementation=self.config.implementation,
        )

        self.density_fns = []
        num_prop_nets = self.config.num_proposal_iterations
        # Build the proposal network(s)
        self.proposal_networks = torch.nn.ModuleList()
        if self.config.use_same_proposal_network:
            assert len(self.config.proposal_net_args_list) == 1, "Only one proposal network is allowed."
            prop_net_args = self.config.proposal_net_args_list[0]
            network = HashMLPDensityField(
                self.scene_box.aabb,
                spatial_distortion=scene_contraction,
                **prop_net_args,
                implementation=self.config.implementation,
            )
            self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for _ in range(num_prop_nets)])
        else:
            for i in range(num_prop_nets):
                prop_net_args = self.config.proposal_net_args_list[min(i, len(self.config.proposal_net_args_list) - 1)]
                network = HashMLPDensityField(
                    self.scene_box.aabb,
                    spatial_distortion=scene_contraction,
                    **prop_net_args,
                    implementation=self.config.implementation,
                )
                self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for network in self.proposal_networks])

        # Samplers
        def update_schedule(step):
            return np.clip(
                np.interp(step, [0, self.config.proposal_warmup], [0, self.config.proposal_update_every]),
                1,
                self.config.proposal_update_every,
            )

        # Change proposal network initial sampler if uniform
        initial_sampler = None  # None is for piecewise as default (see ProposalNetworkSampler)
        if self.config.proposal_initial_sampler == "uniform":
            initial_sampler = UniformSampler(single_jitter=self.config.use_single_jitter)

        self.proposal_sampler = ProposalNetworkSampler(
            num_nerf_samples_per_ray=self.config.num_nerf_samples_per_ray,
            num_proposal_samples_per_ray=self.config.num_proposal_samples_per_ray,
            num_proposal_network_iterations=self.config.num_proposal_iterations,
            single_jitter=self.config.use_single_jitter,
            update_sched=update_schedule,
            initial_sampler=initial_sampler,
        )

        # Collider
        self.collider = NearFarCollider(near_plane=self.config.near_plane, far_plane=self.config.far_plane)

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer()
        self.renderer_normals = NormalsRenderer()

        # shaders
        self.normals_shader = NormalsShader()

        # losses
        self.rgb_loss = MSELoss()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["proposal_networks"] = list(self.proposal_networks.parameters())
        param_groups["fields"] = list(self.field.parameters())
        return param_groups

    def get_training_callbacks(
            self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = []
        if self.config.use_proposal_weight_anneal:
            # anneal the weights of the proposal network before doing PDF sampling
            N = self.config.proposal_weights_anneal_max_num_iters

            def set_anneal(step):
                # https://arxiv.org/pdf/2111.12077.pdf eq. 18
                train_frac = np.clip(step / N, 0, 1)

                def bias(x, b):
                    return b * x / ((b - 1) * x + 1)

                anneal = bias(train_frac, self.config.proposal_weights_anneal_slope)
                self.proposal_sampler.set_anneal(anneal)

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_anneal,
                )
            )
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=self.proposal_sampler.step_cb,
                )
            )
        return callbacks

    def get_outputs(self, ray_bundle: RayBundle):
        ray_samples: RaySamples
        ray_samples_ref: RaySamples
        ray_samples, weights_list, ray_samples_list, ray_samples_ref, weights_list_ref, ray_samples_list_ref = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        field_outputs = self.field.forward(ray_samples, compute_normals=self.config.predict_normals)
        field_outputs_ref = self.field.forward(ray_samples_ref, compute_normals=self.config.predict_normals)
        if self.config.use_gradient_scaling:
            field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)
            field_outputs_ref = scale_gradients_by_distance_squared(field_outputs_ref, ray_samples_ref)

        # visualization(ray_samples, 489, 490)
        # density, _ = self.field.get_density_grid()
        # draw_heatmap(density)

        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])  # [32768, 128, 1]
        weights_ref = ray_samples_ref.get_weights(field_outputs_ref[FieldHeadNames.DENSITY])  # [32768, 128, 1]

        # # search for the first density > threshold, and recompute the weights for depth maps
        # threshold = 25
        # density = (field_outputs[FieldHeadNames.DENSITY])  # [32768, 256, 1]
        #
        # mask = density > threshold
        # cumsum_mask = mask.cumsum(dim=1)
        # first_greater_mask = (cumsum_mask == 1) & mask  # [32768, 256, 1]
        #
        # first_greater_index = torch.argmax(first_greater_mask.int(), dim=1).unsqueeze(-1)  # [32768, 1]
        #
        # # Extract the depth (distance) values for each ray at the first greater sample index
        # steps = (ray_samples.frustums.starts + ray_samples.frustums.ends) / 2
        # depth = torch.gather(steps, 1, first_greater_index).squeeze(-1)  # [32768, 1]

        # plot weights vs depth
        # origins = ray_samples.frustums.origins  # [32768, 128, 3]
        # intersections_1 = ray_samples.frustums.intersections[0]  # [32768, 128, 3]
        # intersections_2 = ray_samples.frustums.intersections[1]  # [32768, 128, 3]
        # norm_dis1 = torch.norm(intersections_1[:, 0, :] - origins[:, 0, :], dim=1)  # [32768]
        # norm_dis2 = torch.norm(intersections_2[:, 0, :] - origins[:, -1, :], dim=1)  # [32768]
        # intersections_3 = ray_cube_intersection(origins=ray_samples.frustums.origins,
        #                                         directions=ray_samples.frustums.directions,
        #                                         cube_size=8.4 * 0.1)
        # norm_dis3 = torch.norm(intersections_3[:, -1, :] - origins[:, -1, :], dim=1)  # [32768]
        # # norm_dis3 = None
        # if not torch.isnan(norm_dis1[200:600]).all():
        #     plot_weights_and_density(weights, field_outputs[FieldHeadNames.DENSITY],
        #                              (ray_samples.frustums.starts + ray_samples.frustums.ends) / 2,
        #                              200, 600, [norm_dis1, norm_dis2], norm_dis3, 0.1)

        weights_list.append(weights)
        weights_list_ref.append(weights_ref)
        ray_samples_list.append(ray_samples)
        ray_samples_list_ref.append(ray_samples_ref)

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], rgb_ref=field_outputs_ref[FieldHeadNames.RGB],
                                weights=weights, weights_ref=weights_ref, ray_samples=ray_samples_ref)
        # depth = self.renderer_depth(weights=weights_for_depth, ray_samples=ray_samples)
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)  # [32768, 1]
        accumulation = self.renderer_accumulation(weights=weights)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
        }

        if self.config.predict_normals:
            normals = self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)
            pred_normals = self.renderer_normals(field_outputs[FieldHeadNames.PRED_NORMALS], weights=weights)
            outputs["normals"] = self.normals_shader(normals)
            outputs["pred_normals"] = self.normals_shader(pred_normals)
        # These use a lot of GPU memory, so we avoid storing them for eval.
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        if self.training and self.config.predict_normals:
            outputs["rendered_orientation_loss"] = orientation_loss(
                weights.detach(), field_outputs[FieldHeadNames.NORMALS], ray_bundle.directions
            )

            outputs["rendered_pred_normal_loss"] = pred_normal_loss(
                weights.detach(),
                field_outputs[FieldHeadNames.NORMALS].detach(),
                field_outputs[FieldHeadNames.PRED_NORMALS],
            )

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])

        return outputs

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = {}
        image = batch["image"].to(self.device)
        metrics_dict["psnr"] = self.psnr(outputs["rgb"], image)
        if self.training:
            metrics_dict["distortion"] = distortion_loss(outputs["weights_list"], outputs["ray_samples_list"])
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = {}
        image = batch["image"].to(self.device)
        loss_dict["rgb_loss"] = self.rgb_loss(image, outputs["rgb"])
        if self.training:
            loss_dict["interlevel_loss"] = self.config.interlevel_loss_mult * interlevel_loss(
                outputs["weights_list"], outputs["ray_samples_list"]
            )
            assert metrics_dict is not None and "distortion" in metrics_dict
            loss_dict["distortion_loss"] = self.config.distortion_loss_mult * metrics_dict["distortion"]
            if self.config.predict_normals:
                # orientation loss for computed normals
                loss_dict["orientation_loss"] = self.config.orientation_loss_mult * torch.mean(
                    outputs["rendered_orientation_loss"]
                )

                # ground truth supervision for normals
                loss_dict["pred_normal_loss"] = self.config.pred_normal_loss_mult * torch.mean(
                    outputs["rendered_pred_normal_loss"]
                )
        return loss_dict

    def get_image_metrics_and_images(
            self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image = batch["image"].to(self.device)
        rgb = outputs["rgb"]
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
        )

        combined_rgb = torch.cat([image, rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        psnr = self.psnr(image, rgb)
        ssim = self.ssim(image, rgb)
        lpips = self.lpips(image, rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        images_dict = {"img": combined_rgb, "accumulation": combined_acc, "depth": combined_depth}

        for i in range(self.config.num_proposal_iterations):
            key = f"prop_depth_{i}"
            prop_depth_i = colormaps.apply_depth_colormap(
                outputs[key],
                accumulation=outputs["accumulation"],
            )
            images_dict[key] = prop_depth_i

        return metrics_dict, images_dict
