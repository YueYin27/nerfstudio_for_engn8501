# Dive into Caustics: 3D Reconstruction of Refractive Caustics

<img src="https://github.com/YueYin27/nerfstudio_for_engn8501/assets/65449627/48e6652a-de05-42f8-951f-31fa84226011" width="100%"/>

<table border="0" style="border: none;">

  <tr>
    <td>
      <img src="https://github.com/YueYin27/nerfstudio_for_engn8501/blob/main/media/nerfacto.gif" alt="Nerfacto" width="100%" />
      <p align="center">Nerfacto</p>
    </td>
    <td>
      <img src="https://github.com/YueYin27/nerfstudio_for_engn8501/blob/main/media/tensorf.gif" alt="TensoRF" width="100%" />
      <p align="center">TensoRF</p>
    </td>
  </tr>
  <tr>
    <td>
      <img src="https://github.com/YueYin27/nerfstudio_for_engn8501/blob/main/media/mipnerf.gif" alt="Mip-NeRF" width="100%" />
      <p align="center">Mip-NeRF</p>
    </td>
    <td>
      <img src="https://github.com/YueYin27/nerfstudio_for_engn8501/blob/main/media/ours.gif" alt="Ours" width="100%" />
      <p align="center">Ours</p>
    </td>
  </tr>
</table>


## Table of Contents

[Code Description](#1-code-description)

[Data Preparation](#2-data-preparation)

[Instructions for Running the Code](#3-instructions-for-running-the-code)
   - [Setup the environment](#1-setup-the-environment)
   - [Train the model](#2-train-the-model)
   - [Evaluate the model](#3-evaluate-the-model)
   - [Render the result as a video](#4-render-the-result-as-a-video)

[Acknowlegdements](#4-acknowledgements)

## 1. Code Description

We based our code on [nerfstudio](https://github.com/nerfstudio-project/nerfstudio). The code we developed is listed below:

#### 1) Added new modules:
[ray_refraction.py](nerfstudio/field_components/ray_refraction.py): Calculate the intersections and surface normals of a ray with a 3d mesh given the .ply file and ray direction, and Index of Refraction(IoR). Use the intersections and normals to compute the direction of the refracted ray. Update the sample points to the new ray direction computed by *Snell's Law*.

[ray_reflection.py](nerfstudio/field_components/ray_reflection.py): Use the intersections and normals calculated in [ray_refraction.py](nerfstudio/field_components/ray_refraction.py) to compute the direction of the reflected ray. Update the sample points to the new ray direction computed by the *Law of Reflection*. 

#### 2) Added new methods:

[rays.py](nerfstudio/cameras/rays.py): `Class RaySamples` -> [`get_refracted_rays()`](https://github.com/YueYin27/nerfstudio_for_engn8501/blob/main/nerfstudio/cameras/rays.py#L185-L222), [`get_reflected_rays()`](https://github.com/YueYin27/nerfstudio_for_engn8501/blob/main/nerfstudio/cameras/rays.py#L224-L245): Call the methods in [ray_reflection.py](nerfstudio/field_components/ray_reflection.py) and [ray_refraction.py](nerfstudio/field_components/ray_refraction.py) and update the ray directions and sample points.

[renderers.py](nerfstudio/model_components/renderers.py): `Class RGBRenderer` -> [`combine_rgb_ref()`](https://github.com/YueYin27/nerfstudio_for_engn8501/blob/main/nerfstudio/model_components/renderers.py#L119-L182): Composite samples along the reflected and refracted ray respectively, and render color image using *Fresnel Equation*.

[losses.py](nerfstudio/model_components/losses.py): `Class DepthLossType` -> [`lossfun_distortion_refractive()`](https://github.com/YueYin27/nerfstudio_for_engn8501/blob/main/nerfstudio/model_components/losses.py#L143-L176): We add the method to apply the modified distortion to our model.

#### 3) Made minor adaptions:

[ray_samplers.py](nerfstudio/model_components/ray_samplers.py): `Class ProposalNetworkSampler` -> [`generate_ray_samples()`](https://github.com/YueYin27/nerfstudio_for_engn8501/blob/main/nerfstudio/model_components/ray_samplers.py#L78-L129): We modify the method to generate two separate ray samplers, one used for reflection and the other used for refraction.

[nerfacto.py](nerfstudio/models/nerfacto.py): `Class NerfactoModel` -> [`get_outputs()`](https://github.com/YueYin27/nerfstudio_for_engn8501/blob/main/nerfstudio/models/nerfacto.py#L380-L468): We modify the method to use the updated [`generate_ray_samples()`](https://github.com/YueYin27/nerfstudio_for_engn8501/blob/main/nerfstudio/model_components/ray_samplers.py#L78-L129) in [ray_samplers.py](nerfstudio/model_components/ray_samplers.py) in nerfacto model.

## 2. Data Preparation

You can download our [dataset](https://github.com/YueYin27/nerfstudio_for_engn8501/tree/main/caustics_bowl_pattern) online from our GitHub repo.

Or, you can install Blender, design a 3D model and run the following command to generate your own dataset:
```
Blender modelname.blend --python dataset_customization/view_train.py -b
Blender modelname.blend --python dataset_customization/view_val.py -b
Blender modelname.blend --python dataset_customization/view_test.py -b
```
Replace the `modelname.blend` with the name of your model.

## 3. Instructions for Running the Code

#### 1) Setup the environment

To run the code, you must first set up the environment for [nerfstudio](https://github.com/nerfstudio-project/nerfstudio).
Detailed instructions can be found [here](https://github.com/nerfstudio-project/nerfstudio#1-installation-setup-the-environment). The installation will take about 40 minutes.

*Note:* The environment configuration on Windows can be a little tricky. One thing we need to be careful about is that since MSVC 2019 use X86 but nvcc use X64, we need to use the command:
```
"C:\Program Files (x86)\Microsoft Visual Studio\2017\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64
```
to make your msvc change to x64 before any installation steps. The reason can be found on [here](https://stackoverflow.com/questions/12843846/problems-when-running-nvcc-from-command-line%5B/url%5D).

#### 2) Train the model

After setting up the environment, the following command will train our model on our synthetic dataset. Keep ``--vis wandb \`` if you use the weights & biases platform to visualise the training process and delete it otherwise.
```
ns-train nerfacto --machine.device-type cuda \
                  --machine.num-devices 1 \
                  --experiment-name caustics \
                  --project-name nerfstudio-caustics \
                  --pipeline.model.background-color random \
                  --pipeline.model.proposal-initial-sampler uniform \
                  --pipeline.model.near-plane 0.05 \
                  --pipeline.model.far-plane 15 \
                  --pipeline.model.num_nerf_samples_per_ray 256 \
                  --pipeline.datamanager.camera-optimizer.mode off \
                  --pipeline.model.use-average-appearance-embedding False \
                  --pipeline.model.distortion-loss-mult 0.1 \
                  --pipeline.model.disable-scene-contraction True \
                  --vis wandb \
         blender-depth-data \
                  --scale-factor 0.1 \
                  --data data/customized/caustics_bowl_pattern/
```

You can use ```ns-train -help ``` to learn more about the command and adjust the hyperparameter settings.

#### 3) Evaluate the model

After the training is finished, you can use the following command to evaluate the model you get(Replace the `OUTPUT_ROOT_PATH` variable with the file path you get from training).
```
OUTPUT_ROOT_PATH=outputs/caustics/nerfacto/2023-11-02_194437

ns-eval --load-config $OUTPUT_ROOT_PATH/config.yml \
        --output-path $OUTPUT_ROOT_PATH/output_test.json
```
### 4) Render the result as a video
Given a pre-trained model checkpoint, you can start the viewer by running
```
ns-viewer --load-config {outputs/.../config.yml}
```
First, we must create a path for the camera to follow. This can be done in the viewer under the "RENDER" tab. Orient your 3D view to the location where you wish the video to start, then press "ADD CAMERA". This will set the first camera keyframe. Continue to new viewpoints, adding additional cameras to create the camera path. 

Once finished, press "RENDER", which will display a modal that contains the command needed to render the video. Create a new terminal and run the command to generate the video.

## 4. Acknowledgements

Our project is built on the [nerfstudio](https://github.com/nerfstudio-project/nerfstudio) framework.
