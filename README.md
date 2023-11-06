## Readme file

### Table of Contents

1. [Code Description](#code-description)

2. [Data Preparation](#data-preparation)

3. [Instructions for Running the Code](#instructions-for-running-the-code)
    - [Setup the environment](#setup-the-environment)
   - [Train the model](#train-the-model)
   - [Evaluate the model](#evaluate-the-model)
   - [Render result as a video](#render-result-as-a-video)
4. [Acknowlegdements](#acknowledgements)

### Code description

We based our code on [nerfstudio](https://github.com/nerfstudio-project/nerfstudio). The code we developed are listed below:

1. Added new modules:

    [ray_reflection.py](nerfstudio/field_components/ray_reflection.py): Calculate the intersections and surface normals of a ray with a 3d mesh given the .ply file and ray direction. Use the intersections and normals to compute the direction of the reflected ray. Update the sample points to the new ray direction computed by the *Law of Reflection*. 

    [ray_refraction.py](nerfstudio/field_components/ray_refraction.py): Calculate the intersections and surface normals of a ray with a 3d mesh given the .ply file and ray direction, and Index of Refraction(IoR). Use the intersections and normals to compute the direction of the refracted ray. Update the sample points to the new ray direction computed by *Snell's Law*.

2. Added new methods:

    [rays.py](nerfstudio/cameras/rays.py): Class RaySamples: `get_refracted_rays()`, `get_reflected_rays()`: Call the methods in [ray_reflection.py](nerfstudio/field_components/ray_reflection.py) and [ray_refraction.py](nerfstudio/field_components/ray_refraction.py) to update the ray directions and sample points.
    
    [renderers.py](nerfstudio/model_components/renderers.py): Class RGBRenderer: `combine_rgb_ref()`: Composite samples along the reflected and refracted ray respectively, and render color image using *Fresnel Equation*.

    [losses.py](nerfstudio/model_components/losses.py): Class DepthLossType: `lossfun_distortion_refractive`: We add the method to apply the modified distortion to our model.

3. Made small adaptions:

    [ray_samplers.py](nerfstudio/model_components/ray_samplers.py):  Class ProposalNetworkSampler: `generate_ray_samples()`: We modify the method to generate two separate ray samplers, one used for reflection and the other used for refraction.

    [nerfacto.py](nerfstudio/models/nerfacto.py): Class NerfactoModel:  `get_outputs()`: We modify the method to use the updated `generate_ray_samples()` in [ray_samplers.py](nerfstudio/model_components/ray_samplers.py) in nerfacto model.

### Dataset Preparation

You can use the default dataset provided in our code, or you can install Blender, design a 3d model and run the following command to generate your own dataset:
```
Blender bowl.blend --python dataset_customization/view_train.py -b
Blender bowl.blend --python dataset_customization/view_val.py -b
Blender bowl.blend --python dataset_customization/view_test.py -b
```
Replace the `bowl.blend` to the name of your model.

### Instructions for Running the Code

#### Setup the environment

To run the code, you first need to set up the environment for [nerfstudio](https://github.com/nerfstudio-project/nerfstudio).
Detailed instructions can be found [here](https://github.com/nerfstudio-project/nerfstudio#1-installation-setup-the-environment). The installation will take about 40 minutes.

*Note:* The environment configuration on Windows can be a little tricky. One thing we need to be careful about is that since MSVC 2019 use X86 but nvcc use X64, we need to use the command: ``"C:\Program Files (x86)\Microsoft Visual Studio\2017\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64`` to make your msvc change to x64 before any installation steps. The reason can be found on [here](https://stackoverflow.com/questions/12843846/problems-when-running-nvcc-from-command-line%5B/url%5D).

#### Train the model

After set up the environment, running the following command will train our model on our synthetic dataset. Keep ``--vis wandb \`` if you use weights & biases plaform to visualise the training process, and delete it otherwise.
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

#### Evaluate the model

After the training is finished, you can use the following command to evaluate the model you get(Replace the `OUTPUT_ROOT_PATH` variable to the file path you get from training).
```
OUTPUT_ROOT_PATH=outputs/caustics/nerfacto/2023-11-02_194437

ns-eval --load-config $OUTPUT_ROOT_PATH/config.yml \
        --output-path $OUTPUT_ROOT_PATH/output_test.json
```
#### Render result as a video
Given a pretrained model checkpoint, you can start the viewer by running
```
ns-viewer --load-config {outputs/.../config.yml}
```
First we must create a path for the camera to follow. This can be done in the viewer under the "RENDER" tab. Orient your 3D view to the location where you wish the video to start, then press "ADD CAMERA". This will set the first camera key frame. Continue to new viewpoints adding additional cameras to create the camera path. 

Once finished, press "RENDER" which will display a modal that contains the command needed to render the video. Create a new terminal and run the command to generate the video.

### Acknowlegdements

Our project is developed on the [nerfstudio](https://github.com/nerfstudio-project/nerfstudio) framework.
