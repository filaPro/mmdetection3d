# TD3D: Top-Down Beats Bottom-Up in 3D Instance Segmentation

> [Top-Down Beats Bottom-Up in 3D Instance Segmentation](https://arxiv.org/abs/2302.02871)

## Abstract

Most 3D instance segmentation methods exploit a bottom-up strategy, typically including resource-exhaustive post-processing. For point grouping, bottom-up methods rely on prior assumptions about the objects in the form of hyperparameters, which are domain-specific and need to be carefully tuned. On the contrary, we address 3D instance segmentation with a TD3D: top-down, fully data-driven, simple approach trained in an end-to-end manner. With its straightforward fully-convolutional pipeline, it performs surprisingly well on the standard benchmarks: ScanNet v2, its extension ScanNet200, and S3DIS. Besides, our method is much faster on inference than the current state-of-the-art grouping-based approaches.

<div align="center">
<img src="https://user-images.githubusercontent.com/6030962/221568290-80a7881b-f041-4e97-b55d-0954b20cb416.png" width="90%"/>
</div>

## Usage

Training and inference in this project were tested with `mmdet3d==1.1.0rc3`.

### Training commands

In MMDet3D's root directory, run the following command to train the model:

```bash
python tools/train.py projects/TD3D/configs/td3d_1xb6_scannet-3d-18class.py
```

### Testing commands

In MMDet3D's root directory, run the following command to test the model:

```bash
python tools/test.py projects/TD3D/configs/td3d_1xb6_scannet-3d-18class.py ${CHECKPOINT_PATH}
```

## Results and models

### ScanNet

|                          Backbone                          | Mem (GB) | Inf time (fps) |   AP@0.25   |   AP@0.5    |                                                                                                                                        Download                                                                                                                                         |
| :--------------------------------------------------------: | :------: | :------------: | :---------: | :---------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [MinkResNet34](./configs/td3d_1xb6_scannet-3d-18class.py) |       |            |  |  | [model](https://download.openmmlab.com/mmdetection3d/v1.1.0_models/td3d/td3d_1xb6_scannet-3d-18class/td3d_1xb6_scannet-3d-18class.pth) \| [log](https://download.openmmlab.com/mmdetection3d/v1.1.0_models/td3d/td3d_1xb6_scannet-3d-18class/td3d_1xb6_scannet-3d-18class.log.json) |

### S3DIS

|                        Backbone                         | Mem (GB) | Inf time (fps) |   AP@0.25   |   AP@0.5    |                                                                                                                                  Download                                                                                                                                   |
| :-----------------------------------------------------: | :------: | :------------: | :---------: | :---------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [MinkResNet34](./configs/td3d_1xb6_s3dis-3d-5class.py) |      |            |  |  | [model](https://download.openmmlab.com/mmdetection3d/v1.1.0_models/td3d/td3d_1xb6_s3dis-3d-5class/td3d_1xb6_s3dis-3d-5class.pth) \| [log](https://download.openmmlab.com/mmdetection3d/v1.1.0_models/td3d/td3d_1xb6_s3dis-3d-5class/td3d_1xb6_s3dis-3d-5class.log.json) |

**Note**

- We recommend to set `det_score_thr` to 0.15 and `nms_pre` to 100 in configs during training to avoid memory problems during validation.
- Inference time is given for a single NVidia GeForce RTX 3090 GPU.

## Citation

```latex
@article{kolodiazhnyi2023td3d,
  title={Top-Down Beats Bottom-Up in 3D Instance Segmentation},
  author={Kolodiazhnyi, Maksim and Rukhovich, Danila and Vorontsova, Anna and Konushin, Anton},
  journal={arXiv preprint arXiv:2302.02871},
  year={2023}
}
```

## Checklist

- [x] Milestone 1: PR-ready, and acceptable to be one of the `projects/`.

  - [x] Finish the code

  - [x] Basic docstrings & proper citation

  - [x] Test-time correctness

  - [x] A full README

- [x] Milestone 2: Indicates a successful model implementation.

  - [x] Training-time correctness

- [ ] Milestone 3: Good to be a part of our core package!

  - [x] Type hints and docstrings

  - [ ] Unit tests

  - [ ] Code polishing

  - [ ] Metafile.yml

- [ ] Move your modules into the core package following the codebase's file hierarchy structure.

- [ ] Refactor your modules into the core package following the codebase's file hierarchy structure.