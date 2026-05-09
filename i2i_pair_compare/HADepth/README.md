# HADepth

This is the official PyTorch implementation of the method  as described in

"HADepth: Highlight-aware monocular depth estimation for endoscopy" submitted to **Signal, Image and Video Processing**.

![Image](assets/network.png)

## ⚙️ Setup

Our experimental configuration and hyperparameters are as follows:

| Parameter             | Configuration           |
| --------------------- | ----------------------- |
| CPU                   | Intel Core i9-10980XE   |
| GPU                   | NVIDIA GeForce RTX 3090 |
| Operating system      | Ubuntu 20.04.5 LTS      |
| CUDA                  | 11.7                    |
| Python version        | 3.9.17                  |
| PyTorch version       | 1.13.0                  |
| Batch size            | 8                       |
| Epochs                | 20                      |
| Initial learning rate | 0.0001                  |
| Optimizer             | Adam                    |

Please download the `depth_anything_vitb14.pth` and place it into the `pretrained_model`.

## 💾 Datasets

The datasets are publicly available at the following URLs:  

**SCARED**: https://endovissub2019-scared.grand-challenge.org/

**ColonDepth**: http://cmic.cs.ucl.ac.uk/ColonoscopyDepth/

For more details on the dataset acquisition of SCARED, please refer to the instructions provided in  [AF-SfMLearner](https://github.com/ShuweiShao/AF-SfMLearner).

## 📦 Pretrained Models

We have released our pretrained models in this link: [google drive ](https://drive.google.com/drive/folders/1vCzhDq-d8ZvCebPcQR6I0IjMf10u6mxM?usp=sharing).

## ⏳ Endovis evaluation
### SCARED

```shell
CUDA_VISIBLE_DEVICES=0 python evaluate_depth.py --data_path <your_data_path> --eval_split endovis --load_weights_folder HADepth_fullmodel --eval_mono
```

### ColonDepth

#### T1-L1
```shell
CUDA_VISIBLE_DEVICES=0 python evaluate_depth.py --data_path <your_data_path> --eval_split t1 --load_weights_folder HADepth_fullmodel --eval_mono
```
#### T2-L2
```shell
CUDA_VISIBLE_DEVICES=0 python evaluate_depth.py --data_path <your_data_path> --eval_split t2 --load_weights_folder HADepth_fullmodel --eval_mono
```
#### T3-L3
```shell
CUDA_VISIBLE_DEVICES=0 python evaluate_depth.py --data_path <your_data_path> --eval_split t3 --load_weights_folder HADepth_fullmodel --eval_mono
```

## 📊 Evaluation

We are currently organizing the code section, and the testing portion of the code will be made publicly available soon.

|        Method        | Abs Rel | 95% CIs | Sq Rel | 95% CIs | RMSE | 95% CIs | RMSE log | 95% CIs | &delta; | 95% CIs |
|:--------------------:|:---------:|:--------:| :----: | :--------: |:---------:|:--------------------:|:--------------------:|:--------------------:|:--------------------:|:--------------------:|
|      Monodepth2      | 0.071     | [0.068, 0.073] | 0.590    | [0.554, 0.627] | 5.606  | [5.404, 5.809] | 0.094      | [0.091, 0.097] | 0.953 | [0.948, 0.957] |
| Endo-SfMLearner | 0.062     | [0.060, 0.065] | 0.606    | [0.551, 0.661] | 5.726  | [5.396, 6.056] | 0.093      | [0.089, 0.097] | 0.957 | [0.952, 0.961] |
|  AF-SfMLearner | 0.059     | [0.057, 0.061] | 0.435    | [0.406, 0.464] | 4.925  | [4.729, 5.122] | 0.082    | [0.079, 0.084] | 0.974 | [0.971, 0.977] |
| EndoDepth-Perf | 0.094     | -    | 0.635    | -   | 5.229  | - | 0.113  | -     | 0.953 | - |
| IID-SfMLearner | 0.057     | [0.055, 0.059] | 0.431  | [0.400, 0.462] | 4.796  | [4.579, 5.013] | 0.079      | [0.076, 0.082] | 0.970 | [0.966, 0.973] |
| Depth Anything | 0.058     | -    | 0.451    | -   | 5.058  | - | 0.081      | -     | 0.974 | - |
|     EndoDAC    | 0.051     | [0.050, 0.053] | 0.355    | [0.327, 0.383] | 4.436  | [4.226, 4.646] | 0.073      | [0.070, 0.075] | 0.979 | [0.977, 0.982] |
|     Ours       | 0.049 | [0.047, 0.051] | 0.326 | [0.299, 0.352] | 4.286 | [4.081, 4.491] | 0.069 | [0.066, 0.071] | 0.984 | [0.981, 0.986] |

## Acknowledgement

We thank the authors of [AF-SfMLearner](https://github.com/ShuweiShao/AF-SfMLearner), [IID-SfMLearner](https://github.com/bobo909/IID-SfmLearner), [Depth Anything](https://github.com/LiheYoung/Depth-Anything) and [EndoDAC](https://github.com/BeileiCui/EndoDAC) for opening source their code.