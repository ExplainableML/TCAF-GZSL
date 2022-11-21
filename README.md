# Temporal and cross-modal attention for audio-visual zero-shot learning

This repository is the official implementation of [Temporal and cross-modal attention for
audio-visual zero-shot learning](https://arxiv.org/abs/2207.09966).

<img src="/img/TCAF.png" width="700" height="400">

## Requirements
Install all required dependencies into a new virtual environment via conda.
```shell
conda env create -f TCAF.yml
```

# Datasets

We base our datasets on the [AVCA repository](https://github.com/ExplainableML/AVCA-GZSL/). The dataset structure is identical to AVCA and the dataset folder is called ```avgzsl_benchmark_non_averaged_datasets```. The only difference is that we use temporal features instead of averaged features. We provide our temporal C3D/VGGish features to download below.

In order to extract the C3D/VGGish features on your own, run the scripts in the ```/cls_feature_extraction``` as follows:
```shell
python3 cls_feature_extraction/get_features_activitynet.py
python3 cls_feature_extraction/get_features_ucf.py
python3 cls_feature_extraction/get_features_vggsound.py
```
Given the files extracted by the above scripts, run the following command to obtain the cls features:

```shell
python3 splitting_scripts_cls/create_features.py
```

Moreover, we adapted the SeLaVi implementation from the [AVCA repository](https://github.com/ExplainableML/AVCA-GZSL/) in order to extract temporal features and to make extraction more parallelizable. For obtaining the SeLaVi features we used the following commands:
```shell
python3 selavi_feature_extraction/get_clusters.py \
--root_dir <path_to_raw_videos> \
--weights_path <path_to_pretrained_selavi_vgg_sound.pth> \
--mode train \
--pretrained False \
--aud_sample_rate 44100 \
--use_mlp False \
--dataset {activity,ucf,vggsound} \
--headcount 2 \
--exp_desc <experiment_description> \
--output_dir <path_to_save_extracted_features> \
--batch_size 1 \
--workers 0

python3 selavi_feature_extraction/merge_features_selavi.py
python3 splitting_scripts_main/create_features_selavi.py

```



## Download features

You can download our temporal supervised (C3D/VGGish) features of all three datasets here:
* [VGGSound-GZSL (C3D/VGGish)](https://s3.mlcloud.uni-tuebingen.de/tcaf-gzsl/vggsound-supervised-temporal.zip) 
* [UCF-GZSL (C3D/VGGish)](https://s3.mlcloud.uni-tuebingen.de/tcaf-gzsl/ucf-supervised-temporal.zip)
* [ActivityNet-GZSL (C3D/VGGish)](https://s3.mlcloud.uni-tuebingen.de/tcaf-gzsl/activitynet-supervised-temporal.zip)

We additionally provide temporal self-supervised (SeLaVi) features, which have been pretrained in self-supervised manner on VGGSound:
* [VGGSound-GZSL (SeLaVi)](https://s3.mlcloud.uni-tuebingen.de/tcaf-gzsl/vggsound-selavi-temporal.zip) 
* [UCF-GZSL (SeLaVi)](https://s3.mlcloud.uni-tuebingen.de/tcaf-gzsl/ucf-selavi-temporal.zip)
* [ActivityNet-GZSL (SeLaVi)](https://s3.mlcloud.uni-tuebingen.de/tcaf-gzsl/activitynet-selavi-temporal.zip)

> Since the VGGSound dataset is also used for the zero-shot learning task, **we recommend the usage of supervised (C3D/VGGish) features** instead of SeLaVi.

The features should be placed inside the ```avgzsl_benchmark_non_averaged_datasets``` folder:
```shell
unzip [DATASET].zip -d avgzsl_benchmark_non_averaged_datasets/
```


# Training
In order to train the model run the following command:
```python3 main.py --cfg CFG_FILE  --root_dir ROOT_DIR --log_dir LOG_DIR --dataset_name DATASET_NAME --run all```

```
arguments:
--cfg CFG_FILE is the file containing all the hyperparameters for the experiments. These can be found in ```config/best/X/best_Y.yaml``` where X indicate whether you want to use cls features or main features. Y indicate the dataset that you want to use.
--root_dir ROOT_DIR indicates the location where the dataset is stored.
--dataset_name {VGGSound, UCF, ActivityNet} indicate the name of the dataset.
--log_dir LOG_DIR indicates where to save the experiments.
--run {'all', 'stage-1', 'stage-2'}. 'all' indicates to run both training stages + evaluation, whereas 'stage-1', 'stage-2' indicates to run only those particular training stages
```


# Evaluation

Evaluation can be done in two ways. Either you train with ```--run all``` which means that after training the evaluation will be done automatically, or you can do it manually.

For manual evaluation run the following command:

```python3 get_evaluation.py --cfg CFG_FILE --load_path_stage_A PATH_STAGE_A --load_path_stage_B PATH_STAGE_B --dataset_name DATASET_NAME --root_dir ROOT_DIR```

```
arguments:
--cfg CFG_FILE is the file containing all the hyperparameters for the experiments. These can be found in ```config/best/X/best_Y.yaml``` where X indicate whether you want to use cls features or main features. Y indicate the dataset that you want to use.
--load_path_stage_A will indicate to the path that contains the network for stage 1
--load_path_stage_B will indicate to the path that contains the network for stage 2
--dataset_name {VGGSound, UCF, ActivityNet} will indicate the name of the dataset
--root_dir points to the location where the dataset is stored
```


# Model weights
The trained models can be downloaded from [here](https://drive.google.com/file/d/1blz6p7qv94V238Qt0w0dBsqXZLGsT84D/view?usp=sharing).

# Results

### GZSL performance on VGGSound-GZSL, UCF-GZSL, ActivityNet-GZSL

| Method             | VGGSound-GZSL          | UCF-GZSL        | ActivityNet-GZSL |
|--------------------|------------------------|-----------------|------------------|
| Attention fusion   |   4.95                 |    24.97        |   5.18           |
| Perceiver          |   4.93                 |     34.11       |   6.92           |
| CJME               |  3.68                  |  28.65          |  7.32            |
| AVGZSLNET          |  5.26                  |  36.51          |  8.30            |
| AVCA               |  8.31                  |  41.34          |  9.92            |
| **TCAF**           |  **8.77**              |  **50.78**      |  **12.20**       |


### ZSL performance on VGGSound-GZSL, UCF-GZSL, ActivityNet-GZSL

| Method             | VGGSound-GZSL          | UCF-GZSL        | ActivityNet-GZSL |
|--------------------|------------------------|-----------------|------------------|
| Attention fusion   |  3.37                  |    20.21        |        4.88      |
| Perceiver          |  3.44                  |     28.12       |        4.47      |
| CJME               |  3.72                  |  29.01          | 6.29             |
| AVGZSLNET          |  4.81                  |  31.51          | 6.39             |
| AVCA               |  6.91                  |  37.72          | 7.58             |
|**TCAF**            |  **7.41**              |  **44.64**      | **7.96**         |

# Project structure
```src``` - Contains the code used throughout the project for dataloaders/models/training/testing.  
```c3d``` - Folder contains the code for the C3D network.  
```audioset_vggish_tensorflow_to_pytorch``` - Contains the code which is used to obtain the audio features using VGGish.  
```cls_feature_extraction``` - Contains the code used to extract the C3D/VGGish features from all 3 datasets.  
```selavi_feature_extraction``` - Contains the code used to extract the SeLaVi features.  
```splitting_scripts_{cls,main}``` - Contains files from spltting our dataset into the required structure.

# References

If you find this code useful, please consider citing:
```
@inproceedings{mercea2022tcaf,
  author    = {Mercea, Otniel-Bogdan and Hummel, Thomas and Koepke, A. Sophia and Akata, Zeynep},
  title     = {Temporal and cross-modal attention for audio-visual zero-shot learning},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2022}
}
```
```
@inproceedings{mercea2022avca,
  author    = {Mercea, Otniel-Bogdan and Riesch, Lukas and Koepke, A. Sophia and Akata, Zeynep},
  title     = {Audio-visual Generalised Zero-shot Learning with Cross-modal Attention and Language},
  booktitle = {Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2022}
}
```
