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

We base our datasets on [AVCA repository](https://github.com/ExplainableML/AVCA-GZSL/). The dataset structure is identical to AVC repository with the only mention that the data folder in this repo is called ```avgzsl_benchmark_non_averaged_datasets```. The only difference is that we use temporal features instead of averaged features.

In order to obtain the C3D/VGGish features, run the scripts in the ```/cls_feature_extraction``` as follows:
```
python cls_feature_extraction/get_features_activitynet.py
python cls_feature_extraction/get_features_ucf.py
python cls_feature_extraction/get_features_vggsound.py
```

Moreover, we adapted the selavi implementation from [AVCA repository](https://github.com/ExplainableML/AVCA-GZSL/). For obtaining the SeLaVi features we used the following command
```
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
```
This selavi script will save each video in a pickle file in order to make this highly parallelizable. In order to put all videos together, run the merge_features_selavi.py:
```
python3 selavi_feature_extraction/merge_features_selavi.py
```

Finally, given the files extracted by the above scripts, run the following commands 

```python3 splitting_scripts_main/create_features_selavi.py``` to obtain the selavi features

```python3 splitting_scripts_cls/create_features.py``` to obtain the cls features


## Download features

The features for UCF can be downloaded from [here](https://drive.google.com/file/d/1h7ysUITXVKka8qppZtU8_AzlC63jK5V_/view?usp=sharing). These features should be placed inside the ```avgzsl_benchmark_non_averaged_datasets``` folder, similar to [AVCA repository](https://github.com/ExplainableML/AVCA-GZSL/).


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

```python3 get_evaluation.py --load_path_stage_A PATH_STAGE_A --load_path_stage_B PATH_STAGE_B --dataset_name DATASET_NAME --root_dir ROOT_DIR```

```arguments:
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

```audioset_vggish_tensorflow_to_pytorch``` - Contains the code which is used to obtain the audio features using VGGish.

```c3d``` - Folder contains the code for the C3D network.

```selavi_feature_extraction``` - Contains the code used to extract the SeLaVi features.

```src``` - Contains the code used throughout the project for dataloaders/models/training/testing.

```cls_feature_extraction``` - Contains the code used to extract the C3D/VGGish features from all 3 datasets.

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
