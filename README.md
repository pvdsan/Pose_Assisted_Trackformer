# Pose Assisted TrackFormer
This is an adaptation of the original Trackformer code done under an academic project at Georgia State University, Atlanta.

## Results of the Work

### MOT17-02
![MOT17-02 Comparison](MOT17_02_Comparsion.gif)

### MOT17-04
![MOT17-04 Comparison](MOT17_04_Comparsion.gif)

The left side is the baseline model and the right side is the pose-assisted model. MOT17-02 is where the baseline model performs better wheras the pose-assisted model performs better in MOT17-04

## Abstract

Multi-object tracking (MOT) is a critical component in applications ranging from autonomous driving and surveillance to sports analytics. Despite advancements in transformer-based MOT models like TrackFormer, challenges remain, especially in crowded, dynamic environments where maintaining identity consistency, managing occlusions, and differentiating visually similar individuals are essential. Pose estimation provides unique spatial and temporal cues, aiding in identity consistency by distinguishing individuals based on body posture. Incorporating pose estimation into MOT allows for the integration of keypoint data with bounding box-based tracking, leading to a hybrid model that combines spatial accuracy with temporal consistency.

<div align="center">
    <img src="docs/method.png" alt="TrackFormer casts multi-object tracking as a set prediction problem performing joint detection and tracking-by-attention. The architecture consists of a CNN for image feature extraction, a Transformer encoder for image feature encoding and a Transformer decoder which applies self- and encoder-decoder attention to produce output embeddings with bounding box and class information."/>
</div>

<div align="center">
    <img src="Training_3.png" alt="The Pose Assisted Model "/>
</div>





## Installation

We refer to our [docs/INSTALL.md](docs/INSTALL.md) for detailed installation instructions.

## Train TrackFormer

We refer to our [docs/TRAIN.md](docs/TRAIN.md) for detailed training instructions.

## Evaluate TrackFormer

In order to evaluate TrackFormer on a multi-object tracking dataset, we provide the `src/track.py` script which supports several datasets and splits interchangle via the `dataset_name` argument (See `src/datasets/tracking/factory.py` for an overview of all datasets.) The default tracking configuration is specified in `cfgs/track.yaml`. To facilitate the reproducibility of our results, we provide evaluation metrics for both the train and test set.

### MOT17

#### Private detections

```
python src/track.py with reid
```

<center>

| MOT17     | MOTA         | IDF1           |       MT     |     ML     |     FP       |     FN              |  ID SW.      |
|  :---:    | :---:        |     :---:      |    :---:     | :---:      |    :---:     |   :---:             |  :---:       |
| **Train** |     74.2     |     71.7       |     849      | 177        |      7431    |      78057          |  1449        |
| **Test**  |     74.1     |     68.0       |    1113      | 246        |     34602    |     108777          |  2829        |

</center>

#### Public detections (DPM, FRCNN, SDP)

```
python src/track.py with \
    reid \
    tracker_cfg.public_detections=min_iou_0_5 \
    obj_detect_checkpoint_file=models/mot17_deformable_multi_frame/checkpoint_epoch_50.pth
```

<center>

| MOT17     | MOTA         | IDF1           |       MT     |     ML     |     FP       |     FN              |  ID SW.      |
|  :---:    | :---:        |     :---:      |    :---:     | :---:      |    :---:     |   :---:             |  :---:       |
| **Train** |     64.6     |     63.7       |    621       | 675        |     4827     |     111958          |  2556        |
| **Test**  |     62.3     |     57.6       |    688       | 638        |     16591    |     192123          |  4018        |

</center>

### MOT20

#### Private detections

```
python src/track.py with \
    reid \
    dataset_name=MOT20-ALL \
    obj_detect_checkpoint_file=models/mot20_crowdhuman_deformable_multi_frame/checkpoint_epoch_50.pth
```

<center>

| MOT20     | MOTA         | IDF1           |       MT     |     ML     |     FP       |     FN              |  ID SW.      |
|  :---:    | :---:        |     :---:      |    :---:     | :---:      |    :---:     |   :---:             |  :---:       |
| **Train** |     81.0     |     73.3       |    1540      | 124        |     20807    |     192665          |  1961        |
| **Test**  |     68.6     |     65.7       |     666      | 181        |     20348    |     140373          |  1532        |

</center>

### MOTS20

```
python src/track.py with \
    dataset_name=MOTS20-ALL \
    obj_detect_checkpoint_file=models/mots20_train_masks/checkpoint.pth
```

Our tracking script only applies MOT17 metrics evaluation but outputs MOTS20 mask prediction files. To evaluate these download the official [MOTChallengeEvalKit](https://github.com/dendorferpatrick/MOTChallengeEvalKit).

<center>

| MOTS20    | sMOTSA         | IDF1           |       FP     |     FN     |     IDs      |
|  :---:    | :---:          |     :---:      |    :---:     | :---:      |    :---:     |
| **Train** |     --         |     --         |    --        |   --       |     --       |
| **Test**  |     54.9       |     63.6       |    2233      | 7195       |     278      |

</center>

### Demo

To facilitate the application of TrackFormer, we provide a demo interface which allows for a quick processing of a given video sequence.

```
ffmpeg -i data/snakeboard/snakeboard.mp4 -vf fps=30 data/snakeboard/%06d.png

python src/track.py with \
    dataset_name=DEMO \
    data_root_dir=data/snakeboard \
    output_dir=data/snakeboard \
    write_images=pretty
```

<div align="center">
    <img src="docs/snakeboard.gif" alt="Snakeboard demo" width="600"/>
</div>
