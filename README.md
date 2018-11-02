# 3d-captioning

Automatically generate description for 3D shapes in ShapeNetCore, which is a subset of [ShapeNet](https://www.shapenet.org/) and contains tables and chairs. We perform 3D captioning task on 64 shape volumes.

Captioning models we applied:
<br/>__FC__ from ["show and tell"](https://arxiv.org/pdf/1411.4555.pdf)
<br/>__Att2all__ from ["show and tell"](https://arxiv.org/pdf/1411.4555.pdf)
<br/>__Att2in__ from ["Self-critical Sequence Training for Image Captioning"](https://arxiv.org/pdf/1612.00563.pdf)
<br/>__Adaptive__ from ["Knowing When to Look: Adaptive Attention via
A Visual Sentinel for Image Captioning"](https://arxiv.org/pdf/1411.4555.pdf)

## Requirements
- ShapeNetCore dataset. You can get the dataset [here](http://text2shape.stanford.edu/dataset/shapenet/nrrd_256_filter_div_64_solid.zip)
- Instead of using the random split, we use the same split as in [Text2Shape](https://arxiv.org/abs/1803.08495). Get the split [here](https://drive.google.com/drive/folders/1nlDBAqdyIzhOXaU0ZpVND8zpktrIUhYc?usp=sharing)
- Python 3.x
- Numpy
- Pytorch 0.4 or newer
- TensorboardX

## Setup
- put the dataset in `data/`
- change the configuration in `lib/configs.py`. 

## Training

### Pre-training the encoders
```shell
python train_embedding.py
```

### Train the captioning models
```shell
python train_caption.py --path=<path-to-embedding-root-folder>
```

## Testing

### Save pre-trained embeddings
```shell
python scripts/save_embedding.py --path=<path-to-embedding-root-folder>
```

### Generate and save shape captions
```shell
python scripts/save_caption.py --caption=<path-to-caption-root-folder> --embedding=<path-to-embedding-root-folder>
```

## Results
![alt text](https://github.com/daveredrum/3d-captioning/blob/master/demo/summ_cap.png)
