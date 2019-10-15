# Vessel Graph Network (VGN)
This is the code for ["Deep Vessel Segmentation by Learning Graphical Connectivity"](https://www.sciencedirect.com/science/article/pii/S1361841519300982).

## Dependency
* Python 2.7.12
* Tensorflow 1.12
* networkx 2.0
* scipy 1.1.0
* mahotas 1.4.3
* matplotlib 2.2.4
* easydict 1.7
* scikit-image 0.14.2
* scikit-fmm 0.0.9
* scikit-learn 0.19.0

## Datasets
* The VGN is evaluated on four retinal image datasets, namely the [DRIVE](https://www.isi.uu.nl/Research/Databases/DRIVE/), [STARE](http://cecas.clemson.edu/~ahoover/stare/), [CHASE_DB1](https://blogs.kingston.ac.uk/retinal/chasedb1/), and [HRF](https://www5.cs.fau.de/research/data/fundus-images/) datasets, which all are publicly available.
* The coronary artery X-ray angiography (CA-XRA) dataset we additionally used for evaluation can not be shared regrettably.

## Precomputed Results
We provide precomputed results of the VGN on the four retinal image datasets. [[OneDrive]](https://1drv.ms/u/s!AmnLATyiwjphhZ0BquyksorE0YV7nA?e=OmHhGW)

## Testing a Model
1. Download available trained models. [[OneDrive]](https://1drv.ms/u/s!AmnLATyiwjphhZ0CYhSYOqHmnQw4UQ?e=eRgvcq)
2. Run a test script among `test_CNN.py`, `test_CNN_HRF.py`, `test_VGN.py`, or `test_VGN_HRF.py`, with appropriate input arguments including the path for the downloaded model.

## Training a Model
We use a sequential training scheme composed of an initial pretraining of the CNN followed by joint training, including fine-tuning of the CNN module, of the whole VGN. Before the joint training, training graphs must be constructed from vessel probability maps inferred from the pretrained CNN.

### CNN Pretraining
(This step can be skipped by using a pretrained model we share.)
1. Download an ImageNet pretrained model. [[OneDrive]](https://1drv.ms/u/s!AmnLATyiwjphhZ0AqBHI2Y0nALUdoQ?e=NG4kVS)
2. Run `train_CNN.py` with appropriate input arguments including the path for the downloaded pretrained model.

### Training Graph Construction
1. Run `GenGraph/make_graph_db.py`.

### VGN Training
1. Place the generated graphs ('.graph_res') and vessel probability images ('_prob.png') inferred from the pretrained CNN in a new directory 'args.save_root/graph'
2. Run `train_VGN.py` with appropriate input arguments including the path for the pretrained CNN model.

## Demo Videos
Two example results, each of which is from the STARE and CHASE_DB1 datasets. The images in each row from left to right are, the original input image, GT, result. The last column is slowly changed from the baseline CNN result to the VGN result to better show the difference.
![](results/im0239.gif)
![](results/Image_12R.gif)

## Citation
```
@article{shin_media19,
  title = "Deep vessel segmentation by learning graphical connectivity",
  journal = "Medical Image Analysis",
  volume = "58",
  pages = "101556",
  year = "2019",
  issn = "1361-8415",
  doi = "https://doi.org/10.1016/j.media.2019.101556",
  url = "http://www.sciencedirect.com/science/article/pii/S1361841519300982",
  author = "Seung Yeon Shin and Soochahn Lee and Il Dong Yun and Kyoung Mu Lee",
}
```
