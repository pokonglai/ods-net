# Panoramic depth maps from omni-directional stereo (ODS) images
This repo is the official implementation of the paper *"Real-time panoramic depth maps from omni-directional stereo images for 6 DoF videos in virtual reality"* which predicts a panoramic depth map from a pair of ODS images taken from 3D 360&deg; sensors.

This work was presented at [IEEE VR 2019](http://ieeevr.org/2019/).

# Getting started

Dataset and trained models can be found at this link: https://drive.google.com/open?id=1W3ur97m_GVFHOqPHor4S3rRUW4HY2q48

The main files of interest will be ```train.py``` and ```predict.py```. 

Downloading the full dataset is not required to test the training of the model as sample data is provided in the folder ```data```.

Pre-trained weights for using ```predict.py``` can be found in the link above. A sample video is also provided in the folder ```testing_videos```.


# Citation

If you use significant portions of our code or ideas from our paper in your research, please cite our work:
```
@inproceedings{lai2019pano,
  title={Real-time panoramic depth maps from omni-directional stereo images for 6 DoF videos in virtual reality},
  author={Lai, Po Kong and Xie, Shuang and Lang, Jochen and Laganière, Robert},
  booktitle={2019 IEEE Virtual Reality (VR)},
  year={2019},
  organization={IEEE}
}
```

# Questions or Comments
Please direct any questions or comments to me; I am happy to help in any way I can. You can email me directly at plai036@uottawa.ca.
