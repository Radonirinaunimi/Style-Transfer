<p align="center">
  <img alt="Logo" src="https://github.com/Radonirinaunimi/Style-Transfer/blob/master/logo/logo.png" width=500>
</p>
<p align="center">
  <img alt="License" src="https://img.shields.io/github/license/Radonirinaunimi/Style-Transfer?style=flat-square">
  <img alt="repo size" src="https://img.shields.io/github/repo-size/Radonirinaunimi/Style-Transfer?style=flat-square">
</p>

#### Description
----------------

**Timst** is a python package based on [pyTorch]() that extracts the features of an image and tranfers them into another; such a technique is known as *image style transfer*. The following implementation is an implementation based on the following [scientific paper](https://arxiv.org/pdf/1508.06576.pdf). The architecture is based on [Convolutional Neural Network](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf) (CNN) which is one of the applications of [Deep Learning](https://en.wikipedia.org/wiki/Deep_learning).


#### How to install
-------------------

There are different ways the package can be installed: 
* By clonning this repository and running the following command in the terminal (you might require <kbd>sudo</kbd> privilege)
```bash
git clone https://github.com/Radonirinaunimi/Style-Transfer
cd Style-Transfer/
python setup.py install --user
```
* By installing it through the Python Package Index (PyPI)
```bash
pip install timst --upgrade
```

#### How to use
---------------

To use **timst**, just run the following:
```bash
timst -i [IMAGE_TO_BE_STYLED] -s [STYLE_TO_BE_APPLIED] [-n NUMBER_OF_ITERATIONS]
```

#### For bugs and feature request
---------------------------------

Open an [issue](https://github.com/Radonirinaunimi/Style-Transfer/issues/new/choose) or a [pull request](https://github.com/Radonirinaunimi/Style-Transfer/compare).
