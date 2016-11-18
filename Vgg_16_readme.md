##VGG16 model for Keras

This is the [Keras](http://keras.io/) model of the 16-layer network used by the VGG team in the ILSVRC-2014 competition.

It has been obtained by directly converting the [Caffe model](https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md) provived by the authors.

Details about the network architecture can be found in the following arXiv paper:

    Very Deep Convolutional Networks for Large-Scale Image Recognition
    K. Simonyan, A. Zisserman
    arXiv:1409.1556


In the paper, the VGG-16 model is denoted as configuration `D`. It achieves 7.5% top-5 error on ILSVRC-2012-val, 7.4% top-5 error on ILSVRC-2012-test.

Please cite the paper if you use the models.

###Contents:

model and usage demo: see `vgg-16_keras.py`

weights: [vgg16_weights.h5](https://drive.google.com/file/d/0Bz7KyqmuGsilT0J5dmRCM0ROVHc/view?usp=sharing)
