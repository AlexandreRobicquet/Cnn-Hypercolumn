from matplotlib import pyplot as plt
import theano
import cv2
import numpy as np
import scipy as sp
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.optimizers import SGD
from sklearn.manifold import TSNE
from sklearn import manifold
from sklearn import cluster
from sklearn.preprocessing import StandardScaler

from hypercolumn import *


model = VGG_16('vgg16_weights.h5')
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy')


im_original = cv2.resize(cv2.imread('madruga.jpg'), (224, 224))
im = im_original.transpose((2,0,1))
im = np.expand_dims(im, axis=0)
im_converted = cv2.cvtColor(im_original, cv2.COLOR_BGR2RGB)
plt.imshow(im_converted)


out = model.predict(im)
plt.plot(out.ravel())



### Extracting feature from the 3rd layer

get_feature = theano.function([model.layers[0].input], model.layers[3].get_output(train=False), allow_input_downcast=False)
feat = get_feature(im)
plt.imshow(feat[0][2])

### Extracting feature from the 15th layer

get_feature = theano.function([model.layers[0].input], model.layers[15].get_output(train=False), allow_input_downcast=False)
feat = get_feature(im)
plt.imshow(feat[0][13])



## Extracting hypercolumn

layers_extract = [3, 8]
hc = extract_hypercolumn(model, layers_extract, im)
