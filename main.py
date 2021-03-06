#!/usr/bin/python
# -*- coding: utf-8 -*-

from hypercolumn import *


model = VGG_16('vgg16_weights.h5')
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy')


im_original = cv2.resize(cv2.imread('smoke.jpg'), (224, 224))
im = im_original.transpose((2,0,1))
im = np.expand_dims(im, axis=0)
im_converted = cv2.cvtColor(im_original, cv2.COLOR_BGR2RGB)
plt.imshow(im_converted)
# plt.show()

out = model.predict(im)
plt.plot(out.ravel())



### Extracting feature from the 3rd layer
feat = get_feature(model,3,im)
plt.imshow(feat[0][0][2])


### Extracting feature from the 15th layer
feat = get_feature(model,15,im)
plt.imshow(feat[0][0][13])



## Extracting hypercolumn
layers_extract = [3, 8]
hc = extract_hypercolumn(model, layers_extract, im)
ave = np.average(hc.transpose(1, 2, 0), axis=2)
plt.imshow(ave)



## Simple hypercolumn pixel clustering

"""
m = hc.transpose(1,2,0).reshape(50176, -1)
kmeans = cluster.KMeans(n_clusters=2, max_iter=300, n_jobs=5, precompute_distances=True)
cluster_labels = kmeans .fit_predict(m)
imcluster = np.zeros((224,224))
imcluster = imcluster.reshape((224*224,))
imcluster = cluster_labels
plt.imshow(imcluster.reshape(224, 224), cmap="hot")

#plt.show()

"""
