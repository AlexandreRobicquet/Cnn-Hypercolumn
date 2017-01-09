# Cnn-Hypercolumn


main code: Cnn-Hypercolumn.py
more information in 
### Content

model and usage demo: see `vgg-16_keras.py`

weights: [vgg16_weights.h5](https://drive.google.com/file/d/0Bz7KyqmuGsilT0J5dmRCM0ROVHc/view?usp=sharing)

### Instruction

to get started, you will need to proceed this way:

pip install -r requirements.txt  # Install dependencies



if you want to work in a virtual environement temporarly:

```python
sudo pip install virtualenv      # This may already be installed
virtualenv .env                  # Create a virtual environment
source .env/bin/activate         # Activate the virtual environment
pip install -r requirements.txt  # Install dependencies
```

work for a while....

```unix
deactivate
```

### WARNING

The last part in main.py should be run on a computer with enough RAM or python will be killed

```python

# Simple hypercolumn pixel clustering

m = hc.transpose(1,2,0).reshape(50176, -1)
kmeans = cluster.KMeans(n_clusters=2, max_iter=300, n_jobs=5, precompute_distances=True)
cluster_labels = kmeans .fit_predict(m)
imcluster = np.zeros((224,224))
imcluster = imcluster.reshape((224*224,))
imcluster = cluster_labels
plt.imshow(imcluster.reshape(224, 224), cmap="hot")

#plt.show()

```


## OpenCv - cv2
#### direct way
To install and use cv2, please follow the instruction at the following address

http://www.pyimagesearch.com/2015/06/15/install-opencv-3-0-and-python-2-7-on-osx/

#### Anaconda

```python
conda install opencv
```

