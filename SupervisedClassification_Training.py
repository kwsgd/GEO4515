# http://remote-sensing.org/image-classification-with-python/
# http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html


###################################################
# Program that generates training data, and 
# use it to build a supervised ML model
###################################################

# from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier

import os, shutil, sys

import skimage.io as io
import numpy      as np

from sklearn.ensemble  import RandomForestClassifier, GradientBoostingClassifier
from sklearn.externals import joblib

###
import earthpy           as et
import earthpy.spatial   as es
import earthpy.plot      as ep
import rasterio          as rio
from rasterio.windows import Window
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
###

# set up your directories
rootdir = 'Prosjektdata/'
# path to training data
path_pix = 'TrainingData/'
# path to model
path_model = 'SupervisedModels/'
# path to classification results
path_class = 'ClassificationResults/'



with rio.open('Prosjektdata/1993_tm_oslo.tif') as src:
	band1 = src.read(1, window=Window(70, 850, 700, 1000)).astype(float)
	band2 = src.read(2, window=Window(70, 850, 700, 1000)).astype(float)
	band3 = src.read(3, window=Window(70, 850, 700, 1000)).astype(float)
	band4 = src.read(4, window=Window(70, 850, 700, 1000)).astype(float)
	band5 = src.read(5, window=Window(70, 850, 700, 1000)).astype(float)
	band7 = src.read(6, window=Window(70, 850, 700, 1000)).astype(float)
	meta = src.meta


'''
with rio.open('Prosjektdata/1993_tm_oslo.tif') as src:
	band1 = src.read(1).astype(float)
	band2 = src.read(2).astype(float)
	band3 = src.read(3).astype(float)
	band4 = src.read(4).astype(float)
	band5 = src.read(5).astype(float)
	band7 = src.read(6).astype(float)
	meta = src.meta
'''

ndvi = es.normalized_diff(band4, band3) # (band4-band3)/(band4+band3)

titles = ["Normalized Difference Vegetation Index (NDVI)"]

#ep.plot_bands(ndvi, cmap="RdYlGn", cols=1, title=titles, vmin=-1, vmax=1)

ndvi_class_bins    = [-np.inf, 0, 0.25, 0.5, 0.75, np.inf]
ndvi_landsat_class = np.digitize(ndvi, ndvi_class_bins)

ndvi_landsat_class = np.ma.masked_where(np.ma.getmask(ndvi), ndvi_landsat_class)
np.unique(ndvi_landsat_class)

# Define color map
nbr_colors = ["gray", "y", "yellowgreen", "g", "darkgreen"]
nbr_cmap = ListedColormap(nbr_colors)

# Define class names
ndvi_cat_names = [
    "No Vegetation",
    "Bare Area",
    "Low Vegetation",
    "Moderate Vegetation",
    "High Vegetation",
]

classes = np.unique(ndvi_landsat_class)
classes = classes.tolist()
print(classes)


profile = meta
#print(profile)
#print(ndvi_landsat_class.shape)
profile.update(dtype=rio.uint8, count=6, width=700, height=1000)
#profile.update(dtype=rio.uint8, count=6)


with rio.open(path_pix+'veg_training.tif', 'w', **profile) as outf:
	outf.write(ndvi_landsat_class.astype(rio.uint8), 1)


def training():
	# path to TIFF
	raster  = rootdir + '1993_tm_oslo.tif'
	# path to the corresponding pixel samples (training data) 
	samples = path_pix + 'veg_training.tif'

	# read in raster
	img_ds = rio.open('Prosjektdata/1993_tm_oslo.tif')   #.read()
	img_ds = img_ds.read(window=Window(70, 850, 700, 1000))
	#img_ds = io.imread(raster)
	# convert to 16bit numpy array 
	img    = np.array(img_ds, dtype='int16')
	print(img.shape)
	#print(img[0].shape, img[1].shape)
	img = np.concatenate(img[0], axis=0)
	#img    = np.stack([band1,band2,band3,band4,band5,band7]).astype(rio.uint16)
	#img.swapaxes(2,0)
	print(img.shape)
	#print(img, 'hei')

	# do the same with your sample pixels
	roi_ds = io.imread(samples)
	roi    = np.array(roi_ds, dtype='int8')
	print(roi.shape)
	# read in your labels
	labels = np.unique(roi[roi > 0])
	print('The training data include {n} classes: {classes}'.format(n=labels.size, classes=labels))
	# compose your X,Y data (dataset - training data)
	#X = img[roi > 0, :]
	X = img.reshape(-1,1)
	#X = np.array((img[0], labels))
	Y = roi[roi > 0]
	print(X.shape)
	print(Y.shape)

	# assign class weights (class 1 has the weight 3, etc.)
	weights = {1:3, 2:2, 3:2, 4:2, 5:1}

	# build your Random Forest Classifier 
	rf = RandomForestClassifier(class_weight=weights,
								n_estimators=100,
								criterion='gini',
								max_depth=4, 
								min_samples_split=2,
								min_samples_leaf=1,
								max_features='auto',
								bootstrap=True,
								oob_score=True,
								n_jobs=1,
								random_state=None,
								verbose=True)  

	# alternatively you may try out a Gradient Boosting Classifier 
	# It is much less RAM consuming and considers weak training data      
	""" 
	rf = GradientBoostingClassifier(n_estimators = 300,
									min_samples_leaf = 1,
									min_samples_split = 4,
									max_depth = 4,
									max_features = 'auto',
									learning_rate = 0.8,
									subsample = 1,
									random_state = None,
									warm_start = True)
	"""

	# now fit your training data with the original dataset
	rf = rf.fit(X,Y)

	# export your Random Forest / Gradient Boosting Model
	model = path_model + "model.pkl"
	joblib.dump(rf, model)

#training()


def classification():
	# Read worldfile of original dataset
	#tfw_old = rootdir + '1993_tm_oslo.tfwx'
	raster  = rootdir + '1993_tm_oslo.tif'


	# Read Data
	#img_ds = rio.open('Prosjektdata/1993_tm_oslo.tif').read()
	img_ds = rio.open('Prosjektdata/1993_tm_oslo.tif')   #.read()
	img_ds = img_ds.read(window=Window(100, 950, 400, 800))
	#img_ds = io.imread(raster)
	img = np.array(img_ds, dtype='int16')
	print(img.shape)
	img = np.concatenate(img[0], axis=0)
	#img = np.concatenate(img[1], axis=1)
	#img = np.concatenate(img, axis=1)
	print(img.shape)

	# call your random forest model
	rf = path_model + 'model.pkl'
	clf = joblib.load(rf) 

	# Classification of array and save as image (23 refers to the number of multitemporal NDVI bands in the stack)
	#new_shape = (img.shape[0] * img.shape[1], img.shape[2])
	#img_as_array = img[:, :, :23].reshape(new_shape)
	#print(img_as_array)

	img_as_array = img.reshape(-1,1)

	class_prediction = clf.predict(img_as_array)
	#class_prediction = class_prediction.reshape(img[:, :, 0].shape)

	# now export your classificaiton
	classification = path_class  + "classification.tif"
	io.imsave(classification, class_prediction)

	# Assign Worldfile to classified image
	#tfw_new = classification.split(".tif")[0] + ".tfw"
	#shutil.copy(tfw_old, tfw_new)

classification()
