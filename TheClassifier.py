import os, shutil, sys

import matplotlib.pyplot as plt
import earthpy.spatial   as es
import earthpy.plot      as ep
import skimage.io 		 as io
import rasterio          as rio
import earthpy           as et
import numpy      		 as np

from sklearn.ensemble  import RandomForestClassifier, GradientBoostingClassifier
from matplotlib.colors import ListedColormap
#from sklearn.externals import joblib
from rasterio.windows  import Window
from sklearn.model_selection import train_test_split

############################################
## Foreløpige tanker om hva vi skal gjøre ##
############################################


# 1) https://learn.arcgis.com/en/projects/predict-seagrass-habitats-with-machine-learning/#create-a-training-dataset
# ArcGIS Pro includes a default conda environment; arcgispro-py3.
# Maybe we have to install this to make it easier for ourselves... 

# 2) https://developers.arcgis.com/python/api-reference/
# 3) https://pro.arcgis.com/en/pro-app/help/main/welcome-to-the-arcgis-pro-app-help.htm
# 4) https://pro.arcgis.com/en/pro-app/arcpy/main/arcgis-pro-arcpy-reference.htm



# ==================
# PREPARE INPUT DATA
# ==================
# 
# Check for missing values (NaN):
# -------------------------------
# 	- If missing values -> fill values to complete the raw data (f.eks. K-nearest neighbour interpolation).
# 
# Since we create our own feature classes (not f.eks. real collected data as in the example in link 1),
# I dont think that missing values are a big problem in our case?
# But maybe its useful to check the bands for missing values?
#
# Could the few 'really red points/dots' in the VegIndexClass plot,
# be caused by missing pixel values? 
# The dots should either way be mentioned/discussed in the VegIndex ex.


##################################################################################
#          OBS - CONFUSING (I should have realized this earlier...).             #
# As in IN5410, we are not given a specific dataset we can use as train/test.    #
# So the "training data" we create, is later split into a train set and test set.# 
##################################################################################


# ========================
# DESCRIPTION OF VARIABLES
# ========================
# 
#
# Input raster(s):
# ----------------
# input (feature) layer : data we can use to create "training data" (can also be a input table).
# input channels        : each input channel correspond to a .tif layer / band. 
#
# input we will use     : 1993_tm_oslo.tif (croppet version)
#                         - this is a multidimensional raster with 6 layers/bands/channels
#
# So, the ''input (feature) layer'' and ''input channel'' is the same, I think. 
# Sometimes the input raster has just one layer/band, 
# other times we work with multidimensional rasters (kinda a 'collection of .tifs').
# 
#
# Empty layers (8bit layers according to exercise):
# -------------------------------------------------
# training channel : selected pixels (samples) 
# output   channel : layer with 1 or 0 / Truth or False
# 
# 	- We extract a selection of pixel values (corresponding to water, forest etc.)
# 	  from our input raster, and store these in our training channel.
# 
# 	- For each pixel value (or 'sample'), we assign the corresponding output, 'correct prediction'.
#   
#   - These channels we create, will become the dataset (the "training data") we need. 
#
# -----------------------------------------------------------------------------------------------------
# QUESTION: choosing pixels/samples etc.
# -----------------------------------------------------------------------------------------------------
#           - Only choose 'one value' of the original img/array (like 324), or need to be (r, g, b) ish?
#           - A pixel is represented by rbg, right?
#
#           - The exercise talk about choosing RGB (f.eks. 321), 
#             so maybe this have something to do with this?
#
#           - When changing from greyscale (?) picture/array to color image, 
#             it makes sense that the array would change to represent the coloured image, right..?
#           - Generally needs to understand pixels etc. better... 
# -----------------------------------------------------------------------------------------------------


# =======================================
# CHOOSING AND CREATING (FEATURE) CLASSES
# =======================================
#
# We need to choose reasonably classes: forest, water, agricultural land, urban/industry,...
# 
# I think we can reuse much of the same principales as in VegetationIndex, 
# but for the water class, we should probably calculate NDWI (water index)? 
# - Similar as VegIndex: only use NIR and Green band instead
# 
# Can also chech out ArcGis Pro and/or other tools,
# to see if classes can be extracted in a easier / nicer way.


# ==========================
# CREATING THE TRAINING DATA 
# ==========================
# 
# Look into how to create polygons:
# ---------------------------------
# 	- Can we have one polygon with say 30 samples (10 water, 10 forest, 10 'other')?
# 
#   - Or do we need to have one polygon for each class (water, forest,...)?
#   - The way the exercise describes it, it seems that we should have one polygram for each class.
#   - Maybe: create a multilayered raster of all the class polygons
# 
# 
# We can determine if a pixel/sample overlaps with a polygon for the given class:
# -----------------------------------------------------------------------------
# 	- if so  ->  set corresponding output channel = 1 (True,  pixel matches the class)
# 	- if not ->  set corresponding output channel = 0 (False, pixel does not match the class)
# 
# By doing this, we generate the labeled "training dataset" for a given class.
#
# Often: the "training data" is created in a geoprocessing tool (see link 1 above, or Focus etc.).
# "The training data" can then be exported, and used as labeled data in python with scikit learn etc. 
# to create a supervised machine learning model like we are used to. 
# This could be a good option if we dont manage to create the "training dataset" in python?


# =========================================
# BUILDING AND TESTING THE CLASSIFIER MODEL
# =========================================
# 
# Now we have created the "training dataset" we need to build our classifier,
# and we split the dataset as usual:
# 	- training set: X_train and y_train
#   - testing set : X_test and y_test 
# 
# Shape of X_train should be (n_samples, n_features)
# Shape of y_train should be (n_samples,)
# 

# To choose the best model, the exercise suggests things like:
# ------------------------------------------------------------
# 	- Signature separability (I have a link somewhere that may be useful, must find it...)
# 	- Scatter plot (Sample: Selected classes)
# 	- Adjust channels on X and Y axes
#   - Signature statistics, and/or histogram.
#   - Classification preview. Choose the classification method which seems best to you.
#
#   ----------------------------------------------------------------
#   We will probably just focus on making one model at first I guess
#   ----------------------------------------------------------------   
#   
#   - So... Basically, use some tools to study how good/bad the the model(s) is
#   - 'Adjust channels': meaning we should look at model performance for every band, or what?
#   - Should the model be good for all classes and bands..? One model for each band, each class..?
#   - Or is the intention to apply a model on rbg image created by the bands? Since we must choose RGB?
#   - Or are we only interested in the read, blue and green band? But it seems like all should be used...
# 
# 
# Maybe look at FYS-STK for inspiration, and how to discuss our results etc.
# train/test performance plot maybe?
# Can we spot if the model is overfitting/underfitting?
# Correlation matrix is probably useless, since we dont have 'features' for each class?
# 
# Instead of just evaluate the model performance on the test data, 
# we could try the classifier on the input raster (the cropped image)?


####################################################################################################
####################################################################################################

# https://spectraldifferences.wordpress.com/2014/09/07/object-based-classification-using-random-forests/

# https://github.com/perrygeo/pyimpute
# https://developers.arcgis.com/python/api-reference/arcgis.raster.analytics.html?highlight=classify#arcgis.raster.analytics.classify
# https://pro.arcgis.com/en/pro-app/help/main/welcome-to-the-arcgis-pro-app-help.htm
# https://developers.arcgis.com/python/guide/install-and-set-up/
# https://developers.arcgis.com/python/sample-notebooks/

# maybe useful when we have created polygons:
# https://rasterio.readthedocs.io/en/latest/topics/masking-by-shapefile.html

# Maybe installing ArcGIS API for Python

# Random Forest works best with a large training data set to train the forest 
# So if we create our own "training data", our train set may be too small 
# for the forest to work good enough, we should keep this in mind when getting results
# Maybe another model would then be better, or using a geoprocessing tool to  create "training data"

# number of bands are kinda like the 'features'

####################################################################################################
####################################################################################################

# Creating "training data"
# ------------------------
# Using the forest classes

'''
# We have to normaize the data?
xTrain = xTrain / 255.0
xTest = xTest / 255.0
'''

'''
# Renaming the targets to 0 and 1 instead of Confirmed/False Positives
target[target == 'CONFIRMED']      = 1
target[target == 'FALSE POSITIVE'] = 0
'''

# next idea -> use only rgb bands 
# use group/segmentation to create to split up
# assign some true value....

####################################################################################################
####################################################################################################

# Not exactly as the exercise, but using the cropped image with 
# forest index classification, use this as the 'training data' with 'true prediction'
# then trying the model on the whole uncropped picture 
# er vel ndvi og class 

# Opening the original .tif file 
original_file = rio.open('Prosjektdata/1993_tm_oslo.tif')

# The metadata of the original file 
meta_data = original_file.meta

#print(meta_data)
#{'driver': 'GTiff', 'dtype': 'uint8', 'nodata': None, 'width': 2245, 'height': 2246,'count': 6,
#'crs': CRS.from_dict(init='epsg:32632'), 'transform': Affine(28.5, 0.0, 554011.5,0.0, -28.5, 6691999.5)}

# Reading and cropping to create our input raster: shape (6, 1000, 700)
input_raster = original_file.read(window=Window(70, 850, 700, 1000))  #.astype(float)


# We need to change the dimensions 
# features : (n_samples, n_features)
# labels   : (n_samples,)


ndvi = es.normalized_diff(input_raster[4], input_raster[3])

#np.where(np.isnan(X))
ndvi = np.nan_to_num(ndvi) 

# Creating classes
ndvi_class_bins    = [-np.inf, 0, 0.25, 0.5, 0.75, np.inf]
ndvi_landsat_class = np.digitize(ndvi, ndvi_class_bins)

# Apply the nodata mask to the newly classified NDVI data
ndvi_landsat_class = np.ma.masked_where(np.ma.getmask(ndvi), ndvi_landsat_class)
np.unique(ndvi_landsat_class)
#print(ndvi_landsat_class.shape) (1000, 700)


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

# Get list of classes
classes = np.unique(ndvi_landsat_class)
classes = classes.tolist()
classes.insert(0,1)
classes = classes[0:5]
print(classes, 'classes')

#ep.plot_bands(input_raster[4])


#features = np.concatenate(input_raster[4], axis=0).reshape(-1,1)
targets  = np.concatenate(ndvi_landsat_class)
features = np.concatenate(ndvi).reshape(-1,1)

print(features.shape)
print(targets.shape)

X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.33, random_state=42)


# build your Random Forest Classifier 
rf = RandomForestClassifier(class_weight=None,
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




# now fit your training data with the original dataset, må bruke org bilde pga reshape
print(features)
print(targets)
rf.fit(features,targets)


# test data as the uncropped ndvi
test_data = original_file.read()
ndvi_test = es.normalized_diff(test_data[4], test_data[3]) #np.where(np.isnan(X))
ndvi_test = np.nan_to_num(ndvi_test) 
test_image = np.concatenate(ndvi_test).reshape(-1,1)

#test_image = np.concatenate(original_file.read(4)).reshape(-1,1)
print(test_image.shape)                                 # (2246, 2245) - >  (5042270,1)
#print(np.ravel(test_image))

prediction = rf.predict(test_image)

prediction = np.ravel(prediction)
print(prediction.shape)

bilde = prediction.reshape(2246, 2245)
print(bilde.shape, 'bi')

#ep.plot_bands(bilde)
#plt.show()

# Define color map
nbr_colors = ["gray", "y", "yellowgreen", "g", "darkgreen"]
nbr_cmap = ListedColormap(nbr_colors)

# Plot your data
fig, ax = plt.subplots()
im      = ax.imshow(bilde)

ep.draw_legend(im_ax=im, classes=classes, titles=ndvi_cat_names)
ax.set_title("hmmmm",fontsize=14)
ax.set_axis_off()

# Auto adjust subplot to fit figure size
plt.tight_layout()
plt.show()







'''
#print(input_raster.swapaxes(0,2).shape)
#print(np.concatenate(input_raster[0], axis=0).shape)

#n_samples = np.concatenate(input_raster[0], axis=0)
#print(len(n_samples))

features = input_raster.swapaxes(0,2)
features = features.swapaxes(0,1)
features = np.concatenate((input_raster[0], input_raster[0]), axis=0)
print(features.shape)
'''