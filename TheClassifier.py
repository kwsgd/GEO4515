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
from sklearn.externals import joblib
from rasterio.windows  import Window

############################################
## Foreløpige tanker om hva vi skal gjøre ##
############################################


# 1) https://learn.arcgis.com/en/projects/predict-seagrass-habitats-with-machine-learning/#create-a-training-dataset
# ArcGIS Pro includes a default conda environment; arcgispro-py3.
# Maybe we have to install this to make it easier for ourselves... 

# 2) https://developers.arcgis.com/python/api-reference/
# 3) https://pro.arcgis.com


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
# https://pro.arcgis.com/en/pro-app/help/main/welcome-to-the-arcgis-pro-app-help.htm#ESRI_SECTION1_777CD8A6DFB247FB8E5E3546A5DC2F2A

# https://github.com/perrygeo/pyimpute
# https://developers.arcgis.com/python/api-reference/arcgis.raster.analytics.html?highlight=classify#arcgis.raster.analytics.classify
# https://pro.arcgis.com/en/pro-app/help/main/welcome-to-the-arcgis-pro-app-help.htm

# Creating "training data":

