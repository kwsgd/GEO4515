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


#TrainingData = rio.open('TrainingData/veg_training.tif').read()

with rio.open('ClassificationResults/classification.tif') as src:
    image = src.read()

plt.imshow(image)
#print(image)

'''
# Plot your data
fig, ax = plt.subplots(figsize=(8, 8))
im      = ax.imshow(ndvi_landsat_class, cmap=nbr_cmap)
'''

# This will create a new stacked raster with all bands
#landsat_post_fire_arr, land_meta = es.stack(post_fire_tifs_list,landsat_post_fire_path)
# View output numpy array
#print(landsat_post_fire_arr)

