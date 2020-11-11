# GEO4515 - Project 2020 - Water Index
# -----------------------------------------------------------------------------
import os, sys

from matplotlib.colors 	 import ListedColormap
from glob			   	 import glob

import functions		 as func
import numpy             as np
import matplotlib.pyplot as plt
import earthpy           as et
import earthpy.spatial   as es
import earthpy.plot      as ep
import rasterio          as rio
from rasterio.windows 	 import Window  # Bruke denne til å croppe!! window=Window(0, 850, 700,1000)

# https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/ndwi/
# Note: NDWI index is often used synonymously with the NDMI index, 
# often using NIR-SWIR combination as one of the two options. 
# NDMI seems to be consistently described using NIR-SWIR combination. 
# As the indices with these two combinations work very differently, 
# with NIR-SWIR highlighting differences in water content of leaves, 
# and GREEN-NIR highlighting differences in water content of water bodies, 
# we have decided to separate the indices on our repository as NDMI using NIR-SWIR, and NDWI using GREEN-NIR.
# -----------------------------------------------------------------------------
# NDWI: how to map water bodies? 
# - NDWI uses green and near infrared bands to highlight water bodies

# Opening the original .tif file 
original_file = rio.open('Prosjektdata/1993_tm_oslo.tif')

input_raster = original_file.read(window=Window(70, 850, 700, 1000)).astype(float)
#input_raster = original_file.read().astype(float)


ndwi = es.normalized_diff(input_raster[1], input_raster[3]) # (band2-band4)/(band2+band4)
#ndmi = es.normalized_diff(input_raster[3], input_raster[4]) # (band4-band4)/(band4+band5)

###############################################################################
# Plot NDWI With Colorbar Legend of Continuous Values
# ----------------------------------------------------

titles = ["Normalized Difference Water Index (NDWI)"]

# Turn off bytescale scaling due to float values for NDVI
ep.plot_bands(ndwi, cmap="RdYlBu", cols=1, title=titles, vmin=-1, vmax=1)
###############################################################################
# Classify NDWI
# -------------

# Create classes and apply to NDWI results
# why does it seem like such a low threshold is best?
# why not closer to 0.5? Is it because its landsat and not sentinel?
ndwi_class_bins    = [-np.inf, 0.1, np.inf]
ndwi_landsat_class = np.digitize(ndwi, ndwi_class_bins)

# Apply the nodata mask to the newly classified NDWI data
ndwi_landsat_class = np.ma.masked_where(np.ma.getmask(ndwi), ndwi_landsat_class)
np.unique(ndwi_landsat_class)

###############################################################################
# Plot Classified NDWI With Categorical Legend - EarthPy Draw_Legend()
# --------------------------------------------------------------------

# Define color map
nbr_colors = ["gray", "royalblue"]
nbr_cmap = ListedColormap(nbr_colors)

print(nbr_cmap)

# Define class names
ndwi_cat_names = [
    "Not Water",
    "Water",
]

# Get list of classes
classes = np.unique(ndwi_landsat_class)
classes = classes.tolist()
#classes.insert(0,1)

# The mask returns a value of none in the classes. remove that
#classes = classes[0:5]

print(classes)

# Plot your data
fig, ax = plt.subplots(figsize=(8, 8))
im      = ax.imshow(ndwi_landsat_class, cmap=nbr_cmap)

ep.draw_legend(im_ax=im, classes=classes, titles=ndwi_cat_names)
ax.set_title("Normalized Difference Vegetation Index (NDVI) Classes",fontsize=14)
ax.set_axis_off()

# Auto adjust subplot to fit figure size
plt.tight_layout()
plt.show()
