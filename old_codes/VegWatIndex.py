# GEO4515 - Project 2020 - Vegetation & Water Index
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
from rasterio.windows 	 import Window

# ----------------------------------------------------------------------------------
# It would of course be easier to just say that ndvi classified as 'no vegetation'
# is probably a water body, and color that blue instead of grey.
# However, what if something is not vegetation class or water?
# And in this way, we also illustrate how to calculate ndwi...
# But we should ask about this, and if it makes sense, and the thresholds... 
# ----------------------------------------------------------------------------------


# Opening the original .tif file 
original_file = rio.open('Prosjektdata/1993_tm_oslo.tif')
#original_file = rio.open('Prosjektdata/2000_etm_oslo.tif')

input_raster = original_file.read(window=Window(70, 850, 700, 1000)).astype(float) # tyrifjorden
#input_raster = original_file.read().astype(float)


ndwi = es.normalized_diff(input_raster[1], input_raster[3]) # (band2-band4)/(band2+band4)
ndvi = es.normalized_diff(input_raster[3], input_raster[2])

# The weird mystical structures has values around:
# [-np.inf, 0.1, 0.80] = weird structure class 2 
print(np.min(ndvi))
print(np.max(ndwi))

# Classify NDVI as classes [1, 2, 3, 4, 5], where class 1 = no vegetation 
# -----------------------------------------------------------------------

ndvi_class_bins    = [-np.inf, 0, 0.25, 0.5, 0.75, np.inf]
ndvi_landsat_class = np.digitize(ndvi, ndvi_class_bins)

# Create 'no water' (class zero), and 'water' (class 1)
ndwi_class_bins    = [-np.inf, 0.1, np.inf]
ndwi_landsat_class = np.digitize(ndwi, ndwi_class_bins)-1

# The image width and length
len_i = ndvi.shape[0]; print(len_i)
len_j = ndvi.shape[1]; print(len_j)

# Array to be filled with classses [0, 1, 2, 3, 4, 5]
VegWatClass = np.zeros((len_i, len_j))

# Splitting the ndvi class 'no vegetation' into 'water' and 'unknown'

for i in range(len_i):
	for j in range(len_j):

		# if ndvi says not vegetation
		if ndvi_landsat_class[i,j] == 1:

			# ndwi says its not water either
			if ndwi_landsat_class[i,j] == 0:
				VegWatClass[i,j] = 0 

			# ndwi says this is water
			else:
				VegWatClass[i,j] = 1

		# if ndvi says vegetation, use same ndvi classes 		
		else:
			VegWatClass[i,j] = ndvi_landsat_class[i,j]

# Checking the number of unknown pixels 
number_unknown = VegWatClass[VegWatClass == 0]
print('unknown pixels:', len(number_unknown))

#########################################
# Plot Classified NDVI and the NDWI Class
# ---------------------------------------

# Define color map for NDVI
nd_colors = ["gray", "royalblue", "y", "yellowgreen", "g", "darkgreen"]
nd_cmap = ListedColormap(nd_colors)

# Define class names
cat_names = [
    "Unknown",
    "Water",
    "Bare Area",
    "Low Vegetation",
    "Moderate Vegetation",
    "High Vegetation",
]


# Get a list of the unique classes 
classes = np.unique(VegWatClass)
classes = classes.tolist()
print(classes)


# Plot the NDVI classification and NDWI class
fig, ax = plt.subplots(figsize=(8, 8))
im      = ax.imshow(VegWatClass, cmap=nd_cmap)

ep.draw_legend(im_ax=im, classes=classes, titles=cat_names)
ax.set_title("Normalized Difference Vegetation Index (NDVI) with Water Class",fontsize=14)
ax.set_axis_off()

# Auto adjust subplot to fit figure size
plt.tight_layout()
plt.show()
