# GEO4515 - Project 2020 - Vegetation Index
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

# https://earthpy.readthedocs.io/en/latest/gallery_vignettes/plot_calculate_classify_ndvi.html#sphx-glr-download-gallery-vignettes-plot-calculate-classify-ndvi-py
# https://directory.eoportal.org/web/eoportal/satellite-missions/l/landsat-7
# 1993_tm_oslo
# https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/ndwi/
# -----------------------------------------------------------------------------

n 	  = 6 	# 6 bands

yr = input("Enter year (YYYY): ")

if yr == '1993':
    file = 'Prosjektdata/1993_tm_oslo.tif'
elif yr == '2000':
    file = 'Prosjektdata/2000_etm_oslo.tif'
else:
    print("Input: 1993 or 2000")

bands = func.GetDataAndBands(n, crop=True, x1=70, x2=850, y1=700, y2=1000, file=file)
#bands = func.GetDataAndBands(n, file='Prosjektdata/1993_tm_oslo.tif')
print(bands)

#all_bands = np.dstack((band1,band2,band3,band4,band5,band7)).swapaxes(2,0)
#all_bands = np.stack([band1,band2,band3,band4,band5,band7])
#ep.plot_bands(all_bands)

###############################################################################
# Calculate Normalized Difference Vegetation Index (NDVI)
# -------------------------------------------------------
#
# You can calculate NDVI for your dataset using the
# ``normalized_diff`` function from the ``earthpy.spatial`` module.
# Math will be calculated (b1-b2) / (b1 + b2).

# Fra oppgavetekst:
#NDVI = (NIR-VIS)/(NIR+VIS),
#e.g. (TM4-TM3)/(TM4+TM3), or (TM4-TM2)/(TM4-TM2)
# As shown below, Normalized Difference Vegetation Index (NDVI) uses the NIR and red channels in its formula.

#ndvi = es.normalized_diff(band4, band3) # (band4-band3)/(band4+band3)
ndvi = es.normalized_diff(bands[3], bands[2]) # (band4-band3)/(band4+band3)
print(ndvi)
print(np.min(ndvi))
print(np.max(ndvi))

###############################################################################
# Plot NDVI With Colorbar Legend of Continuous Values
# ----------------------------------------------------
#
# You can plot NDVI with a colorbar legend of continuous values using the
# ``plot_bands`` function from the ``earthpy.plot`` module.

titles = ["Normalized Difference Vegetation Index (NDVI) Year: %s" %yr]

# Turn off bytescale scaling due to float values for NDVI
ep.plot_bands(ndvi, cmap="RdYlGn", cols=1, title=titles, vmin=-1, vmax=1)
###############################################################################
# Classify NDVI
# -------------
#
# Next, you can categorize (or classify) the NDVI results into useful classes.
# Values under 0 will be classified together as no vegetation. Additional classes
# will be created for bare area and low, moderate, and high vegetation areas.

# Create classes and apply to NDVI results
#ndvi_class_bins    = [-np.inf, 0, 0.1, 0.25, 0.4, np.inf]
ndvi_class_bins    = [-np.inf, 0, 0.25, 0.5, 0.75, np.inf]
ndvi_landsat_class = np.digitize(ndvi, ndvi_class_bins)

# Apply the nodata mask to the newly classified NDVI data, dont need this?
ndvi_landsat_class = np.ma.masked_where(np.ma.getmask(ndvi), ndvi_landsat_class)
np.unique(ndvi_landsat_class)

###############################################################################
# Plot Classified NDVI With Categorical Legend - EarthPy Draw_Legend()
# --------------------------------------------------------------------
#
# You can plot the classified NDVI with a categorical legend using the
# ``draw_legend()`` function from the ``earthpy.plot`` module.

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
#classes.insert(0,1)

# The mask returns a value of none in the classes. remove that
#classes = classes[0:5]

print(classes)

# Plot your data
fig, ax = plt.subplots(figsize=(8, 8))
im      = ax.imshow(ndvi_landsat_class, cmap=nbr_cmap)

ep.draw_legend(im_ax=im, classes=classes, titles=ndvi_cat_names)
ax.set_title("Normalized Difference Vegetation Index (NDVI) Classes. Year: %s" %yr,fontsize=14)
ax.set_axis_off()

# Auto adjust subplot to fit figure size
plt.tight_layout()
plt.show()
