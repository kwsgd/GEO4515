# https://www.hatarilabs.com/ih-en/land-cover-change-analysis-with-python-and-gdal-tutorial

import os, sys, gdal 

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

bands_1993 = rio.open('Prosjektdata/1993_tm_oslo.tif').read().astype(float)
bands_2000 = rio.open('Prosjektdata/2000_etm_oslo.tif').read().astype(float)


# =============================================
# NDI(CHANGE) = (AFTER-BEFORE) / (AFTER+BEFORE)
# =============================================

'''
titles = ['Blue','Green','Red','Near-Infrared','Short-wave Infrared','Short-wave Infrared']

NDI_change = (bands_2000-bands_1993)/(bands_2000+bands_1993)

ep.plot_bands(NDI_change, cmap="RdYlGn", cols=2, title=titles, vmin=-1, vmax=1)

# =============================================
# NDI of NDVI = (AFTER-BEFORE) / (AFTER+BEFORE)
# =============================================

NDVI_before = es.normalized_diff(bands_1993[3], bands_1993[2])
NDVI_after  = es.normalized_diff(bands_2000[3], bands_2000[2])
NDI_of_NDVI = (NDVI_after-NDVI_before)/(NDVI_after+NDVI_before)

ep.plot_bands(NDI_of_NDVI, cmap="RdYlGn", cols=2, title='NDI of NDVI', vmin=-1, vmax=1)

# =============================
# DIFFERENCE = (AFTER - BEFORE)
# =============================

# https://www.usgs.gov/faqs/what-are-best-landsat-spectral-bands-use-my-research?qt-news_science_products=0#qt-news_science_products

# Band 5 - Short-wave Infrared (SWIR)
# -----------------------------------
# Discriminates moisture content of soil and vegetation; penetrates thin clouds

SWIR_before = bands_1993[4] # band 5
SWIR_after  = bands_2000[4] # band 5

SWIR_difference = SWIR_after - SWIR_before

ep.plot_bands(SWIR_difference, cmap="seismic_r", title='SWIR difference [1993 - 2000]')

# Band 4 - Near Infrared
# ----------------------
# Emphasizes biomass content and shorelines
# change colormap

NIR_before = bands_1993[3] # band 4
NIR_after  = bands_2000[3] # band 4

NIR_difference = NIR_after - NIR_before

ep.plot_bands(NIR_difference, cmap="Greens", title='NIR difference [1993 - 2000]')

# Band 3 - Red
# ----------------------
# Discriminates vegetation slopes

RedBefore   = bands_1993[2] # band 3
RedAfter    = bands_2000[2] # band 3

DiffRed = RedAfter - RedBefore

ep.plot_bands(DiffRed, cmap="RdGy_r", title='Red difference [1993 - 2000]')

# Band 2 - Green
# ----------------------
# Emphasizes peak vegetation, which is useful for assessing plant vigor

GreenBefore = bands_1993[1] # band 2
GreenAfter  = bands_2000[1] # band 2

DiffGreen = GreenAfter - GreenBefore

ep.plot_bands(DiffGreen, cmap="YlGn", title='Green difference [1993 - 2000]')

# Band 1 - Blue
# ----------------------
# Bathymetric mapping, distinguishing soil from vegetation and deciduous from coniferous vegetation

BlueBefore  = bands_1993[0] # band 1
BlueAfter   = bands_2000[0] # band 1

DiffBlue = BlueAfter - BlueBefore

ep.plot_bands(DiffGreen, cmap="PuOr", title='Blue difference [1993 - 2000]')

# =========================
# RATIO =  (AFTER / BEFORE)
# =========================

# Band 3 - Red
# ----------------------
# Discriminates vegetation slopes

RedBefore   = bands_1993[2] # band 3
RedAfter    = bands_2000[2] # band 3

RedRatio = RedAfter/RedBefore

ep.plot_bands(RedRatio, cmap="seismic", title='Red ratio [1993 - 2000]')


#DiffBands = np.stack([DiffBlue, DiffGreen, DiffRed])
#ep.plot_rgb(DiffBands, rgb=[2,1,0], title="RGB Difference", stretch=True)

'''
# False Color: NIR, rød, grønn
# Plantene sin helse
#ep.plot_rgb(bands_2000, rgb=[3,2,1], title="False Color (NIR, red, green) in 2000", stretch=True)
#ep.plot_rgb(bands_1993, rgb=[3,2,1], title="False Color (NIR, red, green) in 1993", stretch=True)

#ep.plot_rgb(bands_1993, rgb=[2,1,0], title="True Color in 1993", stretch=True)
#ep.plot_rgb(bands_2000, rgb=[2,1,0], title="True Color in 2000", stretch=True)




