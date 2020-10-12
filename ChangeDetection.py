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


# Sentinel har bedre oppløsning, kanskje bedre for classification


#n_1993 	   = 6 	# 6 bands
#bands_1993 = func.GetDataAndBands(n_1993, crop=False, x1=70, x2=850, y1=700, y2=1000, file='Prosjektdata/1993_tm_oslo.tif')

#n_2000 	   = 6 	# 6 bands
#bands_2000 = func.GetDataAndBands(n_2000, crop=False, x1=70, x2=850, y1=700, y2=1000, file='Prosjektdata/2000_etm_oslo.tif')

bands_1993 = rio.open('Prosjektdata/1993_tm_oslo.tif').read().astype(float)
bands_2000 = rio.open('Prosjektdata/2000_etm_oslo.tif').read().astype(float)


print(bands_1993)
print(bands_2000)

# ndvi = es.normalized_diff(bands[3], bands[2]) # (band4-band3)/(band4+band3)
# må sjekke om det er band 4 og 3 som skal brukes i begge 

# NDI(CHANGE) = (AFTER-BEFORE) / (AFTER+BEFORE)

titles = ['Band 1','Band 2','Band 3','Band 4','Band 5','Band 7']

NDVI_before = es.normalized_diff(bands_1993[3], bands_1993[2])
NDVI_after  = es.normalized_diff(bands_2000[3], bands_2000[2])
NDI_of_NDVI = (NDVI_after-NDVI_before)/(NDVI_after+NDVI_before)

NDI_change = (bands_2000-bands_1993)/(bands_2000+bands_1993)

ep.plot_bands(NDI_of_NDVI, cmap="RdYlGn", cols=2, title='NDI of NDVI', vmin=-1, vmax=1)
ep.plot_bands(NDI_change, cmap="RdYlGn", cols=2, title=titles, vmin=-1, vmax=1)

#ep.plot_rgb(NDI_change, rgb=[2,1,0], title="Change RGB", stretch=True)


