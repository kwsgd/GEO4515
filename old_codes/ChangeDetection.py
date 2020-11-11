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

# # https://www.usgs.gov/faqs/what-are-best-landsat-spectral-bands-use-my-research?qt-news_science_products=0#qt-news_science_products

bands_1993 = rio.open('Prosjektdata/1993_tm_oslo.tif').read().astype(float)
bands_2000 = rio.open('Prosjektdata/2000_etm_oslo.tif').read().astype(float)


def NormalizedDifferenceIndex(after, before, title='', cmap=''):

	"""
	NDI(CHANGE) = (AFTER-BEFORE) / (AFTER+BEFORE)
	"""

	#titles = ['Blue','Green','Red','Near-Infrared','Short-wave Infrared','Short-wave Infrared']
	#NDI_change = (bands_2000-bands_1993)/(bands_2000+bands_1993)
	#ep.plot_bands(NDI_change, cmap="RdYlGn", cols=2, title=titles, vmin=-1, vmax=1)

	NDI = (after-before)/(after+before)
	ep.plot_bands(NDI, cmap=cmap, cols=2, title=titles, vmin=-1, vmax=1)


def NDI_of_NDVI(ndvi_after, ndvi_before, title='NDI of NDVI', cmap='RdYlGn_r'):
	"""
 	NDI of NDVI = (NDVI_AFTER-NDVI_BEFORE) / (NDVI_AFTER+NDVI_BEFORE)
	"""

	#NDVI_before = es.normalized_diff(bands_1993[3], bands_1993[2])
	#NDVI_after  = es.normalized_diff(bands_2000[3], bands_2000[2])
	NDI_of_NDVI = (ndvi_after-ndvi_before)/(ndvi_after+ndvi_before)

	#ep.plot_bands(NDVI_before, cmap="RdYlGn",  cols=1, title='NDVI 1993', vmin=-1, vmax=1)
	#ep.plot_bands(NDVI_after,  cmap="RdYlGn",  cols=1, title='NDVI 2000', vmin=-1, vmax=1)
	#ep.plot_bands(NDI_of_NDVI, cmap="RdYlGn_r", cols=1, title='NDI of NDVI', vmin=-1, vmax=1)
	ep.plot_bands(NDI_of_NDVI, cmap=cmap, cols=1, title=title, vmin=-1, vmax=1)


def Difference(after, before, title='', cmap=''):
	"""
	DIFFERENCE = (AFTER - BEFORE)
	"""

	difference = (after - before)
	ep.plot_bands(SWIR_difference, cmap=cmap, title=title)



def Ratio(after, before, title='', cmap=''):
	"""
	RATIO =  (AFTER / BEFORE)
	"""

	ratio = after/before
	ep.plot_bands(ratio, cmap=cmap, title=title)


def MultitemporalFalseColourComposites(band_nr):
	"""
	Hmm, not sure if I understand this... 
	"""

	im1993 = rio.open('Prosjektdata/1993_tm_oslo.tif').read(band_nr).astype(float)
	#im2000 = rio.open('Prosjektdata/1993_tm_oslo.tif').read().astype(float)
	#im1973 = rio.open('Prosjektdata/1993_tm_oslo.tif').read().astype(float)

	# 1973 as red, 1993 as green and 2000 as blue
	im_bands = np.stack([im1973, im1993, im2000])  # something...
	#im_bands = es.stack()

	ep.plot_rgb(im_bands, rgb=[2,1,0], title="False Color (NIR, red, green) in 2000", stretch=True)
	pass


# --------------------------------------------------------------------------------------------------------

# ===================================
# Band 5 - Short-wave Infrared (SWIR)
# ===================================
# Discriminates moisture content of soil and vegetation; penetrates thin clouds

SWIR_before = bands_1993[4] # band 5
SWIR_after  = bands_2000[4] # band 5

Difference(SWIR_after, SWIR_before, title='SWIR difference [1993 - 2000]', cmap='seismic_r')

# --------------------------------------------------------------------------------------------------------

# ======================
# Band 4 - Near Infrared
# ======================
# Emphasizes biomass content and shorelines

NIR_before = bands_1993[3] # band 4
NIR_after  = bands_2000[3] # band 4

# change colormap?
Difference(NIR_after, NIR_before, title='NIR difference [1993 - 2000]', cmap='Greens')

# --------------------------------------------------------------------------------------------------------

# ============
# Band 3 - Red
# ============
# Discriminates vegetation slopes

RedBefore   = bands_1993[2] # band 3
RedAfter    = bands_2000[2] # band 3

Difference(RedAfter, RedBefore, title='Red difference [1993 - 2000]', cmap='RdGy_r')
Ratio(RedAfter, RedBefore, title='seismic', cmap='Red ratio [1993 - 2000]')

# --------------------------------------------------------------------------------------------------------

# ==============
# Band 2 - Green
# ==============
# Emphasizes peak vegetation, which is useful for assessing plant vigor

GreenBefore = bands_1993[1] # band 2
GreenAfter  = bands_2000[1] # band 2

Difference(GreenAfter, GreenBefore, title='Green difference [1993 - 2000]', cmap='YlGn')

# --------------------------------------------------------------------------------------------------------

# =============
# Band 1 - Blue
# =============
# Bathymetric mapping, distinguishing soil from vegetation and deciduous from coniferous vegetation

BlueBefore  = bands_1993[0] # band 1
BlueAfter   = bands_2000[0] # band 1

Difference(BlueAfter, BlueBefore, title='Blue difference [1993 - 2000]', cmap='PuOr')



# ============================
# False Color: NIR, rød, grønn
# ============================
# Plantene sin helse

TrueColor = False
FalseColr = False

if TrueColor:
	ep.plot_rgb(bands_1993, rgb=[2,1,0], title="True Color in 1993", stretch=True)
	ep.plot_rgb(bands_2000, rgb=[2,1,0], title="True Color in 2000", stretch=True)

if FalseColor:
	ep.plot_rgb(bands_2000, rgb=[3,2,1], title="False Color (NIR, red, green) in 2000", stretch=True)
	ep.plot_rgb(bands_1993, rgb=[3,2,1], title="False Color (NIR, red, green) in 1993", stretch=True)



# Something Andi mentioned about RGB difference plot....?
# That could be cool, and easier in python than the other tools?
#DiffBands = np.stack([DiffBlue, DiffGreen, DiffRed])
#ep.plot_rgb(DiffBands, rgb=[2,1,0], title="RGB Difference", stretch=True)
