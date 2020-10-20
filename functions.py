# GEO4515 - Project 2020 - Functions
# -----------------------------------------------------------------------------
import os, gdal, sys, glob

from PIL 				 import Image, TiffImagePlugin
from rasterio.windows 	 import Window

import matplotlib.pyplot as plt
import numpy             as np
import geopandas         as gpd
import rasterio          as rio
import earthpy           as et
import earthpy.spatial   as es
import earthpy.plot      as ep
# -----------------------------------------------------------------------------

def GetDataAndBands(n, crop=False, x1=100, x2=200, y1=100, y2=200, file=''):
    """
    Function to get (pre-chosen) data and separate data into bands
    Returns list of bands

    DENNE KAN LAGES MER GENERELL
    """
    band = []	# Har det noe aa si om det er array eller liste..????
    # Import data and separate bands
    if crop == True:
        with rio.open(file) as src: # 'Prosjektdata/1993_tm_oslo.tif'
            for i in range(1, n+1):				# Make 6 bands, 1-5 and 7
                #w = src.read(1, window=Window(0, 0, 512, 256))
                #band.append(src.read(i, window=Window(70, 850, 700, 1000)).astype(float))
                band.append(src.read(i, window=Window(x1, x2, y1, y2)).astype(float))
    else:
        with rio.open(file) as src:

            for i in range(1, n+1):				# Make 6 bands, 1-5 and 7
                band.append(src.read(i))

    return band

def ColorComposition(bands, yr='', r=2, g=1, b=0):
    """
    Plot true/false colour composite. True = [2, 1, 0] (r, g, b)
    The function plots the result, but does not automatically save the image!
    """

    if r==2 and g == 1 and b==0:
        print("'True' color composition. %s" %yr)
        ep.plot_rgb(bands, rgb=[r,g,b], figsize=(7,7), title="RGB image (True). Year: %s" %yr, stretch=True)

    else:
        print("False color composition %s" %yr)
        ep.plot_rgb(bands, rgb=[r,g,b], figsize=(7,7), title="RGB image (False). R=%s, G=%s, B=%s (Python index). Year: %s" %(r, g, b, yr), stretch=True)

def CropImg(img, x1, x2, y1, y2): # Usikker om de ble riktig vei...
    """
    Function for cropping an image. Returns the cropped image
    """
    return img[y1:y2, x1:x2]
