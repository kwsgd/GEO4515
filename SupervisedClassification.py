import os, shutil, sys, gdal

import matplotlib.pyplot as plt
import earthpy.spatial   as es
import earthpy.plot      as ep
import skimage.io 		 as io
import geopandas 		 as gpd
import rasterio          as rio
import earthpy           as et
import numpy      		 as np



original_file = rio.open('Prosjektdata/1993_tm_oslo.tif')
meta_data 	  = original_file.meta

width  = meta_data['width']
height = meta_data['height']
print(width, height)

input_raster = original_file.read().astype(float)


Truth_Data = gpd.read_file('Truth_Data.shp'); print(Truth_Data)
classes    = Truth_Data['landcovers'].unique()

print('The Truth Data include {n} classes: {labels}'.format(n=classes.size, labels=classes))

class_ids = np.arange(classes.size + 1)
class_ids = np.unique(classes + 1)
print(class_ids)