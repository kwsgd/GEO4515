import os

# Glob for file manipulation
from glob import glob

import matplotlib.pyplot as plt
import numpy             as np
import geopandas         as gpd
import rasterio          as rio
import earthpy           as et
import earthpy.spatial   as es
import earthpy.plot      as ep

import sys 

# Mange ulike datasets man kan importere 
# https://earthpy.readthedocs.io/en/latest/earthpy-data-subsets.html

# https://www.earthdatascience.org/courses/use-data-open-source-python/multispectral-remote-sensing/landsat-in-Python/

# conda install geopandas
# conda install rasterio
# conda install -c conda-forge earthpy

# Extracted output to C:\Users\Anna-\earth-analytics\data\cold-springs-fire\

# Download data and set working directory
data = et.data.get_data('cold-springs-fire')
os.chdir(os.path.join(et.io.HOME, 'earth-analytics', 'data'))

# Create the path to your data
landsat_post_fire_path = os.path.join("cold-springs-fire",
                                      "landsat_collect",
                                      "LC080340322016072301T1-SC20180214145802",
                                      "crop")

print(landsat_post_fire_path)

# Generate a list of just the tif files
post_fire_tifs_list = glob(os.path.join(landsat_post_fire_path,
                                        "*band*.tif"))

# Sort the data to ensure bands are in the correct order
post_fire_tifs_list.sort()


# Create an output array of all the landsat data stacked
landsat_post_fire_path = os.path.join("cold-springs-fire",
                                      "outputs",
                                      "landsat_post_fire.tif")


# This will create a new stacked raster with all bands
landsat_post_fire_arr, land_meta = es.stack(post_fire_tifs_list,
                                            landsat_post_fire_path)


# View output numpy array
print(landsat_post_fire_arr)

# Plot all bands
ep.plot_bands(landsat_post_fire_arr)
plt.show()


# Plot all band histograms using earthpy
colors      = ['black', 'blue', 'green', 'red', 'black', 'black', 'black']
band_titles = ['Band 1', 'Blue', 'Green', 'Red', 'NIR', 'Band 6', 'Band7']

ep.hist(landsat_post_fire_arr, colors=colors, title=band_titles)

'''
fig1, axs1 = plt.subplots(4, 2)
axs1[0, 0].scatter(landsat_post_fire_arr[0, :, 0], landsat_post_fire_arr[0, :, 1])
axs1[0, 0].set_title('Band 1')
axs1[0, 1].scatter(landsat_post_fire_arr[1, :, 0], landsat_post_fire_arr[1, :, 1])
axs1[0, 1].set_title('Blue')
axs1[1, 0].scatter(landsat_post_fire_arr[2, :, 0], landsat_post_fire_arr[2, :, 1])
axs1[1, 0].set_title('Green')
axs1[1, 1].scatter(landsat_post_fire_arr[4, :, 0], landsat_post_fire_arr[4, :, 1])
axs1[1, 1].set_title('Red')
axs1[2, 0].scatter(landsat_post_fire_arr[5, :, 0], landsat_post_fire_arr[5, :, 1])
axs1[2, 0].set_title('Band 6')
axs1[2, 1].scatter(landsat_post_fire_arr[6, :, 0], landsat_post_fire_arr[6, :, 1])
axs1[2, 1].set_title('Band 7')
axs1[3, 0].scatter(landsat_post_fire_arr[4, :, 0], landsat_post_fire_arr[4, :, 1])
axs1[3, 0].set_title('NIR')

plt.tight_layout()
plt.show()

print(landsat_post_fire_arr)
print(np.shape(landsat_post_fire_arr))
'''

#ep.plot_rgb(landsat_post_fire_arr,
#            rgb=[4, 3, 2],
#            title="CIR Landsat Image Pre-Cold Springs Fire",
#            figsize=(10, 10))
#plt.show()


#ep.plot_rgb(landsat_post_fire_arr,
#            rgb=[3, 2, 1],
#            title="RGB Composite Image\n Post Fire Landsat Data")
#plt.show()

#ep.plot_rgb(landsat_post_fire_arr,
#            rgb=[3, 2, 1],
#            title="Landsat RGB Image\n Linear Stretch Applied",
#            stretch=True,
#            str_clip=1)
#plt.show()

# Adjust the amount of linear stretch to futher brighten the image
#ep.plot_rgb(landsat_post_fire_arr,
#            rgb=[3, 2, 1],
#            title="Landsat RGB Image\n Linear Stretch Applied",
#            stretch=True,
#            str_clip=4)
#plt.show()