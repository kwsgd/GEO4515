# GEO4515 - Project 2020 - Spectral signatures
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

import functions		 as func
# -----------------------------------------------------------------------------

n_bands = 6

yr = input("Enter year (YYYY): ")

if yr == '1993':
    file = 'Prosjektdata/1993_tm_oslo.tif'
elif yr == '2000':
    file = 'Prosjektdata/2000_etm_oslo.tif'
else:
    print("Input: 1993 or 2000")

bands = func.GetDataAndBands(n=n_bands, crop=False, x1=70, x2=850, y1=700, y2=1000, file=file)

# All bands in a stack. Band 1, 2, 3, 4, 5 and 7
all_bands = np.stack([bands[0],bands[1],bands[2],bands[3],bands[4],bands[5]])
#print(all_bands.shape)	# (6, 2246, 2245)

### True/False color composites
# -----------------------------------------------------------------------------
'''
func.ColorComposition(bands=all_bands, yr=yr)
func.ColorComposition(bands=all_bands, yr=yr, r=5, g=2, b=1)
func.ColorComposition(bands=all_bands, yr=yr, r=3, g=2, b=1) # Veg. health?
'''

### Spectral signatures
# -----------------------------------------------------------------------------

# Empty lists for cropped images
band_crop 	= []
water		= []
forest 		= []

# Cropp original image

for i in range(n_bands):
	band_crop.append(func.CropImg(img=bands[i], x1=0, x2=700, y1=850, y2=1800))

# Crop out an water area
for i in range(n_bands):
	water.append(np.mean(func.CropImg(img=band_crop[i], x1=210, x2=260, y1=415, y2=440)))

# Crop out an forest area
for i in range(n_bands):
	forest.append(np.mean(func.CropImg(img=band_crop[i], x1=310, x2=360, y1=685, y2=710)))

#print(band_crop)
#print(water)
#print(forest)

band_list = ['Band 1','Band 2','Band 3','Band 4','Band 5','Band 7']

plt.plot(band_list, water, label='Water')
plt.plot(band_list, forest, label='Forest')
plt.title("Spectral signatures. Year: %s" %yr, fontsize=15)
plt.xlabel("Wavelength", fontsize=15); plt.ylabel("DN", fontsize=15)
plt.legend(fontsize=15)
plt.savefig("Results/band_list_%s.png" %yr); plt.show()







#band_crop = band1[850:1800, 0:700]

#im = ax.imshow(band1.squeeze())
#im_crop = plt.imshow(band_crop)
#ep.colorbar(im)
#ax.set(title="blabla")

#ax.set_axis_off()

#band_titles = ['Band 1']

#plt.hist(water_crop[0], water_crop[1])
#plt.show()

#for i in range(len(all_bands)):
#	print(all_bands[i])


# Dette er den eneste tif-filen jeg får til å plotte
# På alle de andre får jeg feilmeldinger
#im = Image.open('Prosjektdata/1973_mss_oslo.tif')
#im.show()

# 2000_etm_oslo.tif
#
# rio.open('2000_etm_oslo.tif')


'''
in_path = 'C:/Users/Anna-/Documents/GEO4515/Prosjektdata/'
input_filename = '2000_etm_oslo.tif'

out_path = 'C:/Users/Anna-/Documents/GEO4515/Oslo/'
output_filename = 'tile_'

tile_size_x = 256
tile_size_y = 512

ds = gdal.Open(in_path + input_filename)
band = ds.GetRasterBand(1)
xsize = band.XSize
ysize = band.YSize

oslo_list = []

for i in range(0, xsize, tile_size_x):
    for j in range(0, ysize, tile_size_y):
        com_string = "gdal_translate -of GTIFF -srcwin " + str(i)+ ", " + str(j) + ", " + str(tile_size_x) + ", " + str(tile_size_y) + " " + str(in_path) + str(input_filename) + " " + str(out_path) + str(output_filename) + str(i) + "_" + str(j) + ".tif"
        os.system(com_string)
'''

'''
image_list = []
for filename in glob.glob('Oslo/*.tif'): #assuming gif
    im=Image.open(filename)
    image_list.append(im)
print(image_list)
print(len(image_list))
'''
