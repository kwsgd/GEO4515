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

yr = input("Enter year (1993 or 2000): ")

if yr == '1993':
    file = 'Prosjektdata/1993_tm_oslo.tif'
elif yr == '2000':
    file = 'Prosjektdata/2000_etm_oslo.tif'
else:
    print("Input: 1993 or 2000")

# Rundt det vannet :P
bands = func.GetDataAndBands(n=n_bands, crop=True, c_off=30, r_off=800, w=800, h=500, file=file)
#bands = func.GetDataAndBands(n=n_bands, crop=True, c_off=70, r_off=850, w=700, h=1000, file=file)

# All bands in a stack. Band 1, 2, 3, 4, 5 and 7
all_bands = np.stack([bands[0],bands[1],bands[2],bands[3],bands[4],bands[5]])

### True/False color composites
# -----------------------------------------------------------------------------

func.ColorComposition(bands=all_bands, yr=yr)
#func.ColorComposition(bands=all_bands, yr=yr, r=5, g=2, b=1)
func.ColorComposition(bands=all_bands, yr=yr, r=3, g=2, b=1) # Veg. health?


### Spectral signatures
# -----------------------------------------------------------------------------

# Empty lists for cropped images
#band_crop 	= []
#water		= []
#forest 		= []
band_list   = ['Band 1','Band 2','Band 3','Band 4','Band 5','Band 7']

'''
# Cropp original image
for i in range(n_bands):
	#band_crop.append(func.CropImg(img=bands[i], x1=0, x2=700, y1=850, y2=1800))
    band_crop.append(func.CropImg(img=bands[i], x1=0, x2=700, y1=850, y2=1800))

# Crop out an forest area
for i in range(n_bands):
	#forest.append(np.mean(func.CropImg(img=bands[i], x1=310, x2=360, y1=685, y2=710)))
    forest.append(np.mean(func.CropImg(img=bands[i], x1=10, x2=60, y1=50, y2=250)))
'''

samples = 5

'''
Vi maa finne noyaktige omraader for samples...!!!
'''

def Water(x1, samples, n, bands):

    x1      = x1
    water   = [[0] * n for i in range(samples)]

    # Crop out an water area
    for j in range(samples):
        for i in range(n_bands):
            x2 = x1+8
        	#water.append(np.mean(func.CropImg(img=bands[i], x1=210, x2=260, y1=415, y2=440)))
            water[j][i] = np.mean(func.CropImg(img=bands[i], x1=x1, x2=x2, y1=455, y2=490))

            x1 = x2+2
    return water

def Forest(y1, samples, n, bands):
    y1       = y1
    forest   = [[0] * n_bands for i in range(samples)]

    # Crop out an water area
    for j in range(samples):
        for i in range(n_bands):
            y2 = y1+8
            forest[j][i] = np.mean(func.CropImg(img=bands[i], x1=10, x2=40, y1=y1, y2=y2))

            y1 = y2+2
    return forest

def mean(n, samples, bands):
    mean = []
    std  = []
    for i in range(n):
        #sum = 0
        mean_ = []
        for j in range(samples):
            #sum = sum + bands[j][i]
            mean_.append(bands[j][i])

        #mean.append(sum/samples)
        mean.append(np.mean(mean_))
        std.append(np.std(mean_))

    return mean, std


agg = []
# Crop out an agg area
for i in range(n_bands):
    agg.append(np.mean(func.CropImg(img=bands[i], x1=566, x2=582, y1=183, y2=203)))
    # Dette likner mer paa skog.... Sliter litt med x og y verdiene

water  = Water(x1=235, samples=samples, n=n_bands, bands=bands)
forest = Forest(y1=50, samples=samples, n=n_bands, bands=bands)

water_mean, water_std  = mean(n=n_bands, samples=samples, bands=water)
forest_mean, forest_std = mean(n=n_bands, samples=samples, bands=forest)

plt.plot(band_list, water_mean, 'b', label='Water')
plt.plot(band_list, forest_mean, 'g', label='Forest')
plt.plot(band_list, agg, 'y', label='Agg..?')
plt.errorbar(band_list, water_mean, yerr=water_std, label='Error', fmt='.', color='black', ecolor='b') #lightgray
plt.errorbar(band_list, forest_mean, yerr=forest_std, label='Error', fmt='.', color='black', ecolor='g')
plt.title("Spectral signatures, mean value for each band \n Year: %s" %yr, fontsize=15)
plt.xlabel("Wavelength [$\\mu$m]", fontsize=15); plt.ylabel("DN", fontsize=15)
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
