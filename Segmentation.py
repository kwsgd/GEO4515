# https://www.redfoxgis.com/single-post/2015/06/28/How-to-perform-land-cover-classification-using-image-segmentation-in-Python

  
import matplotlib.pyplot as plt
import numpy 			 as np 

from skimage.segmentation import quickshift 
from skimage 			  import io 

import os
import pci
from pci.api import gobs, datasource as ds

pix_file = 'Prosjektdata/1993_tm_oslo.pix'

# open the dataset in write mode
with ds.open_dataset(irvine_pix, ds.eAM_WRITE) as dataset:
    # create a BasicWriter to read and write channels [1,2,3]
    writer = ds.BasicWriter(dataset, [1,2,3])

    # read the entire raster
    raster = writer.read_raster(0, 0, writer.width, writer.height)

    # read the mask from segment 12 matching the dimensions of raster
    mask = dataset.read_mask(12, raster, raster)

    # get the LUTIO for segment 2
    lutio = dataset.get_lut_io(2)

    # read the lut from the segment
    lut = lutio.read_lut()

    # iterate over the pixels, and apply the lut where the mask is valid
    for y in range(raster.height):
        for x in range(raster.width):
            if mask.is_valid(x, y):
                for c in range(raster.chans_count):
                    newval = lut.get_value(raster.get_pixel(x, y, c))
                    raster.set_pixel(x, y, c, int(newval))

    # write the modified raster back to disk
    writer.write_raster(raster)




