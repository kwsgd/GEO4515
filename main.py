# GEO4515 - Project 2020 - Main program
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

# Ikke satt opp ennaa
