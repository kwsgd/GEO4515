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




import rasterio 	as rio
import earthpy.plot as ep

bands_1993 = rio.open('Prosjektdata/1993_tm_oslo.tif').read().astype(float)
bands_2000 = rio.open('Prosjektdata/2000_etm_oslo.tif').read().astype(float)

ep.plot_rgb(bands_2000, rgb=[3,2,1], title="False Color (NIR, red, green) in 2000", stretch=True)
ep.plot_rgb(bands_1993, rgb=[3,2,1], title="False Color (NIR, red, green) in 1993", stretch=True)

ep.plot_rgb(bands_1993, rgb=[2,1,0], title="True Color in 1993", stretch=True)
ep.plot_rgb(bands_2000, rgb=[2,1,0], title="True Color in 2000", stretch=True)




import earthpy.spatial   as es
import numpy             as np
ndvi   = es.normalized_diff(bands[3], bands[2])
titles = ["Normalized Difference Vegetation Index (NDVI)"]

# Turn off bytescale scaling due to float values for NDVI
ep.plot_bands(ndvi, cmap="RdYlGn", cols=1, title=titles, vmin=-1, vmax=1)
ndvi_class_bins    = [-np.inf, 0, 0.25, 0.5, 0.75, np.inf]
ndvi_landsat_class = np.digitize(ndvi, ndvi_class_bins)

# Apply the nodata mask to the newly classified NDVI data
ndvi_landsat_class = np.ma.masked_where(np.ma.getmask(ndvi), ndvi_landsat_class)
np.unique(ndvi_landsat_class)




from sklearn.ensemble  		 import RandomForestClassifier
from sklearn.model_selection import train_test_split

rf = RandomForestClassifier(class_weight=None,
							n_estimators=100,
							criterion='gini',
							max_depth=4, 
							min_samples_split=2,
							min_samples_leaf=1,
							max_features='auto',
							bootstrap=True,
							oob_score=True,
							n_jobs=1,
							random_state=None,
							verbose=True)

rf.fit(X_train, y_train)

prediction = rf.predict(test_image)