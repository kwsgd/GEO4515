import os, shutil, sys, gdal, fiona, scipy

import matplotlib.pyplot as plt
import earthpy.spatial   as es
import earthpy.plot      as ep
import skimage.io 		 as io
import geopandas 		 as gpd
import rasterio          as rio
import earthpy           as et
import numpy      		 as np

from skimage 				 import exposure
from sklearn.model_selection import train_test_split
from sklearn.ensemble        import RandomForestClassifier
from skimage.segmentation    import quickshift, mark_boundaries
from rasterio.plot 			 import reshape_as_raster, reshape_as_image
from matplotlib.colors       import ListedColormap

# https://opensourceoptions.com/blog/python-geographic-object-based-image-analysis-geobia-part-2-image-classification/
# http://patrickgray.me/open-geo-tutorial/chapter_5_classification.html
# ^ can probably use rio.mask.mask to get the pixels instead, would look nicer

original_file = rio.open('Prosjektdata/1993_tm_oslo.tif')
meta_data 	  = original_file.meta

width  = meta_data['width']
height = meta_data['height']

input_raster = original_file.read().astype(float)


landcover_data  = gpd.read_file('Truth_Data.shp'); print(landcover_data)
pixels_data     = gpd.read_file('New_shapefile.shp')

pixels_data.insert(0, column='landcovers', value=landcover_data['landcovers'].values)

Truth_Data = pixels_data
Truth_Data.drop(columns=['geometry'], axis=1, inplace=True)

classes = Truth_Data['landcovers'].unique()

print('The Truth Data include {n} classes: {labels}'.format(n=classes.size, labels=classes))

class_ids = np.arange(classes.size)+1; #print(class_ids)

Truth_Data['id'] = Truth_Data['landcovers'].map(dict(zip(classes, class_ids)))

# Printing the final DataFrame 
print(Truth_Data.loc[130,:])
sys.exit()

# Inputs/features is the 6 landsat bands, and target is the class id:
features = Truth_Data.loc[:, (Truth_Data.columns != 'landcovers') & \
							 (Truth_Data.columns != 'id')].values
							 
target   = Truth_Data.loc[:,  Truth_Data.columns == 'id'].values


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


#X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.33) # random_state=42
#model       = rf.fit(X_train,y_train.ravel())
#predictions = rf.predict(X_test)
#acc_test    = rf.score(X_test, y_test); print(acc_test)  # mean accuracy on test data



model    = rf.fit(features,target)
all_data = reshape_as_image(input_raster)

predictions_all = rf.predict(all_data.reshape(-1, 6))
#np.save('pred_all_.npy', predictions_all) # only run if not created before
predictions_all = np.load('pred_all_.npy')


# reshaping to image size
bilde = predictions_all.reshape(2246, 2245)

# Define color map
nbr_colors = ['blue', 'grey', 'darkgreen']
nbr_cmap   = ListedColormap(nbr_colors)

# Define class names
cat_names = ['water','urban/agriculture','forest']

classes_ids = class_ids.tolist()

# Plot your data
fig, ax = plt.subplots()
im      = ax.imshow(bilde, cmap=nbr_cmap)

ep.draw_legend(im_ax=im, classes=class_ids, titles=cat_names)
ax.set_title("Supervised Classification",fontsize=14)
ax.set_axis_off()

# Auto adjust subplot to fit figure size
plt.tight_layout()
plt.show()