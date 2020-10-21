# https://www.redfoxgis.com/single-post/2015/06/28/How-to-perform-land-cover-classification-using-image-segmentation-in-Python

  
import os, shutil, sys, gdal, fiona, scipy

import matplotlib.pyplot as plt
import earthpy.spatial   as es
import earthpy.plot      as ep
import skimage.io 		 as io
import geopandas 		 as gpd
import rasterio          as rio
import earthpy           as et
import numpy      		 as np

from skimage import exposure
from skimage.segmentation import quickshift, mark_boundaries

# https://opensourceoptions.com/blog/python-geographic-object-based-image-analysis-geobia/

landsat_fn = 'Prosjektdata/1993_tm_oslo.tif'
 
driverTiff = gdal.GetDriverByName('GTiff')
landsat_ds = gdal.Open(landsat_fn)
nbands     = landsat_ds.RasterCount
band_data  = []

print('bands', landsat_ds.RasterCount, 'rows', landsat_ds.RasterYSize, 'columns',
      landsat_ds.RasterXSize)

for i in range(1, nbands+1):
    band = landsat_ds.GetRasterBand(i).ReadAsArray()
    band_data.append(band)
band_data = np.dstack(band_data)


landcover_data  = gpd.read_file('Truth_Data.shp')
pixels_data     = gpd.read_file('New_shapefile.shp')
pixels_data.insert(0, column='landcovers', value=landcover_data['landcovers'].values)
Truth_Data = pixels_data
Truth_Data.drop(columns=['geometry'], axis=1, inplace=True)
classes_ = Truth_Data['landcovers'].unique()
classes = np.arange(classes_.size)+1; #print(class_ids)
Truth_Data['id'] = Truth_Data['landcovers'].map(dict(zip(classes_, classes)))


def segmentation(img):
	#segments = slic(img, n_segments=500000, compactness=0.1)
	segments = quickshift(img, convert2lab=False)

	# save segments to raster
	driverTiff  = gdal.GetDriverByName('GTiff')
	segments_fn = 'segments.tif'
	segments_ds = driverTiff.Create(segments_fn,
									landsat_ds.RasterXSize,
									landsat_ds.RasterYSize,
	                                1,
	                                gdal.GDT_Float32)

	segments_ds.SetGeoTransform(landsat_ds.GetGeoTransform())
	segments_ds.SetProjection(landsat_ds.GetProjectionRef())
	segments_ds.GetRasterBand(1).WriteArray(segments)
	segments_ds = None



def segment_features(segment_pixels):
    features = []
    npixels, nbands = segment_pixels.shape
    for b in range(nbands):
        stats = scipy.stats.describe(segment_pixels[:, b])
        band_stats = list(stats.minmax) + list(stats)[2:]
        if npixels == 1:
            # in this case the variance = nan, change it 0.0
            band_stats[3] = 0.0
        features += band_stats
    return features


def something(img, segments_arr):
	segment_ids = np.unique(segments_arr)
	objects = []
	object_ids = []

	for id in segment_ids:
	    segment_pixels = img[segments_arr == id]
	    object_features = segment_features(segment_pixels)
	    objects.append(object_features)
	    object_ids.append(id)
	return objects, object_ids



img = exposure.rescale_intensity(band_data)  # shape (width, height, channels)

# only run if segments.tif file is not created:
# segmentation(img)


segments_file = gdal.Open('segments.tif')
segments_arr  = segments_file.GetRasterBand(1).ReadAsArray()

# only run if not saved:
#objects, object_ids = something(img, segments_arr)
#objects    = np.array(objects); np.save('objects.npy', objects)
#object_ids = np.array(object_ids); np.save('object_ids.npy', object_ids)

segments   = np.load('objects.npy')
segment_id = np.load('object_ids.npy')

train_img = np.copy(segments)
threshold = train_img.max() + 1

class_label = classes_

segments_per_class = {}

for klass in classes:
    class_label = threshold + klass
    for segment_id in segments_per_class[klass]:
        train_img[train_img == segment_id] = class_label
 
train_img[train_img <= threshold] = 0
train_img[train_img > threshold] -= threshold
 
training_objects = []
training_labels  = []
 
for klass in classes:
    class_train_object = [v for i, v in enumerate(objects) if segment_ids[i] in segments_per_class[klass]]
    training_labels += [klass] * len(class_train_object)
    training_objects += class_train_object
    print('Training objects for class', klass, ':', len(class_train_object))


classifier = RandomForestClassifier(n_jobs=-1)
classifier.fit(training_objects, training_labels)
print('Fitting Random Forest Classifier')
predicted = classifier.predict(segments)
print('Predicting Classifications')
 
clf = np.copy(segments)
for segment_id, klass in zip(segment_ids, predicted):
    clf[clf == segment_id] = klass
 
print('Prediction applied to numpy array')
 
mask = np.sum(img, axis=2)
mask[mask > 0.0] = 1.0
mask[mask == 0.0] = -1.0
clf = np.multiply(clf, mask)
clf[clf < 0] = -9999.0
 
print('Saving classificaiton to raster with gdal')
 
clfds = driverTiff.Create('ObjectBasedClassification.tif',
						  landsat_ds.RasterXSize,
						  landsat_ds.RasterYSize,
                          1,
                          gdal.GDT_Float32)
clfds.SetGeoTransform(landsat_ds.GetGeoTransform())
clfds.SetProjection(landsat_ds.GetProjection())
clfds.GetRasterBand(1).SetNoDataValue(-9999.0)
clfds.GetRasterBand(1).WriteArray(clf)
clfds = None
 
print('Done!')