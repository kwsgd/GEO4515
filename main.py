# GEO4515 - Project 2020
# -----------------------------------------------------------------------------
import os, gdal, sys, argparse

from sklearn.model_selection    import train_test_split
from matplotlib.colors 	        import ListedColormap
from sklearn.ensemble           import RandomForestClassifier
from rasterio.plot 	            import reshape_as_image
from rasterio.windows 	        import Window

import matplotlib.pyplot as plt
import numpy             as np
import geopandas         as gpd
import rasterio          as rio
import earthpy           as et
import earthpy.spatial   as es
import earthpy.plot      as ep
import pandas            as pd
# -----------------------------------------------------------------------------

def GetDataAndBands_2(file, x1, y1, x2, y2, n, crop=False):
    """Create array with the pixel values for each band

	file : the entire .tif-image
	crop : chosen subset of file given by (x1, x2, y1, y2)

	Description of Window-method:
		Window(col_off, row_off, width, height)
			-- col_off (x1) : x-position of original used as start (x=0) in window
			-- row_ff  (y1) : y-position of original used as start (y=0) in window
			-- width   (x2) : the desired width/length  of window/arr in x-direction
			-- height  (y2) : the desired height/length of window/arr in y-direction

	if crop :
		returned array has shape (n_bands, y2, x2),
	"""
    band = []
    # Import data and separate bands
    if crop == True:
        with rio.open(file) as src:
            for i in range(1, n+1):		# For 6 bands, 1-5 and 7
                band.append(src.read(i, window=Window(x1, y1, x2, y2)).astype(float))
    else:
        with rio.open(file) as src:
            for i in range(1, n+1):
                band.append(src.read(i))

    return band


def GetDataAndBands(file, x1=70, x2=700, y1=850, y2=1000, crop=False):
	"""Create array with the pixel values for each band

	file : the entire .tif-image
	crop : chosen subset of file given by (x1, x2, y1, y2)

	Description of Window-method:
		Window(col_off, row_off, width, height)
			-- col_off (x1) : x-position of original used as start (x=0) in window
			-- row_ff  (y1) : y-position of original used as start (y=0) in window
			-- width   (x2) : the desired width/length  of window/arr in x-direction
			-- height  (y2) : the desired height/length of window/arr in y-direction

	if crop :
		returned array has shape (n_bands, y2, x2)
	"""

	# Opening the original file:
	original_file = rio.open(file)

	if crop:
		window     = Window(col_off=x1, row_off=y1, width=x2, height=y2)
		raster_arr = original_file.read(window=window).astype(float)
	else:
		raster_arr = original_file.read().astype(float)

	original_file.close()

	return raster_arr

def ColorComposition(bands, yr='', r=2, g=1, b=0):
	"""Plot true/false colour composite.

	True = [2, 1, 0] (r, g, b)
	The function plots the result, but does not automatically save the image!
	"""

	if r==2 and g == 1 and b==0:
		ep.plot_rgb(bands, rgb=[r,g,b],figsize=(7,7),\
			title="RGB image (True). R=%s, G=%s, B=%s. Year: %s" %(r+1, g+1, b+1, yr), stretch=True)

	else:
		ep.plot_rgb(bands, rgb=[r,g,b], figsize=(7,7),\
			title="RGB image (False). R=%s, G=%s, B=%s. Year: %s" %(r+1, g+1, b+1, yr), stretch=True)


def CropImg(img, x1, x2, y1, y2):
	"""Function thats crops an image and returns it"""

	crop = img[int(y1):int(y2), int(x1):int(x2)]
	return crop


def Water(x1, samples, n, bands):
    """Water Samples"""

    x1      = x1
    water   = [[0] * n for i in range(samples)]

    # Crop out an water area
    for j in range(samples):
        for i in range(n_bands):
            x2 = x1+8
            water[j][i] = np.mean(CropImg(img=bands[i], x1=x1, x2=x2, y1=455, y2=490))

            x1 = x2+2
    return water

def Forest(y1, samples, n, bands):
    """Forest Samples"""

    y1       = y1
    forest   = [[0] * n_bands for i in range(samples)]

    # Crop out an forest area
    for j in range(samples):
        for i in range(n_bands):
            y2 = y1+8
            forest[j][i] = np.mean(CropImg(img=bands[i], x1=10, x2=40, y1=y1, y2=y2))

            y1 = y2+2
    return forest

def mean(n, samples, bands):
    """ Returns mean and std """

    mean = []; std  = []

    for i in range(n):
        mean_ = []
        for j in range(samples):
            mean_.append(bands[j][i])

        mean.append(np.mean(mean_))
        std.append(np.std(mean_))
    return mean, std


def SpectralSignatures(bands, band_list, n_bands, year='', save=False):
    """ Create lists of the mean value for landcover samples"""

    samples = 5

    agg = []
    # Crop out an agg area
    for i in range(n_bands):
        agg.append(np.mean(CropImg(img=bands[i], x1=566, x2=582, y1=183, y2=203)))

    water  = Water(x1=235, samples=samples, n=n_bands, bands=bands)
    forest = Forest(y1=50, samples=samples, n=n_bands, bands=bands)

    water_mean, water_std  = mean(n=n_bands, samples=samples, bands=water)
    forest_mean, forest_std = mean(n=n_bands, samples=samples, bands=forest)

    plt.plot(band_list, water_mean, 'b', label='Water')
    plt.plot(band_list, forest_mean, 'g', label='Forest')
    plt.plot(band_list, agg, 'y', label='Agg..?')
    plt.errorbar(band_list, water_mean, yerr=water_std, label='Error', fmt='.', color='black', ecolor='b') #lightgray
    plt.errorbar(band_list, forest_mean, yerr=forest_std, label='Error', fmt='.', color='black', ecolor='g')
    plt.title("Spectral signatures, mean value for each band \n Year: %s" %year, fontsize=15)
    plt.xlabel("Wavelength [$\\mu$m]", fontsize=15); plt.ylabel("DN", fontsize=15)
    plt.legend(fontsize=15)
    if save:
        plt.savefig("Results/band_list_%s.png" %year)
    plt.show()

def SpectralSignatures2(band_list, year, save=False):
    """ Spectral Signatures with Band Pixel Values from GGis"""

    if year == '1993':
        samples_1993 =  gpd.read_file('shapefiles_1993/merged.shp')
        dictionary   = {'ClassNames':samples_1993['landcovers'],\
                        'Band1':samples_1993['1993_tm__1'],\
                        'Band2':samples_1993['1993_tm__2'],\
                        'Band3':samples_1993['1993_tm__3'],\
                        'Band4':samples_1993['1993_tm__4'],\
                        'Band5':samples_1993['1993_tm__5'],\
                        'Band7':samples_1993['1993_tm__6']}
    elif year == '2000':
        samples_2000 =  gpd.read_file('shapefiles_2000/merged.shp')
        dictionary   = {'ClassNames':samples_2000['landcovers'],\
                        'Band1':samples_2000['2000_etm_1'],\
                        'Band2':samples_2000['2000_etm_2'],\
                        'Band3':samples_2000['2000_etm_3'],\
                        'Band4':samples_2000['2000_etm_4'],\
                        'Band5':samples_2000['2000_etm_5'],\
                        'Band7':samples_2000['2000_etm_6']}
    else:
    	print('Data not available for this year')

    df = pd.DataFrame(dictionary)
    df.set_index(['ClassNames'], inplace=True)

    water_data       = df[df.index == 'water'].copy()
    forest_data      = df[df.index == 'forest'].copy()
    urban_data       = df[df.index == 'urban'].copy()
    agriculture_data = df[df.index == 'agriculture'].copy()
    mean_water       = water_data.mean().to_numpy()
    mean_forest      = forest_data.mean().to_numpy()
    mean_urban       = urban_data.mean().to_numpy()
    mean_agriculture = agriculture_data.mean().to_numpy()
    std_water        = water_data.std().to_numpy()
    std_forest       = forest_data.std().to_numpy()
    std_urban        = urban_data.std().to_numpy()
    std_agriculture  = agriculture_data.std().to_numpy()


    plt.subplots(figsize=(10,6))
    plt.plot(band_list, mean_water, 'b', label='Water')
    plt.plot(band_list, mean_forest, 'g', label='Forest')
    plt.plot(band_list, mean_agriculture, 'y', label='Agriculture')
    plt.plot(band_list, mean_urban, 'm', label='Urban')
    plt.errorbar(band_list, mean_water, yerr=std_water, label='Error', fmt='.', color='black', ecolor='b') #lightgray
    plt.errorbar(band_list, mean_forest, yerr=std_water, label='Error', fmt='.', color='black', ecolor='g')
    plt.errorbar(band_list, mean_agriculture, yerr=std_agriculture, label='Error', fmt='.', color='black', ecolor='y')
    plt.errorbar(band_list, mean_urban, yerr=std_urban, label='Error', fmt='.', color='black', ecolor='m')
    plt.title("Spectral signatures, mean value for each band \n Year: %s" %year, fontsize=15)
    plt.xlabel("Wavelength [$\\mu$m]", fontsize=15);plt.ylabel("DN", fontsize=15)
    plt.xticks(fontsize=13);plt.yticks(fontsize=13)
    plt.legend(bbox_to_anchor=(1,0.5), loc='center left', fontsize=15)
    plt.tight_layout()
    if save:
    	plt.savefig('Results/band_list_QGis_%s.png' %year)
    plt.show()


def NDVI(bands, year):
	"""Normalized Difference Vegetation Index (NDVI)"""

	#ndvi = es.normalized_diff(band4, band3)      # (band4-band3)/(band4+band3)
	ndvi = es.normalized_diff(bands[3], bands[2]) # (band4-band3)/(band4+band3)

	titles = ['Normalized Difference Vegetation Index (NDVI) Year: %s' %year]

	# Turn off bytescale scaling due to float values for NDVI
	ep.plot_bands(ndvi, cmap="RdYlGn", cols=1, title=titles, vmin=-1, vmax=1)

	# Create classes and apply to NDVI results
	ndvi_class_bins    = [-np.inf, 0, 0.25, 0.5, 0.75, np.inf]
	ndvi_landsat_class = np.digitize(ndvi, ndvi_class_bins)

	# Define color map and class names
	ndvi_colors = ListedColormap(['gray', 'y', 'yellowgreen', 'g', 'darkgreen'])
	ndvi_names = ['0.00 No Vegetation','0.25 Bare Area','0.50 Low Vegetation','0.75 Moderate Vegetation','1.00 High Vegetation']

	# Get list of classes
	classes = np.unique(ndvi_landsat_class).tolist()

	fig, ax = plt.subplots(figsize=(8, 8))
	im      = ax.imshow(ndvi_landsat_class, cmap=ndvi_colors)

	ep.draw_legend(im_ax=im, classes=classes, titles=ndvi_names)
	ax.set_title('Normalized Difference Vegetation Index (NDVI) Classes. \nYear: %s' %yr,fontsize=14)
	ax.set_axis_off(); plt.tight_layout()
	plt.show()


def NDI_of_NDVI(ndvi_after, ndvi_before, title='NDI of NDVI', cmap='RdYlGn_r'):
	"""NDI of NDVI = (NDVI_AFTER-NDVI_BEFORE) / (NDVI_AFTER+NDVI_BEFORE)"""

	NDI_of_NDVI = (ndvi_after-ndvi_before)/(ndvi_after+ndvi_before)
	ep.plot_bands(NDI_of_NDVI, cmap=cmap, cols=1, title=title, vmin=-1, vmax=1)

def NDI(after, before, title='', cmap=''):
	"""NDI = (AFTER - BEFORE)/(AFTER + BEFORE)"""

	NDI = (after - before)/(after + before)
	ep.plot_bands(NDI, cmap=cmap, title=title)


def Difference(after, before, title='', cmap=''):
	"""DIFFERENCE = (AFTER - BEFORE)"""

	difference = (after - before)
	ep.plot_bands(difference, cmap=cmap, title=title)


def Ratio(after, before, title='', cmap=''):
	"""RATIO = (AFTER / BEFORE)"""

	ratio = after/before
	ep.plot_bands(ratio, cmap=cmap, title=title)


def ChangeDetection(bands_1993, bands_2000, band_name='SWIR', ndi_of_ndvi=False):
    """ Calculates And Plot Change Detections.

    Band 1 : Blue
			 - Bathymetric mapping, distinguishing soil from vegetation and deciduous from coniferous vegetation

	Band 2 : Green
			 - Emphasizes peak vegetation, which is useful for assessing plant vigor

	Band 3 : Red
			 -  Discriminates vegetation slopes

	Band 4 : Near Infrared (NIR)
			 -  Emphasizes biomass content and shorelines

	Band 5 : Short-wave Infrared (SWIR)
		     - Discriminates moisture content of soil and vegetation (penetrates thin clouds)
	"""

    if band_name == 'SWIR':

        before, after = bands_1993[4], bands_2000[4]
        cmap_diff     = 'seismic_r'
        cmap_rat 	  = 'seismic'

    elif band_name == 'NIR':

        before, after = bands_1993[3], bands_2000[3]
        cmap_diff     = 'Greens'

    elif band_name == 'Red':

        before, after = bands_1993[2], bands_2000[2]
        cmap_diff     = 'RdGy_r'
        cmap_rat 	  = 'seismic'

    elif band_name == 'Green':

        before, after = bands_1993[1], bands_2000[1]
        cmap_diff     = 'YlGn'

    elif band_name == 'Blue':

        before, after = bands_1993[0], bands_2000[0]
        cmap_diff     = 'PuOr'

    else:
        print("Invalid band_number, use one from ['Blue', 'Green', 'Red', 'NIR', 'SWIR']")

    Difference(after, before, title='%s difference [1993 - 2000]' %band_name, cmap=cmap_diff)
    Ratio(after, before,      title='%s ratio [1993 - 2000]' %band_name, cmap=cmap_rat) #only defined cmap_rat for red
    NDI(after, before,        title='%s NDI [1993 - 2000]' %band_name, cmap=cmap_diff)

    if ndi_of_ndvi:
        # suppress RuntimeWarning (probably because of NaN)
        np.seterr(divide='ignore', invalid='ignore')
        ndvi_before = es.normalized_diff(bands_1993[3], bands_1993[2])
        ndvi_after  = es.normalized_diff(bands_2000[3], bands_2000[2])
        NDI_of_NDVI(ndvi_after, ndvi_before, title='NDI of NDVI', cmap='RdYlGn_r')


def SupervisedClassification(raster_arr, width, height, ColorTemplate, ClassTemplate, year=1993):
	"""Pixel-based Supervised Classification of Image"""

	if year == 1993:
		#path_class = 'shapefiles_1993/Truth_Data.shp'        # shapefile of class names
		#path_pix   = 'shapefiles_1993/New_shapefile.shp'     # shapefile of pixel values
		path_class = 'shapefiles_1993/PointSamples.shp'       # shapefile of class names
		path_pix   = 'shapefiles_1993/4classesBandPixels.shp' # shapefile of pixel values
		weights = {1:1, 2:0.5, 3:3, 4:1}
	elif year == 2000:
		#path_class = 'shapefiles_2000/TrainingData.shp'      # shapefile of class names
		#path_pix   = 'shapefiles_2000/BandPixels.shp'        # shapefile of pixel values
		path_class = 'shapefiles_2000/PointSamples.shp'       # shapefile of class names
		path_pix   = 'shapefiles_2000/4classesBandPixels.shp' # shapefile of pixel values
		weights = {1:1, 2:0.3, 3:3, 4:2}
	else:
		print('Training Data (shapefiles) for this year is not available');sys.exit()

	landcover_data = gpd.read_file(path_class)  # dataframe containing class names
	pixels_data    = gpd.read_file(path_pix)    # dataframe containing pixel values

	pixels_data.insert(0, column='landcovers', value=landcover_data['landcovers'].values)
	pixels_data.drop(columns=['geometry'], axis=1, inplace=True)

	Truth_Data  = pixels_data.copy()
	classes     = list(Truth_Data['landcovers'].unique()) # unique classes in dataframe
	class_ids   = list(np.arange(1, len(classes)+1))      # integer classes (ids) for classification
	MatchNameID = dict(zip(classes, class_ids))           # create a dictionary of (names,ids)

	if MatchNameID != ClassTemplate:
		# if MatchNamesID has all expected keys, update to correct (names,ids)
		if np.all([key in MatchNameID.keys() for key in ClassTemplate.keys()]) == True:
			MatchNameID.update(ClassTemplate)
		else:
			print('missing expected landcover names');sys.exit()

	Truth_Data['id'] = Truth_Data['landcovers'].map(MatchNameID)

	# the final dataframe containing the Training/Truth Data:
	print('\nThe Truth Data include {n} classes: {labels}\n'.format(n=len(classes), labels=classes))
	print(Truth_Data)

	# inputs/features is the 6 landsat bands, and target is the class id:
	features = Truth_Data.loc[:,(Truth_Data.columns!='landcovers')&(Truth_Data.columns!='id')].values
	target   = Truth_Data.loc[:,Truth_Data.columns=='id'].values

	X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.33, random_state=4515)

	# build model and predict image:
	rf       = RandomForestClassifier(class_weight=weights,n_estimators=100,max_depth=4,oob_score=True,verbose=True,random_state=4515)
	model    = rf.fit(features,target.ravel())
	print(model.feature_importances_)

	#model    = rf.fit(X_train,y_train.ravel())
	#print(model.feature_importances_)
	#print(rf.score(X_test, y_test.ravel()))
	#sys.exit()

	in_im    = reshape_as_image(raster_arr)      # reshape as image
	pred_im  = rf.predict(in_im.reshape(-1, 6))
	out_im   = pred_im.reshape(height, width)    # reshaping to image size

	# define color map and class names:
	colors  = ListedColormap([color for color in ColorTemplate.values()])
	names   = [name for name in ClassTemplate.keys()]

	# plotting the predicted classes:
	fig, ax = plt.subplots()
	plot_im = ax.imshow(out_im, cmap=colors)
	ep.draw_legend(im_ax=plot_im, classes=class_ids, titles=names)
	ax.set_title('Supervised Classification - %s'%year, fontsize=14)
	ax.set_axis_off();plt.tight_layout();plt.show()
	return Truth_Data, classes


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Landsat Data - Oslo')

    # positional argument for year (must be included):
    parser.add_argument('year', type=int, choices=[1993, 2000], help='The year of Landsat Data')

    # Optional arguments:
    parser.add_argument('-TC', '--TrueColor', action='store_true', help='True Color Plot')
    parser.add_argument('-FC', '--FC', nargs='?', const='[3,2,1]', help='False Color Plot, default: [r=3,g=2,b=1]')
    parser.add_argument('-SS', '--SpecSignature', action='store_true', help='Plot Spectral Signatures')
    parser.add_argument('-VI', '--NDVIandClass', action='store_true', help='Plot NDVI and NDVI classes')
    parser.add_argument('-SC', '--SuperClass', action='store_true', help='Supervised Classification')
    parser.add_argument('-CD', '--ChangeDetect', action='store_true', help='Plot Change Detections')

    if len(sys.argv) < 2:
        sys.argv.append('--help')
    elif len(sys.argv) == 2:
        print('\nplease choose at least one positional arguments'); sys.exit()

    args           = parser.parse_args()
    yr 			   = args.year
    TrueColor 	   = args.TrueColor
    FalseColor     = args.FC
    ChangeDetect   = args.ChangeDetect
    SpecSignatures = args.SpecSignature
    NDVI_and_Class = args.NDVIandClass
    Classification = args.SuperClass

    print('\n{heading} {year}'.format(heading=parser.description, year=yr));print('-'*24)

    if yr == 1993:
        file = 'Prosjektdata/1993_tm_oslo.tif'
    elif yr == 2000:
        file = 'Prosjektdata/2000_etm_oslo.tif'

    band_names  = ['Band 1','Band 2','Band 3','Band 4','Band 5','Band 7']
    array_bands = GetDataAndBands(file=file, crop=True, x1=30, x2=800, y1=800, y2=500) # [x1=30, x2=830, y1=800, y2=1300] in CropImg

    #bands = GetDataAndBands_2(file=file, x1=30, y1=800, x2=800, y2=500, n=6, crop=True)
    # All bands in a stack. Band 1, 2, 3, 4, 5 and 7
    #array_bands = np.stack([bands[0],bands[1],bands[2],bands[3],bands[4],bands[5]])

    # shape of crop: (6, 500, 800)
    n_bands = array_bands.shape[0]
    height  = array_bands.shape[1]
    width   = array_bands.shape[2]

# -----------------------------------------------------------------------------

    if TrueColor:
        print('\n{text} {year}'.format(text="'True' color composition", year=yr))
        ColorComposition(array_bands, yr=yr, r=2, g=1, b=0)


    if FalseColor:
        print('\n{text} {year} with [r,g,b] = {rgb}'.format(text="'False' color composition", year=yr, rgb=FalseColor))
        FalseColor = FalseColor.strip('][').split(',')
        r, g, b    = FalseColor[0], FalseColor[1], FalseColor[2]
        ColorComposition(array_bands, yr=yr, r=int(r), g=int(g), b=int(b))


    if SpecSignatures:
        print('\n{text} {year}'.format(text="Spectral Signatures", year=yr))

        SpectralSignatures2(band_names, year=str(yr), save=True)


    if NDVI_and_Class:
        print('\n{text} {year}'.format(text="NDVI and NDVI classes", year=yr))
        NDVI(array_bands, yr)


    if Classification:
        print('\n{text} {year}'.format(text="Pixel-based Supervised Classification", year=yr))
        #CT  = {'water':'blue', 'urban':'grey', 'forest':'darkgreen'}
        #CT  = {'water':'tab:blue', 'urban':'tab:grey', 'forest':'tab:green'}
        CT  = {'water':'darkslategray','urban':'bisque','forest':'darkolivegreen','agriculture':'goldenrod'}
        IT  = {'water':1,'urban':2,'forest':3,'agriculture':4}
        TruthData, classes = SupervisedClassification(array_bands, width, height, CT, IT, year=yr)


    if ChangeDetect:
        print('\n{text}'.format(text="Change detection"))

        if yr == 1993:
            file_2000   = 'Prosjektdata/2000_etm_oslo.tif'
            bands_2000  = GetDataAndBands(file=file_2000, crop=True, x1=30, x2=800, y1=800, y2=500)
            bands_1993  = array_bands
            ChangeDetection(bands_1993, bands_2000, band_name='SWIR', ndi_of_ndvi=True)

        else:
            file_1993   = 'Prosjektdata/1993_tm_oslo.tif'
            bands_1993  = GetDataAndBands(file=file_1993, crop=True, x1=30, x2=800, y1=800, y2=500)
            bands_2000  = array_bands
            ChangeDetection(bands_1993, bands_2000, band_name='SWIR', ndi_of_ndvi=True)
