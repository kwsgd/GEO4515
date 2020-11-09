# GEO4515 - Project 2020 - Main program
# -----------------------------------------------------------------------------
import os, gdal, sys, argparse

from matplotlib.colors 	 import ListedColormap
from sklearn.ensemble    import RandomForestClassifier
from rasterio.plot 	     import reshape_as_image
from rasterio.windows 	 import Window

import matplotlib.pyplot as plt
import numpy             as np
import geopandas         as gpd
import rasterio          as rio
import earthpy           as et
import earthpy.spatial   as es
import earthpy.plot      as ep
# -----------------------------------------------------------------------------

def GetDataAndBands(file, x1, y1, x2, y2, n, crop=False):
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
    band = []	# Har det noe aa si om det er array eller liste..????
    # Import data and separate bands
    if crop == True:
        with rio.open(file) as src: # 'Prosjektdata/1993_tm_oslo.tif'
            for i in range(1, n+1):				# Make 6 bands, 1-5 and 7
                band.append(src.read(i, window=Window(x1, y1, x2, y2)).astype(float))
    else:
        with rio.open(file) as src:
            for i in range(1, n+1):				# Make 6 bands, 1-5 and 7
                band.append(src.read(i))

    return band

''' FIKK IKKE TIL DENNE MED -SS, MEN DEN SER BEDRE UT....
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
		returned array has shape (n_bands, y2, x2),
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
'''

def ColorComposition(bands, yr='', r=2, g=1, b=0):
	"""Plot true/false colour composite.

	True = [2, 1, 0] (r, g, b)
	The function plots the result, but does not automatically save the image!
	"""

	if r==2 and g == 1 and b==0:
		ep.plot_rgb(bands, rgb=[r,g,b], figsize=(7,7), title="RGB image (True). R=%s, G=%s, B=%s. Year: %s" %(r+1, g+1, b+1, yr), stretch=True)

	else:
		ep.plot_rgb(bands, rgb=[r,g,b], figsize=(7,7), title="RGB image (False). R=%s, G=%s, B=%s. Year: %s" %(r+1, g+1, b+1, yr), stretch=True)


def CropImg(img, x1, x2, y1, y2):
	"""Function thats crops and image and returns it"""

	crop = img[int(y1):int(y2), int(x1):int(x2)]

	return crop


def Water(x1, samples, n, bands):

    x1      = x1
    water   = [[0] * n for i in range(samples)]

    # Crop out an water area
    for j in range(samples):
        for i in range(n_bands):
            x2 = x1+8
            #water.append(np.mean(func.CropImg(img=bands[i], x1=210, x2=260, y1=415, y2=440)))
            water[j][i] = np.mean(CropImg(img=bands[i], x1=x1, x2=x2, y1=455, y2=490))

            x1 = x2+2
    return water

def Forest(y1, samples, n, bands):

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

    '''
    Vi maa finne noyaktige omraader for samples...!!!
    '''

    agg = []
    # Crop out an agg area
    for i in range(n_bands):
        agg.append(np.mean(CropImg(img=bands[i], x1=566, x2=582, y1=183, y2=203)))
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
    if save:
        plt.savefig("Results/band_list_%s.png" %yr)
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

	# Define color map
	nbr_colors = ['gray', 'y', 'yellowgreen', 'g', 'darkgreen']
	nbr_cmap   = ListedColormap(nbr_colors)

	# Define class names
	ndvi_cat_names = ['No Vegetation','Bare Area','Low Vegetation','Moderate Vegetation','High Vegetation']

	# Get list of classes
	classes = np.unique(ndvi_landsat_class).tolist()

	fig, ax = plt.subplots(figsize=(8, 8))
	im      = ax.imshow(ndvi_landsat_class, cmap=nbr_cmap)

	ep.draw_legend(im_ax=im, classes=classes, titles=ndvi_cat_names)
	ax.set_title('Normalized Difference Vegetation Index (NDVI) Classes. Year: %s' %yr,fontsize=14)
	ax.set_axis_off(); plt.tight_layout()
	plt.show()


def NDI_of_NDVI(ndvi_after, ndvi_before, title='NDI of NDVI', cmap='RdYlGn_r'):
	"""NDI of NDVI = (NDVI_AFTER-NDVI_BEFORE) / (NDVI_AFTER+NDVI_BEFORE)"""

	NDI_of_NDVI = (ndvi_after-ndvi_before)/(ndvi_after+ndvi_before)
	ep.plot_bands(NDI_of_NDVI, cmap=cmap, cols=1, title=title, vmin=-1, vmax=1)


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

	#Ratio(after, before, title='%s ratio [1993 - 2000]' %band_name, cmap=cmap_rat) only defined cmap_rat for red

	if ndi_of_ndvi:
		# suppress RuntimeWarning (probably because of NaN)
		np.seterr(divide='ignore', invalid='ignore')
		ndvi_before = es.normalized_diff(bands_1993[3], bands_1993[2])
		ndvi_after  = es.normalized_diff(bands_2000[3], bands_2000[2])
		NDI_of_NDVI(ndvi_after, ndvi_before, title='NDI of NDVI', cmap='RdYlGn_r')


def SupervisedClassification(raster_arr, width, height):
	"""Supervised Classification of Image"""

	landcover_data  = gpd.read_file('Truth_Data.shp')     # dataframe containing class names
	pixels_data     = gpd.read_file('New_shapefile.shp')  # dataframe containing pixel values

	pixels_data.insert(0, column='landcovers', value=landcover_data['landcovers'].values)

	Truth_Data = pixels_data
	Truth_Data.drop(columns=['geometry'], axis=1, inplace=True)

	# created integer classes (ids) for classification
	classes   = Truth_Data['landcovers'].unique()
	class_ids = np.arange(classes.size)+1

	Truth_Data['id'] = Truth_Data['landcovers'].map(dict(zip(classes, class_ids)))

	print('\nThe Truth Data include {n} classes: {labels}\n'.format(n=classes.size, labels=classes))
	print(Truth_Data)

	# Inputs/features is the 6 landsat bands, and target is the class id:
	features = Truth_Data.loc[:, (Truth_Data.columns != 'landcovers') & \
								 (Truth_Data.columns != 'id')].values

	target   = Truth_Data.loc[:,  Truth_Data.columns == 'id'].values

	# reshape array to image:
	image = reshape_as_image(raster_arr)

	# build model and make predict image:
	rf       = RandomForestClassifier(n_estimators=100,max_depth=4,oob_score=True,n_jobs=1,verbose=True)
	model    = rf.fit(features,target.ravel())
	pred_im  = rf.predict(image.reshape(-1, 6))

	# reshaping prediction to image size:
	bilde = pred_im.reshape(height, width)

	# define color map:
	nbr_colors = ['blue', 'grey', 'darkgreen']
	nbr_cmap   = ListedColormap(nbr_colors)

	# define class names:
	cat_names   = ['water','urban/agriculture','forest']
	classes_ids = class_ids.tolist()

	# plotting the predicted classes:
	fig, ax = plt.subplots()
	im      = ax.imshow(bilde, cmap=nbr_cmap)

	ep.draw_legend(im_ax=im, classes=class_ids, titles=cat_names)
	ax.set_title("Supervised Classification",fontsize=14)
	ax.set_axis_off(); plt.tight_layout()
	plt.show()

	return Truth_Data, classes




if __name__ == '__main__':

    # Examples of running:

    # plot false color with other than default:
    #python main.py 1993 -FC [5,4,3]
    #python main.py 1993 -FC [4,2,1]

    # plot true and false color:
    #python main.py 1993 -TC -FC
    #python main.py 1993 -TC -FC [4,2,1]

    # Spectral Signatures and Change Detection:
    #python main.py 1993 --SpecSignatures -CD

    parser = argparse.ArgumentParser(description='Landsat Data - Oslo')

    # positional argument for year (must be included):
    parser.add_argument('year', type=int, choices=[1993, 2000], help='The year of Landsat Data')

    # Optional arguments:
    parser.add_argument('-TC', '--TrueColor',  action='store_true', help='True Color Plot')
    parser.add_argument('-FC', '--FalseColor', nargs='?', const='[4,3,2]', help='False Color Plot')

    parser.add_argument('-SS',   '--SpecSignatures', action='store_true', help='Plot Spectral Signatures')
    parser.add_argument('-NDVI', '--NDVIandClass',   action='store_true', help='Plot NDVI and NDVI classes')

    parser.add_argument('-SC', '--Classification', action='store_true', help='Supervised Classification')
    parser.add_argument('-CD', '--ChangeDetect',   action='store_true', help='Plot ChangeDetections')


    if len(sys.argv) < 2:
        sys.argv.append('--help')
    elif len(sys.argv) == 2:
        print('\nplease choose at least one positional arguments'); sys.exit()


    args           = parser.parse_args()
    yr 			   = args.year
    TrueColor 	   = args.TrueColor
    FalseColor     = args.FalseColor
    ChangeDetect   = args.ChangeDetect
    SpecSignatures = args.SpecSignatures
    NDVI_and_Class = args.NDVIandClass
    Classification = args.Classification


    print('\n{heading} {year}'.format(heading=parser.description, year=yr))

    if yr == 1993:
        file = 'Prosjektdata/1993_tm_oslo.tif'
    elif yr == 2000:
        file = 'Prosjektdata/2000_etm_oslo.tif'

    band_names  = ['Band 1','Band 2','Band 3','Band 4','Band 5','Band 7']
    #array_bands = GetDataAndBands(file=file, crop=True, x1=70, x2=700, y1=850, y2=1000) # similar to x1=70, x2=770, y1=850, y2=1850 in CropImg
    bands = GetDataAndBands(file=file, x1=30, y1=800, x2=800, y2=500, n=6, crop=True)
    # All bands in a stack. Band 1, 2, 3, 4, 5 and 7
    array_bands = np.stack([bands[0],bands[1],bands[2],bands[3],bands[4],bands[5]])

    # if deafult crop, shape = (6, 1000, 700)
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
        #r, g, b    = FalseColor[0], FalseColor[1], FalseColor[2]
        #ColorComposition(array_bands, yr=yr, r=int(r), g=int(g), b=int(b))
        ColorComposition(array_bands, yr=yr, r=3, g=2, b=1)


    if SpecSignatures:
        print('\n{text} {year}'.format(text="Spectral Signatures", year=yr))
        SpectralSignatures(bands, band_names, n_bands, year=yr, save=True)


    if NDVI_and_Class:
        print('\n{text} {year}'.format(text="NDVI and NDVI classes", year=yr))
        NDVI(array_bands, yr)


    if Classification:
        print('\n{text} {year}'.format(text="Pixel-based Supervised Classification", year=yr))
        TruthData, classes = SupervisedClassification(array_bands, width, height)


    if ChangeDetect:
        print('\n{text}'.format(text="Change detection"))

        if yr == 1993:
            file_2000   = 'Prosjektdata/2000_etm_oslo.tif'
            bands_2000  = GetDataAndBands(file=file_2000, x1=30, y1=800, x2=800, y2=500, n=6, crop=True)
            bands_1993  = array_bands
            ChangeDetection(bands_1993, bands_2000, band_name='SWIR', ndi_of_ndvi=True)

        else:

            file_1993   = 'Prosjektdata/1993_tm_oslo.tif'
            bands_1993  = GetDataAndBands(file=file_1993, x1=30, y1=800, x2=800, y2=500, n=6, crop=True)
            bands_2000  = array_bands
            ChangeDetection(bands_1993, bands_2000, band_name='SWIR', ndi_of_ndvi=True)























    # Spørsmål til oppgavene:
    # 1) Holder det med vann og skog på spectral signatures?
    # 2) Noe spesielt vi skal prøve å få frem på false color?
    # 3) Hva er det rød på NDVI?
    # 4) Kan det programmet (Focus) eksportere training data?
    # 5) Change detection: skal vi ha plot for alle bandsa?

    # NDWI: how to map water bodies?
    # - NDWI uses green and near infrared bands to highlight water bodies


    '''
    # Setting up directory paths, good idea or not?
    rootdir   = 'GEO4515/'      # kanskje unødvendig, men ved 'bakoverflytting' trengs det kanskje?

    DataDir   = 'ProsjektData/' # Path to all data used, eventuelt bare kalle 'Data'
                                # adda 'LabeledTrainingData' - hååååper å klare og lage dataene snart..
                                # føler jeg kommer stadig nærmere as... men ja, mye nytt jeg ikke helt skjønner

    ResultDir = 'Results/'      # Path to all results, with folders for each task maybe?

    CodesDir  = 'Codes/'        # alle python scrips inn her etterhvert kanskje?
                                # Må sikkert fikse lit på programmene først uten å fucka opp litt
                                # from CodesDir import functions.py as F
    '''

    '''
    # random example function in function.py

    def ExampleFunc(bands, r=3, g=2, b=1, infolder=None, getfile=None, outfolder=None, newfile=None, save=False):
        """
        # Spørs jo veldig på funksjonen da,
        # noen ganger bedre å skrive filnavn i funksjonen også,
        # særlig outputs kanskje om man skal ha flere
        # forutsetter vel også litt if tests, print statements om ikke filnavn er gitt osv osv...
        """

        SelectATiff = infolder+getfile
        # en tilfeldig eksempel funksjon
        ColorComposition(bands, r=2, g=1, b=0):

        SelectATiff = infolder+getfile

        if SaveResult == True
        if newfile==None:
            outfile = 'plot_r[%g]_g[%g]_b[%g].png' %(r, g, b) # husker helt med % på string xD
            SaveRes = outfolder+outfile   # saved in ResultDir/SpectralSignatures/

        # hm, noe sånt ish hvis tr
        # elif newfile==str:
    '''

    '''
    # in main.py:
    import function.py as F
    # example for running task1
    outfile = 'experimentplot'     [%g]_g[%g]_b[%g].png' %(r, g, b)
    ExampleFunc(r=3, g=2, b=1,
				infolder=Datadir,
				getfile='1993_tm_oslo.tif')
				outfolder=ResultDir+'SpectralSignatures/''
				outfile='JustTriedSomething'
    '''
