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

# Setting up directory paths, good idea or not?

rootdir   = 'GEO4515/'      # kanskje unødvendig, men ved 'bakoverflytting' trengs det kanskje?

DataDir   = 'ProsjektData/' # Path to all data used, eventuelt bare kalle 'Data'
							# adda 'LabeledTrainingData' - hååååper å klare og lage dataene snart..
							# føler jeg kommer stadig nærmere as... men ja, mye nytt jeg ikke helt skjønner

ResultDir = 'Results/'      # Path to all results, with folders for each task maybe?

CodesDir  = 'Codes/'        # alle python scrips inn her etterhvert kanskje? 
							# Må sikkert fikse lit på programmene først uten å fucka opp litt 
						    # from CodesDir import functions.py as F


























# overtrøtt leking med generelle paths for en meningsløsfunksjon xD
# men tenkte noen genrelle paths, og filename som input argumenter i funksjoner osv
# meeen må nesten se an om/hvis det blir fornuftig/oversiktlig etterhvert eller ikke xD
# feel absolutt feel free to edit, det jeg har skrevet 
# bare å flytte langt ned om du skal jobbe med noe her altså xD

'''
random example function in function.py

def ExampleFunc(bands, r=3, g=2, b=1, infolder=None, getfile=None, outfolder=None, newfile=None, save=False):
	'''
	# Spørs jo veldig på funksjonen da,
	# noen ganger bedre å skrive filnavn i funksjonen også,
	# særlig outputs kanskje om man skal ha flere
	# forutsetter vel også litt if tests, print statements om ikke filnavn er gitt osv osv...
	'''
	
	SelectATiff = infolder+getfile
	# en tilfeldig eksempel funksjon
	ColorComposition(bands, r=2, g=1, b=0):

	SelectATiff = infolder+getfile

	if SaveResult == True
	if newfile==None:
			outfile = 'plot_r[%g]_g[%g]_b[%g].png' %(r, g, b) # husker helt med % på string xD
			SaveRes = outfolder+outfile   # saved in ResultDir/SpectralSignatures/
	
	# hm, noe sånt ish hvis tr
	elif newfile==str: 

# in main.py:
import function.py as F
# example for running task1 
outfile = 'experimentplot'     [%g]_g[%g]_b[%g].png' %(r, g, b) # husker helt med % på string xD
ExampleFunc(r=3, g=2, b=1,
			infolder=Datadir,
			getfile='1993_tm_oslo.tif')
			outfolder=ResultDir+'SpectralSignatures/''
			outfile='JustTriedSomething'
'''

