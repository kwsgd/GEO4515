# GEO4515 - Remote Sensing
University of Oslo - UiO
--------------------------

### **Description of Bands:**

Index 0 : Band 1 - Blue  
Index 1 : Band 2 - Green  
Index 2 : Band 3 - Red  
Index 3 : Band 4 - Near Infrared (NIR)  
Index 4 : Band 5 - Short-wave Infrared (SWIR)  
Index 5 : Band 7 - Short-wave Infrared (SWIR)  
---------------------------------------------

### How To Run Script

#### **The Help Message:**

usage: main.py [-h] [-TC] [-FC [FC]] [-SS] [-VI] [-SC] [-CD] {1993,2000}

Landsat Data - Oslo

positional arguments:  
  {1993,2000}           The year of Landsat Data

optional arguments:  
  -h, --help            show this help message and exit  
  -TC, --TrueColor      True Color Plot  
  -FC [FC], --FC [FC]   False Color Plot, default: [r=3,g=2,b=1]  
  -SS, --SpecSignature  Plot Spectral Signatures  
  -VI, --NDVIandClass   Plot NDVI and NDVI classes  
  -SC, --SuperClass     Supervised Classification  
  -CD, --ChangeDetect   Plot Change Detections  
---------------------------------------------------------------

#### **Additional Information:**
* positional argument for year is required (1993 or 2000)
* you should use at least one optional argument
* optional arguments can be mixed and used at the same time
* If year=1993 and --ChangeDetect is called -> data for 2000 is also created
* If year=2000 and --ChangeDetect is called -> data for 1993 is also created
------------------------------------------------------------------

#### **Run Script - Examples:**

**True Color and False Color (with default):**
* python main.py 1993 -TC -FC

**False Color (with other band combinations):**
* python main.py 1993 -FC [5,4,3]
* python main.py 1993 -FC [4,2,1]

**NDVI with NDVI classes:**
* python main.py 1993 --NDVIandClass
* python main.py 1993 -VI

**Supervised Classification of Image (pixel-based):**
* python main.py 1993 --SuperClass
* python main.py 1993 -SC

**Spectral Signatures and Change Detection:**
* python main.py 1993 --SpecSignature -CD
* python main.py 1993 --ChangeDetect -SS