This project was developed 
This current README version is from 7/24/20.
Last updated: 8/5/20

Overview: 










MLHelpers.py:
Currently completed with poor performance on . 


Bundle2Helpers.py:



userQuery.py:
Code for an input query from a user. It simply requires a file with available coordinates, coordinates as a string in EPSG:4326, or a shape file, a tif/tiff file, or jp2 file. From there, satellite imagery is gathered, and 


userQueryHelpers.py:





Training Data:
Training data is divided by specific crop. Currently there are 5 directories, however only 2 contain substantial training data, corn and wheat. Each crop directory contains test, train, and validation. train is currently the only non-empty folder. 
The directory is organized into input batches, containing one satellite image (multiple bands) per folder in train, named as "year+2 letter crop label+ crop name +  






