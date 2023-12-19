## Failure Detection ## 
### Purpose ###
This script aims to detect faulty areas in the print-job and colour-code them based on severity:
- obvious cracks and marks are red
- areas with miniature scratches that are more than 14 degrees from the vertical are yellow
- rest is green

The script aims to return the full area of the body, the percentage error of red, and the percentage error of yellow. 

### If you don't already have a way to run python you can install Anaconda + Spyder: ###
  1. Download anaconda from this url: https://www.anaconda.com/download/
  2. If you are on a work computer, you must open the terminal and type: conda config –-set ssl_verify "crt"
     (get the crt from IT)
  3. If your anaconda navigator doesn't have spyder, type this into terminal: conda install -c anaconda spyder 

### Need to install opencv and scipy on python to use: ###
- for pip:
  - pip install opencv-python
  - python -m pip install scipy
 
- for conda (on anaconda + spyder):
  - conda install -c conda-forge
  - conda install scipy

### How to Use: ###
- change imgDir, outDir, excelDir, and scale
- run
