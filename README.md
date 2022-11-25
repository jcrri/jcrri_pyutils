# =============================================================================
# PyWake / TOPFARM external utils
# =============================================================================

Description: Tools for pywake / topfarm that can be imported externally. For
instance, an AEP component that can record as a text file and stores the ite-
ration number.

# =============================================================================
# Required dependencies
# =============================================================================

geotable --> pip install geotable
osgeo/gdal --> conda install -c conda-forge gdal
utm --> pip install utm

# =============================================================================
# Usage example
# =============================================================================

import sys
sys.path.append(r'C:\Users\jcrri\Documents\TOPFARM\jcrri_pyutils')
from klm2bounds import kml2list
