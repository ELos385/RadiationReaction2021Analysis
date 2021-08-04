# RadiationReaction2021
Analysis for the 2021 radiation reaction experiment 

# Layout
The bulk of the code will be organised as follows: each diagnostic should have its own subdirectory in modules, script, etc. Each diagnostic in the modules directory will have its own object, with associated functions and attributes. The scripts in the modules directory are not intended to be run on their own. They should be called by analysis scripts in the scripts folder. 

Code which can be used across multiple diagnostics should be stored in the lib folder. Subdirectories should be grouped by functionality: for example subdirectory lib/Image_tools would contain a class or function for image processing.

Calibration data or metadata should be stored in /calib/. Subfolders should be used for each diagnostic.

# Installation

If you are cloning the repository for the first time, rename _config.py to config.py and update its contents with your own details.
To access the shared configurations, modules and libs, suggested code for the top of your scripts:
```
import sys
sys.path.append('../../') # this should point to the top level directory
from setup import *
```





