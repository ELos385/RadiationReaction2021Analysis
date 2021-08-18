# RadiationReaction2021
Analysis for the 2021 radiation reaction experiment 

# Layout
The bulk of the code will be organised as follows: each diagnostic should have its own subdirectory in modules, script, etc. Each diagnostic in the modules directory will have its own object, with associated functions and attributes. The scripts in the modules directory are not intended to be run on their own. They should be called by analysis scripts in the scripts folder. 

Code which can be used across multiple diagnostics should be stored in the lib folder. Subdirectories should be grouped by functionality: for example subdirectory lib/Image_tools would contain a class or function for image processing.

Calibration data or metadata should be stored in /calib/. Subfolders should be used for each diagnostic.

# Installation

If you are cloning the repository for the first time, rename _config.py to config.py and update its contents with your own details.
To access the shared configurations, modules and libs, use the following code at the top of your scripts:
```python
import sys
sys.path.append('../../') # this should point to the top level directory
from setup import *
```
This allows you to access classes and functions from the shared library folders, e.g.
```python
from lib.data_pipeline import DataPipeline
```
Common global configuration variables are defined in the setup.py file. If you define new global configuration variables, please use all caps, e.g. ROOT_DATA_FOLDER.

# General Comments
- Code should be written in python 3 in a object oriented format as much as possible.
- Try to keep code readable: use comments and [Docstrings](https://www.python.org/dev/peps/pep-0257/)
- No need to reinvent the wheel: check with previous people and online experiment analysis for previous code.
- I've add a /local/ folder to the .gitignore file. If you clone the repository to your own computer, you can create this folder and anything in it should stay local.

# Some Git Help
["Hello World" starter](https://guides.github.com/activities/hello-world/)

[Understanding the flow of Github](https://guides.github.com/introduction/flow/)

[Markdown cheatsheet](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet)

[Github Desktop](https://desktop.github.com/)

[Object Oriented Programming in Python help](https://www.programiz.com/python-programming/object-oriented-programming)

