import sys
sys.path.append('../../')
from setup import *
from glob import glob

def get_dirs(diag, date=None, head=False):
    """quick function to find all directories in a diag's data in MIRAGE.
    
    Allows you to quickly iterate DataPipeline over mulitple runs.

    Returns 
    -------
        list
        list of found directories
        Doesn't include trailing '/' for directories
    
    Inputs
    ------
    diag 
        str
        name of diag folder to look into in MIRAGE
    
    date
        str
        If given, then will find all runs within this date,
        if not (default is None), then will find all dates within this diagnsotic.
        
    head
        bool
        If True, then returns folders with their full filepaths.
        Default is False so just returns the folders
    
    """
    filepath = ROOT_DATA_FOLDER + '/' + diag + '/'
    if date is None:
        pass
    else:
        filepath += date + '/'
    dirs = glob(filepath+"*/")
    dirs = [i[:-1] for i in dirs ]
    
    if head is False:
        len_to_cut = len(filepath)
        dirs = [i[len_to_cut:] for i in dirs]
    
    return dirs