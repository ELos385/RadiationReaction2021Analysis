import sys
sys.path.append('../../')
from setup import *

import numpy as np
from datetime import datetime as dt   

from lib.general_tools import *

def get_sql_data(dateruns, shots):
    """For a given list of shots, get the corresponding gsns and timestamps 
    from the data.sqlite file
    
    Returns
    -------
    tuple (of len == 2)
        first item: 
            list
            list of gsns of shots as float, so can be np.nan if not found.
    
        second item:
            list
            list of timestamps of shots as datetime.datetime objects.
    
    The datetime module makes it easier to work out relative time differences
    
    Inputs
    -------
    dateruns
        list
        list of daterun strs (e.g. ['20210618/run15', ... ])

    shots
        list
        list of corresponding shots (e.g. [12, ...])
    
    dateruns and shots should be of same size. 
    If len(dateruns)==1, then assumes all shots are from that one run.
    Else it return None
    """
    sql = Read_SQL_shot_summary()
    sql_data = sql.get_all()
    sql_dt_format = "%Y-%m-%d %H:%M:%S.%f"
    
    if len(dateruns)==len(shots):
        pass
    elif len(dateruns)==1:
        # assume all shots from that one specified daterun
        dr = dateruns[0]
        dateruns = [dr for i in shots]
    else:
        print('Error occurred in defining shots.')
        return None, None
    
    dateruns = [str(i) for i in dateruns]
    shots = [str(i) for i in shots]
    
    gsns = np.full_like(shots, fill_value=np.nan, dtype=float)
    timestamps = np.full_like(shots, fill_value=np.nan, dtype=np.object)
    
    for idx, (r, s) in enumerate(zip(dateruns, shots)):        
        try:
            ids = (sql_data['run']==r) & (sql_data['shot_or_burst']==s)
            timestamp = np.array(sql_data[ids]['timestamp'])[0]
            ts = dt.strptime(timestamp, sql_dt_format)
            timestamps[idx] = ts
            
            gsn = np.array(sql_data[ids]['gsn'])[0]
            gsns[idx] = gsn
            
        except(IndexError):
            # happens if there no entry satisfied clause for ids
            print('Failed for %s %s' % (r, s))
        
    return gsns, timestamps