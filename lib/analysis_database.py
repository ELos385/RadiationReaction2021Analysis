import pandas as pd
import numpy as np
import os

def entries2columns(y):
    N = np.shape(y)
    if len(N)==1:
        y_reshaped = [[i] for i in y]
    else:
        y_reshaped = [[i[n] for i in y] for n in range(N[1])]
    return y_reshaped

def data2dict(keys,data):
    return dict(zip(keys, data))


class Database:
    index_names = ('run', 'shot')
    
    def __init__(self,file_path,columns=None,always_load=True,overwrite=False):
        self.file_path = file_path
        if columns is not None:
            self.columns = columns
        else:
            self.columns = self.get_columns()

        if always_load:
            if os.path.isfile(file_path):
                self.load_dataframe()
            else:
                self.dataframe = None
        else:
            self.dataframe = None

        self.always_load = always_load
        self.overwrite = overwrite
        
        
    def save_entry(self,index_list,data):
        ''' pass entrys one at a time to append to the file '''
        indexes = self.make_index(index_list)
        data_dict = data2dict(self.columns,data)
        
        if os.path.isfile(self.file_path):
            if self.always_load:
                self.load_dataframe()
            elif self.dataframe is None:
                self.load_dataframe()

            entry_exists = self._check4indexes(indexes)
            if entry_exists:
                if self.overwrite:
                    self.dataframe.loc[tuple(index_list)] = data
                else:
                    print('Entry exists and did not overwrite')
            else:
                self.dataframe = self.dataframe.append(pd.DataFrame([data_dict],index=indexes))
        else:
            self.dataframe = pd.DataFrame([data_dict],index=indexes)
        self.save_dataframe()
        
    def save_multiple_entries(self,index_lists, data_lists):
        if os.path.isfile(self.file_path):
            self.load_dataframe()
        data_dict = data2dict(self.columns,entries2columns(data_lists))
        indexes = self.make_index(index_lists)

        new_dataframe = pd.DataFrame(data_dict,index=indexes)
        if self.dataframe is None:
            self.dataframe = new_dataframe
        else:
            self.dataframe = new_dataframe.combine_first(self.dataframe)
        self.save_dataframe()
        
    def load_dataframe(self):
        # self.dataframe = pd.read_hdf(self.hdf_file,self.group_key)
        self.dataframe = pd.read_pickle(self.file_path)
        
    def _check4indexes(self,indexes):
        return indexes[0] in self.dataframe.index
    
    def make_index(self,index_list):
        return pd.MultiIndex.from_arrays(entries2columns(index_list),names=self.index_names)
    
    def save_dataframe(self):
        # self.dataframe.to_hdf(self.hdf_file,key=self.group_key)
        self.dataframe.to_pickle(self.file_path,protocol=4)
