"""A pipeline for processing data in parallel.

A pipeline consists of a data loading step, a data processing (map) step, and a
data combination (reduce) step. The data loading and processing is carried out
in parallel using multi-threading or multi-processing, then the combination is
performed by the calling thread or process.
"""

import sys
sys.path.append('../../')
from . import base_path
from .loader import get_data_loader

import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, wait
from itertools import groupby

import numpy
import pandas

def _process(path, data_loader, map_func):
    # This is the worker function that executes on the other threads/processes

    data = data_loader.load_data(path)
    if data is not None:
        return map_func(data)

class DataPipeline:
    """A custom data processing pipeline.

    Arguments:
     * diagnostic name: used to select the correct path and the correct data
       loader plugin
     * map function: a one-argument callable that processes the loaded data
     * reduce function: a callable that combines the data from a single burst,
       or None to concatenate all shots together

        update: added single_shot_mode argument to handle runs containing only data files
    """

    def __init__(self, diag_name: str, map_func=lambda x: x, reduce_func=None, single_shot_mode=False):
        self.base_path = os.path.join(base_path(), diag_name)
        self.data_loader = get_data_loader(diag_name)
        self.map_func = map_func
        self.reduce_func = reduce_func
        self.single_shot_mode = single_shot_mode

    def run(self, run_name, burst_num=None, *, parallel='thread', max_workers=None,
            use_pandas=False):
        """Run the data processing pipeline on the given run.

        Arguments:
         * run name: the name of the run to process
         * burst number: if specified, only process one burst

        Keyword-only arguments:
         * parallel: the kind of parallelism to employ:
          - 'thread': use multiple threads. Reliable but limited speed
          - 'process': use multiple processes. Faster (no GIL) but may not work
            in interactive programs.
         * use_pandas: if True, return a pandas.Series instead of an (ids, data)
           pair. This only works if the data is 1-dimensional

        """

        data_ids, paths = self.list_data(run_name, burst_num)

        if parallel == 'thread':
            executor = ThreadPoolExecutor(max_workers)
            chunksize = 1
        elif parallel == 'process':
            executor = ProcessPoolExecutor(max_workers)
            if max_workers is None:
                max_workers = 32
            chunksize = len(paths) // max_workers
        else:
            raise ValueError('invalid value for parallel: {}'.format(parallel))

        futures = []

        with executor:
            data = executor.map(lambda p: _process(p, self.data_loader, self.map_func), paths,
                    chunksize=chunksize)

        id_data_pairs = [(id, d) for id, d in zip(data_ids, data) if d is not None]
        data_ids = [id for id, _ in id_data_pairs]
        data = [d for _, d in id_data_pairs]

        if (self.reduce_func is None) or (self.single_shot_mode):
            result = (data_ids, numpy.stack(data))
        else:
            if burst_num is None:
                # Reduce bursts one at a time
                burst_ids = []
                burst_data = []
                for b, ids in groupby(enumerate(data_ids), lambda x: x[1][0]):
                    idxs = [x[0] for x in ids]
                    burst_ids.append(b)
                    start_idx = idxs[0]
                    end_idx = start_idx + len(idxs)
                    burst_data.append(self.reduce_func(data[start_idx:end_idx]))
                result = (burst_ids, numpy.stack(burst_data))
            else:
                # Only one burst to reduce
                if use_pandas:
                    # This is a special case
                    return self.reduce_func(data)
                else:
                    return [None], self.reduce_func(data)

        if use_pandas:
            index, data = result
            if len(data.shape) != 1:
                raise ValueError('cannot convert multidimensional data to pandas form')
            if self.reduce_func is None and burst_num is None:
                # This is the only case where we keep both shot and burst
                # numbers, and need to use a MultiIndex
                index = pandas.MultiIndex.from_tuples(index, names=['burst', 'shot'])
            return pandas.Series(data, index=index)
        else:
            return result

    def list_data(self, run_name, burst_num=None):
        """List all the files to be processed.

        Returns a pair of lists. The first contains data ids (shot numbers or
        burst/shot number pairs). The second contains paths.

        Added single shot mode handling
        """

        info = []

        base_path = os.path.join(self.base_path, run_name)

        if self.single_shot_mode:
            for shot, path in self._enumerate_single_shots(base_path):
                info.append( (shot, path) )

        else:
            if burst_num is None:
                for name in os.listdir(base_path):
                    if name.startswith('Burst'):
                        try:
                            b = int(name[5:])
                        except ValueError:
                            continue
                        for shot, path in self._enumerate_burst(base_path, b):
                            info.append( ( (b, shot), path) )
            else:
                for shot, path in self._enumerate_burst(base_path, burst_num):
                    info.append( (shot, path) )

        info.sort() # Use lexical sorting, which does exactly what we want

        data_ids, paths = zip(*info)

        return data_ids, paths

    def _enumerate_burst(self, base_path, burst_num):
        path = os.path.join(base_path, f'Burst{burst_num:03}')
        extensions = self.data_loader.extensions

        for name in os.listdir(path):
            base_name, ext = os.path.splitext(name)
            if not ext.lower().lstrip('.') in extensions:
                continue
            if not base_name.startswith('Shot'):
                continue
            try:
                shot = int(base_name[5:])
            except ValueError:
                continue
            yield shot, os.path.join(path, name)

    def _enumerate_single_shots(self, path):
        """ Modfied from _enumerate_burst to handle single shot mode
        """
        #path = os.path.join(base_path, f'Burst{burst_num}')
        extensions = self.data_loader.extensions

        for name in os.listdir(path):
            base_name, ext = os.path.splitext(name)
            if not ext.lower().lstrip('.') in extensions:
                continue
            if not base_name.startswith('Shot'):
                continue
            try:
                shot = int(base_name[4:])
            except ValueError:
                continue
            yield shot, os.path.join(path, name)
