"""A utility module for building live plotting programs.

This module helps you to write programs that connect to a MIRAGE server and
plot all data as it arrives. It works with pyqtgraph.
"""

from queue import Queue, Full
import traceback

import mirage_ui
import pyqtgraph as pg

import numpy as np
from skimage.io import imread

from PyQt5 import QtCore

class DataProcessor(QtCore.QThread):
    """A QObject that processes data live from MIRAGE.

    To use this class, subclass it and provide two methods: process_data() and
    render_data(). After creating an instance, connect the download_queue_ready
    signal of the ServerConnection to the new_data slot and call start().

    process_data() is passed the relative URL of a data file and executes in a
    separate thread. It can then process it and store the results in an
    instance variable.

    render_data() is called in the main thread after process_data() returns. It
    takes no arguments, and is expected to take the data out of the instance
    variable where process_data() stashed it and plot it.
    """

    def __init__(self, *args, diag_name=None, queue_size=4):
        """Keyword arguments:
        * diag_name: if this is set, only process data from the given diagnostic
        * queue_size: the number of files to keep in the processing queue.
        While the queue is full, new files are ignored. Defaults to 4.
        """

        super().__init__(*args)

        self.queue = Queue(4)
        self.diag_name = diag_name

    @QtCore.pyqtSlot(str, int)
    def new_data(self, url, _size):
        if self.diag_name is None or url.startswith(self.diag_name + '/'):
            try:
                self.queue.put_nowait(url)
            except Full:
                pass

    def run(self):
        while True:
            url = self.queue.get()
            if url is None:
                break
            try:
                self.process_data(url)
            except Exception:
                traceback.print_exc()
            else:
                QtCore.QMetaObject.invokeMethod(self, '_render_data')

    @QtCore.pyqtSlot()
    def _render_data(self):
        self.render_data()


class MultiDataProcessor(QtCore.QThread):
    """A QObject that processes data live from MIRAGE.

    To use this class, subclass it and provide two methods: process_data() and
    render_data(). After creating an instance, connect the download_queue_ready
    signal of the ServerConnection to the new_data slot and call start().

    process_data() is passed the relative URL of a data file and executes in a
    separate thread. It can then process it and store the results in an
    instance variable.

    render_data() is called in the main thread after process_data() returns. It
    takes no arguments, and is expected to take the data out of the instance
    variable where process_data() stashed it and plot it.
    """

    def __init__(self, *args, diag_name_list=None, queue_size=4):
        """Keyword arguments:
        * diag_name_list: if this is set, only process data from the given diagnostics
        * queue_size: the number of files to keep in the processing queue.
        While the queue is full, new files are ignored. Defaults to 4.
        """

        super().__init__(*args)

        self.queue = Queue(4)
        self.diag_name_list = diag_name_list
        self.diag_arrived = [False]*len(diag_name_list)
        self.url_list = ['']*len(diag_name_list)

    @QtCore.pyqtSlot(str, int)
    def new_data(self, url, _size):
        for n,diag_name in enumerate(self.diag_name_list):
            if url.startswith(diag_name + '/'):
                self.diag_arrived[n] = True
                self.url_list[n] = url
                if all(self.diag_arrived):
                    self.diag_arrived = [False]*len(self.diag_name_list)
                    
                    try:
                        self.queue.put_nowait(self.url_list)
                    except Full:
                        pass
                    self.url_list = ['']*len(self.diag_name_list)

    def run(self):
        while True:
            url = self.queue.get()
            if url is None:
                break
            try:
                self.process_data(url)
            except Exception:
                traceback.print_exc()
            else:
                QtCore.QMetaObject.invokeMethod(self, '_render_data')

    @QtCore.pyqtSlot()
    def _render_data(self):
        self.render_data()

class SimpleLineGraph(pg.GraphicsLayoutWidget):
    """A simple line graph widget.

    This is suitable for plotting a single value per laser shot as a line
    graph. To use, subclass it and provide a process_data() method. This takes
    a single argument (the relative url), runs in a separate thread, and is
    expected to return a single value calculated from the given file.
    """

    class Processor(DataProcessor):
        def render_data(self):
            self.curve.setData(self.data)

    def __init__(self, server, diag_name, history):
        """Arguments:
        * server: the ServerConnection object to use
        * diag_name: the diagnostic name to use
        * history: the number of points to plot as a rolling graph
        """

        super().__init__()

        p = self.addPlot()
        self.data = np.zeros(history) * np.nan
        self.data[0] = 0
        curve = p.plot(self.data)

        data_processor = self.Processor(self, diag_name=diag_name)
        data_processor.data = self.data
        data_processor.curve = curve
        data_processor.process_data = self._process_data
        data_processor.start()

        server.download_queue_ready.connect(data_processor.new_data)

    def _process_data(self, url):
        val = self.process_data(url)
        self.data[:-1] = self.data[1:]
        self.data[-1] = val

class ImageView(pg.ImageView):
    """A simple false-colour view of camera data.

    This can be used as-is. It does not require sub-classing, but it does
    require information on how to find the data.
    """

    class Processor(DataProcessor):
        def process_data(self, url):
            path = self.base_url + '/' + url
            self.data = imread(path)
            if self.format == 'gcam-12bit' and self.data.dtype == np.uint16:
                self.data &= 0xfff

        def render_data(self):
            self.img_view.setImage(self.data, autoRange=False, autoLevels=False, autoHistogramRange=False)

    def __init__(self, server, diag_name, base_url, format=''):
        """Arguments:
        * server: the ServerConnection object to use
        * diag_name: the diagnostic name to use
        * base_url: the base URL or path at which data can be found
        * format: can be used to indicate special data formats. Most image file
        formats are recognised, but some data requires special processing

        Available formats:
        * 'gcam-12bit': 12-bit data recorded by gCam. This has a large
        zero-offset applied to it. 8-bit data is treated normally.
        """

        super().__init__()

        data_processor = self.Processor(self, diag_name=diag_name)
        data_processor.base_url = base_url
        data_processor.format = format
        data_processor.img_view = self
        data_processor.start()

        server.download_queue_ready.connect(data_processor.new_data)
