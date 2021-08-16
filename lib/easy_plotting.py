"""A utility module to make live plotting even easier.

Create an DockView and call one of its add_ methods, passing it the plot name,
the diagnostic name, and your processing function.

Each processing function is given the absolute path name or URL, and it is
expected to return a processed value. The type of value depends on the type of
plot. All processing functions will run in a separate thread.
"""

import os, re, sys
sys.path.append('../..')
sys.path.append('../')

import urllib.parse

from . import live_plotting

import numpy as np
import pyqtgraph as pg
from pyqtgraph.dockarea import DockArea, Dock
from RR2021.general_tools import save_object, load_object

URL_REGEX = re.compile(r'\w+://')

def _make_path(base_path, relative_path):
    if URL_REGEX.match(base_path):
        return urllib.parse.urljoin(base_path, relative_path)
    else:
        relative_path = urllib.parse.unquote(relative_path)
        return os.path.join(base_path, relative_path)

class ScalarHistoryPlot(live_plotting.SimpleLineGraph):
    """A plot of the history of a scalar value.

    The processing function is expected to return a single scalar value.
    """

    def __init__(self, server, diag_name, base_path, verbose, history, processing_func):
        super().__init__(server, diag_name, history)
        self.base_path = base_path
        self.user_process_data = processing_func

    def process_data(self, url):
        path = _make_path(self.base_path, url)
        if os.path.isfile(path):
            return self.user_process_data(path)
        else:
            return None

class LinePlot(pg.GraphicsLayoutWidget):
    """A plot showing a 1D array.

    This is suitable for plotting a vector or 1-D array per laser shot. The
    processing function may return a numpy array (or array-like) or it may
    return a 2-element tuple, containing the x and y data.
    """

    class Processor(live_plotting.DataProcessor):
        def process_data(self, url):
            path = _make_path(self.base_path, url)
            if os.path.isfile(path):
                data = self.user_process_data(path)

                if isinstance(data, tuple):
                    x_data, y_data = data
                else:
                    y_data = data
                    x_data = np.arange(len(y_data))
                self.x_data = x_data
                self.y_data = y_data
            else:
                if hasattr(self, 'y_data'):
                    self.y_data = self.y_data*0



        def render_data(self):
            self.curve.setData(self.x_data, self.y_data)

    def __init__(self, server, diag_name, base_path, verbose, processing_func):
        super().__init__()

        p = self.addPlot()
        self.p = p
        curve = p.plot([0])

        data_processor = self.Processor(self, diag_name=diag_name)
        data_processor.base_path = base_path
        data_processor.curve = curve
        data_processor.user_process_data = processing_func
        data_processor.start()

        data_processor.verbose = verbose

        server.download_queue_ready.connect(data_processor.new_data)

class MultiLinePlot(pg.GraphicsLayoutWidget):
    """A plot showing a 1D array.
    
    This is suitable for plotting a vector or 1-D array per laser shot. The
    processing function may return a numpy array (or array-like) or it may
    return a 2-element tuple, containing the x and y data.
    """

    class Processor(live_plotting.MultiDataProcessor):
        def process_data(self, url_list):
            files_found = []
            path_list = []
            for url in url_list:
                path = _make_path(self.base_path, url)
                files_found.append(os.path.isfile(path))
                path_list.append(path)
            
            if all(files_found):
                data_list = self.user_process_data(path_list)
                x_data_list = []
                y_data_list = []
                for data in data_list:
                    if isinstance(data, tuple):
                        x_data, y_data = data
                    else:
                        y_data = data
                        x_data = np.arange(len(y_data))
                    x_data_list.append(x_data)
                    y_data_list.append(y_data)
            else:
                if hasattr(self, 'y_data'):
                    y_data_list =  []
                    for data in self.y_data_list:
                        y_data_list.append(data*0)
            self.x_data_list = x_data_list
            self.y_data_list = y_data_list
            
                
        def render_data(self):
            for n in range(self.num_lines):
                self.curve[n].setData(self.x_data_list[n], self.y_data_list[n],pen=self.pen_list[n])


    def __init__(self, server, diag_name_list, base_path, verbose, processing_func, line_colors=None):
        super().__init__()
        
        num_lines = len(diag_name_list)
        p = self.addPlot()
        curve = []
    
        for n in range(num_lines):
            curve.append(p.plot())
        self.p = p
        pen_list = []
        if line_colors is not None:
            for n in range(num_lines):
                pen_list.append(pg.mkPen(color=line_colors[n],width=1))
                curve[n].setPen(pg.mkPen(line_colors[n]))
        else:
            for n in range(num_lines):
                pen_list.append(pg.mkPen(width=1))
                

        data_processor = self.Processor(self, diag_name_list=diag_name_list)

        data_processor.pen_list = pen_list   
        data_processor.num_lines = num_lines
        data_processor.base_path = base_path
        data_processor.curve = curve
        data_processor.user_process_data = processing_func
        data_processor.start()

        data_processor.verbose = verbose

        server.download_queue_ready.connect(data_processor.new_data)

class ImagePlot(pg.ImageView):
    """A plot showing a 2D array (a matrix or image).

    The processing function returns a 2D array only.

    The x_axis and y_axis parameters are lists of pairs of values. Each pair is
    a pixel number and an axis label.
    """

    class Processor(live_plotting.DataProcessor):
        def process_data(self, url):
            
            path = _make_path(self.base_path, url)
            if os.path.isfile(path):
                self.data = self.user_process_data(path)
                if self.verbose:
                    print('Grabbed: ', path)
            else:
                if hasattr(self, 'data'):
                    self.data = self.data*0
                if self.verbose:
                    print('Failed to grab: ', path)
               

        def render_data(self):
            """ added error handling
            """
            if hasattr(self, 'data'):
                self.img_view.setImage(self.data, autoRange=False, autoLevels=False, autoHistogramRange=False)
            else:
                pass

    def __init__(self, server, diag_name, base_path, verbose, processing_func, *, x_axis=None, y_axis=None):
        if x_axis or y_axis:
            view = pg.PlotItem()
            if x_axis:
                view.getAxis('bottom').setTicks([x_axis, []])
            if y_axis:
                view.getAxis('left').setTicks([y_axis, []])
            super().__init__(view=view)
        else:
            super().__init__()

        data_processor = self.Processor(self, diag_name=diag_name)
        data_processor.base_path = base_path
        data_processor.img_view = self
        data_processor.user_process_data = processing_func
        data_processor.start()

        data_processor.verbose = verbose

        server.download_queue_ready.connect(data_processor.new_data)

class DockView(DockArea):
    """A view to hold multiple dockable plots.

    Can be treated as a pyqtgraph DockArea, or you can use the add_ convenience
    functions.

    (20210215) verbose attr added to help debugging. Needs to be propagated to other objects.
    """
    
    def __init__(self, server, base_path, verbose=False):
        super().__init__()
        self.dock_arrangement_file = 'dock_state.pkl'
        server.save_view.clicked.connect(self.save_dock_arrangement)
        self.server = server
        self.base_path = base_path
        self.verbose = bool(verbose)
        # server.exit_button.clicked.connect(self.exit)
    # def exit(self):
    #     print('Byeeee')
    #     for dock in self.docks.values():
    #         dock.setParent(None)        
    #     print('toodles')
    #     self.server.close()
    #     print('off now then')
        
    #     self.close()
    #     self.clear()
        

    def add_scalar_history_plot(self, dock_name, diag_name, history, func):
        dock = Dock(dock_name, size=(400, 400))
        self.addDock(dock)
        dock.addWidget(ScalarHistoryPlot(self.server, diag_name, self.base_path, self.verbose, history, func))
    
    def add_line_plot(self, dock_name, diag_name, func):
        dock = Dock(dock_name, size=(400, 400))
        self.addDock(dock)
        dock.addWidget(LinePlot(self.server, diag_name, self.base_path, self.verbose, func))

    def add_multiline_plot(self, dock_name, diag_name_list, func, line_colors=None):
        dock = Dock(dock_name, size=(400, 400))
        self.addDock(dock)
        dock.addWidget(MultiLinePlot(self.server, diag_name_list, self.base_path, self.verbose, func, line_colors = line_colors))

    def add_image_plot(self, dock_name, diag_name, func):
        dock = Dock(dock_name, size=(400, 400))
        self.addDock(dock)
        dock.addWidget(ImagePlot(self.server, diag_name, self.base_path, self.verbose, func))

    def add_image_plot_with_axes(self, dock_name, diag_name, func, x_axis, y_axis):
        dock = Dock(dock_name, size=(400, 400))
        self.addDock(dock)
        dock.addWidget(ImagePlot(self.server, diag_name, self.base_path, self.verbose, func,
            x_axis=x_axis, y_axis=y_axis))

    def save_dock_arrangement(self):
        dock_state = self.saveState()
        save_object(dock_state,self.dock_arrangement_file)

    def load_dock_arrangement(self):
        if os.path.isfile(self.dock_arrangement_file):
            dock_state = load_object(self.dock_arrangement_file)
            self.restoreState(dock_state)



