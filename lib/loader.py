from skimage.io import imread
import numpy
from PIL import Image

DATA_LOADER_REGISTRY = {}

def register_data_loader(diag_name, loader_class):
    """Register a given data loader class to be used with a given diagnostic."""

    DATA_LOADER_REGISTRY[diag_name] = loader_class

def get_data_loader(diag_name):
    """Get the data loader class for use with the given diagnostic."""

    return DATA_LOADER_REGISTRY[diag_name]

class GCamDataLoader:
    """A loader for 8-bit and 12-bit data saved by gCam."""

    extensions = ['tif', 'tiff']

    @classmethod
    def load_data(cls, path):
        data = imread(path)

        if len(data.shape) != 2:
            raise ValueError('cannot load RGB data')

        if data.dtype == numpy.uint16:
            # 12-bit data in a 16-bit image
            # The format is 0x8XXX
            # So just mask out the top nybble and we're good to go
            data &= 0xfff

        return data

class FallibleGCamDataLoader(GCamDataLoader):
    """Like GCamDataLoader, but ignores errors."""

    @classmethod
    def load_data(cls, path):
        try:
            return super().load_data(path)
        except Exception:
            pass

class ImageDataLoader:
    """A loader for any data that can be loaded by imread.

    But actually only looks at PNG, TIFF, JPEG, BMP.
    """

    extensions = ['tif', 'tiff', 'jpeg', 'jpg', 'png', 'bmp']

    @classmethod
    def load_data(cls, path):
        return imread(path)

class TraceDataLoader:
    """A loader for csv data from oscilloscopes, picoscopes, and other """

    extensions = ['csv', 'txt']

    @classmethod
    def load_data(cls, path):
        return numpy.genfromtxt(path, delimiter=',', skip_header=3)

class LundatronLoader:
    extensions = ['tif']
    @classmethod
    def load_data(cls, path):
        return numpy.array(Image.open(path)).astype(float)
