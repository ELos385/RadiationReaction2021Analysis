import sys
if '../../' not in sys.path: sys.path.append('../../')                          # Add root folder path


class Clicker():
    """
        holds the interactive plot that registers where clicks have occurred
        (but not drags, so they don't get confused with zooms etc.)
    """
    #https://stackoverflow.com/questions/48446351/distinguish-button-press-event-from-drag-and-zoom-clicks-in-matplotlib
    def __init__(self, ax, func, button=1):
        self.ax=ax
        self.func=func
        self.button=button
        self.press=False
        self.move = False
        self.c1=self.ax.figure.canvas.mpl_connect('button_press_event', self.onpress)
        self.c2=self.ax.figure.canvas.mpl_connect('button_release_event', self.onrelease)
        self.c3=self.ax.figure.canvas.mpl_connect('motion_notify_event', self.onmove)

        self.listo = []

    def onclick(self,event):
        if event.inaxes == self.ax:
            if event.button == self.button:
                self.func(event, self.ax)
    def onpress(self,event):
        self.press=True
    def onmove(self,event):
        if self.press:
            self.move=True
    def onrelease(self,event):
        if self.press and not self.move:
            self.onclick(event)
        self.press=False; self.move=False

def list_clicks_on_ax(ax, printout=True):
    """Given an axis with an image on, function keeps track of where the user
    has clikced on the image. Quick for marking up masks / areas on images.

    returns
        click - an instance of Clicker class that allows interaction
        (This has to be returned for the link to the plot to be sustained).
        full_list - list of points clicked. Will update as you click.

    ax is a matplotlib axis object that you wish to add the clicker functionality
    too. E.g.

    fig, ax = plt.subplots()
    ax.imshow(im)
    _, full_list = make_clicker(ax)

    printout is a boolean that if True, the point clicked will be printed to
    the command line each time.
    """
    full_list = []
    def func(event, ax):
        #c = int(event.xdata), int(event.ydata)
        c = float(event.xdata), float(event.ydata)
        full_list.append(c)
        if printout==True:
            print(str(c), ' '*(12 - len(str(c))), len(full_list))

    click = Clicker(ax, func, button=1)
    return click, full_list
