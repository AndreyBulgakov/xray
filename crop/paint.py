# coding=utf-8
import matplotlib
matplotlib.use('WebAgg')
import pylab
from matplotlib import pyplot as plt
from matplotlib.widgets import RadioButtons, Button
from matplotlib import gridspec

import polyutils
from ZoomMatplot import ZoomPan

# Path to save polygons
poly_dir = 'polygons/'


class PolygonBuilder:
    def __init__(self, lung, color):

        self.lungX = []
        self.lungY = []

        self.lung = lung

        self.lung.set_color(color)
        self.lung.set_marker('^')

        self.cid = lung.figure.canvas.mpl_connect('button_press_event', self.on_click)

        self.isActive = False

    def refresh(self):
        self.lung.set_data(self.lungX, self.lungY)
        self.lung.figure.canvas.draw()

    def on_click(self, event):
        if event.button == 1 and self.isActive:
            self.draw_line_for_lung(event, self.lung, self.lungX, self.lungY)

    def draw_line_for_lung(self, event, lung, lungX, lungY):
        # WTF. May be we can delete that?
        if event.inaxes != self.lung.axes: return
        lungX.append(int(event.xdata))
        lungY.append(int(event.ydata))

        xbegin = lungX + [lungX[0]]
        ybegin = lungY + [lungY[0]]

        lung.set_data(xbegin, ybegin)
        lung.figure.canvas.draw()

    def set_active(self, isActive):
        self.isActive = isActive

    def switch_active(self):
        self.isActive = not self.isActive

def show(imagename, imagearr):
    # Disable toolbar
    matplotlib.rcParams['toolbar'] = 'None'
    # fig = plt.figure(figsize=(20, 10))
    # gs = gridspec.GridSpec(1, 2, width_ratios=[10, 1])
    # ax = plt.subplot(gs[0])
    fig, ax = plt.subplots(figsize=(20, 10))
    fig.tight_layout()

    ax.set_title('Click to build polygon')
    leftLung, = ax.plot([0], [0])  # empty line
    rightLung, = ax.plot([0], [0])  # empty line

    # Create Polygon builder
    leftLungBuilder = PolygonBuilder(leftLung, 'r')
    rightLungBuilder = PolygonBuilder(rightLung, 'r')

    # Zoom
    scale = 1.1
    zp = ZoomPan()
    figZoom = zp.zoom_factory(ax, base_scale=scale)
    figPan = zp.pan_factory(ax)

    # RadioButtons
    rax = fig.add_axes([0.05, 0.7, 0.08, 0.10])
    rax.set_title('Select Lung')
    radio = RadioButtons(rax, ('Left', 'Right'))
    radio.set_active(0)
    leftLungBuilder.set_active(True)

    def on_radio(label):
        leftLungBuilder.switch_active()
        rightLungBuilder.switch_active()

    radio.on_clicked(on_radio)

    # Clear Left
    axclearLeft = fig.add_axes([0.05, 0.6, 0.08, 0.10])
    def clearLeft(event):
        leftLungBuilder.lungX = []
        leftLungBuilder.lungY = []
        leftLungBuilder.refresh()
    clearLeftbtn = Button(axclearLeft, 'Clear Left')
    clearLeftbtn.on_clicked(clearLeft)

    # Clear Right
    axclearRight = fig.add_axes([0.05, 0.5, 0.08, 0.10])
    def clearRight(event):
        rightLungBuilder.lungX = []
        rightLungBuilder.lungY = []
        rightLungBuilder.refresh()

    clearRightbtn = Button(axclearRight, 'Clear Right')
    clearRightbtn.on_clicked(clearRight)

    # Save button
    axsave = fig.add_axes([0.05, 0.4, 0.08, 0.10])
    def save(event):
        left = polyutils.createPoly('left', xarr=leftLungBuilder.lungX, yarr=leftLungBuilder.lungY)
        right = polyutils.createPoly('right', xarr=rightLungBuilder.lungX, yarr=rightLungBuilder.lungY)
        polyutils.save_polys(poly_dir + imagename + '.polys', left, right)

    savebtn = Button(axsave, 'Save')
    savebtn.on_clicked(save)


    ax.imshow(imagearr, cmap=pylab.cm.bone)

    # mpld3.show()
    plt.show()
