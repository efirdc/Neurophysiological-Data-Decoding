from vedo import Volume, dataurl

import vedo
from vedo.addons import addScalarBar
from vedo.plotter import Plotter
from vedo.pyplot import cornerHistogram
from vedo.utils import mag, precision, linInterpolate, isSequence
from vedo.colors import printc, colorMap, getColor
from vedo.shapes import Text2D, Line, Ribbon, Spline
from vedo.pointcloud import Points, fitPlane
from vedo import settings
from vedo.io import loadImageData
import numpy as np
import os

from pathlib import Path
import h5py

dataset_path = Path('X:\\Datasets\\2021 TC2See fMRI Data\\')
project_path = dataset_path / 'project'
derivatives_path = dataset_path / 'derivatives'

ssd_dataset_path = Path('C:\\Datasets\\2021 TC2See fMRI Data\\')
ssd_derivatives_path = ssd_dataset_path / 'derivatives'

f = h5py.File(ssd_derivatives_path / 'feature-selection-maps.hdf5', 'r')

subject = 'sub-02'
cache_name = 'window-4-6'
model_name = 'ViT-B=32'
feature_name = 'transformer.resblocks.0'
selection_mode = 'mean-top-5'

key = '/'.join([subject, cache_name, model_name, feature_name, selection_mode, 'scores'])

data = f[key][0]

volume = Volume(data).mode(1).c('jet')

# globals
_cmap_slicer='gist_ncar_r'
_alphaslider0, _alphaslider1, _alphaslider2 = 0.33, 0.66, 1  # defaults
_kact=0


def RayCastPlotter(volume):
    """
    Generate a ``Plotter`` window for Volume rendering using ray casting.
    Returns the ``Plotter`` object.
    """
    vp = settings.plotter_instance
    if not vp:
        vp = Plotter(axes=4, bg='bb')

    volumeProperty = volume.GetProperty()
    img = volume.imagedata()

    if volume.dimensions()[2]<3:
        print("Error in raycaster: not enough depth", volume.dimensions())
        return vp
    # printc("GPU Ray-casting tool", c="b", invert=1)

    smin, smax = img.GetScalarRange()

    x0alpha = smin + (smax - smin) * 0.25
    x1alpha = smin + (smax - smin) * 0.5
    x2alpha = smin + (smax - smin) * 1.0

    ############################## color map slider
    # Create transfer mapping scalar value to color
    cmaps = ["jet",
             "viridis",
             "bone",
             "hot",
             "plasma",
             "winter",
             "cool",
             "gist_earth",
             "coolwarm",
             "tab10",
             ]
    cols_cmaps = []
    for cm in cmaps:
        cols = colorMap(range(0, 21), cm, 0, 20)  # sample 20 colors
        cols_cmaps.append(cols)
    Ncols = len(cmaps)
    csl = (0.9, 0.9, 0.9)
    if sum(getColor(vp.renderer.GetBackground())) > 1.5:
        csl = (0.1, 0.1, 0.1)

    def sliderColorMap(widget, event):
        sliderRep = widget.GetRepresentation()
        k = int(sliderRep.GetValue())
        sliderRep.SetTitleText(cmaps[k])
        volume.color(cmaps[k])

    w1 = vp.addSlider2D(
        sliderColorMap,
        0,
        Ncols - 1,
        value=0,
        showValue=0,
        title=cmaps[0],
        c=csl,
        pos=[(0.8, 0.05), (0.965, 0.05)],
        )
    w1.GetRepresentation().SetTitleHeight(0.018)

    ############################## alpha sliders
    # Create transfer mapping scalar value to opacity
    opacityTransferFunction = volumeProperty.GetScalarOpacity()

    def setOTF():
        opacityTransferFunction.RemoveAllPoints()
        opacityTransferFunction.AddPoint(smin, 0.0)
        opacityTransferFunction.AddPoint(smin + (smax - smin) * 0.1, 0.0)
        opacityTransferFunction.AddPoint(x0alpha, _alphaslider0)
        opacityTransferFunction.AddPoint(x1alpha, _alphaslider1)
        opacityTransferFunction.AddPoint(x2alpha, _alphaslider2)

    setOTF()

    def sliderA0(widget, event):
        global _alphaslider0
        _alphaslider0 = widget.GetRepresentation().GetValue()
        setOTF()

    vp.addSlider2D(sliderA0, 0, 1,
                   value=_alphaslider0,
                   pos=[(0.84, 0.1), (0.84, 0.26)],
                   c=csl, showValue=0)

    def sliderA1(widget, event):
        global _alphaslider1
        _alphaslider1 = widget.GetRepresentation().GetValue()
        setOTF()

    vp.addSlider2D(sliderA1, 0, 1,
                   value=_alphaslider1,
                   pos=[(0.89, 0.1), (0.89, 0.26)],
                   c=csl, showValue=0)

    def sliderA2(widget, event):
        global _alphaslider2
        _alphaslider2 = widget.GetRepresentation().GetValue()
        setOTF()

    w2 = vp.addSlider2D(sliderA2, 0, 1,
                        value=_alphaslider2,
                        pos=[(0.96, 0.1), (0.96, 0.26)],
                        c=csl, showValue=0,
                        title="Opacity levels")
    w2.GetRepresentation().SetTitleHeight(0.016)

    # add a button
    def buttonfuncMode():
        s = volume.mode()
        snew = (s + 1) % 2
        volume.mode(snew)
        bum.switch()

    bum = vp.addButton(
        buttonfuncMode,
        pos=(0.7, 0.035),
        states=["composite", "max proj."],
        c=["bb", "gray"],
        bc=["gray", "bb"],  # colors of states
        font="",
        size=16,
        bold=0,
        italic=False,
    )
    bum.status(volume.mode())

    # def CheckAbort(obj, event):
    #     if obj.GetEventPending() != 0:
    #         obj.SetAbortRender(1)
    # vp.window.AddObserver("AbortCheckEvent", CheckAbort)

    # add histogram of scalar
    plot = cornerHistogram(volume,
                           bins=25, logscale=1, c=(.7,.7,.7), bg=(.7,.7,.7), pos=(0.78, 0.065),
                           lines=True, dots=False,
                           nmax=3.1415e+06, # subsample otherwise is too slow
                           )

    # xbins = np.linspace(smin, smax, 25)
    # yvals = volume.histogram(bins=25, logscale=1)
    # plot = cornerPlot(np.c_[xbins, yvals],
    #     c=(.7,.7,.7), bg=(.7,.7,.7), pos=(0.78, 0.065), s=0.4,
    #     lines=True, dots=False,
    # )

    plot.GetPosition2Coordinate().SetValue(0.197, 0.20, 0)
    plot.GetXAxisActor2D().SetFontFactor(0.7)
    plot.GetProperty().SetOpacity(0.5)
    vp.add([plot, volume])
    return vp

plt = RayCastPlotter(volume)
volume._update(loadImageData(f[key][1]))
plt.show(viewup="z", bg='black', bg2='blackboard', axes=15).close()


