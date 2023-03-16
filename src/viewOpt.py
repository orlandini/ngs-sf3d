from ngsolve.internal import *

def loadView():
    ngsolve.internal.visoptions.vecfunction="J"
    ngsolve.internal.visoptions.scalfunction="sol3d:0"

    ngsolve.internal.visoptions.invcolor=0
    ngsolve.internal.visoptions.autoscale=1
    ngsolve.internal.visoptions.mminval=0
    #ngsolve.internal.visoptions.mmaxval=1.4
    ngsolve.internal.visoptions.mmaxval=13e3

    ngsolve.internal.visoptions.clipsolution="scal"
    ngsolve.internal.viewoptions.clipping.enable=1
    ngsolve.internal.viewoptions.clipping.nx=0
    ngsolve.internal.viewoptions.clipping.ny=0
    ngsolve.internal.viewoptions.clipping.nz=-1
    ngsolve.internal.viewoptions.clipping.onlydomain=0
    ngsolve.internal.viewoptions.clipping.notdomain=0
    ngsolve.internal.viewoptions.clipping.dist=0.0001
    ngsolve.internal.visoptions.deformation=0
    ngsolve.internal.visoptions.scaledeform1=10
    ngsolve.internal.visoptions.scaledeform2=1

    ngsolve.internal.Rotate(0, 0)
    ngsolve.internal.Zoom(0)
    ngsolve.internal.Move(0, 0)
    ngsolve.internal.visoptions.gridsize=40


    ngsolve.internal.visoptions.numfieldlines=100
    ngsolve.internal.visoptions.imaginary = 0



    #ngsolve.internal.SnapShot('test.ppm')