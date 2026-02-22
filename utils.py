import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar

def mshow(a):
    if isinstance(a, cp.ndarray):
        a = a.get()
    fig, axs = plt.subplots(1, 1, figsize=(6,6))
    im = axs.imshow(a, cmap="gray")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    plt.show()


def mshow_complex(a,voxelsize=None):
    if isinstance(a, cp.ndarray):
        a = a.get()
    fig, axs = plt.subplots(1, 2, figsize=(14,6))
    axs[0].set_title('delta')
    axs[1].set_title('beta')
    im = axs[0].imshow(a.real, cmap="gray")
    
    if voxelsize is not None:
        scalebar = ScaleBar(0.015, "um", length_fraction=0.25, font_properties={
        "family": "serif",
        },  # For more information, see the cell below
        location="lower right")
        axs[0].add_artist(scalebar)

    fig.colorbar(im, fraction=0.046, pad=0.04)
    im = axs[1].imshow(a.imag, cmap="gray")
    scalebar = ScaleBar(0.015, "um", length_fraction=0.25, font_properties={
        "family": "serif",
    },  # For more information, see the cell below        
    location="lower right")
    # axs[1].add_artist(scalebar)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    plt.show()

def mshow_polar(a):
    if isinstance(a, cp.ndarray):
        a = a.get()
    fig, axs = plt.subplots(1, 2, figsize=(14,6))
    im = axs[0].imshow(np.abs(a), cmap="gray")
    axs[0].set_title("abs")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    im = axs[1].imshow(np.angle(a), cmap="gray")
    axs[1].set_title("phase")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    plt.show()
        
def mshow_pos(pos):
    if isinstance(pos, cp.ndarray):
        pos = pos.get()
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.set_title('position error')
    ax.plot(pos[:,:,1].flatten(), ".",label='x')
    ax.plot(pos[:,:,0].flatten(), ".",label='y')
    ax.grid()    
    ax.legend()
    plt.show()
    
def mshow_approx(t,err_real,err_approx):
    plt.figure(figsize=(4, 4))
    plt.plot(t, err_real, "o-", label="real")
    plt.plot(t, err_approx, "x-", label="approx")
    plt.legend()
    plt.grid()
    plt.show()


def reprod(a,b):
    return a.real*b.real+a.imag*b.imag
    

def redot(a, b,axis=None):
    # if axis is None:        
    #     res = cp.vdot(a.view('float32'),b.view('float32'))
    # else:
    res = cp.sum(reprod(a, b),axis=axis)
    return res