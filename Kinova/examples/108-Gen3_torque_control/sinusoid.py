
import numpy as jnp
from numpy import pi, sin, cos, linalg
from params import *
from homogeneousTransforms import *
import csv

def sinusoid(t,origin,freq,amp):            #provides sinusoid based on centre, amplitude, and frequency
    #plane is xz plane.
    var0 = origin

    var0_vec = var0*jnp.ones(jnp.size(t))

    var = amp*sin(2*pi*freq*t) + var0       #cos guarantees 0 is centre? harmonic oscillaiton kinda deal

    return var

def sinusoid_instant(t,origin,freq,amp):
    #plane is xz plane.
    var0 = origin

    var = amp*sin(2*pi*freq*t) + var0       #returns one data point only.

    return var