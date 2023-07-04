
import jax.numpy as jnp
from jax.numpy import pi, sin, cos, linalg
import jax
from functools import partial
from params import *
from homogeneousTransforms import *
from scipy.linalg import solve_continuous_lyapunov
from jax.scipy.linalg import sqrtm
# from main import massMatrix
# from massMatrix import massMatrix
# from effectorFKM import FKM
# from copy import deepcopy
from jax import lax
import csv

def sinusoid(t,origin,freq,amp):            #provides sinusoid based on centre, amplitude, and frequency
    #plane is xz plane.
    var0 = origin
    var0_vec = var0*jnp.ones(jnp.size(t))

    var = amp*sin(2*pi*freq*t) + var0

    return var