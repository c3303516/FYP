import jax.numpy as jnp
from jax.numpy import pi, sin, cos, linalg
from jax import grad
from params import *



def potentialEnergy(q,s):
    V = -m1*jnp.transpose(g0)@s.rc100 -m2*jnp.transpose(g0)@s.rc200 -m3*jnp.transpose(g0)@s.rc300

    print('V', V)

    dV = grad(V)                #this might have to stay in main
    dVdq = dV(q)
    return V, dVdq