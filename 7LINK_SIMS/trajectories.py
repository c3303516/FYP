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



# class Traj:
def point(t,xd,freq,amp):
    x0 = xd.at[0].get()
    y0 = xd.at[1].get()
    z0 = xd.at[2].get()
    
    x0_vec = x0*jnp.ones(jnp.size(t))
    y0_vec = y0*jnp.ones(jnp.size(t))
    z0_vec = z0*jnp.ones(jnp.size(t))
    xe = jnp.array([x0_vec,y0_vec,z0_vec])
    return xe


def planar_circle(t,origin,freq,amp):
    #plane is xz plane.
    x0 = origin.at[0].get()
    z0 = origin.at[1].get()

    x0_vec = x0*jnp.ones(jnp.size(t))
    z0_vec = z0*jnp.ones(jnp.size(t))

    x = amp*sin(2*pi*freq*t) + x0       #check this
    z = amp*cos(2*pi*freq*t) + z0
    # print(x)

    vx = 2*amp*pi*freq*cos(2*pi*freq*t)
    vz = -2*amp*pi*freq*sin(2*pi*freq*t)

    xe = jnp.array([x,jnp.zeros(jnp.size(t)), z])
    ve = jnp.array([vx,jnp.zeros(jnp.size(t)), vz]) #this doesn't get used as i havent figured out a way to shift it to qdot
    return xe

def sinusoid_x(t,origin,freq,amp):
    #plane is xz plane.
    x0 = origin.at[0].get()
    z0 = origin.at[1].get()

    x0_vec = x0*jnp.ones(jnp.size(t))
    # z0_vec = z0*jnp.ones(jnp.size(t))
    # print('vec',x0_vec)

    x = amp*sin(2*pi*freq*t) + x0       #check this
    # z = amp*cos(2*pi*freq*t) + z0

    vx = 2*amp*pi*freq*cos(2*pi*freq*t)
    # vz = -2*amp*pi*freq*sin(2*pi*freq*t)

    xe = jnp.array([x,jnp.zeros(jnp.size(t)), jnp.zeros(jnp.size(t))])
    # ve = jnp.array([vx,jnp.zeros(jnp.size(t)),jnp.zeros(jnp.size(t))])
    return xe