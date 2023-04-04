import jax.numpy as jnp
from jax.numpy import pi, sin, cos, linalg
import jax
from copy import deepcopy
from functools import partial
# from main import dynamics

# @partial(jax.jit, static_argnames=['s'])
def rk4(xt,constants, dt,func,gC,contA,dMdq,dVdq, s):
    t = 1
    k1 = func(t,xt,constants,dMdq,dVdq,gC,contA, s)
    # print('A071', s.A07)
    k2 = func(t,xt + (k1 * dt) / 2,constants,dMdq,dVdq,gC,contA, s)
    # print('A072', s.A07)
    k3 = func(t,xt + (k2 * dt) / 2,constants,dMdq,dVdq,gC,contA, s)
    # print('A073', s.A07)
    k4 = func(t,xt + (k3 * dt),constants,dMdq,dVdq,gC,contA, s)
    # print(contA)
    return xt + (k1/6 + k2/3 + k3/3 + k4/6) * dt