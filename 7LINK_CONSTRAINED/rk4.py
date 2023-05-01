import jax.numpy as jnp
from jax.numpy import pi, sin, cos, linalg
import jax
from copy import deepcopy
from functools import partial
# from main import dynamics

# @partial(jax.jit, static_argnames=['s'])
def rk4(xt, dt,func,gC,contA,dMdq,dVdq, s):
    t = 1
    k1 = func(xt,dMdq,dVdq,gC,contA, s)
    # print('A071', s.A07)
    k2 = func(xt + (k1 * dt) / 2,dMdq,dVdq,gC,contA, s)
    # print('A072', s.A07)
    k3 = func(xt + (k2 * dt) / 2,dMdq,dVdq,gC,contA, s)
    # print('A073', s.A07)
    k4 = func(xt + (k3 * dt),dMdq,dVdq,gC,contA, s)
    # print(contA)
    return xt + (k1/6 + k2/3 + k3/3 + k4/6) * dt