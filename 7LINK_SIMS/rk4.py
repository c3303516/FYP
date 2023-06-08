import jax.numpy as jnp
from jax.numpy import pi, sin, cos, linalg
import jax
from copy import deepcopy
from functools import partial
# from main import dynamics

# # @partial(jax.jit, static_argnames=['s'])
def rk4(xt,func,dt,*args):
# def rk4(xt,func,dt,*arguments):
    # print('size xt',jnp.size(xt))
    #check what arguments are being sent. Massive purg incoming
    k1 = func(xt,*args)
    # print('A071', s.A07)
    k2 = func(xt + (k1 * dt) / 2.,*args)
    # print('A072', s.A07)
    k3 = func(xt + (k2 * dt) / 2.,*args)
    # print('A073', s.A07)
    k4 = func(xt + (k3 * dt),*args)
    # print(x_tilde)

    # print('size k1',jnp.shape(k1))
    # print('size k2',jnp.shape(k2))
    # print('size k3',jnp.shape(k3))
    # print('size k4',jnp.shape(k4))
    # print('size result',jnp.shape(xt + (k1/6. + k2/3. + k3/3. + k4/6.) * dt))

    return xt + (k1/6. + k2/3. + k3/3. + k4/6.) * dt