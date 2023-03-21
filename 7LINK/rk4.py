import jax.numpy as jnp
from jax.numpy import pi, sin, cos, linalg

# from main import dynamics


def rk4(xt, dt,func,contA, s):
    k1 = func(xt,contA, s)
    k2 = func(xt + (k1 * dt) / 2,contA, s)
    k3 = func(xt + (k2 * dt) / 2,contA, s)
    k4 = func(xt + (k3 * dt),contA, s)
    # print(contA)
    return xt + (k1/6 + k2/3 + k3/3 + k4/6) * dt