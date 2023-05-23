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



#Momentum observer dynamics. Takes measurement of q and outputs mometum estimate
#k is tuning parameter. phi is piecewise constant observer state

def observer(q,phi,k_obv,Cq,Dq,Tq,xp):

    phi = phi + k_obv
    phat = xp + phi*q
    xp_dot = (Cq - Dq - phi*Tq)@phat - Tq@dVq + Gq@(u-u0)



    return