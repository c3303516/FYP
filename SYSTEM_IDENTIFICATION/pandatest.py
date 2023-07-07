import jax
from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.numpy import pi, sin, cos, linalg
from jax import grad, jacobian, jacfwd
from effectorFKM import endEffector
from massMatrix_holonomic import massMatrix_holonomic
from dynamics_momentumTransform import dynamics_Transform
import trajectories
from rk4 import rk4
from params import robotParams
from copy import deepcopy
from scipy.linalg import solve_continuous_lyapunov
from jax.scipy.linalg import sqrtm
from scipy.optimize import least_squares
from functools import partial
from gravcomp import gq
from sinusoid import sinusoid
import sys
import csv
import pandas as pd


data = pd.read_csv("7LINK_IMPLEMENTATION/data/freeswing_inverted_v1",sep=",",header=None, skiprows=3)       #last inputs go past the details of the csv.
print(data.head())      #prints first 5 rows. tail() prints last 5

data_array = data.to_numpy()

print(data_array)

# time = data_array.at[:,[2]].get()
time = data_array[:,1]
print(time)