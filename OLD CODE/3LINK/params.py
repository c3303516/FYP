from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as np

# class Params:
    
#     def __init__(self):

def robotParams(s):
    s.g = 9.81
    s.m1 = 1.5
    s.m2 = 1
    s.m3 = 0.8

    s.l1 = 1.5
    s.l2 = 0.8
    s.l3 = 0.5
    s.c1 = 0.6
    s.c2 = 0.4
    s.c3 = 0.25

    Iz1 = s.m1*s.l1*s.l1/12
    Iz2 = s.m2*s.l2*s.l2/12
    Iz3 = s.m3*s.l3*s.l3/12

    s.I1 = np.array([[0., 0., 0.],
            [0., 0., 0.],
            [0., 0., Iz1]])
    s.I2 = np.array([[0., 0., 0.],
            [0., 0., 0.],
            [0., 0., Iz2]])
    s.I3 = np.array([[0., 0., 0.],
                    [0., Iz3, 0.],
                    [0., 0., 0.]])
    return s
# a = 2
# b = 3