from cmath import pi, sin
import jax.numpy as np
from jax import grad

from params import *
from homogeneousTransforms import *

def massMatrix(q,s):
    Jc2 = s.Jc2
    Jc3 = s.Jc3

    R02 = s.A02[0:3,0:3]
    R03 = s.A03[0:3,0:3]
    # print('R02----', R02)
    # testM = np.block([
    #     [np.multiply(m2,np.eye(3,3)), np.zeros((3,3))],
    #     [np.zeros((3,3)),            R02.T@I2@R02 ]
    # ])
    # print('TestM -----',testM)
    M2 = Jc2.T@np.block([
        [np.multiply(m2,np.eye(3,3)), np.zeros((3,3))],
        [np.zeros((3,3)),            R02.T@I2@R02 ]
    ])@Jc2
    M3 = Jc3.T@np.block([
        [np.multiply(m3,np.eye(3,3)), np.zeros((3,3))],
        [np.zeros((3,3)),            R03.T@I3@R03 ]
    ])@Jc3

    Mq = M2 + M3
    s.Mq = Mq

    


