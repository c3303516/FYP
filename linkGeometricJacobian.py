from cmath import pi, sin
import autograd.numpy as np
from autograd import grad

from params import *
from homogeneousTransforms import *

def GeoJac(q,s):

    R01 = s.A01[0:3,0:3]     #rotation matrices
    R12 = s.A12[0:3,0:3]

    A0c1 = rotx(-pi/2)*trany(-c1)*rotz(q[0])
    A0c2 = s.A01*trany(-c2)*rotz(q[1])
    A0c3 = s.A02*trany(-c3)*rotx(pi/2)

    A0c1 = A0c1.real
    A0c2 = A0c2.real
    A0c3 = A0c3.real

    r100   = s.A01[0:3,3]
    r200   = s.A02[0:3,3]

    rc100   = A0c1[0:3,3]
    rc200   = A0c2[0:3,3]
    rc300   = A0c3[0:3,3]

    s.rc100 = rc100
    s.rc200 = rc200
    s.rc300 = rc300

    z00 = np.matrix([[0], [0], [1]])
    # print(z00)
    # print(R01)
    z01 = R01@z00
    z02 = R01@R12@z00

    # print(rc200-r100)
    

    ske1 = skew(z01)
    # print('sk1',ske1)
    ske2 = skew(z02)
    Jc2   = np.block([
        [ske1@(rc200-r100), np.zeros((3,1))],
        [z01,               np.zeros((3,1))]
        ])
    Jc3   = np.block([
        [ske1@(rc300-r100), ske2@(rc300-r200)],
        [z01,               z02]
        ])

    s.Jc2 = np.asmatrix(Jc2)
    s.Jc3 = np.asmatrix(Jc3)
    

    

