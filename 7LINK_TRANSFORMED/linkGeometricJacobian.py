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

    q1 = q0.at[0].get()
    q2 = q0.at[1].get()

    A0c1 = tranx(s.c1[0])*trany(c1[1])*tranz(c1[2]) 
    A0c2 = s.A01*tranx(s.c2[0])*trany(c2[1])*tranz(c2[2])
    A0c3 = s.A02*tranx(s.c3[0])*trany(c3[1])*tranz(c3[2])
    A0c4 = s.A03*tranx(s.c4[0])*trany(c4[1])*tranz(c4[2])
    A0c5 = s.A04*tranx(s.c5[0])*trany(c5[1])*tranz(c5[2])
    A0c6 = s.A05*tranx(s.c6[0])*trany(c6[1])*tranz(c6[2])
    A0c7 = s.A06*tranx(s.c7[0])*trany(c7[1])*tranz(c7[2])
    A0c8 = s.A07*tranx(s.c8[0])*trany(c8[1])*tranz(c8[2])
    A0cG = s.A0E*tranz(s.cGripper[2])

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
    

    

