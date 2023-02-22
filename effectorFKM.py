
import autograd.numpy as np
from params import *
from autograd.numpy import pi, cos, sin
from homogeneousTransforms import *


def FKM(q,s):

    A01 = rotx(-pi/2)@trany(-l1)@rotz(q[0])
    print('break')
    A12 = trany(-l2)@rotz(q[1])
    A23 = trany(-l3)@rotx(pi/2)

    print(A01)

    A01 = A01.real
    A12 = A12.real
    A23 = A23.real


    A02 = A01*A12
    A03 = A02*A23

    s.A01 = A01
    s.A12 = A12
    s.A23 = A23

    s.A01 = A01
    s.A02 = A02
    s.A03 = A03
    

    # FKM = np.array([(A01),(A12),(A23)])
    # print(FKM)

    r030 = A03[0:3,3]

    a = np.sqrt(A03[2,1]*A03[2,1] + A03[2,2]*A03[2,2])
    psi  = np.arctan2(A03[1,0],A03[0,0])
    theta = np.arctan2(-A03[2,0], a)
    phi = np.arctan2(A03[2,1],A03[2,2])

    xe = np.matrix([[r030], [phi], [theta], [psi]], dtype=object)
    return xe




# q = np.matrix([[0],[0]])
# xe = effectorFKM(q)
# print(s.A03)