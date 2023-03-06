import jax.numpy as jnp
from jax.numpy import pi, sin, cos, linalg
from params import *

from params import *
from homogeneousTransforms import *


def FKM(q1,q2,s):

    A01 = rotx(-pi/2)@trany(-l1)@rotz(q1)
    A12 = trany(-l2)@rotz(q2)
    A23 = trany(-l3)@rotx(pi/2)

    A02 = A01@A12
    A03 = A02@A23


    s.A01 = A01
    s.A12 = A12
    s.A23 = A23

    s.A01 = A01
    s.A02 = A02
    s.A03 = A03
    #end effector pose
    r030 = A03[0:3,[3]]

    a = jnp.sqrt(A03[2,1]*A03[2,1] + A03[2,2]*A03[2,2])
    psi  = jnp.arctan2(A03[1,0],A03[0,0])
    theta = jnp.arctan2(-A03[2,0], a)
    phi = jnp.arctan2(A03[2,1],A03[2,2])

    s.xe = jnp.block([[r030], [phi], [theta], [psi]])
    return s




# q = np.matrix([[0],[0]])
# xe = effectorFKM(q)
# print(s.A03)