import jax.numpy as jnp
from jax.numpy import pi, sin, cos, linalg

from params import *
from homogeneousTransforms import *
from jax.config import config
config.update("jax_enable_x64", True)

def massMatrix(q, s):
    q1 = q.at[0].get()
    q2 = q.at[1].get()

    # Geometric Jacobians
    R01 = s.A01[0:3,0:3]     #rotation matrices
    R12 = s.A12[0:3,0:3]

    A0c1 = rotx(-pi/2)@trany(-c1)@rotz(q1)
    A0c2 = s.A01@trany(-c2)@rotz(q2)
    A0c3 = s.A02@trany(-c3)@rotx(pi/2)

    r100   = s.A01[0:3,[3]]
    r200   = s.A02[0:3,[3]]

    rc100   = A0c1[0:3,[3]]
    rc200   = A0c2[0:3,[3]]
    rc300   = A0c3[0:3,[3]]

    z00 = jnp.array([[0.], [0.], [1.]])
    z01 = R01@z00
    z02 = R01@R12@z00

    ske1 = skew(z01)
    ske2 = skew(z02)

    Jc2   = jnp.block([
        [ske1@(rc200-r100), jnp.zeros((3,1))],
        [z01,               jnp.zeros((3,1))]
        ])
    Jc3   = jnp.block([
        [ske1@(rc300-r100), ske2@(rc300-r200)],
        [z01,               z02]
        ])

    s.Jc2 = Jc2
    s.Jc3 = Jc3
    # Mass Matrix
    R02 = s.A02[0:3,0:3]
    R03 = s.A03[0:3,0:3]

    M2 = Jc2.T@jnp.block([
        [jnp.multiply(m2,jnp.eye(3,3)), jnp.zeros((3,3))],
        [jnp.zeros((3,3)),            R02.T@I2@R02 ]
    ])@Jc2
    M3 = Jc3.T@jnp.block([
        [jnp.multiply(m3,jnp.eye(3,3)), jnp.zeros((3,3))],
        [jnp.zeros((3,3)),            R03.T@I3@R03 ]
    ])@Jc3

    Mq = M2 + M3
    s.Mq = Mq
    return Mq, s

    


