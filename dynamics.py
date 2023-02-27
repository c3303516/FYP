import jax.numpy as jnp
from jax.numpy import pi, sin, cos, linalg
from params import *
from massMatrix import massMatrix
from effectorFKM import FKM


def dynamics(q, s):
    # Gravitation torque
    FKM(q,s)
    Mq = massMatrix(q, s)

    gq = gravTorque(q,s)
    b = jnp.transpose(linalg.solve(jnp.transpose(Mq), jnp.transpose(s.dMdq1)))
    dMinvdq1 = linalg.solve(-Mq, b)

    b = jnp.transpose(linalg.solve(jnp.transpose(Mq), jnp.transpose(s.dMdq2)))
    dMinvdq2 = linalg.solve(-Mq, b)
    return 

def gravTorque(q,s):
    Jc2 = s.Jc2
    Jc3 = s.Jc3
    # g0 = jnp.array([[0],[0],[-g]])
    g0 = s.g0
    tauc2 = jnp.block([
        [jnp.multiply(m2,g0)],
        [jnp.zeros((3,1))]
    ])
    tauc3 = jnp.block([
        [jnp.multiply(m3,g0)],
        [jnp.zeros((3,1))]
    ])

    gq = jnp.transpose(-((jnp.transpose(tauc2))@Jc2 + (jnp.transpose(tauc3))@Jc3))
    return gq