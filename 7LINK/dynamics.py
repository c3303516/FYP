import jax.numpy as jnp
from jax.numpy import pi, sin, cos, linalg
from params import *
# from main import massMatrix
from effectorFKM import FKM


def dynamics_test(x, s):
    q1 = x.at[(0,0)].get()
    q2 = x.at[(1,0)].get()
    q = jnp.array([
        q1, q2
    ])
    p = jnp.array([
        x.at[(2,0)].get(), x.at[(3,0)].get()
        ])
    # Gravitation torque
    FKM(q1,q2,s)
    Mq = massMatrix(q, s)
    s.g0 = jnp.array([[0],[0],[-g]])
    g0 = s.g0
    gq = gravTorque(q,s)
    dVdq = s.dV(q)
    dVdq1 = dVdq.at[0].get()
    dVdq2 = dVdq.at[1].get()

    Mq = s.Mq
    dMdq1 = s.dMdq1
    dMdq2 = s.dMdq2

    b = jnp.transpose(linalg.solve(jnp.transpose(Mq), jnp.transpose(dMdq1)))
    dMinvdq1 = linalg.solve(-Mq, b)

    b = jnp.transpose(linalg.solve(jnp.transpose(Mq), jnp.transpose(dMdq2)))
    dMinvdq2 = linalg.solve(-Mq, b)

    # print('dVdq', jnp.transpose(dVdq))
    dHdq = 0.5*(jnp.array([
        [jnp.transpose(p)@dMinvdq1@p],
        [jnp.transpose(p)@dMinvdq2@p],
    ])) + jnp.array([[dVdq1], [dVdq2]])    # addition now works and gives same shape, however numerical values are incorrect

    # print('dHdq', dHdq)

    dHdp = 0.5*jnp.block([
    [(jnp.transpose(p)@linalg.solve(s.Mq,jnp.array([[1.],[0.]]))) + (jnp.array([1.,0.])@linalg.solve(s.Mq,p))],
    [jnp.transpose(p)@linalg.solve(s.Mq,jnp.array([[0.],[1.]])) + jnp.array([0.,1.])@linalg.solve(s.Mq,p)],
    ]) 

    # print('dHdp', dHdp)
    D = 0.5*jnp.eye(2)
    xdot = jnp.block([
        [jnp.zeros((2,2)), jnp.eye(2)],
        [-jnp.eye(2),      -D ],
    ])@jnp.block([[dHdq],[dHdp]])

    return xdot

def gravTorque(s):
    Jc2 = s.Jc2
    Jc3 = s.Jc3
    Jc4 = s.Jc4
    Jc5 = s.Jc5
    Jc6 = s.Jc6
    Jc7 = s.Jc7
    Jc8 = s.Jc8

    # g0 = jnp.array([[0],[0],[-g]])
    g0 = s.g0
    tauc2 = jnp.block([
        [jnp.multiply(s.m2,g0)],
        [jnp.zeros((3,1))]
    ])
    tauc3 = jnp.block([
        [jnp.multiply(s.m3,g0)],
        [jnp.zeros((3,1))]
    ])
    tauc4 = jnp.block([
        [jnp.multiply(s.m4,g0)],
        [jnp.zeros((3,1))]
    ])
    tauc5 = jnp.block([
        [jnp.multiply(s.m5,g0)],
        [jnp.zeros((3,1))]
    ])
    tauc6 = jnp.block([
        [jnp.multiply(s.m6,g0)],
        [jnp.zeros((3,1))]
    ])
    tauc7 = jnp.block([
        [jnp.multiply(s.m7,g0)],
        [jnp.zeros((3,1))]
    ])
    tauc8 = jnp.block([
        [jnp.multiply(s.m8,g0)],
        [jnp.zeros((3,1))]
    ])

    gq = jnp.transpose(-((jnp.transpose(tauc2))@Jc2 + (jnp.transpose(tauc3))@Jc3 + (jnp.transpose(tauc4))@Jc4 + (jnp.transpose(tauc5))@Jc5 + (jnp.transpose(tauc6))@Jc6 + (jnp.transpose(tauc7))@Jc7 + (jnp.transpose(tauc8))@Jc8))
    return gq