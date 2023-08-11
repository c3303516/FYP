from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.numpy import pi, sin, cos, linalg
from params import *
from massMatrix import massMatrix
from effectorFKM import FKM
import jax
from functools import partial



@partial(jax.jit, static_argnames=['s'])
def dynamics_test(x, s):
    q1 = x.at[(0,0)].get()
    q2 = x.at[(1,0)].get()
    q = jnp.array([
        q1, q2
    ])
    p1 = x.at[(2,0)].get()
    p2 = x.at[(3,0)].get()
    p = jnp.array([
        p1,p2
        ])
    # print('q1',q1)
    # print('q2',q2)
    # print('p',p)
    # print('q',q)
    p = jnp.transpose(p)
    q = jnp.transpose(q)

    s = FKM(q,s)

    Mq, s = massMatrix(q, s)
    # Gravitation torque

    s.g0 = jnp.array([[0],[0],[-g]])
    g0 = s.g0
    gq = gravTorque(s)
    s.gq = gq.at[0].get()
    # print('gq',s.gq)
    dVdq = s.dV(q)
    dVdq1 = dVdq.at[0].get()
    dVdq2 = dVdq.at[1].get()

    # Mass matrix inverse
    # Mq = s.Mq
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

    jnp.set_printoptions(precision=15)
    # print('dMdq1',dMdq1)
    # print('dMdq2',dMdq2)
    # print('dMinvdq1',dMinvdq1)
    # print('dMinvdq2',dMinvdq2)
    # print('dHdq', dHdq)

    dHdp = 0.5*jnp.block([
    [(jnp.transpose(p)@linalg.solve(Mq,jnp.array([[1.],[0.]]))) + (jnp.array([1.,0.])@linalg.solve(Mq,p))],
    [(jnp.transpose(p)@linalg.solve(Mq,jnp.array([[0.],[1.]]))) + (jnp.array([0.,1.])@linalg.solve(Mq,p))],
    ]) 

    # print('dHdp', dHdp)
    D = 0.5*jnp.eye(2)
    xdot = jnp.block([
        [jnp.zeros((2,2)), jnp.eye(2)],
        [-jnp.eye(2),      -D ],
    ])@jnp.block([[dHdq],[dHdp]]) + jnp.block([[jnp.zeros((2,1))],[gq]])

    # +  [zeros(2,1);(u)]   #need to implement this grav torque control action


    return xdot

def gravTorque(s):
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