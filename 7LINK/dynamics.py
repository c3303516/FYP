import jax.numpy as jnp
from jax.numpy import pi, sin, cos, linalg
import jax
from functools import partial
from params import *
# from main import massMatrix
from massMatrix import massMatrix
from effectorFKM import FKM
from copy import deepcopy

@partial(jax.jit, static_argnames=['s'])
def dynamics_test(x, s):
    q1 = x.at[(0,0)].get()
    q2 = x.at[(1,0)].get()
    q3 = x.at[(2,0)].get()
    q4 = x.at[(3,0)].get()
    q5 = x.at[(4,0)].get()
    q6 = x.at[(5,0)].get()
    q7 = x.at[(6,0)].get()

    p1 = x.at[(7,0)].get()
    p2 = x.at[(8,0)].get()
    p3 = x.at[(9,0)].get()
    p4 = x.at[(10,0)].get()
    p5 = x.at[(11,0)].get()
    p6 = x.at[(12,0)].get()
    p7 = x.at[(13,0)].get()

    q = jnp.array([
        q1, q2, q3, q4, q5, q6, q7
        ])
    p = jnp.array([
        p1,p2,p3,p4,p5,p6,p7
        ])
    # print('q1',q1)
    # print('q2',q2)
    # print('p',p)
    # print('q',q)

    s = FKM(q,deepcopy(s))

    Mq, s = massMatrix(q, deepcopy(s))
    # Gravitation torque
    s.g0 = jnp.array([[0],[0],[-s.g]])
    g0 = s.g0
    
    dVdq = s.dV(q,s)
    dVdq1 = dVdq.at[0].get()
    dVdq2 = dVdq.at[1].get()
    dVdq3 = dVdq.at[2].get()
    dVdq4 = dVdq.at[3].get()
    dVdq5 = dVdq.at[4].get()
    dVdq6 = dVdq.at[5].get()
    dVdq7 = dVdq.at[6].get()


    # Mass matrix inverse
    # Mq = s.Mq
    dMdq1 = s.dMdq1
    dMdq2 = s.dMdq2
    dMdq3 = s.dMdq3
    dMdq4 = s.dMdq4
    dMdq5 = s.dMdq5
    dMdq6 = s.dMdq6
    dMdq7 = s.dMdq7

    temp1 = jnp.transpose(linalg.solve(jnp.transpose(Mq), jnp.transpose(dMdq1)))
    temp2 = jnp.transpose(linalg.solve(jnp.transpose(Mq), jnp.transpose(dMdq2)))
    temp3 = jnp.transpose(linalg.solve(jnp.transpose(Mq), jnp.transpose(dMdq3)))
    temp4 = jnp.transpose(linalg.solve(jnp.transpose(Mq), jnp.transpose(dMdq4)))
    temp5 = jnp.transpose(linalg.solve(jnp.transpose(Mq), jnp.transpose(dMdq5)))
    temp6 = jnp.transpose(linalg.solve(jnp.transpose(Mq), jnp.transpose(dMdq6)))
    temp7 = jnp.transpose(linalg.solve(jnp.transpose(Mq), jnp.transpose(dMdq7)))
    
    dMinvdq1 = linalg.solve(-Mq, temp1)
    dMinvdq2 = linalg.solve(-Mq, temp2)
    dMinvdq3 = linalg.solve(-Mq, temp3)
    dMinvdq4 = linalg.solve(-Mq, temp4)
    dMinvdq5 = linalg.solve(-Mq, temp5)
    dMinvdq6 = linalg.solve(-Mq, temp6)
    dMinvdq7 = linalg.solve(-Mq, temp7)


    # print('dVdq', jnp.transpose(dVdq))
    dHdq = 0.5*(jnp.array([
        [jnp.transpose(p)@dMinvdq1@p],
        [jnp.transpose(p)@dMinvdq2@p],
        [jnp.transpose(p)@dMinvdq3@p],
        [jnp.transpose(p)@dMinvdq4@p],
        [jnp.transpose(p)@dMinvdq5@p],
        [jnp.transpose(p)@dMinvdq6@p],
        [jnp.transpose(p)@dMinvdq7@p]
    ])) + jnp.array([[dVdq1], [dVdq2], [dVdq3], [dVdq4], [dVdq5], [dVdq6], [dVdq7]])    # addition now works and gives same shape, however numerical values are incorrect

    # print('dHdq', dHdq)

    dHdp = 0.5*jnp.block([
    [jnp.transpose(p)@linalg.solve(s.Mq,jnp.array([[1.],[0.],[0.],[0.],[0.],[0.],[0.]])) + (jnp.array([1.,0.,0.,0.,0.,0.,0.])@linalg.solve(s.Mq,p))],
    [jnp.transpose(p)@linalg.solve(s.Mq,jnp.array([[0.],[1.],[0.],[0.],[0.],[0.],[0.]])) + (jnp.array([0.,1.,0.,0.,0.,0.,0.])@linalg.solve(s.Mq,p))],
    [jnp.transpose(p)@linalg.solve(s.Mq,jnp.array([[0.],[0.],[1.],[0.],[0.],[0.],[0.]])) + (jnp.array([0.,0.,1.,0.,0.,0.,0.])@linalg.solve(s.Mq,p))],
    [jnp.transpose(p)@linalg.solve(s.Mq,jnp.array([[0.],[0.],[0.],[1.],[0.],[0.],[0.]])) + (jnp.array([0.,0.,0.,1.,0.,0.,0.])@linalg.solve(s.Mq,p))],
    [jnp.transpose(p)@linalg.solve(s.Mq,jnp.array([[0.],[0.],[0.],[0.],[1.],[0.],[0.]])) + (jnp.array([0.,0.,0.,0.,1.,0.,0.])@linalg.solve(s.Mq,p))],
    [jnp.transpose(p)@linalg.solve(s.Mq,jnp.array([[0.],[0.],[0.],[0.],[0.],[1.],[0.]])) + (jnp.array([0.,0.,0.,0.,0.,1.,0.])@linalg.solve(s.Mq,p))],
    [jnp.transpose(p)@linalg.solve(s.Mq,jnp.array([[0.],[0.],[0.],[0.],[0.],[0.],[1.]])) + (jnp.array([0.,0.,0.,0.,0.,0.,1.])@linalg.solve(s.Mq,p))]
    ]) 

    # print('dHdp', dHdp)

    tau = inputTorque(q,s)
    D = 0.5*jnp.eye(7)
    xdot = jnp.block([
        [jnp.zeros((7,7)), jnp.eye(7)],
        [-jnp.eye(7),      -D ],
    ])@jnp.block([[dHdq],[dHdp]])  + tau     #CHECK YOU ACTUALLY PUT THE GRAVITY IN 

    # print('xdot', xdot)

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

def inputTorque(q,s):

    # gq = gravTorque(s)

    if s.controlActive == 0:
        controlAction = jnp.zeros((7,1))
    
    if s.gravityCompensation == 1:
        gq = gravTorque(s)
    else:
        gq = jnp.zeros((7,1))
        
    u = controlAction + gq

    tau = jnp.block([[jnp.zeros((7,1))],[u]])
    return tau
