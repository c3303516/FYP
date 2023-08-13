import jax.numpy as jnp
from jax.numpy import pi, sin, cos, linalg
import jax
from functools import partial
from params import *
from homogeneousTransforms import *
from scipy.linalg import solve_continuous_lyapunov
from jax.scipy.linalg import sqrtm
# from main import massMatrix
# from massMatrix import massMatrix
# from effectorFKM import FKM
# from copy import deepcopy
from jax import lax
import csv

from scipy.integrate import RK45


# @partial(jax.jit, static_argnames=['s'])
@jax.jit
def dynamics(x,v,D,Mq,dMdq_values,dVdq): #need to put in constants

    q1 = x.at[0,0].get()      #make the full q vector
    q2 = x.at[1,0].get()
    q3 = x.at[2,0].get()
    q4 = x.at[3,0].get()
    q5 = x.at[4,0].get()
    q6 = x.at[5,0].get()
    q7 = x.at[6,0].get()
    p1 = x.at[7,0].get()      #make the full q vector
    p2 = x.at[8,0].get()
    p3 = x.at[9,0].get()
    p4 = x.at[10,0].get()
    p5 = x.at[11,0].get()
    p6 = x.at[12,0].get()
    p7 = x.at[13,0].get()

    q0 = jnp.array([     #this is qhat. q0 denotes before momentum transform
        [q1], [q2], [q3], [q4], [q5], [q6], [q7]
        ])
    p0 = jnp.array([     #this function already inputs transformed p
        [p1], [p2], [p3], [p4], [p5], [p6], [p7]
        ])

    # print('p0',p0)
    dVdq1 = dVdq.at[0].get()
    dVdq2 = dVdq.at[1].get()
    dVdq3 = dVdq.at[2].get()

    # Mass matrix inverse

    dMdq1 = dMdq_values.at[0].get()
    dMdq2 = dMdq_values.at[1].get()
    dMdq3 = dMdq_values.at[2].get()
    dMdq4 = dMdq_values.at[3].get()
    dMdq5 = dMdq_values.at[4].get()
    dMdq6 = dMdq_values.at[5].get()
    dMdq7 = dMdq_values.at[6].get()

    # testing = jnp.transpose(Mq)          # what is causing the error?? Mq or dMdq1here doensn't throw an erro
    temp1 = jnp.transpose(linalg.solve(jnp.transpose(Mq), jnp.transpose(dMdq1)))
    temp2 = jnp.transpose(linalg.solve(jnp.transpose(Mq), jnp.transpose(dMdq2)))
    temp3 = jnp.transpose(linalg.solve(jnp.transpose(Mq), jnp.transpose(dMdq3)))
    temp4 = jnp.transpose(linalg.solve(jnp.transpose(Mq), jnp.transpose(dMdq4)))
    temp5 = jnp.transpose(linalg.solve(jnp.transpose(Mq), jnp.transpose(dMdq5)))
    temp6 = jnp.transpose(linalg.solve(jnp.transpose(Mq), jnp.transpose(dMdq6)))
    temp7 = jnp.transpose(linalg.solve(jnp.transpose(Mq), jnp.transpose(dMdq7)))
    
    dMinvdq1 = -linalg.solve(Mq, temp1)
    dMinvdq2 = -linalg.solve(Mq, temp2)
    dMinvdq3 = -linalg.solve(Mq, temp3)
    dMinvdq4 = -linalg.solve(Mq, temp4)
    dMinvdq5 = -linalg.solve(Mq, temp5)
    dMinvdq6 = -linalg.solve(Mq, temp6)
    dMinvdq7 = -linalg.solve(Mq, temp7)


    # print(dMinvdq1)
    # print('dVdq', dVdq)
    # print(jnp.array([[dVdq1], [dVdq2], [dVdq3]]))

    Htemp1 = jnp.transpose(p0.at[:,0].get())@dMinvdq1@p0.at[:,0].get()
    Htemp2 = jnp.transpose(p0.at[:,0].get())@dMinvdq2@p0.at[:,0].get()
    Htemp3 = jnp.transpose(p0.at[:,0].get())@dMinvdq3@p0.at[:,0].get()
    Htemp4 = jnp.transpose(p0.at[:,0].get())@dMinvdq4@p0.at[:,0].get()
    Htemp5 = jnp.transpose(p0.at[:,0].get())@dMinvdq5@p0.at[:,0].get()
    Htemp6 = jnp.transpose(p0.at[:,0].get())@dMinvdq6@p0.at[:,0].get()
    Htemp7 = jnp.transpose(p0.at[:,0].get())@dMinvdq7@p0.at[:,0].get()
    # print('Htemp', Htemp2)

    # dHdq = 0.5*(jnp.array([       #how i used to do it. fixing p0 array stuff mess this up
    #     [jnp.transpose(p0)@dMinvdq1@p0],
    #     [jnp.transpose(p0)@dMinvdq2@p0],
    #     [jnp.transpose(p0)@dMinvdq3@p0],
    # ])) + jnp.array([[dVdq1], [dVdq2], [dVdq3]])    # addition now works and gives same shape, however numerical values are incorrect

    dHdq = 0.5*jnp.array([
        [Htemp1],
        [Htemp2],
        [Htemp3],
        [Htemp4],
        [Htemp5],
        [Htemp6],
        [Htemp7],
    ]) + dVdq# jnp.array([[dVdq1], [dVdq2], [dVdq3]]) 


    # print('dHdq', dHdq)


    dHdp = 0.5*jnp.block([
    [jnp.transpose(p0)@linalg.solve(Mq,jnp.array([[1.],[0.],[0.],[0.],[0.],[0.],[0.]])) + (jnp.array([1.,0.,0.,0.,0.,0.,0.])@linalg.solve(Mq,p0))],
    [jnp.transpose(p0)@linalg.solve(Mq,jnp.array([[0.],[1.],[0.],[0.],[0.],[0.],[0.]])) + (jnp.array([0.,1.,0.,0.,0.,0.,0.])@linalg.solve(Mq,p0))],
    [jnp.transpose(p0)@linalg.solve(Mq,jnp.array([[0.],[0.],[1.],[0.],[0.],[0.],[0.]])) + (jnp.array([0.,0.,1.,0.,0.,0.,0.])@linalg.solve(Mq,p0))],
    [jnp.transpose(p0)@linalg.solve(Mq,jnp.array([[0.],[0.],[0.],[1.],[0.],[0.],[0.]])) + (jnp.array([0.,0.,0.,1.,0.,0.,0.])@linalg.solve(Mq,p0))],
    [jnp.transpose(p0)@linalg.solve(Mq,jnp.array([[0.],[0.],[0.],[0.],[1.],[0.],[0.]])) + (jnp.array([0.,0.,0.,0.,1.,0.,0.])@linalg.solve(Mq,p0))],
    [jnp.transpose(p0)@linalg.solve(Mq,jnp.array([[0.],[0.],[0.],[0.],[0.],[1.],[0.]])) + (jnp.array([0.,0.,0.,0.,0.,1.,0.])@linalg.solve(Mq,p0))],
    [jnp.transpose(p0)@linalg.solve(Mq,jnp.array([[0.],[0.],[0.],[0.],[0.],[0.],[1.]])) + (jnp.array([0.,0.,0.,0.,0.,0.,1.])@linalg.solve(Mq,p0))]
    ]) 

    # print('dHdp', dHdp)
      #multiply by the boolean to change
    tau = jnp.block([[jnp.zeros((7,1))],[v]])

    xdot = jnp.block([
        [jnp.zeros((7,7)), jnp.eye(7)],
        [-jnp.eye(7),      -D ]
    ])@jnp.block([[dHdq],[dHdp]])  + tau     #CHECK YOU ACTUALLY PUT THE GRAVITY IN 

    # print('xdot', xdot)


    return xdot
