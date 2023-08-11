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
def dynamics_test(x,v,D,Mq,dMdq_values,dVdq): #need to put in constants

    q2 = x.at[0,0].get()      #make the full q vector
    q4 = x.at[1,0].get()
    q6 = x.at[2,0].get()
    p2 = x.at[3,0].get()
    p4 = x.at[4,0].get()
    p6 = x.at[5,0].get()

    q0 = jnp.array([     #this is qhat. q0 denotes before momentum transform
        [q2], [q4], [q6]
        ])
    p0 = jnp.array([     #this function already inputs transformed p
        [p2],[p4],[p6]
        ])


    dVdq1 = dVdq.at[0].get()
    dVdq2 = dVdq.at[1].get()
    dVdq3 = dVdq.at[2].get()

    # Mass matrix inverse

    dMdq1 = dMdq_values.at[0].get()
    dMdq2 = dMdq_values.at[1].get()
    dMdq3 = dMdq_values.at[2].get()

    # testing = jnp.transpose(Mq)          # what is causing the error?? Mq or dMdq1here doensn't throw an erro
    temp1 = jnp.transpose(linalg.solve(jnp.transpose(Mq), jnp.transpose(dMdq1)))
    temp2 = jnp.transpose(linalg.solve(jnp.transpose(Mq), jnp.transpose(dMdq2)))
    temp3 = jnp.transpose(linalg.solve(jnp.transpose(Mq), jnp.transpose(dMdq3)))
    
    dMinvdq1 = -linalg.solve(Mq, temp1)
    dMinvdq2 = -linalg.solve(Mq, temp2)
    dMinvdq3 = -linalg.solve(Mq, temp3)


    # print(dMinvdq1)
    # print('dVdq', dVdq)
    # print(jnp.array([[dVdq1], [dVdq2], [dVdq3]]))

    Htemp1 = jnp.transpose(p0)@dMinvdq1@p0
    Htemp2 = jnp.transpose(p0)@dMinvdq2@p0
    Htemp3 = jnp.transpose(p0)@dMinvdq3@p0

    # dHdq = 0.5*(jnp.array([       #how i used to do it. fixing p0 array stuff mess this up
    #     [jnp.transpose(p0)@dMinvdq1@p0],
    #     [jnp.transpose(p0)@dMinvdq2@p0],
    #     [jnp.transpose(p0)@dMinvdq3@p0],
    # ])) + jnp.array([[dVdq1], [dVdq2], [dVdq3]])    # addition now works and gives same shape, however numerical values are incorrect

    dHdq = 0.5*jnp.array([
        [Htemp1.at[0,0].get()],
        [Htemp2.at[0,0].get()],
        [Htemp3.at[0,0].get()],
    ]) + jnp.array([[dVdq1], [dVdq2], [dVdq3]]) 


    # print('dHdq', dHdq)

    dHdp = 0.5*jnp.block([
    [jnp.transpose(p0)@linalg.solve(Mq,jnp.array([[1.],[0.],[0.]])) + (jnp.array([1.,0.,0.])@linalg.solve(Mq,p0))],
    [jnp.transpose(p0)@linalg.solve(Mq,jnp.array([[0.],[1.],[0.]])) + (jnp.array([0.,1.,0.])@linalg.solve(Mq,p0))],
    [jnp.transpose(p0)@linalg.solve(Mq,jnp.array([[0.],[0.],[1.]])) + (jnp.array([0.,0.,1.])@linalg.solve(Mq,p0))],
    ]) 


    gq_hat = jnp.array([gq.at[1].get(), #might need to check this
                        gq.at[3].get(),
                        gq.at[5].get(),])

    u = controlAction + gravComp*gq_hat         #multiply by the boolean to change
    tau = jnp.block([[jnp.zeros((3,1))],[u]])

    xdot = jnp.block([
        [jnp.zeros((3,3)), jnp.eye(3)],
        [-jnp.eye(3),      -D ]
    ])@jnp.block([[dHdq],[dHdp]])  + tau     #CHECK YOU ACTUALLY PUT THE GRAVITY IN 

    # print('xdot', xdot)


    return xdot
