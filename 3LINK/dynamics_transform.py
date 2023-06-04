import jax.numpy as jnp
from jax.numpy import pi, sin, cos, linalg
import jax
from functools import partial
from params import *
from homogeneousTransforms import *
from scipy.linalg import solve_continuous_lyapunov
from jax.scipy.linalg import sqrtm


@jax.jit
def dynamics_transform(x,v,D,Tq,dTqinvdq_values,dVdq):

    q1 = x.at[0,0].get()      #make the full q vector
    q2 = x.at[1,0].get()
    p1 = x.at[2,0].get()
    p2 = x.at[3,0].get()

    q0 = jnp.array([     #this is qhat. q0 denotes before momentum transform
        [q1], [q2]
        ])
    p = jnp.array([     #this function already inputs transformed p
        [p1],[p2]
        ])

    dVdq1 = dVdq.at[0,0].get()
    dVdq2 = dVdq.at[1,0].get()

    
    gq_hat = jnp.array([[dVdq1],[dVdq2]])


    #Dynamics after Transform

    dTqinvdq1 = dTqinvdq_values.at[0].get()
    dTqinvdq2 = dTqinvdq_values.at[1].get()

    dTqinv_pdq1 = dTqinvdq1@p
    dTqinv_pdq2 = dTqinvdq2@p

    temp = jnp.block([dTqinv_pdq1, dTqinv_pdq2])
    tempT = jnp.transpose(temp)
    # print('temp',temp)
    tempc = tempT - temp

    Cq = Tq@tempc@Tq
    Dq = Tq@D@Tq
    
    dV = jnp.array([[dVdq1], [dVdq2]])
    # print('dVdq', jnp.transpose(dVdq))
    tau = jnp.block([[jnp.zeros((2,1))],[v]])       #add  control input
    
    xdot_transform = jnp.block([        #transformed dynamic equations
        [jnp.zeros((2,2)), Tq],
        [-Tq,      Cq-Dq],
    ])@jnp.block([[dV],[p]])  + tau 

    return xdot_transform

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