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


# @partial(jax.jit, static_argnames=['s'])
# def dynamics_Transform(x,Mq,Tq,dMdq_values,dTqinvdq_values,dVdq, gravComp,x_err,s): #need to put in constants
@jax.jit
def dynamics_Transform(x,D,Tq,dTqinvdq_values,dVdq,gravComp,x_err,Kp,Kd,alpha): #need to put in constants

    q2 = x.at[0,0].get()      #make the full q vector
    q4 = x.at[1,0].get()
    q6 = x.at[2,0].get()
    p2 = x.at[3,0].get()
    p4 = x.at[4,0].get()
    p6 = x.at[5,0].get()

    
    q0 = jnp.array([     #this is qhat. q0 denotes before momentum transform
        [q2], [q4], [q6]
        ])
    p = jnp.array([     #this function already inputs transformed p
        [p2],[p4],[p6]
        ])

    # q0 = jnp.transpose(q0)

    # print('dynamics q0',q0)
    # Gravitation torque
    # g0 = jnp.array([[0],[0],[-s.g]])

    dVdq1 = dVdq.at[0,0].get()
    dVdq2 = dVdq.at[1,0].get()
    dVdq3 = dVdq.at[2,0].get()
    # print(dVdq1)

    # gq = gravTorque(s,Jc2,Jc3,Jc4,Jc5,Jc6,Jc7,Jc8)
    gq_hat = jnp.array([[dVdq1],[dVdq2],[dVdq3]])

    # D = 0.5*jnp.eye(3)

    #Dynamics after Transform

    dTqinvdq1 = dTqinvdq_values.at[0].get()
    dTqinvdq2 = dTqinvdq_values.at[1].get()
    dTqinvdq3 = dTqinvdq_values.at[2].get()

    
    dTqinv_pdq1 = dTqinvdq1@p
    dTqinv_pdq2 = dTqinvdq2@p
    dTqinv_pdq3 = dTqinvdq3@p

    temp = jnp.block([dTqinv_pdq1, dTqinv_pdq2, dTqinv_pdq3])
    tempT = jnp.transpose(temp)
    # print('temp',temp)
    tempc = tempT - temp

    Cq = Tq@tempc@Tq
    # print(Cq)

    # tempDq = linalg.solve(jnp.transpose(Tqinv),jnp.transpose(D))
    # Dq = linalg.solve(Tqinv,tempDq)
    Dq = Tq@D@Tq
    # print('Dq',Dq)

    #Hamiltonian
    dV = jnp.array([[dVdq1], [dVdq2], [dVdq3]])
    # print('dV',dV)

    # print('Cq-Dq', Cq-Dq)
    ##############CONTROLLER#########################
    q_tilde = x_err.at[0:3].get()
    p_tilde = x_err.at[3:6].get()
    # print(q_tilde,p_tilde)

    #build control law v
    D_hat = jnp.zeros((3,3))
    v = alpha*(Cq - D_hat - Kd)@Kp@(q_tilde + alpha*p_tilde) - Tq@Kp@(q_tilde + alpha*p_tilde) - Kd@p_tilde
    # print(v)

    u = gravComp*gq_hat    #multiply by the boolean to change
    # print('u',u)
    # print(gq_hat - jnp.array([[dVdq1],[dVdq2],[dVdq3]]))                #how could i forget this is ZERO.
    
     #torque input will be given by gravcomp torque plus control function.
    u_hat = Tq@u  + v      #changes into new momentum coordinates
    #Gq0 is identity(3) for u  torque input.  Gq = Tq*Gq0 = Tq*eye = Tq

    ## Stabilisation (equation 21 in paper)
    # Tqinv = linalg.solve(Tq,jnp.eye(3))
    # Ginv = Tqinv
    # u_hat = Ginv@(Tq@jnp.array([[dVdq1],[dVdq2],[dVdq3]]))# + v)
    # print(u_hat)

    tau = jnp.block([[jnp.zeros((3,1))],[u_hat]])


    xdot_transform = jnp.block([        #transformed dynamic equations
        [jnp.zeros((3,3)), Tq],
        [-Tq,      Cq-Dq],
    ])@jnp.block([[dV],[p]])  + tau 

    return xdot_transform     #i might not have realised this actually returns pdot. fix main loop to adjust


def observer(q,p,phat,phi,k_obv,Cq,Dq,Tq,xp, dVq, Gq, u,dTqinvdq_values):

    #observer needs to be constructed in main. efforts will go there

    #C(q) is found as part of the dynamic model, same as Tq. However, now need in terms of phat, not p. 
    dTqinvdq1 = dTqinvdq_values.at[0].get()
    dTqinvdq2 = dTqinvdq_values.at[1].get()
    dTqinvdq3 = dTqinvdq_values.at[2].get()

    dTqinv_phatdq1 = dTqinvdq1@phat
    dTqinv_phatdq2 = dTqinvdq2@phat
    dTqinv_phatdq3 = dTqinvdq3@phat

    temphat = jnp.block([dTqinv_phatdq1, dTqinv_phatdq2, dTqinv_phatdq3])
    temphatT = jnp.transpose(temphat)
    # print('temp',temp)
    tempchat = temphatT - temphat
    Cq_phat = Tq@tempchat@Tq

    u0 = 0
    xp_dot = (Cq_phat - Dq - phi*Tq)@phat - Tq@dVq + Gq@(u-u0)

    phi = phi + k_obv           #this happens for an 'instantaneos change'
    xp = xp - k_obv*q

    phat = xp + phi*q       #estimation of momentum states


    return  phat,xp_dot

# def gravTorque(s,Jc2,Jc3,Jc4,Jc5,Jc6,Jc7,Jc8):

#     g0 = jnp.array([[0],[0],[-s.g]])
#     # g0 = s.g0
#     tauc2 = jnp.block([
#         [jnp.multiply(s.m2,g0)],
#         [jnp.zeros((3,1))]
#     ])
#     tauc3 = jnp.block([
#         [jnp.multiply(s.m3,g0)],
#         [jnp.zeros((3,1))]
#     ])
#     tauc4 = jnp.block([
#         [jnp.multiply(s.m4,g0)],
#         [jnp.zeros((3,1))]
#     ])
#     tauc5 = jnp.block([
#         [jnp.multiply(s.m5,g0)],
#         [jnp.zeros((3,1))]
#     ])
#     tauc6 = jnp.block([
#         [jnp.multiply(s.m6,g0)],
#         [jnp.zeros((3,1))]
#     ])
#     tauc7 = jnp.block([
#         [jnp.multiply(s.m7,g0)],
#         [jnp.zeros((3,1))]
#     ])
#     tauc8 = jnp.block([
#         [jnp.multiply(s.m8,g0)],
#         [jnp.zeros((3,1))]
#     ])

#     gq = jnp.transpose(-((jnp.transpose(tauc2))@Jc2 + (jnp.transpose(tauc3))@Jc3 + (jnp.transpose(tauc4))@Jc4 + (jnp.transpose(tauc5))@Jc5 + (jnp.transpose(tauc6))@Jc6 + (jnp.transpose(tauc7))@Jc7 + (jnp.transpose(tauc8))@Jc8))
#     return gq

# def inputTorque(q,gq, s):

    # gq = gravTorque(s)

    # if s.controlActive == 0:
    # controlAction = jnp.zeros((7,1))
 
    # controlAction = jnp.array([[0.],[0.],[0.],[5.],[0.],[0.],[0.]])
        # print('cA',controlAction)

    # operand = jnp.array([0.])
    # controlAction = lax.cond(t<0, lambda x: jnp.array([[0.],[0.],[0.],[5.],[0.],[0.],[0.]]), lambda x: jnp.zeros((7,1)), operand)

    u = s.controlAction + gq
    # print('u',u,'gq',gq)
    # u = gq
    tau = jnp.block([[jnp.zeros((7,1))],[u]])
    return tau
