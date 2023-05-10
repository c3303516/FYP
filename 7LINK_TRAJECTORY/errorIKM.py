import jax.numpy as jnp
from jax.numpy import pi, sin, cos, linalg
import jax
from functools import partial
from params import *
from homogeneousTransforms import *
from scipy.linalg import solve_continuous_lyapunov
from jax.scipy.linalg import sqrtm
import csv


def errorIKM(q, q0, xestar,s):

    #q0 is init condition. q is iteration to solve for xstar
    #xstar is currently just positon, no angles
    #EFFECTOR IKM
    q2 = q.at[(0,0)].get()      #make the full q vector
    q4 = q.at[(1,0)].get()
    q6 = q.at[(2,0)].get()

    q = jnp.array([     #this is qhat. q0 denotes before momentum transform
        q2, q4, q6
        ])

    q = jnp.transpose(q)
    qconstants = s.constants
    q_bold = qconstants.at[0].get()     #constrained variabless
    
    q1 = q_bold.at[0].get()
    q3 = q_bold.at[1].get()
    q5 = q_bold.at[2].get()
    q7 = q_bold.at[3].get()

    A01 = tranz(s.l1)@rotx(pi)@rotz(q1)          
    A12 = rotx(pi/2)@tranz(-s.d1)@trany(-s.l2)@rotz(q2)
    A23 = rotx(-pi/2)@trany(s.d2)@tranz(-s.l3)@rotz(q3)
    A34 = rotx(pi/2)@tranz(-s.d2)@trany(-s.l4)@rotz(q4)
    A45 = rotx(-pi/2)@trany(s.d2)@tranz(-s.l5)@rotz(q5)
    A56 = rotx(pi/2)@trany(-s.l6)@rotz(q6)
    A67 = rotx(-pi/2)@tranz(-s.l7)@rotz(q7)
    A7E = rotx(pi)@tranz(s.l8)    
    AEG = tranz(s.lGripper)

    A02 = A01@A12
    A03 = A02@A23
    A04 = A03@A34
    A05 = A04@A45
    A06 = A05@A56
    A07 = A06@A67
    A0E = A07@A7E
    #end effector pose
    r0E0 = A0E[0:3,[3]]

    #note: q is the solver's 'guess', qstar is the trajectory
    e_q = q - q0
    pose_weights = jnp.array([1,1,1])
    e_pose = r0E0 - xestar          #only position for now, no angles
    sqK = 100*jnp.diag(pose_weights)
    q_weights = jnp.array([5,3,1])
    sqW = 1*jnp.diag(q_weights)

    sqM = jnp.block([
        [sqK,               jnp.zeros((3, 3))],
        [jnp.zeros((3, 3)),  sqW              ]
    ])

#     s = effectorAnalyticalJacobian(q, param, s);      #this is here for future reference if I want to fix angles
#     JA = s.JA;
#     J = [sqW;
#          sqK*JA];

    e = sqM@jnp.block([e_q,e_pose])
                       #coudl use jacobian and feed through. might decrease time
    return e