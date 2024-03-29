import jax
from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.numpy import pi, sin, cos, linalg
from jax import grad, jacobian, jacfwd
import numpy as np
from effectorFKM import endEffector
from massMatrix_holonomic import massMatrix_holonomic
from dynamics_momentumTransform import dynamics_Transform
import trajectories
from rk4 import rk4
from params import robotParams
from copy import deepcopy
from scipy.linalg import solve_continuous_lyapunov
from jax.scipy.linalg import sqrtm
from scipy.optimize import least_squares
from functools import partial
from gravcomp import gq
import sys
import csv
import pandas as pd

class self:
    def __init___(self):
        return 'initialised'

################ HOMOGENEOUS TRANSFORMS ###############################

def skew(u):
    ans = jnp.block([[0., -u[2], u[1]],
                    [u[2], 0., -u[0]],
                    [-u[1], u[0], 0.]])
    return ans

def hatSE3(x):
    A = skew(x[3:5])
    return A


def rotx(mu):
    A = jnp.block([[1., 0., 0., 0.],
                   [0., jnp.cos(mu), -jnp.sin(mu), 0.],
                   [0., jnp.sin(mu), jnp.cos(mu), 0.],
                   [0., 0., 0., 1.]])
    return A

def roty(mu):
    A = jnp.block([[jnp.cos(mu), 0., jnp.sin(mu), 0.],
                   [0., 1., 0., 0.],
                   [-jnp.sin(mu), 0., jnp.cos(mu), 0.],
                   [0., 0., 0., 1.]])
    return A

def rotz(mu):
    A = jnp.block([[jnp.cos(mu), -jnp.sin(mu), 0., 0.],
                   [jnp.sin(mu), jnp.cos(mu), 0., 0.],
                   [0., 0., 1., 0.],
                   [0., 0., 0., 1.]])

    return A

def rotx_small(mu):
    A = jnp.block([[1., 0., 0.],
                [0., mu, -mu],
                [0., mu, mu],
                ])
    return A

def roty_small(mu):
    A = jnp.block([[mu, 0., mu],
                   [0., 1., 0.],
                   [-mu, 0., mu]
                   ])
    return A

def rotz_small(mu):
    A = jnp.block([[mu, -mu, 0.],
                   [mu, mu, 0.],
                   [0., 0., 1.]
                   ])
    return A

def tranx(mu):
    A = jnp.array([[1., 0., 0., mu],
                   [0., 1., 0., 0.],
                   [0., 0., 1., 0.],
                   [0., 0., 0., 1.]])
    return A

def trany(mu):
    A = jnp.array([[1., 0., 0., 0.],
                   [0., 1., 0., mu],
                   [0., 0., 1., 0.],
                   [0., 0., 0., 1.]])
    return A

def tranz(mu):
    A = jnp.array([[1., 0., 0., 0.],
                   [0., 1., 0., 0.],
                   [0., 0., 1., mu],
                   [0., 0., 0., 1.]])
    return A

def hatSE3(x):
    S = jnp.block([
        skew(x[3:6]), x[0:3], jnp.zeros((4,1))
    ])
    return S

####### MASS MATRIX #############

def massMatrix_continuous(q_hat,qconstants):
   
    dFcdq = jnp.array([
        [0.,0.,0.],
        [0.,0.,0.],
        [0.,0.,0.],
        [0.,0.,0.],
    ])
    q_bold = dFcdq@q_hat + qconstants.at[0].get()
    # print(q_bold)

    q1 = q_bold.at[0,0].get()
    q3 = q_bold.at[1,0].get()
    q5 = q_bold.at[2,0].get()
    q7 = q_bold.at[3,0].get()

    q2 = q_hat.at[0,0].get()
    q4 = q_hat.at[1,0].get()
    q6 = q_hat.at[2,0].get()
    # print('q1',q1)

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
    A0G = A0E@AEG

    # Geometric Jacobians

    c1 = s.c1
    c1x = c1.at[0].get()
    c1y = c1.at[1].get()
    c1z = c1.at[2].get()
    c2 = s.c2
    c2x = c2.at[0].get()
    c2y = c2.at[1].get()
    c2z = c2.at[2].get()
    c3 = s.c3
    c3x = c3.at[0].get()
    c3y = c3.at[1].get()
    c3z = c3.at[2].get()
    c4 = s.c4
    c4x = c4.at[0].get()
    c4y = c4.at[1].get()
    c4z = c4.at[2].get()
    c5 = s.c5
    c5x = c5.at[0].get()
    c5y = c5.at[1].get()
    c5z = c5.at[2].get()
    c6 = s.c6
    c6x = c6.at[0].get()
    c6y = c6.at[1].get()
    c6z = c6.at[2].get()
    c7 = s.c7
    c7x = c7.at[0].get()
    c7y = c7.at[1].get()
    c7z = c7.at[2].get()
    c8 = s.c8
    c8x = c8.at[0].get()
    c8y = c8.at[1].get()
    c8z = c8.at[2].get()
    cG = s.cGripper
    cGz = cG.at[2].get()



    A0c1 = tranx(c1x)@trany(c1y)@tranz(c1z)
    A0c2 = A01@tranx(c2x)@trany(c2y)@tranz(c2z)
    A0c3 = A02@tranx(c3x)@trany(c3y)@tranz(c3z)  
    A0c4 = A03@tranx(c4x)@trany(c4y)@tranz(c4z)
    A0c5 = A04@tranx(c5x)@trany(c5y)@tranz(c5z)
    A0c6 = A05@tranx(c6x)@trany(c6y)@tranz(c6z)
    A0c7 = A06@tranx(c7x)@trany(c7y)@tranz(c7z)
    A0c8 = A07@tranx(c8x)@trany(c8y)@tranz(c8z)
    A0cG = A0E@tranz(cGz)



            # Geometric Jacobians
    R01 = A01[0:3,0:3]     #rotation matrices
    R12 = A12[0:3,0:3]
    R23 = A23[0:3,0:3]
    R34 = A34[0:3,0:3]
    R45 = A45[0:3,0:3]
    R56 = A56[0:3,0:3]
    R67 = A67[0:3,0:3]
    R7E = A7E[0:3,0:3]
    REG = AEG[0:3,0:3]

    r100   = A01[0:3,[3]]
    r200   = A02[0:3,[3]]
    r300   = A03[0:3,[3]]
    r400   = A04[0:3,[3]]
    r500   = A05[0:3,[3]]
    r600   = A06[0:3,[3]]
    r700   = A07[0:3,[3]]
    r800   = A0E[0:3,[3]]
    rG00   = A0G[0:3,[3]]


    rc100   = A0c1[0:3,[3]]
    rc200   = A0c2[0:3,[3]]
    rc300   = A0c3[0:3,[3]]
    rc400   = A0c4[0:3,[3]]
    rc500   = A0c5[0:3,[3]]
    rc600   = A0c6[0:3,[3]]
    rc700   = A0c7[0:3,[3]]
    rc800   = A0c8[0:3,[3]]
    rcG00   = A0cG[0:3,[3]]

    z00 = jnp.array([[0.], [0.], [1.]])
    z01 = R01@z00
    z02 = R01@R12@z00
    z03 = R01@R12@R23@z00
    z04 = R01@R12@R23@R34@z00
    z05 = R01@R12@R23@R34@R45@z00
    z06 = R01@R12@R23@R34@R45@R56@z00
    z07 = R01@R12@R23@R34@R45@R56@R67@z00
    z08 = R01@R12@R23@R34@R45@R56@R67@R7E@z00
    z0G = R01@R12@R23@R34@R45@R56@R67@R7E@REG@z00

    ske1 = skew(z01)
    ske2 = skew(z02)
    ske3 = skew(z03)
    ske4 = skew(z04)
    ske5 = skew(z05)
    ske6 = skew(z06)
    ske7 = skew(z07)
    ske8 = skew(z08)
    skeG = skew(z0G)


    Jc2   = jnp.block([
        [ske1@(rc200-r100),   jnp.zeros((3,1)),   jnp.zeros((3,1)),   jnp.zeros((3,1)),   jnp.zeros((3,1)),   jnp.zeros((3,1)),   jnp.zeros((3,1))],   #jnp.zeros((3,1))],
        [z01,                 jnp.zeros((3,1)),   jnp.zeros((3,1)),   jnp.zeros((3,1)),   jnp.zeros((3,1)),   jnp.zeros((3,1)),   jnp.zeros((3,1))]  # jnp.zeros((3,1))]
        ])
    Jc3   = jnp.block([
        [ske1@(rc300-r100),  ske2@(rc300-r200),   jnp.zeros((3,1)),   jnp.zeros((3,1)),   jnp.zeros((3,1)),   jnp.zeros((3,1)),   jnp.zeros((3,1))],  # jnp.zeros((3,1))],
        [z01,                z02              ,   jnp.zeros((3,1)),   jnp.zeros((3,1)),   jnp.zeros((3,1)),   jnp.zeros((3,1)),   jnp.zeros((3,1))]  # jnp.zeros((3,1))]
        ])
    Jc4   = jnp.block([
        [ske1@(rc400-r100),  ske2@(rc400-r200),  ske3@(rc400-r300),   jnp.zeros((3,1)),   jnp.zeros((3,1)),   jnp.zeros((3,1)),   jnp.zeros((3,1))], #  jnp.zeros((3,1))],
        [z01,                z02,                 z03             ,   jnp.zeros((3,1)),   jnp.zeros((3,1)),   jnp.zeros((3,1)),   jnp.zeros((3,1))]   #jnp.zeros((3,1))]
        ])
    Jc5   = jnp.block([
        [ske1@(rc500-r100),  ske2@(rc500-r200),  ske3@(rc500-r300),  ske4@(rc500-r400),   jnp.zeros((3,1)),   jnp.zeros((3,1)),   jnp.zeros((3,1))],  # jnp.zeros((3,1))],
        [z01,                z02,                z03,                z04              ,   jnp.zeros((3,1)),   jnp.zeros((3,1)),   jnp.zeros((3,1))]  # jnp.zeros((3,1))]
        ])
    Jc6   = jnp.block([
        [ske1@(rc600-r100),  ske2@(rc600-r200),  ske3@(rc600-r300),  ske4@(rc600-r400),  ske5@(rc600-r500),   jnp.zeros((3,1)),   jnp.zeros((3,1))],   #jnp.zeros((3,1))],
        [z01,                z02,                z03,                z04,                z05              ,   jnp.zeros((3,1)),   jnp.zeros((3,1))] #  jnp.zeros((3,1))]
        ])
    Jc7   = jnp.block([
        [ske1@(rc700-r100),  ske2@(rc700-r200),  ske3@(rc700-r300),  ske4@(rc700-r400),  ske5@(rc700-r500),  ske6@(rc700-r600),   jnp.zeros((3,1))],  #jnp.zeros((3,1))],
        [z01,                z02              ,  z03,                z04,                z05,                z06              ,   jnp.zeros((3,1))]   #jnp.zeros((3,1))]
        ])
    Jc8   = jnp.block([
        [ske1@(rc800-r100),  ske2@(rc800-r200),  ske3@(rc800-r300),  ske4@(rc800-r400),  ske5@(rc800-r500),  ske6@(rc800-r600),  ske7@(rc800-r700)], #  jnp.zeros((3,1))],
        [z01,                z02,                z03              ,  z04,                z05,                z06,                z07              ]  #,   jnp.zeros((3,1))]
        ])

    # Mass Matrix
    R02 = A02[0:3,0:3]
    R03 = A03[0:3,0:3]
    R04 = A04[0:3,0:3]
    R05 = A05[0:3,0:3]
    R06 = A06[0:3,0:3]
    R07 = A07[0:3,0:3]
    R08 = A0E[0:3,0:3]
    R0G = A0G[0:3,0:3]

    I2 = s.I2
    I3 = s.I3
    I4 = s.I4
    I5 = s.I5
    I6 = s.I6
    I7 = s.I7
    I8 = s.I8
    IG = s.IG


    M2 = Jc2.T@jnp.block([
        [jnp.multiply(s.m2,jnp.eye(3,3)), jnp.zeros((3,3))],
        [jnp.zeros((3,3)),            R02.T@I2@R02 ]
    ])@Jc2
    M3 = Jc3.T@jnp.block([
        [jnp.multiply(s.m3,jnp.eye(3,3)), jnp.zeros((3,3))],
        [jnp.zeros((3,3)),            R03.T@I3@R03 ]
    ])@Jc3
    M4 = Jc4.T@jnp.block([
        [jnp.multiply(s.m4,jnp.eye(3,3)), jnp.zeros((3,3))],
        [jnp.zeros((3,3)),            R04.T@I4@R04 ]
    ])@Jc4
    M5 = Jc5.T@jnp.block([
        [jnp.multiply(s.m5,jnp.eye(3,3)), jnp.zeros((3,3))],
        [jnp.zeros((3,3)),            R05.T@I5@R05 ]
    ])@Jc5
    M6 = Jc6.T@jnp.block([
        [jnp.multiply(s.m6,jnp.eye(3,3)), jnp.zeros((3,3))],
        [jnp.zeros((3,3)),            R06.T@I6@R06 ]
    ])@Jc6
    M7 = Jc7.T@jnp.block([
        [jnp.multiply(s.m7,jnp.eye(3,3)), jnp.zeros((3,3))],
        [jnp.zeros((3,3)),            R07.T@I7@R07 ]
    ])@Jc7
    M8 = Jc8.T@jnp.block([
        [jnp.multiply(s.m8,jnp.eye(3,3)), jnp.zeros((3,3))],
        [jnp.zeros((3,3)),            R08.T@I8@R08 ]
    ])@Jc8

    Mq = M2 + M3 + M4 + M5 + M6 + M7 + M8# + MG

    return Mq
    

################## DMDQ FUNCTIONS ##########################
def MqPrime(q_hat,constants):

    Mq = massMatrix_continuous(q_hat,constants)
    A = jnp.array([
        [0.,0.,0.],
        [1.,0.,0.],
        [0.,0.,0.],
        [0.,1.,0.],
        [0.,0.,0.],
        [0.,0.,1.],
        [0.,0.,0.],
    ])
    
    Mq_hat = jnp.transpose(A)@Mq@A

    Mprime = jnp.zeros(Mq_hat.size)
    # print('sizeMq', jnp.shape(Mq_hat))
    # print('sizeMprime',jnp.shape(Mprime))
    m,n = jnp.shape(Mq_hat)
    for i in range(m):
        # print('i', i)
        Mprime = Mprime.at[n*i:n*(i+1)].set(Mq_hat.at[0:m,i].get())
        
    return Mprime

def TqiPrime(q_hat,constants):           #used in creating a Tqinv stack for dTinvdq calcs. #Legacy
    Mq = massMatrix_continuous(q_hat,constants)
    A = jnp.array([
        [0.,0.,0.],
        [1.,0.,0.],
        [0.,0.,0.],
        [0.,1.,0.],
        [0.,0.,0.],
        [0.,0.,1.],
        [0.,0.,0.],
    ])
    
    Mq_hat = jnp.transpose(A)@Mq@A
    Tqi = jnp.real(sqrtm(Mq_hat))

    Tiprime = jnp.zeros(Tqi.size)
    # print('sizeMq', jnp.shape(Mq_hat))
    # print('sizeMprime',jnp.shape(Mprime))
    m,n = jnp.shape(Tqi)
    for i in range(m):
        # print('i', i)
        Tiprime = Tiprime.at[n*i:n*(i+1)].set(Tqi.at[0:m,i].get())
        
    return Tiprime

@jax.jit
def unravel(dMdq_temp):         #This rearranges the square matrix that gets input after jacobian calculation
    # could probably generalise this for any array
    (m,n,l) = jnp.shape(dMdq_temp)
    dMdq1 = jnp.zeros((n,n))
    dMdq2 = jnp.zeros((n,n))
    dMdq3 = jnp.zeros((n,n))
    # print('dmdqshpae',jnp.shape(dMdq1))

    for i in range(n):
        dMdq1 = dMdq1.at[0:n,i].set(dMdq_temp.at[n*i:n*(i+1),0,0].get())
        dMdq2 = dMdq2.at[0:n,i].set(dMdq_temp.at[n*i:n*(i+1),1,0].get())
        dMdq3 = dMdq3.at[0:n,i].set(dMdq_temp.at[n*i:n*(i+1),2,0].get())

    return dMdq1, dMdq2, dMdq3 

# @partial(jax.jit, static_argnames=['Ti'])
@jax.jit
def dTi(Ti,dM):
    dTi = solve_continuous_lyapunov(Ti,dM)
    return dTi


@jax.jit
def Vq(q_hat, qconstants):
    #Function has to do FKM again to enable autograd to work
    dFcdq = jnp.array([
        [0.,0.,0.],
        [0.,0.,0.],
        [0.,0.,0.],
        [0.,0.,0.],
    ])
    q_bold = dFcdq@q_hat + qconstants.at[0].get()

    q1 = q_bold.at[0].get()
    q3 = q_bold.at[1].get()
    q5 = q_bold.at[2].get()
    q7 = q_bold.at[3].get()

    q2 = q_hat.at[0].get()
    q4 = q_hat.at[1].get()
    q6 = q_hat.at[2].get()

    A01 = tranz(s.l1)@rotx(pi)@rotz(q1)          
    A12 = rotx(pi/2)@tranz(-s.d1)@trany(-s.l2)@rotz(q2)
    A23 = rotx(-pi/2)@trany(s.d2)@tranz(-s.l3)@rotz(q3)
    A34 = rotx(pi/2)@tranz(-s.d2)@trany(-s.l4)@rotz(q4)
    A45 = rotx(-pi/2)@trany(s.d2)@tranz(-s.l5)@rotz(q5)
    A56 = rotx(pi/2)@trany(-s.l6)@rotz(q6)
    A67 = rotx(-pi/2)@tranz(-s.l7)@rotz(q7)

    A02 = A01@A12
    A03 = A02@A23
    A04 = A03@A34
    A05 = A04@A45
    A06 = A05@A56
    A07 = A06@A67
    # A0E = A07@A7E
    # A0G = A0E@AEG

    c1 = s.c1
    c1x = c1.at[0].get()
    c1y = c1.at[1].get()
    c1z = c1.at[2].get()
    c2 = s.c2
    c2x = c2.at[0].get()
    c2y = c2.at[1].get()
    c2z = c2.at[2].get()
    c3 = s.c3
    c3x = c3.at[0].get()
    c3y = c3.at[1].get()
    c3z = c3.at[2].get()
    c4 = s.c4
    c4x = c4.at[0].get()
    c4y = c4.at[1].get()
    c4z = c4.at[2].get()
    c5 = s.c5
    c5x = c5.at[0].get()
    c5y = c5.at[1].get()
    c5z = c5.at[2].get()
    c6 = s.c6
    c6x = c6.at[0].get()
    c6y = c6.at[1].get()
    c6z = c6.at[2].get()
    c7 = s.c7
    c7x = c7.at[0].get()
    c7y = c7.at[1].get()
    c7z = c7.at[2].get()
    c8 = s.c8
    c8x = c8.at[0].get()
    c8y = c8.at[1].get()
    c8z = c8.at[2].get()
    cG = s.cGripper
    cGz = cG.at[2].get()

    A0c1 = tranx(c1x)@trany(c1y)@tranz(c1z)
    A0c2 = A01@tranx(c2x)@trany(c2y)@tranz(c2z)
    A0c3 = A02@tranx(c3x)@trany(c3y)@tranz(c3z)  
    A0c4 = A03@tranx(c4x)@trany(c4y)@tranz(c4z)
    A0c5 = A04@tranx(c5x)@trany(c5y)@tranz(c5z)
    A0c6 = A05@tranx(c6x)@trany(c6y)@tranz(c6z)
    A0c7 = A06@tranx(c7x)@trany(c7y)@tranz(c7z)
    A0c8 = A07@tranx(c8x)@trany(c8y)@tranz(c8z)
    rc100   = A0c1[0:3,3]
    rc200   = A0c2[0:3,3]
    rc300   = A0c3[0:3,3]
    rc400   = A0c4[0:3,3]
    rc500   = A0c5[0:3,3]
    rc600   = A0c6[0:3,3]
    rc700   = A0c7[0:3,3]
    rc800   = A0c8[0:3,3]

    g0 = jnp.array([[0.],[0.],[-s.g]])
    gprime = jnp.transpose(g0)

    V = -s.m1*gprime@rc100-s.m2*gprime@rc200-s.m3*gprime@rc300-s.m4*gprime@rc400-s.m5*gprime@rc500-s.m6*gprime@rc600 -s.m7*gprime@rc700 -s.m8*gprime@rc800 #-s.mGripper*gprime@rcG00
    # s.V = V
    return V.at[0].get()


def holonomicConstraint(q_hat, qconstants):
    dFcdq = jnp.array([
        [0.,0.,0.],
        [0.,0.,0.],
        [0.,0.,0.],
        [0.,0.,0.],
    ])
    q_bold = dFcdq@q_hat + qconstants
    q_bold = q_bold.at[0].get()         #remove the extra array
    return q_bold



def diff_finite(q_d,t_span):
    #this function approxmates xdot from IKM solution q_d. The final momentum will always be zero, as the approxmition will shorten the array
    l = jnp.size(t_span)
    qdot = jnp.zeros(jnp.shape(q_d))        #end will be zero
    # print('qdotsize',jnp.shape(qdot))

    for i in range(l):
        qdot = qdot.at[:,[i]].set(q_d.at[:,[i+1]].get()- q_d.at[:,[i]].get())
    return qdot


    ####################### ODE SOLVER #################################


    # args = (v,D,constants)          ,Tq,dTqinv_block,dVdq)
def ode_dynamics_wrapper(xt,control_input,Damp,const):    
#This function allows the system dynamics to be integrated with the RK4 function. Everything as a function of q
    qt = jnp.array([[xt.at[0,0].get()],   #unpack states
                   [xt.at[1,0].get()],
                   [xt.at[2,0].get()]])

    Mqt, Tqt, Tqinvt = massMatrix_holonomic(qt,s)   #Get Mq, Tq and Tqinv for function to get dTqdq
    dMdqt = massMatrixJac(qt,const)
    dMdq1t, dMdq2t, dMdq3t = unravel(dMdqt)

    dTqidq1t = solve_continuous_lyapunov(Tqinvt,dMdq1t)
    dTqidq2t = solve_continuous_lyapunov(Tqinvt,dMdq2t)
    dTqidq3t = solve_continuous_lyapunov(Tqinvt,dMdq3t)

    dTqinvt = jnp.array([dTqidq1t,dTqidq2t,dTqidq3t])
    # dMdq_block = jnp.array([dMdq1, dMdq2, dMdq3])
    dVdqt = dV_func(qt,const)
    args = (control_input,Damp,Tqt,dTqinvt,dVdqt)

    xt_dot = dynamics_Transform(xt,*args)        #return xdot from dynamics function

    return xt_dot

###############
def sinusoid_instant(t,origin,freq,amp):
    #plane is xz plane.
    var0 = origin

    var = amp*sin(2*pi*freq*t) + var0       #returns one data point only.

    return var

################################# OBSERVER #######################################

def C_SYS(p_sym,p_sym2,Tq,dTqinvdq_val):
    #This function is specifically used in the creation of \bar{C}
    dTqinvdq1 = dTqinvdq_val.at[0].get()
    dTqinvdq2 = dTqinvdq_val.at[1].get()
    dTqinvdq3 = dTqinvdq_val.at[2].get()
    # print('dTqinvdq1',dTqinvdq1)
    dTqinv_phatdq1 = dTqinvdq1@p_sym
    dTqinv_phatdq2 = dTqinvdq2@p_sym
    dTqinv_phatdq3 = dTqinvdq3@p_sym
    # print('dTqinv',jnp.shape(dTqinv_phatdq1))
    temphat = jnp.block([dTqinv_phatdq1, dTqinv_phatdq2, dTqinv_phatdq3])
    temphatT = jnp.transpose(temphat)
    # print('temp',temp)
    Ctemp = temphatT - temphat
    # print('shape Ctemp', jnp.shape(Ctemp))
    # print('Ctemp',Ctemp)
    Cq_phat = Tq@Ctemp@Tq
    # print('shape Cqphat', jnp.shape(Cq_phat))

    # print(jnp.shape(Cq_phat@p_sym2))
    return Cq_phat@p_sym2

@jax.jit
def Cqp(p,Tq,dTqinvdq_val):
    #Calculation of the Coriolis Damping matrix (Check if this is correct name)
    dTqinvdq1 = dTqinvdq_val.at[0].get()
    dTqinvdq2 = dTqinvdq_val.at[1].get()
    dTqinvdq3 = dTqinvdq_val.at[2].get()
    # print('p',p)
    dTqinv_phatdq1 = dTqinvdq1@p
    dTqinv_phatdq2 = dTqinvdq2@p
    dTqinv_phatdq3 = dTqinvdq3@p
    # print('dTqinv',dTqinv_phatdq1)
    temphat = jnp.block([dTqinv_phatdq1, dTqinv_phatdq2, dTqinv_phatdq3])
    temphatT = jnp.transpose(temphat)
    # print('temp',temp)
    Ctemp = temphatT - temphat
    # print(Ctemp)
    Cq = Tq@Ctemp@Tq

    return Cq

@jax.jit
def observer_dynamics(xp,q,phi,u,Cq_phat,D,dVq,Tq):
    Dq = Tq@D@Tq
    # CbSYM(jnp.array([[0.],[0.],[0.]]),phat,Tq,dTqinvdq_values)
    Gq = Tq #previous result - confirm this
    u0 = jnp.zeros((3,1))
    # xp_dot = (Cq_phat - Dq - phi*Tq)@phat - Tq@dVq + Gq@(u+u0)
    #these dynamics substitue phat = xp + phi*q for better performance with RK4 solve
    # print('ObsdVq',dVq)
    xp_dot = (Cq_phat - Dq - phi*Tq)@(xp + phi*q) - Tq@dVq + Gq@u      #Gq@(u+u0) = u as control law isn't multiplied by Gq^-1
    return  xp_dot

#This function allows the observer dynamics to be integrated with the RK4 function. Everything as a function of q

#Try and implement this again. Joel has everything as a function of q and ph, which is what is needed.
def ode_observer_wrapper(xo,phato,phio,cntrl,dampo,consto):
    qo = jnp.array([[xo.at[0,0].get()],   #unpack states
                   [xo.at[1,0].get()],
                   [xo.at[2,0].get()]])
    xpo = jnp.array([[xo.at[3,0].get()],
                     [xo.at[4,0].get()],
                     [xo.at[5,0].get()]])
    # xp = phat - phi*q  -- extract q from this with a constant phat?

    Mqo, Tqo, Tqinvo = massMatrix_holonomic(qo,s) 
    dMdqo = massMatrixJac(qo,consto)
    dMdq1o, dMdq2o, dMdq3o = unravel(dMdqo)

    dTqidq1o = solve_continuous_lyapunov(Tqinvo,dMdq1o)
    dTqidq2o = solve_continuous_lyapunov(Tqinvo,dMdq2o)
    dTqidq3o = solve_continuous_lyapunov(Tqinvo,dMdq3o)

    dTqinvo = jnp.array([dTqidq1o,dTqidq2o,dTqidq3o])
    # dMdq_block = jnp.array([dMdq1, dMdq2, dMdq3])
    dVdqo = dV_func(qo,consto)

    # print('dVdqo',dVdqo)
    phato_dynamic = xpo+phio*qo
    Cqo = Cqp(phato_dynamic,Tqo,dTqinvo)
    # print('v', cntrl)

    # print('qo',qo)

    dxp = observer_dynamics(xpo,qo,phio,cntrl,Cqo,dampo,dVdqo,Tqo)
    # print('dxp',dxp)
    xpo_dot = jnp.block([[jnp.zeros((3,1))],[dxp]])
    # xpo_dot = dxp
    # print('xpodot', xpo_dot)
    return xpo_dot


#rewrite to bring switch out of switch condtition
def switchCond(phat,kappa,phi,Tq,dTqinvdq_values):
    # m,n = jnp.shape(Tq)
    # print('phat',phat)
    Cbar_large = CbSYM(jnp.zeros((3,1)),phat,Tq,dTqinvdq_values)  
    #process Cbar to the correct size, shape and order. Removes the columns, and transpose reorders columns back to how they should be with jac function
    # print(jnp.transpose(Cbar_large.at[:,0,:,0].get()))
    # print('SHape Cbar', jnp.shape(Cbar))
    # Cbar = jnp.transpose(Cbar_large.at[:,0,:,0].get())        #transpose is not needed according to comparison to joels matlab code
    Cbar = Cbar_large.at[:,0,:,0].get()
    min = jnp.amin(jnp.real(linalg.eigvals(phi*Tq - 0.5*(Cbar + jnp.transpose(Cbar)))))-kappa
    return min

def observerSwitch(q,phi,xp,kappa):
    phinew = phi + kappa
    xpnew = xp - kappa*q
    print('Switch Occurred')

    return phinew,xpnew

######################################## MAIN CODE STARTS HERE #################################
######################################## MAIN CODE STARTS HERE #################################
######################################## MAIN CODE STARTS HERE #################################
######################################## MAIN CODE STARTS HERE #################################
######################################## MAIN CODE STARTS HERE #################################




massMatrixJac = jacfwd(MqPrime)

# Mq_print = massMatrix_continuous(q_0,constants)
# print('Mq', jnp.transpose(holonomicTransform)@Mq_print@holonomicTransform)

# dMdq_print = massMatrixJac(q_0,constants)

# dMdq1, dMdq2, dMdq3 = unravel(dMdq_print)

dV_func = jacfwd(Vq,argnums=0)

#compute \barC matrix
CbSYM = jacfwd(C_SYS,argnums=0)


######################################## EXTRACT REAL DATA#############
################ IMPORT REAL DATA ######################

filestring = "7LINK_IMPLEMENTATION/data/freeswing2_v1sin"

data = pd.read_csv(filestring,sep=",",header=None, skiprows=3)       #last inputs go past the details of the csv.
print(data.head())      #prints first 5 rows. tail() prints last 5

data_array = data.to_numpy()
print('Data Size', jnp.shape(data_array))
start = 0      #avoid first time step if == 1
end = 400      #this ill be 1500 later
t = data_array[start:end,[1]]
t = t.astype('float64')
t = jnp.transpose(t)
print('End Time', t.at[:,end].get())

q7HistT = data_array[start:end,3:10]         #only pull out actuators 2, 4, 6
vel3HistT = data_array[start:end,10:13]
controlHistT = data_array[start:end,13:16]
q7HistT = q7HistT.astype('float64')
vel3HistT = vel3HistT.astype('float64')
controlHistT = controlHistT.astype('float64')
q7Hist = jnp.transpose(q7HistT)
vel3Hist = jnp.transpose(vel3HistT) #transpose and adjust to radians
controlHist = jnp.transpose(controlHistT)


qHist = q7Hist[[1,3,5],:]         #extract from actuators that move
q_0 = qHist[:,[0]]
print('q0',q_0)
dt = diff_finite(t,t)

(n,hold) = q_0.shape
l = jnp.size(t)

################################## SIMULATION/PLOT############################################

#INITIAL VALUES
q0_1 = 0.
q0_2 = 0.
q0_3 = 0.

q_initial = jnp.array([[q0_1,q0_2,q0_3]])
# p_initial = jnp.array([1.,0.,-0.4])            #This initial momentum is not momentum transformed
p_initial = jnp.array([0.,0.,0.])    

D = jnp.array([
    [1., 0., 0.],
    [0., 1., 0.],
    [0., 0., 1.],
])
D = 0.*D
print('D',D)


# OBSERVER PARAMETERS
kappa = 1.     #low value to test switches
phi = kappa #phi(0) = k
phat0 = jnp.array([[0.],[0.],[0.]])           #initial momentum estimate
xp0 = phat0 - phi*q_0     #inital xp 
# ObsRate = 200.   #Hz: refresh rate of observer
# dt_obs = 1/ObsRate
# timeObsUpdate = -dt_obs     
timeObsUpdate = 0.      #last time observer updated
# print('Observer dt',dt_obs)
Hobs = 0.

substeps = 1

q_0 = jnp.transpose(q_initial)
s = self()
s = robotParams(s)

constants = jnp.array([                 #These are the positions the wrists are locked to
    [0.],
    [0.],
    [0.],
    [0.],
])
s.constants = constants         #for holonomic transform

holonomicTransform = jnp.array([
        [0.,0.,0.],
        [1.,0.,0.],
        [0.,0.,0.],
        [0.,1.,0.],
        [0.,0.,0.],
        [0.,0.,1.],
        [0.,0.,0.],
    ])


xe0 = endEffector(q_0,s)
print('Initial Position', xe0)  #XYP coords.

#Define Initial Values
Mqh0, Tq0, Tq0inv = massMatrix_holonomic(q_0,s)   #Get Mq, Tq and Tqinv for function to get dTqdq
dMdq0 = massMatrixJac(q_0,constants)
dMdq10, dMdq20, dMdq30 = unravel(dMdq0)
dTq0invdq1 = solve_continuous_lyapunov(Tq0inv,dMdq10)
dTq0invdq2 = solve_continuous_lyapunov(Tq0inv,dMdq20)
dTq0invdq3 = solve_continuous_lyapunov(Tq0inv,dMdq30)
dTqinv0 = jnp.array([dTq0invdq1,dTq0invdq2,dTq0invdq3])

# p_0 = Tq0@(Mqh0@vel3Hist.at[:,0].get())

while switchCond(phat0,kappa,phi,Tq0,dTqinv0) <= 0:         #Find initial phi
    phitemp, xptmp = observerSwitch(q_0,phi,xp0,kappa)
    phi = phitemp
    xp0 = xptmp
    print('xp0', xp0)
    print(phi)

p0 = p_initial@Tq0                  #Transform momentum state. Note that the mutiplication is out of order because p_initial is horizontal.
print('p0',p0)
x0 = jnp.block([[q_initial,p0]])
x0 = jnp.transpose(x0)
print('Initial States', x0)

#Define Storage
(m,hold) = x0.shape
xHist = jnp.zeros((m,l+1))
xeHist = jnp.zeros((m,l))
hamHist = jnp.zeros(l)
kinHist = jnp.zeros(l)
potHist = jnp.zeros(l)
H0Hist = jnp.zeros(l)
HconHist = jnp.zeros(l)
xpHist = jnp.zeros((n,l))
phiHist = jnp.zeros(l)
phatHist = jnp.zeros((n,l+1))
switchHist = jnp.zeros(l)
# controlHist = jnp.zeros((n,l))

#Setting Initial Values
xHist = xHist.at[:,[0]].set(x0)
phatHist = phatHist.at[:,[0]].set(phat0)
xpHist = xpHist.at[:,[0]].set(xp0)
# controlHist = control.at[:,[0]].set(v_)      #controlling 3 states

sigma = 0.5*(pi/180.)     #noise standad deviation

print('SIMULATION LOOP STARTED')

jnp.set_printoptions(precision=15)

for k in range(l):
    time = t.at[:,k].get()
    dt_instant = dt.at[:,k].get()
    print('Time',time)
    dt_sub = dt_instant/substeps
    x = xHist.at[:,[k]].get()

    ## ADD NOISE TO THE POSITION MEASUREMENT
    q = jnp.array([[x.at[0,0].get()],
                   [x.at[1,0].get()],
                   [x.at[2,0].get()]])
    p = jnp.array([[x.at[3,0].get()],        #This is currently returning p, not p0
                   [x.at[4,0].get()],
                   [x.at[5,0].get()]])
    # print(q,p)
    # q_noise = sigma*np.random.randn(3,1)
    # print('q_noise',q_noise)
    # q_measure = q + q_noise

    xp = xpHist.at[:,[k]].get()
    phat = xp + phi*q           #find phat for this timestep

    Mq_hat, Tq, Tqinv = massMatrix_holonomic(q,s)   #Get Mq, Tq and Tqinv for function to get dTqdq

    dMdq = massMatrixJac(q,constants)
    dMdq1, dMdq2, dMdq3 = unravel(dMdq)

    dTqinvdq1 = solve_continuous_lyapunov(Tqinv,dMdq1)
    dTqinvdq2 = solve_continuous_lyapunov(Tqinv,dMdq2)
    dTqinvdq3 = solve_continuous_lyapunov(Tqinv,dMdq3)

    dTqinv_block = jnp.array([dTqinvdq1,dTqinvdq2,dTqinvdq3])
    # dMdq_block = jnp.array([dMdq1, dMdq2, dMdq3])
    dVdq = dV_func(q,constants)

    # print(dVdq)

    # Cqph = Cqp(phat,Tq,dTqinv_block)     #Calculate value of C(q,phat) Matrix.
    # Cqp_real = Cqp(p,Tq,dTqinv_block)     #Calculate value of C(q,p) Matrix.

    # Dhat = Tq@D@Tq          #dhat is damping estimate. Here Dhat = D

    # result = CbSYM(jnp.zeros((3,1)),phat,Tq,dTqinv_block)

    # print('Cbar', result)
    # print('size Cbar', jnp.shape(result))
    cond = switchCond(phat,kappa,phi,Tq,dTqinv_block)   #check if jump is necessary
    # print(cond)
    switchHist = switchHist.at[k].set(cond)
    if cond <= 0:
        phiplus, xpplus = observerSwitch(q,phi,xp,kappa)     #switch to xp+,phi+ values
        phi = phiplus
        xp = xpplus          #update phi and xp with new values


    # phat_plus = xp + phi*q
    # print('Phat Switch', phat - phat_plus)        #check that the switch doesn't affect phat - it doesn't

    ptilde = phat - p       #observer error for k timestep
    # print('p~',ptilde)
    # print('xp',xp)

    v = controlHist.at[:,[k]].get()     #extract control applied to the robot
    
    # print('v',v)
    #OBSERVER ODE SOLVE 
    # timeObs = round((time - timeObsUpdate),3)      #dealing with this float time issue


    print('Observer Updating')
    # print('Time Elapsed', timeObs)
    for i in range(substeps):           #holding q here for substeps might be making things really bad
        x_obs = jnp.array([
                            [q.at[0,0].get()],       #build state vector for observer
                            [q.at[1,0].get()],
                            [q.at[2,0].get()],
                            [xp.at[0,0].get()],
                            [xp.at[1,0].get()],
                            [xp.at[2,0].get()]])
        # print('Time Elapsed', timeObs)
        obs_args = (phat,phi,v,D,constants)#,q_measure)       #sending noise measurement to obs
                # ode_observer_wrapper(xo,phato,phio,cntrl,dampo,consto)
        xp_update = rk4(x_obs,ode_observer_wrapper,dt_instant,*obs_args)          #call rk4 solver to update ode
        xp_k  = jnp.array([[xp_update.at[3,0].get()],
                            [xp_update.at[4,0].get()],
                            [xp_update.at[5,0].get()]])
        # xp_k  = jnp.array([[xp_update.at[0,0].get()],
        #                     [xp_update.at[1,0].get()],
        #                     [xp_update.at[2,0].get()]])
        xp = xp_k
    # else:
        # xp_k = xp           #hold xp value from previous iteration

    # xp_k = xp           #while observer isn't working

    
    Hobs = 0.5*(jnp.transpose(ptilde.at[:,0].get())@ptilde.at[:,0].get())

    #SYSTEM ODE SOLVE
    print('System Updating')
    args = (v,D,constants)
    x_nextstep = x
    for i in range(substeps):       #the fact this doens't update the arguments might but fucking this up. 
        x_step= rk4(x_nextstep,ode_dynamics_wrapper,dt_sub,*args)
        x_nextstep = x_step


    x_k = x_nextstep           #extract final values from ODE solve

    if jnp.isnan(x_k.any()):          #check if simiulation messes up
        print(x_k)
        print('NAN found, exiting loop')
        break

    if jnp.isnan(xp_k.any()):
        print(xp_k)
        print('NAN found, exiting loop')
        break


    #Store Variables for next time step
    xHist = xHist.at[:,[k+1]].set(x_k)            #x for next timestep       
    xpHist = xpHist.at[:,[k+1]].set(xp_k)

    #store current timestep variables
    phatHist = phatHist.at[:,[k]].set(phat)
    #Check Observer dynamics
    H0Hist = H0Hist.at[k].set(Hobs)
    print('H obs', Hobs)
    phiHist = phiHist.at[k].set(phi)
    # controlHist = controlHist.at[:,[k]].set(v)
    # print('v',v.at[:,0].get())
    # HconHist = HconHist.at[k].set(Hcon)

    kinTemp = 0.5*(jnp.transpose(p.at[:,0].get())@p.at[:,0].get())
    potTemp = Vq(q,constants)
    hamTemp = kinTemp + potTemp   

    hamHist = hamHist.at[k].set(hamTemp)
    kinHist = kinHist.at[k].set(kinTemp)     
    potHist = potHist.at[k].set(potTemp)
    # xeHist = xeHist.at[:,k].set(xe)


print(controlHist)
# print(fake)
# print(hamHist)
# print(stop)

############### outputting to csv file#####################
############### outputting to csv file#####################
############### outputting to csv file#####################
############### outputting to csv file#####################
details = ['Testing torques applied to real arm to see if sim converges.']
simInfo = ['dT', dt, 'Substep Number', substeps]
# controlInfo = ['Control',controlActive,'Grav Comp', gravComp,'Control Rate',ContRate,'Kp',Kp,'Kd',Kd,'alpha',alpha]
observerInfo = [ 'Kappa',kappa,'sigma',sigma]
# trackingInfo = ['Trajectory Type', traj, 'Origin', origin, 'Freq nad Amplitude',frequency,amplitude]
header = ['Time', 'State History']
t2 = t.at[0,:].get()        #to prevent the [] in the data
with open('/root/FYP/7LINK_IMPLEMENTATION/data/sim_offline_v1', 'w', newline='') as f:

    writer = csv.writer(f)
    # writer.writerow(simtype)
    writer.writerow(details)
    writer.writerow(simInfo)
    # writer.writerow(controlInfo)
    writer.writerow(observerInfo)
    # writer.writerow(trackingInfo)
    writer.writerow(header)

    # writer.writerow(['Time', t])
    for i in range(l):
        timestamp = t2.at[i].get()               #time
        q1 = xHist.at[0,i].get()                #postion
        q2 = xHist.at[1,i].get()
        q3 = xHist.at[2,i].get()
        p1 = xHist.at[3,i].get()                #momentum (transformed)
        p2 = xHist.at[4,i].get()
        p3 = xHist.at[5,i].get()
        ham = hamHist.at[i].get()               #energy
        kin = kinHist.at[i].get()
        pot = potHist.at[i].get()
        # qd1 = q_d.at[0,i].get()          #used to be xe - now qd, as xe can be calculated in matlab
        # qd2 = q_d.at[1,i].get()                 #Trajectory
        # qd3 = q_d.at[2,i].get()
        phat1 = phatHist.at[0,i].get()          #estimated dmomentum
        phat2 = phatHist.at[1,i].get()
        phat3 = phatHist.at[2,i].get()
        Hobs = H0Hist.at[i].get()               #Observer energy
        ph = phiHist.at[i].get()                #phi
        xp1 = xpHist.at[0,i].get()              #xp
        xp2 = xpHist.at[1,i].get()
        xp3 = xpHist.at[2,i].get()
        sc = switchHist.at[i].get()             #switch condition
        v1 = controlHist.at[0,i].get()          #control values
        v2 = controlHist.at[1,i].get()
        v3 = controlHist.at[2,i].get()
        # Hc = HconHist.at[k].get()
        data = ['Time:', timestamp  , 'x:   ', q1,q2,q3,p1,p2,p3,ham,kin,pot,phat1,phat2,phat3,ph, Hobs,sc,xp1,xp2,xp3,v1,v2,v3]
          # data = ['State',i,':', xHist[k,:]] #xHist.at[k,:].get()]# 'End Effector Pose', xeHist.at[k,:].get()]
        
        writer.writerow(data)


