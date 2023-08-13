import jax
from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.numpy import pi, sin, cos, linalg
from jax import grad, jacobian, jacfwd
from effectorFKM import endEffector
from massMatrix_holonomic import massMatrix_holonomic
from dynamics_momentumTransform import dynamics_Transform
from rk4 import rk4
from params import robotParams
from copy import deepcopy
from scipy.linalg import solve_continuous_lyapunov
from jax.scipy.linalg import sqrtm
from scipy.optimize import least_squares, fmin
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
        [0.],
        [0.],
        [0.],
        [0.],
        [0.],
        [0.],
    ])
    q_bold = dFcdq@q_hat + qconstants.at[0].get()

    q1 = q_bold.at[0].get()
    q3 = q_bold.at[1].get()
    q5 = q_bold.at[2].get()
    q7 = q_bold.at[3].get()
    q6 = q_bold.at[4].get()
    q2 = q_bold.at[5].get()

    q4 = q_hat.at[0].get()          #CHANGE
    # q4 = q_hat.at[1].get()
    # q6 = q_hat.at[2].get()
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
        [jnp.zeros((3,3)),            R01@I2@R01.T ]
    ])@Jc2
    M3 = Jc3.T@jnp.block([
        [jnp.multiply(s.m3,jnp.eye(3,3)), jnp.zeros((3,3))],
        [jnp.zeros((3,3)),            R02@I3@R02.T ]
    ])@Jc3
    M4 = Jc4.T@jnp.block([
        [jnp.multiply(s.m4,jnp.eye(3,3)), jnp.zeros((3,3))],
        [jnp.zeros((3,3)),            R03@I4@R03.T ]
    ])@Jc4
    M5 = Jc5.T@jnp.block([
        [jnp.multiply(s.m5,jnp.eye(3,3)), jnp.zeros((3,3))],
        [jnp.zeros((3,3)),            R04@I5@R04.T ]
    ])@Jc5
    M6 = Jc6.T@jnp.block([
        [jnp.multiply(s.m6,jnp.eye(3,3)), jnp.zeros((3,3))],
        [jnp.zeros((3,3)),            R05@I6@R05.T ]
    ])@Jc6
    M7 = Jc7.T@jnp.block([
        [jnp.multiply(s.m7,jnp.eye(3,3)), jnp.zeros((3,3))],
        [jnp.zeros((3,3)),            R06@I7@R06.T ]
    ])@Jc7
    M8 = Jc8.T@jnp.block([
        [jnp.multiply(s.m8,jnp.eye(3,3)), jnp.zeros((3,3))],
        [jnp.zeros((3,3)),            R07@I8@R07.T ]
    ])@Jc8

    Mq = M2 + M3 + M4 + M5 + M6 + M7 + M8# + MG

    return Mq
    

################## DMDQ FUNCTIONS ##########################
def MqPrime(q_hat,constants):

    Mq = massMatrix_continuous(q_hat,constants)
    holo = jnp.array([      #CHANGE
        [0.],
        [0.],
        [0.],
        [1.],
        [0.],
        [0.],
        [0.],
    ])
    
    Mq_hat = jnp.transpose(holo)@Mq@holo

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
    # dMdq2 = jnp.zeros((n,n))
    # dMdq3 = jnp.zeros((n,n))
    # print('dmdqshpae',jnp.shape(dMdq1))

    for i in range(n):
        dMdq1 = dMdq1.at[0:n,i].set(dMdq_temp.at[n*i:n*(i+1),0,0].get())
        # dMdq2 = dMdq2.at[0:n,i].set(dMdq_temp.at[n*i:n*(i+1),1,0].get())
        # dMdq3 = dMdq3.at[0:n,i].set(dMdq_temp.at[n*i:n*(i+1),2,0].get())

    return dMdq1#, dMdq2, dMdq3 



@jax.jit
def Vq(q_hat, qconstants):
    #Function has to do FKM again to enable autograd to work
    dFcdq = jnp.array([
        [0.],
        [0.],
        [0.],
        [0.],
        [0.],
        [0.],
    ])
    q_bold = dFcdq@q_hat + qconstants.at[0].get()

    q1 = q_bold.at[0].get()
    q3 = q_bold.at[1].get()
    q5 = q_bold.at[2].get()
    q7 = q_bold.at[3].get()
    q6 = q_bold.at[4].get()
    q2 = q_bold.at[5].get()

    q4 = q_hat.at[0].get()      #CHANGE
    # q4 = q_hat.at[1].get()
    # q6 = q_hat.at[2].get()

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
    qt = jnp.array([[xt.at[0,0].get()]])
                       #unpack states
                #    [xt.at[1,0].get()],
                #    [xt.at[2,0].get()]])

    Mqt, Tqt, Tqinvt, Jct = massMatrix_holonomic(qt,s)   #Get Mq, Tq and Tqinv for function to get dTqdq
    dMdqt = massMatrixJac(qt,const)
    # dMdq1t, dMdq2t, dMdq3t = unravel(dMdqt)
    dMdqt = unravel(dMdqt)
    # print('Tqinvt', jnp.shape(Tqinvt))
    
    # print('dMdqt', jnp.shape(dMdqt))

    dTqidq1t = solve_continuous_lyapunov(Tqinvt,dMdqt)
    # dTqidq2t = solve_continuous_lyapunov(Tqinvt,dMdq2t)
    # dTqidq3t = solve_continuous_lyapunov(Tqinvt,dMdq3t)

    dTqinvt = jnp.array([dTqidq1t])#,dTqidq2t,dTqidq3t])
    # dMdq_block = jnp.array([dMdq1, dMdq2, dMdq3])
    dVdqt = dV_func(qt,const)
    args = (control_input,Damp,Tqt,dTqinvt,dVdqt)

    xt_dot = dynamics_Transform(xt,*args)        #return xdot from dynamics function

    return xt_dot



############################ CONTROLLER #############################

@jax.jit
def control(x_err,Tq,Cq,Kp,Kd,alpha,gravComp):
    #This control function provides the 'v' input detailed in the paper. This v is added to gravity compensation.
    q_tilde = x_err.at[0:3].get()
    p_tilde = x_err.at[3:6].get()

    D_hat = jnp.zeros((3,3))
    v = alpha*(Cq - D_hat - Kd)@Kp@(q_tilde + alpha*p_tilde) - Tq@Kp@(q_tilde + alpha*p_tilde) - Kd@p_tilde
    # print(v)

    return v




######################################## MAIN CODE STARTS HERE #################################
######################################## MAIN CODE STARTS HERE #################################
######################################## MAIN CODE STARTS HERE #################################
######################################## MAIN CODE STARTS HERE #################################
######################################## MAIN CODE STARTS HERE #################################

#INITIAL VALUES
# q0_1 = pi/2.
# q0_2 = 0.
# q0_3 = 0.

# q_initial = jnp.array([[q0_1,q0_2,q0_3]])
# p_initial = jnp.array([1.,0.,-0.4])            #This initial momentum is not momentum transformed
# p_initial = jnp.array([0.,0.,0.])    

################ IMPORT REAL DATA ######################
#CHANGE
data = pd.read_csv("7LINK_IMPLEMENTATION/data/sinusoid_inverted_v2",sep=",",header=None, skiprows=3)       #last inputs go past the details of the csv.
print(data.head())      #prints first 5 rows. tail() prints last 5

data_array = data.to_numpy()

start = 1000
end = 1200      #this ill be 2000 later
t = data_array[start:end,[1]]
t = t.astype('float32')
t = jnp.transpose(t)

q7HistT = data_array[start:end,3:10]         #only pull out actuators 2, 4, 6
velHistT = data_array[start:end,10:13]
controlHistT = data_array[start:end,13:16]
# print(xHistT)
q7HistT = q7HistT.astype('float32')
velHistT = velHistT.astype('float32')
controlHistT = controlHistT.astype('float32')
q7Hist = jnp.transpose(q7HistT)
vel3Hist = jnp.transpose(velHistT)
controlHist = jnp.transpose(controlHistT)
# print(jnp.shape(q7Hist))
# print(jnp.shape(velHist))
# print(jnp.shape(controlHist))

qHist = q7Hist[[3],:]         #this will only be the actuator we care about.        ############# CHANGE THIS
velHist = vel3Hist[[1],:]     # vel is 3            #CHANGE

print('size qhist', jnp.shape(qHist))

q_0 = qHist[:,[0]]
print('q0',q_0)
dt = diff_finite(t,t)

# q_0 = jnp.transpose(q_initial)
s = self()
s = robotParams(s)

constants = jnp.array([                 #These are the positions the wrists are locked to. Now includes 2 of the bend actuators
    [0.],
    [0.],
    [0.],
    [0.],
    [0.],
    [0.],
])
s.constants = constants         #for holonomic transform

holonomicTransform = jnp.array([            #CHANGE
        [0.],
        [0.],         #only actuator we car about? might have to reduce the rows. write this out.
        [0.],
        [1.],
        [0.],
        [0.],
        [0.],
    ])

s.holo = holonomicTransform

xe0 = endEffector(q_0,s)
print('Initial Position', xe0)  #XYP coords.

massMatrixJac = jacfwd(MqPrime)

Mq_print = massMatrix_continuous(q_0,constants)
print('Mq', jnp.transpose(holonomicTransform)@Mq_print@holonomicTransform)
dMdq_print = massMatrixJac(q_0,constants)

dMdq1= unravel(dMdq_print)
# V = Vq(q_hat,constants)
# print('V', V)
dV_func = jacfwd(Vq,argnums=0)

# print(fake)


################################## SIMULATION/PLOT############################################

# This simulations uses p and q hat
(n,hold) = q_0.shape
gravComp = 1.       #1 HAS GRAVITY COMP.

#Define Friction
# D = jnp.zeros((3,3))
D = 1.*jnp.eye(n)          #check this imple

# endT = T - dt       #prevent truncaton
# t = jnp.arange(0,T,dt)
l = jnp.size(t)
# print('l',l)

pHist = jnp.zeros((n,l))            #this is for momentum from real data


### Process real data for momentum states. #####

for k in range(l):
    # x = qHist.at[:,[k]].get()
    q = jnp.array([qHist.at[:,[k]].get()])

    Mq_hat, Tq, Tqinv, Jc_hat = massMatrix_holonomic(q,s)   #Get Mq, Tq and Tqinv for function to get dTqdq
    vel = velHist.at[:,[k]].get()
    # print(vel)
    #for momentum
    p0 = Mq_hat@vel
    p = Tq@p0
    # print(p)
    pHist = pHist.at[:,[k]].set(p)



#Define Initial Values
Mqh0, Tq0, Tq0inv, Jc_hat0 = massMatrix_holonomic(q_0,s)   #Get Mq, Tq and Tqinv for function to get dTqdq
# dMdq0 = massMatrixJac(q_0,constants)
# dMdq10, dMdq20, dMdq30 = unravel(dMdq0)
# dTq0invdq1 = solve_continuous_lyapunov(Tq0inv,dMdq10)
# dTq0invdq2 = solve_continuous_lyapunov(Tq0inv,dMdq20)
# dTq0invdq3 = solve_continuous_lyapunov(Tq0inv,dMdq30)
# dTqinv0 = jnp.array([dTq0invdq1,dTq0invdq2,dTq0invdq3])


p0 = Tq0@velHist[:,[0]]
# p0 = p_initial@Tq0                  #Transform momentum state. Note that the mutiplication is out of order because p_initial is horizontal.
print('p0',p0)
x0 = jnp.block([[q_0],[p0]])
# x0 = jnp.transpose(x0)
print('Initial States', x0)

#Define Storage
(m,hold) = x0.shape
xHist_sim = jnp.zeros((m,l+1))
controlHist_sim = jnp.zeros((n,l))

#Setting Initial Values
# xHist_sim = xHist_sim.at[:,[0]].set(x0)
# controlHist = control.at[:,[0]].set(v_)      #controlling 3 states


# print(fake)
print('SIMULATION LOOP STARTED')

jnp.set_printoptions(precision=15)
substeps = 2


## RUNNIN SIM
def simulation(x_init,controlHist,t,dt,dampingParam):
    print('Beginning sim')
    l = jnp.size(t)
    (m,hold) = jnp.shape(x_init)
    # print(jnp.shape(x_init))
    x_sim = jnp.zeros((m,l+1))
    # print('size xsim',jnp.shape(x_sim))
    x_sim = x_sim.at[:,[0]].set(x_init)
    for k in range(l):
        # time = t.at[:,k].get()
        dt_instant = dt.at[:,k].get()
        print('dt',dt_instant)

        v = controlHist.at[[1],[k]].get()     #extract control applied to the robot, REMEMBER TO CHANGE TO ACTUATOR #CHANGE
        # print('shape v',jnp.shape(v))
        x = x_sim.at[:,[k]].get()
        # q = jnp.array([[x.at[0,0].get()],
        #             [x.at[1,0].get()],
        #             [x.at[2,0].get()]])
        # p = jnp.array([[x.at[3,0].get()],        #This is currently returning p, not p0
        #             [x.at[4,0].get()],
        #             [x.at[5,0].get()]])
        # print(q,p)
        # print('shape x', jnp.shape(x))

        # Mq_hat, Tq, Tqinv, Jc_hat = massMatrix_holonomic(q,s)   #Get Mq, Tq and Tqinv for function to get dTqdq

        # dMdq = massMatrixJac(q,constants)
        # dMdq1, dMdq2, dMdq3 = unravel(dMdq)

        # dTqinvdq1 = solve_continuous_lyapunov(Tqinv,dMdq1)
        # dTqinvdq2 = solve_continuous_lyapunov(Tqinv,dMdq2)
        # dTqinvdq3 = solve_continuous_lyapunov(Tqinv,dMdq3)

        # dTqinv_block = jnp.array([dTqinvdq1,dTqinvdq2,dTqinvdq3])
        # dMdq_block = jnp.array([dMdq1, dMdq2, dMdq3])
        # dVdq = dV_func(q,constants)

        D = dampingParam#*jnp.eye(3)

        #SYSTEM ODE SOLVE
        print('System Updating')
        args = (v,D,constants)
        # dt_integration = dt_instant/substeps
        x_next = x
        # for i in range(substeps):       #the fact this doens't update the arguments might but fucking this up. 
        # print('dt',jnp.shape(dt_instant))
            # print('i',i)
        x_step= rk4(x_next,ode_dynamics_wrapper,dt_instant,*args)        #constraints need to be made for this.
            # x_next = x_step

        x_k = x_step          #extract final values from ODE solve
        # print('size xk',jnp.shape(x_k))
        # print('xk',x_k)
        if jnp.isnan(x_k.any()):          #check if simiulation messes up
            print(x_k)
            print('NAN found, exiting loop')
            break

        #Store Variables for next time step
        x_sim = x_sim.at[:,[k+1]].set(x_k)            #x for next timestep       

    # controlHist = controlHist.at[:,[k]].set(v)

    # xeHist = xeHist.at[:,k].set(xe)
    return x_sim                    #THIS WAS WRING VUCKA;LSDFJ

def optimisationCost(dampP,X,simFunc,xstart,controlHist,t,dt):
    
    m,n = jnp.shape(X)
    # print('size x',jnp.shape(X))
    simX = simFunc(xstart,controlHist,t,dt,dampP)
    # print(jnp.shape(X[[0],:]))
    # print(jnp.shape(simX[[0],0:n]))
    cost = linalg.norm(X[[0],:]-simX[[0],0:n])#+linalg.norm(X[[1],:]-simX[[1],(n-1)])+linalg.norm(X[[2],:]-simX[[2],(n-1)])
    return cost

# print('sie qhist', jnp.size(qHist))
args = (qHist,simulation,x0,controlHist,t,dt)
dampingParamInitial = 10.        
dampingParamOpt = fmin(optimisationCost,dampingParamInitial,args,disp=True)

print('Damping Param',dampingParamOpt)

# print(controlHist)
# print(fake)
# print(hamHist)
# print('No save')
# print(fake)


### DAMPING PARAM RESULTS
#2nd joint: 5.0465087890625          NEEED TO DO THIS SIM AGIN
#4th joint: Haven't done optimsation but will assume it is same as first joint.
#6th joint: 4.985522460937505

############### outputting to csv file#####################
############### outputting to csv file#####################
############### outputting to csv file#####################
############### outputting to csv file#####################
print('Saving Data')
# l = jnp.size(self.t)

details = ['Damping Parameter:', dampingParamOpt]
# values = ['Amp/Freqs: v1',self.a1,self.f1,'v2',self.a2,self.f2,'v3',self.a3,self.f3]
# header = ['Time', 'State History']
with open('SYSTEM_IDENTIFICATION/data/dampingParameter2_massmatrixfixed', 'w', newline='') as f:  #CHANGE

    writer = csv.writer(f)
    # writer.writerow(simtype)
    writer.writerow(details)
    # writer.writerow(values)
    # writer.writerow(header)

    # writer.writerow(['Time', t])
    # for i in range(l):
    #     timestamp = self.timeStore[i]           #time
    #     q1 = self.q_storage[0,i]               #postion
    #     q2 = self.q_storage[1,i]
    #     q3 = self.q_storage[2,i]
    #     q4 = self.q_storage[3,i]
    #     q5 = self.q_storage[4,i]
    #     q6 = self.q_storage[5,i]
    #     q7 = self.q_storage[6,i]
    #     qdot1 = self.vel_storage[0,i]               #momentum (transformed)
    #     qdot2 = self.vel_storage[1,i]
    #     qdot3 = self.vel_storage[2,i]
    #     v1 = self.controlHist[0,i]       #control values
    #     v2 = self.controlHist[1,i] 
    #     v3 = self.controlHist[2,i] 
    #     data = ['Time:', timestamp  , 'x:   ', q1,q2,q3,q4,q5,q6,q7,qdot1,qdot2,qdot3, v1,v2,v3]
        
    #     writer.writerow(data)


print('Data Saved!')