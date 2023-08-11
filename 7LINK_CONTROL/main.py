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
import sys
import csv

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

############################# TRACKING PROBLEM ###################################

@partial(jax.jit, static_argnames=['struct'])
def errorIKM(q, q_last, xestar,struct):

    #q0 is start point. q is iteration to solve for xstar
    #xstar is currently just positon, no angles
    #EFFECTOR IKM
    q2 = q[0]      #make the full q vector
    q4 = q[1]
    q6 = q[2]
    # print('struct',struct)
    q = jnp.array([     #this is qhat
        [q2, q4, q6]
        ])

    q = jnp.transpose(q)
    qconstants = struct.constants
    q_bold = qconstants.at[:,0].get()     #constrained variabless
    # print('constants',q_bold)
    
    q1 = q_bold.at[0].get()
    q3 = q_bold.at[1].get()
    q5 = q_bold.at[2].get()
    q7 = q_bold.at[3].get()

    A01 = tranz(struct.l1)@rotx(pi)@rotz(q1)          
    A12 = rotx(pi/2)@tranz(-struct.d1)@trany(-struct.l2)@rotz(q2)
    A23 = rotx(-pi/2)@trany(struct.d2)@tranz(-struct.l3)@rotz(q3)
    A34 = rotx(pi/2)@tranz(-struct.d2)@trany(-struct.l4)@rotz(q4)
    A45 = rotx(-pi/2)@trany(struct.d2)@tranz(-struct.l5)@rotz(q5)
    A56 = rotx(pi/2)@trany(-struct.l6)@rotz(q6)
    A67 = rotx(-pi/2)@tranz(-struct.l7)@rotz(q7)
    A7E = rotx(pi)@tranz(struct.l8)    
    AEG = tranz(struct.lGripper)

    A02 = A01@A12
    A03 = A02@A23
    A04 = A03@A34
    A05 = A04@A45
    A06 = A05@A56
    A07 = A06@A67
    A0E = A07@A7E
    #end effector pose
    r0E0 = A0E[0:3,[3]]

    pose_weights = jnp.array([1.,1.])
    sqK = 1000*jnp.diag(pose_weights)                            #weights on x and z error
    e_pose = jnp.array([r0E0[0]-xestar[0] ,r0E0[2]-xestar[2]] )  #penalised error from xe to xestar

    e_q = q - q_last            #penalises movement from q0
    q_weights = jnp.array([1.,1.,1.])              #penalises specific joint movement. Currently set to use base joint more
    sqW = 1*jnp.diag(q_weights)

    sqM = jnp.block([
        [sqW,               jnp.zeros((3, 2))],
        [jnp.zeros((2,3)),  sqK              ]
    ])
#     s = effectorAnalyticalJacobian(q, param, s);      #this is here for future reference if I want to fix angles
#     JA = s.JA;
#     J = [sqW;
#          sqK*JA];

    e = sqM@jnp.block([[e_q],[e_pose]])         # this isn't quadratic?? FUCK
                       #coudl use jacobian and feed through. might decrease time
    # print(e)
    e = e[:,0]           
    return e


def solveIKM(traj,init_guess,params):
    #solves for displacement only
    guess = init_guess
    # print(jnp.shape(guess))
    m,n = jnp.shape(traj)
    # print('m,n', m,n)
    q_d = jnp.zeros((m,n))

    bound = ([s.qlower.at[1].get(),s.qlower.at[3].get(),s.qlower.at[5].get()],[s.qupper.at[1].get(),s.qupper.at[3].get(),s.qupper.at[5].get()])


    for i in range(n):
        point = traj.at[0:3,i].get()
        # errorIKM(q, q0, xestar,struct):       #function handle for ref
        q0 = jnp.transpose(jnp.array([guess]))
        # print(jnp.shape(q0))
        result = least_squares(errorIKM,guess,bounds=bound,args=(q0,point,s))        #starts guess at guess, and penalised movement from last q_d point
        guess = result.x
        q_d = q_d.at[:,i].set(guess)
        # print(q_d.at[:,i].get())

    return q_d


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





######################################## MAIN CODE STARTS HERE #################################
######################################## MAIN CODE STARTS HERE #################################
######################################## MAIN CODE STARTS HERE #################################
######################################## MAIN CODE STARTS HERE #################################
######################################## MAIN CODE STARTS HERE #################################

#INITIAL VALUES
q0_1 = pi/4.
q0_2 = 0.
q0_3 = 0.

q_initial = jnp.array([[q0_1,q0_2,q0_3]])
p_initial = jnp.array([1.,0.,-0.4])            #This initial momentum is not momentum transformed
# p_initial = jnp.array([0.,0.,0.])    



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

massMatrixJac = jacfwd(MqPrime)

Mq_print = massMatrix_continuous(q_0,constants)
print('Mq', jnp.transpose(holonomicTransform)@Mq_print@holonomicTransform)

dMdq_print = massMatrixJac(q_0,constants)


dMdq1, dMdq2, dMdq3 = unravel(dMdq_print)
# print('dMdq', dMdq1)
# print('dMdq', dMdq2)
# print('dMdq', dMdq3)
# print('size', jnp.shape(dMdq_print))

# print(fake)
# V = Vq(q_hat,constants)
# print('V', V)
dV_func = jacfwd(Vq,argnums=0)


################################## SIMULATION/PLOT############################################

# This simulations uses p and q hat
(n,hold) = q_0.shape

#Initialise Simulation Parameters
dt = 0.005
substeps = 1
# dt_sub = dt/substeps      #no longer doing substeps
T = 3.

controlActive = 1     #CONTROL ACTIONS
gravComp = 1.       #1 HAS GRAVITY COMP.
# #Define tuning parameters
alpha = 0.5 # was 0.1 in before
Kp = 10.*jnp.eye(n)       #saved tunings
Kd = 0.5*jnp.eye(n)

ContRate = 200. #Hz: Controller refresh rate
dt_con = 1/ContRate
print('Controller dt',dt_con)
timeConUpdate = -dt_con     #this forces an initial update at t = 0s
v = jnp.zeros((3,1))
Hcon = 0

#Define Friction
D = jnp.zeros((3,3))
D = 1.*jnp.eye(n)          #check this implentation
# D = jnp.array([
#     [1., 0., 0.],
#     [0., 1., 0.],
#     [0., 0., 1.],
# ])

endT = T - dt       #prevent truncaton
t = jnp.arange(0,T,dt)
l = jnp.size(t)

##################### TRACKING PROBLEM PARAMETERS
#solve IKM to find q_d.
origin = jnp.array([[0.6],[0.6]])            #circle origin, or point track. XZ coords.as system is XZ planar
frequency = 0.2
amplitude = 0.1

traj = 'point'      #Name Trajectory Function
# traj = 'planar_circle'      #Name Trajectory Function
# traj = 'sinusoid_x'      #Name Trajectory Function

traj_func = getattr(trajectories,traj)
xe = traj_func(t,origin,frequency,amplitude)        #frequency and amplitude aren't used in point tracking , but function asks for these parameters. Probably really messy implementation

q_init_guess = jnp.zeros(3)         #initialise initial IKM Guess

#TRAJECTORY TRACK
q_d = solveIKM(xe,q_init_guess,s)      #solve IKM so trajectory is changed to generalised idsplacement coordinates.
# print('q_d',q_d)
# print('qd shape', jnp.shape(q_d))
# print(q_d.at[:,0].get())
dq_d = diff_finite(q_d,t)           #velocity
ddq_d = diff_finite(dq_d,t)         #acceleration

#Define Initial Values
Mqh0, Tq0, Tq0inv = massMatrix_holonomic(q_0,s)   #Get Mq, Tq and Tqinv for function to get dTqdq
dMdq0 = massMatrixJac(q_0,constants)
dMdq10, dMdq20, dMdq30 = unravel(dMdq0)
dTq0invdq1 = solve_continuous_lyapunov(Tq0inv,dMdq10)
dTq0invdq2 = solve_continuous_lyapunov(Tq0inv,dMdq20)
dTq0invdq3 = solve_continuous_lyapunov(Tq0inv,dMdq30)
dTqinv0 = jnp.array([dTq0invdq1,dTq0invdq2,dTq0invdq3])


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
controlHist = jnp.zeros((n,l))

#Setting Initial Values
xHist = xHist.at[:,[0]].set(x0)
# controlHist = control.at[:,[0]].set(v_)      #controlling 3 states

print('SIMULATION LOOP STARTED')

jnp.set_printoptions(precision=15)

for k in range(l):
    time = t.at[k].get()
    print('Time',time)

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

    Cqph = Cqp(p,Tq,dTqinv_block)     #Calculate value of C(q,p) Matrix.

    Dhat = Tq@D@Tq          #dhat is damping estimate. Here Dhat = D

    # result = CbSYM(jnp.zeros((3,1)),phat,Tq,dTqinv_block)

    # print('Cbar', result)
    # print('size Cbar', jnp.shape(result))

    if controlActive == 1:
        timeCon = round((time - timeConUpdate),3)
        if timeCon >= dt_con:    #update controller

            if gravComp == 1:           #this probably should be in the controller update?
                tau = Tq@dVdq           #tranform into momentum trnasform dynamics
            else:
                tau = jnp.zeros((3,1))

            print('Controller Updating')
            qddot = dq_d.at[:,[k]].get()  
            qddotdot = ddq_d.at[:,[k]].get()  
            p_d = Tqinv@qddot                  #as p0 = Mq*qdot, and p = Tq*p0
            
            # p_d = jnp.zeros((3,1))                                  #if this is zero, everything is treated as a point track with position updates.
            # x_d = jnp.block([[q_d.at[:,[k]].get()], [p_d]])
            # err = jnp.block([[q], [p]]) - x_d     #define error    
            errq = q - q_d.at[:,[k]].get()
            errp = p -p_d           #change for real or fake error.
            err = jnp.block([[errq],[errp]])    #define error           now running off phat. THis was p_d before to act only on position error.
            # print('err',err)
            #Find Control Input for current x, xtilde
            # v_input = control(err,Tq,Cqph,Kp,Kd,alpha,gravComp)     #uses Cqp with estimated momentum
            v_input = control(err,Tq,Cqph,Kp,Kd,alpha,gravComp)
            timeConUpdate = time

            Hcon = 0.5*jnp.transpose(errp.at[:,0].get())@errp.at[:,0].get() + 0.5*jnp.transpose(errq.at[:,0].get() + alpha*errp.at[:,0].get())@Kp@(errq.at[:,0].get() + alpha*errp.at[:,0].get())
            print('Hcon',Hcon)
            dTiqdot_dq1 = dTqinvdq1@qddot               #dp_d/dq = dTinv/dq @ qd_dot
            dTiqdot_dq2 = dTqinvdq2@qddot
            dTiqdot_dq3 = dTqinvdq3@qddot

            dp_d_dq = jnp.block([dTiqdot_dq1,dTiqdot_dq2,dTiqdot_dq3])      #this is from product rule
            D_con = Dhat        #holdover if system updates and controller doesn't

            v = -(Cqph - D_con)@p_d + dp_d_dq@Tq@p + Tqinv@qddotdot + tau + v_input          #total control law from equaton 17 now here.
            # print('v',v)

    else:
        v = jnp.zeros((3,1))
        # dVdq = dV_func(q_measure,constants)
        # v = 2*Tq@dVdq       #turn off control and set free swing upwards


    #SYSTEM ODE SOLVE
    print('System Updating')
    args = (v,D,constants)
    x_nextstep = x
    # for i in range(substeps):       #the fact this doens't update the arguments might but fucking this up. 
    x_step= rk4(x,ode_dynamics_wrapper,dt,*args)
    x_nextstep = x_step


    x_k = x_nextstep           #extract final values from ODE solve

    if jnp.isnan(x_k.any()):          #check if simiulation messes up
        print(x_k)
        print('NAN found, exiting loop')
        break


    #Store Variables for next time step
    xHist = xHist.at[:,[k+1]].set(x_k)            #x for next timestep       
    #store current timestep variables

    controlHist = controlHist.at[:,[k]].set(v)
    HconHist = HconHist.at[k].set(Hcon)

    kinTemp = 0.5*(jnp.transpose(p.at[:,0].get())@p.at[:,0].get())
    potTemp = Vq(q,constants)
    hamTemp = kinTemp + potTemp   

    hamHist = hamHist.at[k].set(hamTemp)
    kinHist = kinHist.at[k].set(kinTemp)     
    potHist = potHist.at[k].set(potTemp)

print('FINISHED')

############### outputting to csv file#####################
############### outputting to csv file#####################
############### outputting to csv file#####################
############### outputting to csv file#####################
details = ['controller updates immediately. tHIS HAS THE FIXED MASS MATRIX']
simInfo = ['dT', dt, 'Substep Number', substeps]
controlInfo = ['Control',controlActive,'Grav Comp', gravComp,'Control Rate',ContRate,'Kp',Kp,'Kd',Kd,'alpha',alpha]
observerInfo = ['Observer is not operation. Ignore all observer paramerters here']
trackingInfo = ['Trajectory Type', traj, 'Origin', origin, 'Freq nad Amplitude',frequency,amplitude]
header = ['Time', 'State History']
with open('/root/FYP/7LINK_CONTROL/data/control_point', 'w', newline='') as f:

    writer = csv.writer(f)
    # writer.writerow(simtype)
    writer.writerow(details)
    writer.writerow(simInfo)
    writer.writerow(controlInfo)
    writer.writerow(observerInfo)
    writer.writerow(trackingInfo)
    writer.writerow(header)

    # writer.writerow(['Time', t])
    for i in range(l):
        timestamp = t.at[i].get()               #time
        q1 = xHist.at[0,i].get()                #postion
        q2 = xHist.at[1,i].get()
        q3 = xHist.at[2,i].get()
        p1 = xHist.at[3,i].get()                #momentum (transformed)
        p2 = xHist.at[4,i].get()
        p3 = xHist.at[5,i].get()
        ham = hamHist.at[i].get()               #energy
        kin = kinHist.at[i].get()
        pot = potHist.at[i].get()
        qd1 = q_d.at[0,i].get()          #used to be xe - now qd, as xe can be calculated in matlab
        qd2 = q_d.at[1,i].get()                 #Trajectory
        qd3 = q_d.at[2,i].get()
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
        Hc = HconHist.at[i].get()
        data = ['Time:', timestamp  , 'x:   ', q1,q2,q3,p1,p2,p3,ham,kin,pot,qd1,qd2,qd3,phat1,phat2,phat3,ph, Hobs,sc,xp1,xp2,xp3,v1,v2,v3,Hc]
          # data = ['State',i,':', xHist[k,:]] #xHist.at[k,:].get()]# 'End Effector Pose', xeHist.at[k,:].get()]
        
        writer.writerow(data)


