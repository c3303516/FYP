import jax
from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.numpy import pi, sin, cos, linalg
from jax import grad, jacobian, jacfwd
from effectorFKM import FKM, endEffector
from massMatrix_holonomic import massMatrix_holonomic
from dynamics_momentumTransform import dynamics_Transform
# from errorIKM import errorIKM #now in main!
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


def skew(u):
    ans = jnp.block([[0., -u[2], u[1]],
                    [u[2], 0., -u[0]],
                    [-u[1], u[0], 0.]])
    return ans

def hatSE3(x):
    A = skew(x[3:5])
    return A


def rotx(mu):
    A = jnp.array([[1., 0., 0., 0.],
                   [0., jnp.cos(mu), -jnp.sin(mu), 0.],
                   [0., jnp.sin(mu), jnp.cos(mu), 0.],
                   [0., 0., 0., 1.]])
    return A

def roty(mu):
    A = jnp.array([[jnp.cos(mu), 0., jnp.sin(mu), 0.],
                   [0., 1., 0., 0.],
                   [-jnp.sin(mu), 0., jnp.cos(mu), 0.],
                   [0., 0., 0., 1.]])
    return A

def rotz(mu):
    A = jnp.array([[jnp.cos(mu), -jnp.sin(mu), 0., 0.],
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

def massMatrix_continuous(q_hat,qconstants):
   
    dFcdq = jnp.array([
        [0.,0.,0.],
        [0.,0.,0.],
        [0.,0.,0.],
        [0.,0.,0.],
    ])
    q_bold = dFcdq@q_hat + qconstants.at[0].get()
    # print(q_bold)

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

    Mprime = jnp.zeros([Mq_hat.size])
    m,n = jnp.shape(Mq_hat)
    for i in range(m):
        # print('i', i)
        Mprime = Mprime.at[n*i:n*(i+1)].set(Mq_hat.at[0:m,i].get())
        
    return Mprime

def unravel(dMdq_temp, s):
    # could probably generalise this for any array
    (m,n) = jnp.shape(dMdq_temp)
    dMdq1 = jnp.zeros((n,n))
    dMdq2 = jnp.zeros((n,n))
    dMdq3 = jnp.zeros((n,n))

    for i in range(n):
        # print('i',i)        #i goes from 0-6
        # print('n*i',n*i)
        dMdq1 = dMdq1.at[0:n,i].set(dMdq_temp.at[n*i:n*(i+1),0].get())
        dMdq2 = dMdq2.at[0:n,i].set(dMdq_temp.at[n*i:n*(i+1),1].get())
        dMdq3 = dMdq3.at[0:n,i].set(dMdq_temp.at[n*i:n*(i+1),2].get())


    return dMdq1, dMdq2, dMdq3 

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


@partial(jax.jit, static_argnames=['struct'])
def errorIKM(q, q0, xestar,struct):

    #q0 is start point. q is iteration to solve for xstar
    #xstar is currently just positon, no angles
    #EFFECTOR IKM
    q2 = q[0]      #make the full q vector
    q4 = q[1]
    q6 = q[2]
    # print('struct',struct)
    q = jnp.array([     #this is qhat. q0 denotes before momentum transform
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

    pose_weights = jnp.array([1,1])
    sqK = 100000*jnp.diag(pose_weights)
    e_pose = jnp.array([r0E0[0]-xestar[0] ,r0E0[2]-xestar[2]] )  #penalised error from xe to xestar

    e_q = q - q0            #penalises movement from q0
    q_weights = jnp.array([5,3,1])
    sqW = 1*jnp.diag(q_weights)

    sqM = jnp.block([
        [sqW,               jnp.zeros((3, 2))],
        [jnp.zeros((2,3)),  sqK              ]
    ])
    # print(sqM)

#     s = effectorAnalyticalJacobian(q, param, s);      #this is here for future reference if I want to fix angles
#     JA = s.JA;
#     J = [sqW;
#          sqK*JA];

    e = sqM@jnp.block([[e_q],[e_pose]])
                       #coudl use jacobian and feed through. might decrease time
    # print(e)
    e = e[:,0]           
    return e

# def err(q, q_init_guess, x_d,params):       #this isn't used???
#     # print(q_init_guess)
#     # print('xd',x_d)
#     e = errorIKM(q, q_init_guess, x_d,params)
#     return e

def solveIKM(traj,init_guess,params):
    #solves for displacement only
    guess = init_guess
    print(jnp.shape(guess))
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


def planar_circle(freq,amp,origin, t):
    #plane is xz plane.
    x0 = origin.at[0].get()
    z0 = origin.at[1].get()

    x0_vec = x0*jnp.ones(jnp.size(t))
    z0_vec = z0*jnp.ones(jnp.size(t))
    # print('vec',x0_vec)

    x = amp*sin(2*pi*freq*t) + x0
    z = amp*cos(2*pi*freq*t) + z0

    vx = 2*amp*pi*freq*cos(2*pi*freq*t)
    vz = -2*amp*pi*freq*sin(2*pi*freq*t)

    xe = jnp.array([x,jnp.zeros(jnp.size(t)), z])
    ve = jnp.array([vx,jnp.zeros(jnp.size(t)), vz])
    return xe,ve


def ode_solve(xt,Mq_val,Tq_val,dMdq_block,dTqinv_block,dVdq, dt, substep_no,gC,x_tilde, s):
    # x_step = jnp.zeros(m,1)
    x_nextstep = xt
    substep = dt/substep_no
    for i in range(substep_no):
        x_step= rk4(x_nextstep,dynamics_Transform,substep,Mq_val,Tq_val,dMdq_block,dTqinv_block,dVdq,gC,x_tilde,s)
        x_nextstep = x_step

    x_finalstep = x_nextstep
    return x_finalstep

def create_qdot(q_d,t_span):
    #this function approxmates xdot from IKM solution q_d. The final momentum will always be zero, as the approxmition will shorten the array
    l = jnp.size(t_span)
    qdot = jnp.zeros(jnp.shape(q_d))
    # print('qdotsize',jnp.shape(qdot))

    for i in range(l):
        qdot = qdot.at[:,[i]].set(q_d.at[:,[i+1]].get()- q_d.at[:,[i]].get())
    return qdot


######################################## MAIN CODE STARTS HERE #################################
######################################## MAIN CODE STARTS HERE #################################
######################################## MAIN CODE STARTS HERE #################################
######################################## MAIN CODE STARTS HERE #################################
######################################## MAIN CODE STARTS HERE #################################

#Inital states
q_hat1 = 0.
q_hat2 = 0.
q_hat3 = 0.

q_hat = jnp.array([q_hat1,q_hat2,q_hat3])
p_hat = jnp.array([0.,0.,0.])

x0 = jnp.block([[q_hat,p_hat]])
x0 = jnp.transpose(x0)
# print('shape x0',jnp.shape(x0))

q_hat = jnp.transpose(q_hat)

# print('shape q_hat',jnp.shape(q_hat))

s = self()
s = robotParams(s)

constants = jnp.array([                 #These are the positions the wrists are locked to
    [0.],
    [0.],
    [0.],
    [0.],
])
# constants = constants.at[0].get()

Mq = massMatrix_continuous(q_hat,constants)
# print('Mq',Mq)

holonomicTransform = jnp.array([
        [0.,0.,0.],
        [1.,0.,0.],
        [0.,0.,0.],
        [0.,1.,0.],
        [0.,0.,0.],
        [0.,0.,1.],
        [0.,0.,0.],
    ])
# print(Mq)
Mq_hat = jnp.transpose(holonomicTransform)@Mq@holonomicTransform        #for reduced order mass matrix
print(Mq_hat)

massMatrixJac = jacfwd(MqPrime)
dMdqhat = massMatrixJac(q_hat,constants)
# print(dMdqhat)
# print('size dMdq', jnp.shape(dMdqhat))

dMdqhat1, dMdqhat2, dMdqhat3 = unravel(dMdqhat, s)

dMdq_block = jnp.array([dMdqhat1, dMdqhat2, dMdqhat3])
# print('dMdq:', dMdq_block)

V = Vq(q_hat,constants)
print('V', V)
dV = jacfwd(Vq,argnums=0)

###################################################################################################
# print('STOP HERE STOP HER ESTOP HERE STHOP HERE')
# print(fake)


################################## SIMULATION/PLOT############################################

# This simulations uses p and q hat
(m,n) = x0.shape

s.constants = constants         #for holonomic transform
dt = 0.001
substeps = 1
T = 1.5
controlActive = 1     #CONTROL
gravComp = 1.       #1 HAS GRAVITY COMP. Must be a float to maintain precision

t = jnp.arange(0,T,dt)
l = t.size

xHist = jnp.zeros((6,l+1))
# print('xHist',xHist)
hamHist = jnp.zeros(l)
kinHist = jnp.zeros(l)
potHist = jnp.zeros(l)
# print('hamHist',hamHist)
# print('x0',x0)

xHist = xHist.at[:,[0]].set(x0)
controlHist = jnp.zeros((3,l))      #controlling 3 states

#Tracking Problem
#solve IKM to find q_d.

# x_d_cart = jnp.array([[0.2],[0.],[0.8]])        #define point to reach

circle_origin = jnp.array([0.,0.8])            #define circle parameters
frequency = 1.
circle_radius = 0.2
xe, ve = planar_circle(frequency,circle_radius,circle_origin,t)       #return cartesian coords for circle path.


# print(x_d)

q_init_guess = jnp.zeros(3)
# print('q_guss',q_init_guess)


q_d = solveIKM(xe,q_init_guess,s)      #solve IKM so trajectory is changed to generalised idsplacement coordinates.
print('q_d',q_d)
# print(q_d.at[:,0].get())
dq_d = create_qdot(q_d,t)

# print(dq_d)
# print(stop)
# q_d = jnp.array([pi/6., -pi/3., pi/6.])
# q_tilde = q_hat - q_d
# p_d = jnp.zeros(3)                      #no momentum error


jnp.set_printoptions(precision=15)

for k in range(l):
    x = xHist.at[:,[k]].get()
    q = jnp.array([x.at[0,0].get(),
                   x.at[1,0].get(),
                   x.at[2,0].get()])
    p = jnp.array([x.at[3,0].get(),        #This is currently returning p, not p0
                   x.at[4,0].get(),
                   x.at[5,0].get()])
    # print(q,p)
    
    Mq_hat, Tq, Tqinv, Jc_hat = massMatrix_holonomic(q,s)   #Get Mq, Tq and Tqinv for function to get dTqdq

    dMdq = massMatrixJac(q,constants)       #might inject this mq directly into the dynamics later
    dMdq1, dMdq2, dMdq3 = unravel(dMdq, s)
    # print('Tq',Tq)
    # print('Tqinv',Tqinv)
    # print(stop)
    dTqinvdq1 = solve_continuous_lyapunov(Tqinv,dMdq1)
    dTqinvdq2 = solve_continuous_lyapunov(Tqinv,dMdq2)
    dTqinvdq3 = solve_continuous_lyapunov(Tqinv,dMdq3)

    dTqinv_block = jnp.array([dTqinvdq1,dTqinvdq2,dTqinvdq3])
    # print(type(dMdq1))
    
    dMdq_block = jnp.array([dMdq1, dMdq2, dMdq3])
    # print('block',dMdq_block)
    # V = Vq(q)
    dVdq = dV(q,constants)
    time = t.at[k].get()

    if controlActive == 0:          #reset if control action is down
        err = jnp.zeros((6,1))
    else: 
        p_d = Tqinv@dq_d.at[:,k].get()
        x_d = jnp.block([[q_d.at[:,k].get(), p_d]])
        err = jnp.transpose(jnp.block([[q, p ]])) - jnp.transpose(x_d)     #define error
    
    xtemp = ode_solve(x,Mq_hat,Tq,dMdq_block,dTqinv_block,dVdq,dt, substeps,gravComp, err, s)     #try dormand prince. RK4 isn't good enough

    if jnp.isnan(xtemp.at[0,0].get()):
        print(xtemp.at[0,0].get())
        print('NAN found, exiting loop')
        break


    xHist = xHist.at[:,[k+1]].set(xtemp)
    # print(xtemp)
    # controlHist = controlHist.at[:,[k]].set(controlAction)
    # print('size of p and q', jnp.shape(p),jnp.shape(q))

    kinTemp = 0.5*(jnp.transpose((p))@(p))
    potTemp = Vq(q,constants)
    hamTemp = 0.5*(jnp.transpose((p))@(p)) + potTemp

    hamHist = hamHist.at[k].set(hamTemp)
    kinHist = kinHist.at[k].set(kinTemp)
    potHist = potHist.at[k].set(potTemp)



print(hamHist)
# print(stop)

############### outputting to csv file#####################
############### outputting to csv file#####################
############### outputting to csv file#####################
############### outputting to csv file#####################
details = ['Grav Comp', gravComp, 'dT', dt, 'Substep Number', substeps]
header = ['Time', 'State History']
with open('/root/FYP/7LINK_TRAJECTORY/data/circularpath_veltrack2', 'w', newline='') as f:

    writer = csv.writer(f)
    # writer.writerow(simtype)
    writer.writerow(details)
    writer.writerow(header)

    # writer.writerow(['Time', t])
    for i in range(l):
        q1 = xHist.at[0,i].get()
        q2 = xHist.at[1,i].get()
        q3 = xHist.at[2,i].get()
        p1 = xHist.at[3,i].get()
        p2 = xHist.at[4,i].get()
        p3 = xHist.at[5,i].get()
        ham = hamHist.at[i].get()
        kin = kinHist.at[i].get()
        pot = potHist.at[i].get()
        timestamp = t.at[i].get()
        data = ['Time:', timestamp  , 'x:   ', q1,q2,q3,p1,p2,p3,ham,kin,pot]
          # data = ['State',i,':', xHist[k,:]] #xHist.at[k,:].get()]# 'End Effector Pose', xeHist.at[k,:].get()]
        
        writer.writerow(data)
    # header = ['Time', 'Control History']
    # writer.writerow(details)
    # for i in range(l):
    #     c1 = controlHist.at[0,i].get()
    #     c2 = controlHist.at[1,i].get()
    #     c3 = controlHist.at[2,i].get()

    #     timestamp = t.at[i].get()
    #     data = ['Time:', timestamp, 'Control Action:    ', c1,c2,c3]

    #     writer.writerow(data)
# print('xHist',xHist)    
# print('xeHist',xeHist)

