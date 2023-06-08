import jax
from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.numpy import pi, sin, cos, linalg
from jax import grad, jacobian, jacfwd
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

def unravel(dMdq_temp, s):
    # could probably generalise this for any array
    (m,n,l) = jnp.shape(dMdq_temp)
    dMdq1 = jnp.zeros((n,n))
    dMdq2 = jnp.zeros((n,n))
    dMdq3 = jnp.zeros((n,n))
    # print('dmdqshpae',jnp.shape(dMdq1))

    for i in range(n):
        # print('i',i)        #i goes from 0-6
        # print('n*i',n*i)
        dMdq1 = dMdq1.at[0:n,i].set(dMdq_temp.at[n*i:n*(i+1),0,0].get())
        dMdq2 = dMdq2.at[0:n,i].set(dMdq_temp.at[n*i:n*(i+1),1,0].get())
        dMdq3 = dMdq3.at[0:n,i].set(dMdq_temp.at[n*i:n*(i+1),2,0].get())


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

    e = sqM@jnp.block([[e_q],[e_pose]])
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


def create_qdot(q_d,t_span):
    #this function approxmates xdot from IKM solution q_d. The final momentum will always be zero, as the approxmition will shorten the array
    l = jnp.size(t_span)
    qdot = jnp.zeros(jnp.shape(q_d))
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

    Mqt, Tqt, Tqinvt, Jct = massMatrix_holonomic(qt,s)   #Get Mq, Tq and Tqinv for function to get dTqdq
    dMdqt = massMatrixJac(qt,const)
    dMdq1t, dMdq2t, dMdq3t = unravel(dMdqt, s)

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
    q_tilde = x_err.at[0:3].get()
    p_tilde = x_err.at[3:6].get()
    # print('dV',dVdq)
    # dVdq1 = dVdq.at[0,0].get()
    # dVdq2 = dVdq.at[1,0].get()
    # dVdq3 = dVdq.at[2,0].get()
    # gq_hat = jnp.array([[dVdq1],[dVdq2],[dVdq3]])

    # print(gq_hat-dVdq)

    D_hat = jnp.zeros((3,3))
    v = alpha*(Cq - D_hat - Kd)@Kp@(q_tilde + alpha*p_tilde) - Tq@Kp@(q_tilde + alpha*p_tilde) - Kd@p_tilde
    # print(v)

    return v


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
    xp_dot = (Cq_phat - Dq - phi*Tq)@(xp + phi*q) - Tq@dVq + Gq@(u+u0)
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

    Mqo, Tqo, Tqinvo, Jco = massMatrix_holonomic(qo,s) 
    dMdqo = massMatrixJac(qo,consto)
    dMdq1o, dMdq2o, dMdq3o = unravel(dMdqo, s)

    dTqidq1o = solve_continuous_lyapunov(Tqinvo,dMdq1o)
    dTqidq2o = solve_continuous_lyapunov(Tqinvo,dMdq2o)
    dTqidq3o = solve_continuous_lyapunov(Tqinvo,dMdq3o)

    dTqinvo = jnp.array([dTqidq1o,dTqidq2o,dTqidq3o])
    # dMdq_block = jnp.array([dMdq1, dMdq2, dMdq3])
    dVdqo = dV_func(qo,consto)

    # print('dVdqo',dVdqo)
    phato_dynamic = xpo+phi*qo
    Cqo = Cqp(phato_dynamic,Tqo,dTqinvo)
    # print('v', cntrl)

    # print('qo',qo)

    dxp = observer_dynamics(xpo,qo,phio,cntrl,Cqo,dampo,dVdqo,Tqo)
    # print('dxp',dxp)
    xpo_dot = jnp.block([[jnp.zeros((3,1))],[dxp]])
    # print('xpodot', xpo_dot)
    return xpo_dot


#rewrite to bring switch out of switch condtition
def switchCond(phat,kappa,phi,Tq,dTqinvdq_values):
    m,n = jnp.shape(Tq)
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

#INITIAL VALUES
q0_1 = pi/4.
q0_2 = 0.
q0_3 = 0.

q_0 = jnp.array([[q0_1,q0_2,q0_3]])
p0 = jnp.array([0.,0.,0.])

x0 = jnp.block([[q_0,p0]])
x0 = jnp.transpose(x0)
print('Initial States', x0)

q_0 = jnp.transpose(q_0)
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

# V = Vq(q_hat,constants)
# print('V', V)
dV_func = jacfwd(Vq,argnums=0)

#compute \barC matrix
CbSYM = jacfwd(C_SYS,argnums=0)



################################## SIMULATION/PLOT############################################

# This simulations uses p and q hat
(n,hold) = q_0.shape
(m,hold) = x0.shape

#Initialise Simulation Parameters
dt = 0.005
substeps = 1
# dt_sub = dt/substeps      #no longer doing substeps
T = 2.

controlActive = 0     #CONTROL ACTIONS
gravComp = 1.       #1 HAS GRAVITY COMP.
# #Define tuning parameters
alpha = 0.001
Kp = 200.*jnp.eye(n)
Kd = 50.*jnp.eye(n)
ContRate = 100 #Hz: Controller refresh rate
dt_con = 1/ContRate
print('Controller dt',dt_con)
timeConUpdate = 0.     #this forces an initial update at t = 0s
v_control = jnp.zeros((3,1))

#Define Friction
# D = jnp.zeros((3,3))
D = 1.*jnp.eye(n)          #check this imple

endT = T - dt       #prevent truncaton
t = jnp.arange(0,T,dt)
l = jnp.size(t)

#Define Storage
xHist = jnp.zeros((m,l+1))
xeHist = jnp.zeros((m,l))
hamHist = jnp.zeros(l)
kinHist = jnp.zeros(l)
potHist = jnp.zeros(l)
H0Hist = jnp.zeros(l)
xpHist = jnp.zeros((n,l))
phiHist = jnp.zeros(l)
phatHist = jnp.zeros((n,l+1))
switchHist = jnp.zeros(l)


# OBSERVER PARAMETERS
kappa = 2.     #low value to test switches
phi = kappa #phi(0) = k
phat0 = jnp.array([[0.],[0.],[0.]])           #initial momentum estimate
xp0 = phat0 - phi*q_0     #inital xp 
ObsRate = 200.   #Hz: refresh rate of observer
timeObsUpdate = 0.           #last time observer updated
dt_obs = 1/ObsRate
print('Observer dt',dt_obs)
Hobs = 0.

Mqh0, Tq0, Tq0inv, Jc_hat0 = massMatrix_holonomic(q_0,s)   #Get Mq, Tq and Tqinv for function to get dTqdq
dMdq0 = massMatrixJac(q_0,constants)
dMdq10, dMdq20, dMdq30 = unravel(dMdq0, s)
dTq0invdq1 = solve_continuous_lyapunov(Tq0inv,dMdq10)
dTq0invdq2 = solve_continuous_lyapunov(Tq0inv,dMdq20)
dTq0invdq3 = solve_continuous_lyapunov(Tq0inv,dMdq30)
dTqinv0 = jnp.array([dTq0invdq1,dTq0invdq2,dTq0invdq3])

while switchCond(phat0,kappa,phi,Tq0,dTqinv0) <= 0:         #Find initial phi
    phitemp, xptmp = observerSwitch(q_0,phi,xp0,kappa)
    phi = phitemp
    xp0 = xptmp
    print('xp0', xp0)
    print(phi)

# ### TESTING CBAR CONSTRUCTION IN DIFFERENT WAY

# def Tq_phat(q,p_sym,Tq):          #this needs to be derived wrt q.
#     result = Tq@p_sym
#     return result

# Tq_jac = jacfwd(Tq_phat, argnums=0)


#Setting Initial Values
xHist = xHist.at[:,[0]].set(x0)
phatHist = phatHist.at[:,[0]].set(phat0)
xpHist = xpHist.at[:,[0]].set(xp0)
controlHist = jnp.zeros((3,l))      #controlling 3 states

#TRACKING PROBLEM
#solve IKM to find q_d.
origin = jnp.array([[0.5],[0.7]])            #circle origin, or point track. XZ coords.as system is XZ planar
frequency = 0.2
amplitude = 0.1

traj = 'point'      #Name Trajectory Function
# traj = 'planar_circle'      #Name Trajectory Function
# traj = 'sinusoid_x'      #Name Trajectory Function

traj_func = getattr(trajectories,traj)
xe = traj_func(t,origin,frequency,amplitude)        #frequency and amplitude aren't used in point tracking , but function asks for these parameters. Probably really messy implementation
# print(xe)
q_init_guess = jnp.zeros(3)         #initialise initial IKM Guess
# print('q_guss',q_init_guess)

#TRAJECTORY TRACK
q_d = solveIKM(xe,q_init_guess,s)      #solve IKM so trajectory is changed to generalised idsplacement coordinates.
# print('q_d',q_d)
print('qd shape', jnp.shape(q_d))
# print(q_d.at[:,0].get())
dq_d = create_qdot(q_d,t)

print('SIMULATION LOOP STARTED')

jnp.set_printoptions(precision=15)

for k in range(l):
    time = t.at[k].get()
    print('Time',time)

    x = xHist.at[:,[k]].get()
    q = jnp.array([[x.at[0,0].get()],
                   [x.at[1,0].get()],
                   [x.at[2,0].get()]])
    p = jnp.array([[x.at[3,0].get()],        #This is currently returning p, not p0
                   [x.at[4,0].get()],
                   [x.at[5,0].get()]])
    # print(q,p)

    xp = xpHist.at[:,[k]].get()


    phat = xp + phi*q           #find phat for this timestep

    Mq_hat, Tq, Tqinv, Jc_hat = massMatrix_holonomic(q,s)   #Get Mq, Tq and Tqinv for function to get dTqdq

    dMdq = massMatrixJac(q,constants)
    dMdq1, dMdq2, dMdq3 = unravel(dMdq, s)

    dTqinvdq1 = solve_continuous_lyapunov(Tqinv,dMdq1)
    dTqinvdq2 = solve_continuous_lyapunov(Tqinv,dMdq2)
    dTqinvdq3 = solve_continuous_lyapunov(Tqinv,dMdq3)

    dTqinv_block = jnp.array([dTqinvdq1,dTqinvdq2,dTqinvdq3])
    # dMdq_block = jnp.array([dMdq1, dMdq2, dMdq3])
    dVdq = dV_func(q,constants)

    # print(dVdq)

    Cqph = Cqp(phat,Tq,dTqinv_block)     #Calculate value of C(q,phat) Matrix.
    Cqp_real = Cqp(p,Tq,dTqinv_block)     #Calculate value of C(q,phat) Matrix.

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
    print('p~',ptilde)

    if controlActive == 1:
        timeCon = round((time - timeConUpdate),3)
        if timeCon >= dt_con:    #update controller
            print('Controller Updating')
            p_d = Tqinv@dq_d.at[:,[k]].get()                      #as p0 = Mq*qdot, and p = Tq*p0
            # p_d = jnp.zeros((1,3))
            x_d = jnp.block([[q_d.at[:,[k]].get()], [p_d]])
            err = jnp.block([[q], [p]]) - x_d     #define error
            #Find Control Input for current x, xtilde
            # v_control = control(err,Tq,Cqph,Kp,Kd,alpha,gravComp)     #uses Cqp with estimated momentum
            v_control = control(err,Tq,Cqp_real,Kp,Kd,alpha,gravComp)
            timeConUpdate = time

    else:
        v_control = jnp.zeros((3,1))

    if gravComp == 1:
        tau = dVdq 
    else:
        tau = jnp.zeros((3,1))

    v = (tau + v_control)       #multiplication by Gq occures within sys and obs dynamic functions
    # print('v',v)  

    #update ODE for next time step
    # x_k,xp_k = ode_solve(dt,substeps,x,xp,v,Cqph,D,Tq,dTqinv_block,dVdq,phi)        #values for the kth timestep

    #OBSERVER ODE SOLVE 
    timeObs = round((time - timeObsUpdate),3)      #dealing with this float time issue
    # print('Time Elapsed', timeObs)

    
    if timeObs >= dt_obs:    #update observer
            
        x_obs = jnp.array([[x.at[0,0].get()],       #build state vector for observer
                          [x.at[1,0].get()],
                          [x.at[2,0].get()],
                          [xp.at[0,0].get()],
                          [xp.at[1,0].get()],
                          [xp.at[2,0].get()]])

        Hobs = 0.5*(jnp.transpose(ptilde.at[:,0].get())@ptilde.at[:,0].get())
        # print('Time Elapsed', timeObs)
        timeObsUpdate = time
        print('Observer Updating')
        obs_args = (phat,phi,v,D,constants)
                # ode_observer_wrapper(xo,phato,phio,cntrl,dampo,consto)
        xp_update = rk4(x_obs,ode_observer_wrapper,dt_obs,*obs_args)          #call rk4 solver to update ode
        # print('xp_update', xp_update)
        # obs_args = (phi,v,Cqph,D,dVdq,Tq)     #this acts on th observer dyanmics directly
        # xp_update = rk4(x_obs,observer_dynamics,dt_obs,*obs_args)
        # xp_step = jnp.zeros((3,1))       #just put this here to test controller works
        xp_k  = jnp.array([[xp_update.at[3,0].get()],
                          [xp_update.at[4,0].get()],
                          [xp_update.at[5,0].get()]])

    else:
        xp_k = xp           #hold xp value from previous iteration


    #SYSTEM ODE SOLVE
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

    kinTemp = 0.5*(jnp.transpose(p.at[:,0].get())@p.at[:,0].get())
    potTemp = Vq(q,constants)
    hamTemp = kinTemp + potTemp   

    hamHist = hamHist.at[k].set(hamTemp)
    kinHist = kinHist.at[k].set(kinTemp)        #CHANGE THIS BACK
    potHist = potHist.at[k].set(potTemp)
    # xeHist = xeHist.at[:,k].set(xe)



# print(hamHist)
# print(stop)

############### outputting to csv file#####################
############### outputting to csv file#####################
############### outputting to csv file#####################
############### outputting to csv file#####################
details = ['Grav Comp', gravComp, 'dT', dt, 'Substep Number', substeps,' Obs/Cont Rates', ObsRate,ContRate]
controlConstants = ['Control',controlActive,'Kp',Kp,'Kd',Kd,'alpha',alpha, 'kappa',kappa]
header = ['Time', 'State History']
with open('/root/FYP/7LINK_SIMS/data/gravcomp_observer', 'w', newline='') as f:

    writer = csv.writer(f)
    # writer.writerow(simtype)
    writer.writerow(details)
    writer.writerow(controlConstants)
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
        qd1 = q_d.at[0,i].get()          #used to be xe - now qd, as xe can be calculated in matlab
        qd2 = q_d.at[1,i].get()
        qd3 = q_d.at[2,i].get()
        phat1 = phatHist.at[0,i].get()
        phat2 = phatHist.at[1,i].get()
        phat3 = phatHist.at[2,i].get()
        Hobs = H0Hist.at[i].get()
        ph = phiHist.at[i].get()
        xp1 = xpHist.at[0,i].get()
        xp2 = xpHist.at[1,i].get()
        xp3 = xpHist.at[2,i].get()
        sc = switchHist.at[i].get()
        data = ['Time:', timestamp  , 'x:   ', q1,q2,q3,p1,p2,p3,ham,kin,pot,qd1,qd2,qd3,phat1,phat2,phat3,ph, Hobs,sc,xp1,xp2,xp3]
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

