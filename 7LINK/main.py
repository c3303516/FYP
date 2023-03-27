
from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.numpy import pi, sin, cos, linalg
from jax import grad, jacobian, jacfwd
from effectorFKM import FKM, endEffector
from massMatrix import massMatrix
# from dynamics import  gravTorque
from dynamics import dynamics_test
from rk4 import rk4
from params import robotParams
from copy import deepcopy
from scipy import integrate
import sys

import csv

# import matplotlib.pyplot as plt

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
    # A = jnp.array([[1., 0., 0., 0.],
    #                [0., mu, -mu, 0.],
    #                [0., mu, mu, 0.],
    #                [0., 0., 0., 1.]])
    return A

def roty(mu):
    A = jnp.array([[jnp.cos(mu), 0., jnp.sin(mu), 0.],
                   [0., 1., 0., 0.],
                   [-jnp.sin(mu), 0., jnp.cos(mu), 0.],
                   [0., 0., 0., 1.]])
    # A = jnp.array([[mu, 0., mu, 0.],
    #                [0., 1., 0., 0.],
    #                [mu, 0., mu, 0.],
    #                [0., 0., 0., 1.]])
    return A

def rotz(mu):
    A = jnp.array([[jnp.cos(mu), -jnp.sin(mu), 0., 0.],
                   [jnp.sin(mu), jnp.cos(mu), 0., 0.],
                   [0., 0., 1., 0.],
                   [0., 0., 0., 1.]])
    # A = jnp.array([[mu, -mu, 0., 0.],
    #                [mu, mu, 0., 0.],
    #                [0., 0., 1., 0.],
    #                [0., 0., 0., 1.]])
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

def massMatrix_continuous(q0):
    q1 = q0.at[0].get()
    q2 = q0.at[1].get()
    q3 = q0.at[2].get()
    q4 = q0.at[3].get()
    q5 = q0.at[4].get()
    q6 = q0.at[5].get()
    q7 = q0.at[6].get()

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

    # print('a01',A01)
    # print('a02',A02)
    # print('a03',A03)
    # print('a04',A04)
    # print('a05',A05)
    # print('a06',A0G)
    # print('a07',A07)
    # print('a0e',A0E)
    # print('a0G',A0G)

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

    # print(c2)
    # print(c2x)

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
    # print('rc100',rc100)
    # print('rc200',rc200)
    # print('rc300',rc300)
    # print('rc400',rc400)
    # print('rc500',rc500)
    # print('rc600',rc600)
    # print('rc700',rc700)
    # print('rc800',rc800)

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
    # JcG   = jnp.block([
    #     [ske1@(rcG00-r100),  ske2@(rcG00-r200),  ske3@(rcG00-r200),  ske4@(rcG00-r400),  ske5@(rcG00-r500),  ske6@(rcG00-r600),  ske7@(rcG00-r700),  ske8@(rcG00-r800)],
    #     [z01,                z02,                z03,                z04              ,  z05,                z06,                z07,                z08              ]
    #     ])
    # print('Jc2', Jc2)
    # print('Jc3', Jc3)
    # print('Jc4', Jc4)
    # print('Jc5', Jc5)
    # print('Jc6', Jc6)
    # print('Jc7', Jc7)
    # print('Jc8', Jc8)
    # print('JcG', JcG)

    # s.Jc2 = Jc2
    # s.Jc3 = Jc3
    # s.Jc4 = Jc4
    # s.Jc5 = Jc5
    # s.Jc6 = Jc6
    # s.Jc7 = Jc7
    # s.Jc8 = Jc8
    # s.JcG = JcG
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
    # MG = JcG.T@jnp.block([
    #     [jnp.multiply(s.mGripper,jnp.eye(3,3)), jnp.zeros((3,3))],
    #     [jnp.zeros((3,3)),            R0G.T@IG@R0G ]
    # ])@JcG

    Mq = M2 + M3 + M4 + M5 + M6 + M7 + M8# + MG
    return Mq
    
def MqPrime(q0):

    Mq = massMatrix_continuous(q0)
    Mprime = jnp.zeros([Mq.size])
    m,n = jnp.shape(Mq)
    # print('n', n)
    # print('size Mq', Mq.size)
    for i in range(m):
        # print('i', i)
        Mprime = Mprime.at[n*i:n*(i+1)].set(Mq.at[0:m,i].get())
        
    # Mprime = Mprime.at[0:7].set(Mq.at[0:8,0].get())
    # Mprime = Mprime.at[7:14].set(Mq.at[0:8,1].get())
    # Mprime = Mprime.at[14:21].set(Mq.at[0:8,2].get())
    # Mprime = Mprime.at[21:28].set(Mq.at[0:8,3].get())
    # Mprime = Mprime.at[28:35].set(Mq.at[0:8,4].get())
    # Mprime = Mprime.at[35:42].set(Mq.at[0:8,5].get())
    # Mprime = Mprime.at[42:49].set(Mq.at[0:8,6].get())
    return Mprime

def unravel(dMdq, s):
    # could probably generalise this for any array
    (m,n) = jnp.shape(dMdq)
    # print('m,n',m,n)
    dMdq1 = jnp.zeros((n,n))
    dMdq2 = jnp.zeros((n,n))
    dMdq3 = jnp.zeros((n,n))
    dMdq4 = jnp.zeros((n,n))
    dMdq5 = jnp.zeros((n,n))
    dMdq6 = jnp.zeros((n,n))
    dMdq7 = jnp.zeros((n,n))


    for i in range(n):
        # print('i',i)        #i goes from 0-6
        # print('n*i',n*i)
        dMdq1 = dMdq1.at[0:n,i].set(dMdq.at[n*i:n*(i+1),0].get())
        dMdq2 = dMdq2.at[0:n,i].set(dMdq.at[n*i:n*(i+1),1].get())
        dMdq3 = dMdq3.at[0:n,i].set(dMdq.at[n*i:n*(i+1),2].get())
        dMdq4 = dMdq4.at[0:n,i].set(dMdq.at[n*i:n*(i+1),3].get())
        dMdq5 = dMdq5.at[0:n,i].set(dMdq.at[n*i:n*(i+1),4].get())
        dMdq6 = dMdq6.at[0:n,i].set(dMdq.at[n*i:n*(i+1),5].get())
        dMdq7 = dMdq7.at[0:n,i].set(dMdq.at[n*i:n*(i+1),6].get())
    # print(dMdq1)
    # dMdq2 = dMdq2.at[0:2,0].set(dMdq.at[0:2,1].get())
    # dMdq2 = dMdq2.at[0:2,1].set(dMdq.at[2:4,1].get())
    # print(dMdq2)

    # s.dMdq1 = dMdq1
    # s.dMdq2 = dMdq2
    # s.dMdq3 = dMdq3
    # s.dMdq4 = dMdq4
    # s.dMdq5 = dMdq5
    # s.dMdq6 = dMdq6
    # s.dMdq7 = dMdq7
    #THis function has been edited to return each dMdq as another output
    return dMdq1, dMdq2, dMdq3, dMdq4, dMdq5, dMdq6, dMdq7 

def Vq(q,s):
    #Function has to do FKM again to enable autograd to work
    q1 = q.at[0].get()
    q2 = q.at[1].get()
    q3 = q.at[2].get()
    q4 = q.at[3].get()
    q5 = q.at[4].get()
    q6 = q.at[5].get()
    q7 = q.at[6].get()

    A01 = tranz(s.l1)@rotx(pi)@rotz(q1)          
    A12 = rotx(pi/2)@tranz(-s.d1)@trany(-s.l2)@rotz(q2)
    A23 = rotx(-pi/2)@trany(s.d2)@tranz(-s.l3)@rotz(q3)
    A34 = rotx(pi/2)@tranz(-s.d2)@trany(-s.l4)@rotz(q4)
    A45 = rotx(-pi/2)@trany(s.d2)@tranz(-s.l5)@rotz(q5)
    A56 = rotx(pi/2)@trany(-s.l6)@rotz(q6)
    A67 = rotx(-pi/2)@tranz(-s.l7)@rotz(q7)
    # A7E = rotx(pi)@tranz(s.l8)    
    # AEG = tranz(s.lGripper)

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
    # A0cG = A0E@tranz(cGz)

    # rc100   = A0c1.at[0:3,3].get()
    # rc200   = A0c2.at[0:3,3].get()
    # rc300   = A0c3.at[0:3,3].get()
    # rc400   = A0c4.at[0:3,3].get()
    # rc500   = A0c5.at[0:3,3].get()
    # rc600   = A0c6.at[0:3,3].get()
    # rc700   = A0c7.at[0:3,3].get()
    # rc800   = A0c8.at[0:3,3].get()
    # rcG00   = A0cG.at[0:3,3].get()
    rc100   = A0c1[0:3,3]
    rc200   = A0c2[0:3,3]
    rc300   = A0c3[0:3,3]
    rc400   = A0c4[0:3,3]
    rc500   = A0c5[0:3,3]
    rc600   = A0c6[0:3,3]
    rc700   = A0c7[0:3,3]
    rc800   = A0c8[0:3,3]
    # rcG00   = A0cG[0:3,3]

    # print('rc100',rc100)
    # print('rc200',rc200)
    # print('rc300',rc300)
    # print('rc400',rc400)
    # print('rc500',rc500)
    # print('rc600',rc600)
    # print('rc700',rc700)
    # print('rc800',rc800)
    g0 = jnp.array([[0.],[0.],[-s.g]])
    gprime = jnp.transpose(g0)

    V = -s.m1*gprime@rc100-s.m2*gprime@rc200-s.m3*gprime@rc300-s.m4*gprime@rc400-s.m5*gprime@rc500-s.m6*gprime@rc600 -s.m7*gprime@rc700 -s.m8*gprime@rc800 #-s.mGripper*gprime@rcG00
    s.V = V
    return V.at[0].get()


# def dynamics(x, s):
    q1 = x.at[(0,0)].get()
    q2 = x.at[(1,0)].get()
    q3 = x.at[(2,0)].get()
    q4 = x.at[(3,0)].get()
    q5 = x.at[(4,0)].get()
    q6 = x.at[(5,0)].get()
    q7 = x.at[(6,0)].get()

    p1 = x.at[(7,0)].get()
    p2 = x.at[(8,0)].get()
    p3 = x.at[(9,0)].get()
    p4 = x.at[(10,0)].get()
    p5 = x.at[(11,0)].get()
    p6 = x.at[(12,0)].get()
    p7 = x.at[(13,0)].get()

    q = jnp.array([
        q1, q2, q3, q4, q5, q6, q7
        ])
    p = jnp.array([
        p1,p2,p3,p4,p5,p6,p7
        ])
    # print('q1',q1)
    # print('q2',q2)
    # print('p',p)
    # print('q',q)

    FKM(q,s)

    massMatrix(q, s)
    # Gravitation torque
    s.g0 = jnp.array([[0],[0],[-s.g]])
    g0 = s.g0
    gq = gravTorque(s)
    s.gq = gq
    dVdq = s.dV(q)
    dVdq1 = dVdq.at[0].get()
    dVdq2 = dVdq.at[1].get()
    dVdq3 = dVdq.at[2].get()
    dVdq4 = dVdq.at[3].get()
    dVdq5 = dVdq.at[4].get()
    dVdq6 = dVdq.at[5].get()
    dVdq7 = dVdq.at[6].get()


    # Mass matrix inverse
    Mq = s.Mq
    dMdq1 = s.dMdq1
    dMdq2 = s.dMdq2
    dMdq3 = s.dMdq3
    dMdq4 = s.dMdq4
    dMdq5 = s.dMdq5
    dMdq6 = s.dMdq6
    dMdq7 = s.dMdq7

    temp1 = jnp.transpose(linalg.solve(jnp.transpose(Mq), jnp.transpose(dMdq1)))
    temp2 = jnp.transpose(linalg.solve(jnp.transpose(Mq), jnp.transpose(dMdq2)))
    temp3 = jnp.transpose(linalg.solve(jnp.transpose(Mq), jnp.transpose(dMdq3)))
    temp4 = jnp.transpose(linalg.solve(jnp.transpose(Mq), jnp.transpose(dMdq4)))
    temp5 = jnp.transpose(linalg.solve(jnp.transpose(Mq), jnp.transpose(dMdq5)))
    temp6 = jnp.transpose(linalg.solve(jnp.transpose(Mq), jnp.transpose(dMdq6)))
    temp7 = jnp.transpose(linalg.solve(jnp.transpose(Mq), jnp.transpose(dMdq7)))
    
    dMinvdq1 = linalg.solve(-Mq, temp1)
    dMinvdq2 = linalg.solve(-Mq, temp2)
    dMinvdq3 = linalg.solve(-Mq, temp3)
    dMinvdq4 = linalg.solve(-Mq, temp4)
    dMinvdq5 = linalg.solve(-Mq, temp5)
    dMinvdq6 = linalg.solve(-Mq, temp6)
    dMinvdq7 = linalg.solve(-Mq, temp7)


    # print('dVdq', jnp.transpose(dVdq))
    dHdq = 0.5*(jnp.array([
        [jnp.transpose(p)@dMinvdq1@p],
        [jnp.transpose(p)@dMinvdq2@p],
        [jnp.transpose(p)@dMinvdq3@p],
        [jnp.transpose(p)@dMinvdq4@p],
        [jnp.transpose(p)@dMinvdq5@p],
        [jnp.transpose(p)@dMinvdq6@p],
        [jnp.transpose(p)@dMinvdq7@p]
    ])) + jnp.array([[dVdq1], [dVdq2], [dVdq3], [dVdq4], [dVdq5], [dVdq6], [dVdq7]])    # addition now works and gives same shape, however numerical values are incorrect

    # print('dHdq', dHdq)

    dHdp = 0.5*jnp.block([
    [jnp.transpose(p)@linalg.solve(s.Mq,jnp.array([[1.],[0.],[0.],[0.],[0.],[0.],[0.]])) + (jnp.array([1.,0.,0.,0.,0.,0.,0.])@linalg.solve(s.Mq,p))],
    [jnp.transpose(p)@linalg.solve(s.Mq,jnp.array([[0.],[1.],[0.],[0.],[0.],[0.],[0.]])) + (jnp.array([0.,1.,0.,0.,0.,0.,0.])@linalg.solve(s.Mq,p))],
    [jnp.transpose(p)@linalg.solve(s.Mq,jnp.array([[0.],[0.],[1.],[0.],[0.],[0.],[0.]])) + (jnp.array([0.,0.,1.,0.,0.,0.,0.])@linalg.solve(s.Mq,p))],
    [jnp.transpose(p)@linalg.solve(s.Mq,jnp.array([[0.],[0.],[0.],[1.],[0.],[0.],[0.]])) + (jnp.array([0.,0.,0.,1.,0.,0.,0.])@linalg.solve(s.Mq,p))],
    [jnp.transpose(p)@linalg.solve(s.Mq,jnp.array([[0.],[0.],[0.],[0.],[1.],[0.],[0.]])) + (jnp.array([0.,0.,0.,0.,1.,0.,0.])@linalg.solve(s.Mq,p))],
    [jnp.transpose(p)@linalg.solve(s.Mq,jnp.array([[0.],[0.],[0.],[0.],[0.],[1.],[0.]])) + (jnp.array([0.,0.,0.,0.,0.,1.,0.])@linalg.solve(s.Mq,p))],
    [jnp.transpose(p)@linalg.solve(s.Mq,jnp.array([[0.],[0.],[0.],[0.],[0.],[0.],[1.]])) + (jnp.array([0.,0.,0.,0.,0.,0.,1.])@linalg.solve(s.Mq,p))]
    ]) 

    # print('dHdp', dHdp)
    D = 0.5*jnp.eye(7)
    xdot = jnp.block([
        [jnp.zeros((7,7)), jnp.eye(7)],
        [-jnp.eye(7),      -D ],
    ])@jnp.block([[dHdq],[dHdp]])       #CHECK YOU ACTUALLY PUT THE GRAVITY IN 

    print('xdot', xdot)

    return xdot

def ode_solve(xt,dMdq_block,dVdq, dt, m,contA, s):
    # x_step = jnp.zeros(m,1)
    x_nextstep = xt
    substep = dt/m
    for i in range(m):
        x_step= rk4(x_nextstep,substep,dynamics_test,contA,dMdq_block,dVdq,s)
        x_nextstep = x_step
        # print('xstep', x_step)

    x_finalstep = x_nextstep
    return x_finalstep


## MAIN CODE STARTS HERE

# q1 = 0.
# q2 = pi/2.
# q3 = 0.
# q4 = 0.
# q5 = 0.
# q6 = 0.
# q7 = 0.

#HOME POSITION OF BOT
q1 = 0   
q2 = 0.261799387799149   
q3 = 3.141592653589793   
q4 = 4.014257279586958                   
q5 = 0   
q6 = 0.959931088596881   
q7 = 1.570796326794897


q0 = jnp.array([q1,q2,q3,q4,q5,q6,q7])

p0 = jnp.array([0.,0.,0.,0.,0.,0.,0.])

x0 = jnp.block([[q0,p0]])
x0 = jnp.transpose(x0)
# print('x0',x0)

s = self()
s = robotParams(s)
s.pred = 0
# Mq = massMatrix_continuous(q0)
# print(Mq)
FKM(q0,s)
Mq = massMatrix(q0,s)

jnp.set_printoptions(precision=15)
# print('Mass Matrix',Mq)

# Mq_cont = massMatrix_continuous(q0)
# print('Mq_cont', Mq_cont)
# print('size Mq', jnp.shape(Mq))
# MqP = MqPrime(q0)

# print('MqPrime', MqP)
# print('MqPrime size', jnp.size(MqP))

massMatrixJac = jacfwd(MqPrime)
dMdq = massMatrixJac(q0)
# print('size dMdq', jnp.shape(dMdq))
# unravel(dMdq, s)

dMdq1, dMdq2, dMdq3, dMdq4, dMdq5, dMdq6, dMdq7 = unravel(dMdq, s)

dMdq_block = jnp.array([dMdq1, dMdq2, dMdq3, dMdq4, dMdq5, dMdq6, dMdq7])


# dMdq1_extract = dMdq_block.at[0].get()
# print('dMdq1', dMdq1)
# print('dMdq1 extract', dMdq1_extract)

# print('dMdq1 diff', dMdq1 - dMdq1_extract)
# V = Vq(q0)
# print('V', V)
dV = jacfwd(Vq, argnums= 0)
# print('dV', s.dV(q0))
s.controlActive = 0     #CONTROL
s.controlAction = jnp.array([[0.],[0.],[0.],[0.],[0.],[0.],[0.]])
s.gravityCompensation = 0       #1 HAS GRAVITY COMP
# s.time = 5
# xdot = dynamics_test(x0, s)
# print('xdot', xdot)

# s.time = 0.1
# xdot = dynamics_test(x0, s)
# print('xdot', xdot)

# print('STOP HERE STOP HER ESTOP HERE STHOP HERE')
# print(fake)
# SIMULATION/PLOT
(m,n) = x0.shape

dt = 0.0005
substeps = 20
T = .5
s.controlActive = 0     #CONTROL
s.gravityCompensation = 0       #1 HAS GRAVITY COMP

t = jnp.arange(0,T,dt)
l = t.size

xHist = jnp.zeros((m,l+1))
print('xHist',xHist)
print('x0',x0)

xHist = xHist.at[:,[0]].set(x0)
# xeHist = jnp.zeros((6,l))

for k in range(l):
    x = xHist.at[:,[k]].get()
    q = jnp.array([x.at[0,0].get(),
                   x.at[1,0].get(),
                   x.at[2,0].get(),
                   x.at[3,0].get(),
                   x.at[4,0].get(),
                   x.at[5,0].get(),
                   x.at[6,0].get()])
    p = jnp.array([x.at[7,0].get(),
                   x.at[8,0].get(),
                   x.at[9,0].get(),
                   x.at[10,0].get(),
                   x.at[11,0].get(),
                   x.at[12,0].get(),
                   x.at[13,0].get()])

    dMdq = massMatrixJac(q)
    dMdq1, dMdq2, dMdq3, dMdq4, dMdq5, dMdq6, dMdq7 = unravel(dMdq, s)
    
    dMdq_block = jnp.array([dMdq1, dMdq2, dMdq3, dMdq4, dMdq5, dMdq6, dMdq7])
    # V = Vq(q)
    dVdq = dV(q,s)
    time = t.at[k].get()
    controlAction = jnp.zeros((7,1))

    xtemp = ode_solve(x,dMdq_block,dVdq,dt, substeps,controlAction, s)     #try dormand prince. RK4 isn't good enough
    # xtemp = integrate.solve_ivp(dynamics_test,x.at[:,0].get(),method='RK45',args=(x,controlAction,s))
    # print(xtemp)
    if jnp.isnan(xtemp.at[0,0].get()):
        print(xtemp.at[0,0].get())
        sys.exit('Code is stopped cause of NAN')
    xHist = xHist.at[:,[k+1]].set(xtemp)
    s2 = FKM(xtemp,s)

    
#outputting to csv file

details = ['Grav Comp', s.gravityCompensation, 'dT', dt, 'Substep Number', substeps]
header = ['Time', 'State History']
with open('/root/FYP/7LINK/data/HomePosition_structless_test', 'w', newline='') as f:

    writer = csv.writer(f)
    writer.writerow(details)
    writer.writerow(header)

    # writer.writerow(['Time', t])
    for i in range(l):
        q1 = xHist.at[0,i].get()
        q2 = xHist.at[1,i].get()
        q3 = xHist.at[2,i].get()
        q4 = xHist.at[3,i].get()
        q5 = xHist.at[4,i].get()
        q6 = xHist.at[5,i].get()
        q7 = xHist.at[6,i].get()
        p1 = xHist.at[7,i].get()
        p2 = xHist.at[8,i].get()
        p3 = xHist.at[9,i].get()
        p4 = xHist.at[10,i].get()
        p5 = xHist.at[11,i].get()
        p6 = xHist.at[12,i].get()
        p7 = xHist.at[13,i].get()
        timestamp = t.at[i].get()
        data = ['Time:', timestamp  , 'x:   ', q1,q2,q3,q4,q5,q6,q7,p1,p2,p3,p4,p5,p6,p7]
        # data = ['State',i,':', xHist[k,:]] #xHist.at[k,:].get()]# 'End Effector Pose', xeHist.at[k,:].get()]
        
        writer.writerow(data)

# print('xHist',xHist)    
# print('xeHist',xeHist)

def plot():
    fig, ax = plt.subplots(7,1)
    # ax = fig.subplots()
    ax[0].plot(t, xHist.at[0,:].get())

    ax[1].plot(t, xHist.at[1,:].get())

    ax[2].plot(t, xHist.at[2,:].get())

    ax[3].plot(t, xHist.at[3,:].get())

    ax[4].plot(t, xHist.at[4,:].get())

    ax[5].plot(t, xHist.at[5,:].get())

    ax[6].plot(t, xHist.at[6,:].get())
    fig.savefig('plot_test_7.png')


    fig, ax = plt.subplots(3,1)
    ax[0].plot(t, xeHist.at[0,:].get())

    ax[1].plot(t, xeHist.at[1,:].get())

    ax[2].plot(t, xeHist.at[2,:].get())
    fig.savefig('plot_test_7_endEffector.png')



    ax = plt.figure().add_subplot(projection = '3d')
    # plt.Axes3D.plot(xeHist.at[0,0].get(),xeHist.at[1,0].get(),xeHist.at[2,0].get())
    ax.plot(xeHist.at[0,0].get(),xeHist.at[1,0].get(),xeHist.at[2,0].get(), label='7 Link Manipulator')
    plt.show()
    fig.savefig('plot_test_3D_7Link')

