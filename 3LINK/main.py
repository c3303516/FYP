# from params import *
# from homogeneousTransforms import *
from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.numpy import pi, sin, cos, linalg
from jax import grad, jacobian, jacfwd
from effectorFKM import FKM
from massMatrix import massMatrix
from dynamics import  gravTorque
from rk4 import rk4

import matplotlib.pyplot as plt

class self:
    def __init___(self):
        return 'initialised'

g = 9.81
m1 = 1.5
m2 = 1
m3 = 0.8

l1 = 1.5
l2 = 0.8
l3 = 0.5
c1 = 0.6
c2 = 0.4
c3 = 0.25

Iz1 = m1*l1*l1/12
Iz2 = m2*l2*l2/12
Iz3 = m3*l3*l3/12

I1 = jnp.array([[0., 0., 0.],
          [0., 0., 0.],
          [0., 0., Iz1]])
I2 = jnp.array([[0., 0., 0.],
          [0., 0., 0.],
          [0., 0., Iz2]])
I3 = jnp.array([[0., 0., 0.],
                [0., Iz3, 0.],
                [0., 0., 0.]])

# def fun_sin(x):
#     A = jnp.sin(x)
#     return A

# def fun_cos(x):
#     A = jnp.cos(x)
#     return A


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

    A01 = rotx(-pi/2)@trany(-l1)@rotz(q1)
    A12 = trany(-l2)@rotz(q2)
    A23 = trany(-l3)@rotx(pi/2)
    A02 = A01@A12
    A03 = A02@A23

    # Geometric Jacobians
    R01 = A01[0:3,0:3]     #rotation matrices
    R12 = A12[0:3,0:3]

    A0c1 = rotx(-pi/2)@trany(-c1)@rotz(q1)
    A0c2 = A01@trany(-c2)@rotz(q2)
    A0c3 = A02@trany(-c3)@rotx(pi/2)

    r100   = A01[0:3,[3]]
    r200   = A02[0:3,[3]]

    rc100   = A0c1[0:3,[3]]
    rc200   = A0c2[0:3,[3]]
    rc300   = A0c3[0:3,[3]]

    z00 = jnp.array([[0.], [0.], [1.]])
    z01 = R01@z00
    z02 = R01@R12@z00

    ske1 = skew(z01)
    ske2 = skew(z02)

    Jc2   = jnp.block([
        [ske1@(rc200-r100), jnp.zeros((3,1))],
        [z01,               jnp.zeros((3,1))]
        ])
    Jc3   = jnp.block([
        [ske1@(rc300-r100), ske2@(rc300-r200)],
        [z01,               z02]
        ])

    # Mass Matrix
    R02 = A02[0:3,0:3]
    R03 = A03[0:3,0:3]

    M2 = Jc2.T@jnp.block([
        [jnp.multiply(m2,jnp.eye(3,3)), jnp.zeros((3,3))],
        [jnp.zeros((3,3)),            R02.T@I2@R02 ]
    ])@Jc2
    M3 = Jc3.T@jnp.block([
        [jnp.multiply(m3,jnp.eye(3,3)), jnp.zeros((3,3))],
        [jnp.zeros((3,3)),            R03.T@I3@R03 ]
    ])@Jc3

    Mq = M2 + M3
    return Mq
    
def MqPrime(q0):

    Mq = massMatrix_continuous(q0)
    Mprime = jnp.zeros([Mq.size])
    Mprime = Mprime.at[0:2].set(Mq.at[0:2,0].get())
    Mprime = Mprime.at[2:4].set(Mq.at[0:2,1].get())
    return Mprime


def unravel(dMdq, s):
    # could probably generalise this for any array
    (m,n) = jnp.shape(dMdq)
    dMdq1 = jnp.zeros((2,2))
    dMdq2 = jnp.zeros((2,2))
    # for i in range(n):
    dMdq1 = dMdq1.at[0:2,0].set(dMdq.at[0:2,0].get())
    dMdq1 = dMdq1.at[0:2,1].set(dMdq.at[2:4,0].get())
    # print(dMdq1)
    dMdq2 = dMdq2.at[0:2,0].set(dMdq.at[0:2,1].get())
    dMdq2 = dMdq2.at[0:2,1].set(dMdq.at[2:4,1].get())
    # print(dMdq2)
    s.dMdq2 = dMdq2
    s.dMdq1 = dMdq1
    return s

def Vq(q):
    #Function has to do FKM again to enable autograd to work
    q1 = q.at[0].get()
    q2 = q.at[1].get()

    A01 = rotx(-pi/2)@trany(-l1)@rotz(q1)
    A12 = trany(-l2)@rotz(q2)

    A02 = A01@A12

    A0c1 = rotx(-pi/2)@trany(-c1)@rotz(q1)
    A0c2 = A01@trany(-c2)@rotz(q2)
    A0c3 = A02@trany(-c3)@rotx(pi/2)

    rc100   = A0c1[0:3,3]
    rc200   = A0c2[0:3,3]
    rc300   = A0c3[0:3,3]
    g0 = jnp.array([[0.],[0.],[-g]])
    gprime = jnp.transpose(g0)
    
    V = -m1*gprime@rc100 -m2*gprime@rc200 -m3*gprime@rc300
    s.V = V
    return V.at[0].get()


def dynamics(x, s):
    q1 = x.at[(0,0)].get()
    q2 = x.at[(1,0)].get()
    q = jnp.array([
        q1, q2
    ])
    p = jnp.array([
        x.at[(2,0)].get(), x.at[(3,0)].get()
        ])
    # print('q1',q1)
    # print('q2',q2)
    # print('p',p)
    # print('q',q)

    FKM(q1,q2,s)

    massMatrix(q, s)
    # Gravitation torque

    s.g0 = jnp.array([[0],[0],[-g]])
    g0 = s.g0
    gq = gravTorque(s)
    s.gq = gq
    dVdq = s.dV(q)
    dVdq1 = dVdq.at[0].get()
    dVdq2 = dVdq.at[1].get()

    # Mass matrix inverse
    Mq = s.Mq
    dMdq1 = s.dMdq1
    dMdq2 = s.dMdq2
    b = jnp.transpose(linalg.solve(jnp.transpose(Mq), jnp.transpose(dMdq1)))
    dMinvdq1 = linalg.solve(-Mq, b)

    b = jnp.transpose(linalg.solve(jnp.transpose(Mq), jnp.transpose(dMdq2)))
    dMinvdq2 = linalg.solve(-Mq, b)

    # print('dVdq', jnp.transpose(dVdq))
    dHdq = 0.5*(jnp.array([
        [jnp.transpose(p)@dMinvdq1@p],
        [jnp.transpose(p)@dMinvdq2@p],
    ])) + jnp.array([[dVdq1], [dVdq2]])    # addition now works and gives same shape, however numerical values are incorrect

    # print('dHdq', dHdq)

    dHdp = 0.5*jnp.block([
    [(jnp.transpose(p)@linalg.solve(s.Mq,jnp.array([[1.],[0.]]))) + (jnp.array([1.,0.])@linalg.solve(s.Mq,p))],
    [jnp.transpose(p)@linalg.solve(s.Mq,jnp.array([[0.],[1.]])) + jnp.array([0.,1.])@linalg.solve(s.Mq,p)],
    ]) 

    # print('dHdp', dHdp)
    D = 0.5*jnp.eye(2)
    xdot = jnp.block([
        [jnp.zeros((2,2)), jnp.eye(2)],
        [-jnp.eye(2),      -D ],
    ])@jnp.block([[dHdq],[dHdp]])

    return xdot

def ode_solve(xt, dt, m, s):
    # x_step = jnp.zeros(m,1)
    x_step = xt
    substep = dt/m
    for i in range(m):
        x_step = rk4(x_step,substep,dynamics,s)
        # print('xstep', x_step)
    return x_step, s


## MAIN CODE STARTS HERE

q1 = 0.1
q2 = 0.2
# p0 = jnp.matrix([[0],[0]])
# q0 = jnp.array([[q1],[q2]])
q0 = jnp.array([q1,q2])

p0 = jnp.array([-1.0,0.1])

x0 = jnp.block([[q0,p0]])
x0 = jnp.transpose(x0)
print('x0',x0)

s = self()

# Mq = massMatrix(q0)
# s.Mq = Mq

# # result = massMatrix(q1,q2)
# print(Mq)
# MqPrime(q0)

massMatrixJac = jacfwd(MqPrime)
# dMdq = massMatrixJac(q0)
# unravel(dMdq, s)
V = Vq(q0)
print('V',V)
s.dV = jacfwd(Vq)

# xdot = dynamics(x0, s)

# print('xdot', xdot)

# xt1 = ode_solve(x0,0.01, 7, s)

# print('xt1', xt1)

## SIMULATION/PLOT
(m,n) = x0.shape


dt = 0.01

substeps = 3
T = 0.3

t = jnp.arange(0,T,dt)
l = t.size

# print('t',t)
xHist = jnp.zeros((m,l))
print(xHist)
print(x0)

xHist = xHist.at[:,[0]].set(x0)
xeHist = jnp.zeros((6,l))



# for k in range(l):
#     x = xHist.at[:,[k]].get()
#     q = jnp.array([x.at[(0,0)].get(),
#                     x.at[(1,0)].get()])
#     p = jnp.array([x.at[(2,0)].get(),
#                     x.at[(3,0)].get()])

#     dMdq = massMatrixJac(q)
#     unravel(dMdq, s)
#     # V = Vq(q)
#     xtemp, s = ode_solve(x,dt, substeps, s)
#     # print(xtemp)
#     xHist = xHist.at[:,[k+1]].set(xtemp)
#     xeHist = xeHist.at[:,[k]].set(s.xe)
    

print('xHist',xHist)    
print('xeHist',xeHist)

fig, ax = plt.subplots(4,1)
# ax = fig.subplots()
ax[0].plot(t, xHist.at[0,:].get())

ax[1].plot(t, xHist.at[1,:].get())

ax[2].plot(t, xHist.at[2,:].get())

ax[3].plot(t, xHist.at[3,:].get())
fig.savefig('test.png')


ax = plt.figure().add_subplot(projection = '3d')