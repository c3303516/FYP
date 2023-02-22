# from params import *
# from homogeneousTransforms import *
import jax.numpy as jnp
from jax.numpy import pi, sin, cos
from jax import grad, jacobian, jacfwd
# from autograd.numpy.linalg import matrix_power


# autograd seems not to work with the block feature. This stems from the face that Jc's need to be made from smaller arrays
# the block feature makes this non-differentiable

# class Params:
    
#     def __init__(self):
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

def fun_sin(x):
    A = jnp.sin(x)
    return A

def fun_cos(x):
    A = jnp.cos(x)
    return A


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
                   [0., fun_cos(mu), -fun_sin(mu), 0.],
                   [0., fun_sin(mu), fun_cos(mu), 0.],
                   [0., 0., 0., 1.]])
    # A = jnp.array([[1., 0., 0., 0.],
    #                [0., mu, -mu, 0.],
    #                [0., mu, mu, 0.],
    #                [0., 0., 0., 1.]])
    return A

def roty(mu):
    A = jnp.array([[fun_cos(mu), 0., fun_sin(mu), 0.],
                   [0., 1., 0., 0.],
                   [-fun_sin(mu), 0., fun_cos(mu), 0.],
                   [0., 0., 0., 1.]])
    # A = jnp.array([[mu, 0., mu, 0.],
    #                [0., 1., 0., 0.],
    #                [mu, 0., mu, 0.],
    #                [0., 0., 0., 1.]])
    return A

def rotz(mu):
    A = jnp.array([[fun_cos(mu), -fun_sin(mu), 0., 0.],
                   [fun_sin(mu), fun_cos(mu), 0., 0.],
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

def massMatrix(q0):
    q1 = q0.at[0].get()
    q2 = q0.at[1].get()

    # q = jnp.array([[q1],0.0])
    # q = jnp.matrix(q_arr).T
    # A01 = rotx(-pi/2)@trany(-l1)@rotz(q[0])
    # A12 = trany(-l2)@rotz(q[1])
    # A23 = trany(-l3)@rotx(pi/2)
    

    A01 = rotx(-pi/2)@trany(-l1)@rotz(q1)
    # A01 = rotz(q1)
    A12 = trany(-l2)@rotz(q2)
    A23 = trany(-l3)@rotx(pi/2)
    # print(A01)
    
    # A01 = jnp.cos(jnp.array([[q1, pi/2, pi/2, pi/2],
    #                [pi/2, q1, pi/2, pi/2],
    #                [pi/2, pi/2, pi/2, pi/2],
    #                [pi/2, pi/2, pi/2, pi/2]])) + jnp.sin(jnp.array([[0, -q1, 0., 0.],
    #                [q1, 0., 0., 0.],
    #                [0., 0., 0., 0.],
    #                [0., 0., 0., 0.]])) + jnp.array([[0., 0., 0., 0.],
    #                [0., 0, 0., 0.],
    #                [0., 0., 1., 0.],
    #                [0., 0., 0., 1.]])

    # print('alt')
    # A12 = jnp.cos(jnp.array([[q2, pi/2, pi/2, pi/2],
    #                [pi/2, q2, pi/2, pi/2],
    #                [pi/2, pi/2, pi/2, pi/2],
    #                [pi/2, pi/2, pi/2, pi/2]])) + jnp.sin(jnp.array([[0, -q2, 0., 0.],
    #                [q2, 0., 0., 0.],
    #                [0., 0., 0., 0.],
    #                [0., 0., 0., 0.]])) + jnp.array([[0., 0., 0., 0.],
    #                [0., 0, 0., 0.],
    #                [0., 0., 1., 0.],
    #                [0., 0., 0., 1.]])
    # A23 = trany(-l3)@jnp.cos(jnp.array([[pi/2, pi/2, pi/2, pi/2],
    #                [pi/2, pi/2, pi/2, pi/2],
    #                [pi/2, pi/2, pi/2, pi/2],
    #                [pi/2, pi/2, pi/2, pi/2]])) + jnp.sin(jnp.array([[0, 0., 0., 0.],
    #                [0., 0., -pi/2, 0.],
    #                [0., pi/2, 0., 0.],
    #                [0., 0., 0., 0.]])) + jnp.array([[1., 0., 0., 0.],
    #                [0., 0, 0., 0.],
    #                [0., 0., 0., 0.],
    #                [0., 0., 0., 1.]])
   

    A02 = A01@A12
    A03 = A02@A23
    # r030 = A03[0:3,[3]]

    # a = jnp.sqrt(A03[2,1]*A03[2,1] + A03[2,2]*A03[2,2])
    # psi  = jnp.arctan2(A03[1,0],A03[0,0])
    # theta = jnp.arctan2(-A03[2,0], a)
    # phi = jnp.arctan2(A03[2,1],A03[2,2])

    # xe = jnp.block([[r030], [phi], [theta], [psi]])
    # # print('xe is', xe)

    # Geometric Jacobians
    R01 = A01[0:3,0:3]     #rotation matrices
    R12 = A12[0:3,0:3]
    # R01 = rotx_small(-pi/2)@rotz_small(q1)            #these are constructed using rotations only.
    # R12 = rotz_small(q2)
    # R23 = rotx_small(pi/2)


    # A0c1 = rotx(-pi/2)@trany(-c1)@rotz(q[0])
    # A0c2 = A01@trany(-c2)@rotz(q[1])
    # A0c3 = A02@trany(-c3)@rotx(pi/2)
    A0c1 = rotx(-pi/2)@trany(-c1)@rotz(q1)
    A0c2 = A01@trany(-c2)@rotz(q2)
    A0c3 = A02@trany(-c3)@rotx(pi/2)

    r100   = A01[0:3,[3]]
    r200   = A02[0:3,[3]]
    # v1 = jnp.array([[1],[0],[0]])
    # v2 = jnp.array([[0],[1],[0]])
    # v3 = jnp.array([[0],[0],[1]])
    # v4 = jnp.array([[0],[0],[0],[1]])
    # store = A01@v4
    # r100 = store[0:3]
    # store = A02@v4
    # r200 = store[0:3]

    rc100   = A0c1[0:3,[3]]
    rc200   = A0c2[0:3,[3]]
    rc300   = A0c3[0:3,[3]]

    # store = A0c1@v4
    # rc100 = store[0:3]
    # store = A0c2@v4
    # rc200 = store[0:3]
    # store = A0c3@v4
    # rc300 = store[0:3]

    z00 = jnp.array([[0.], [0.], [1.]])
    z01 = R01@z00
    z02 = R01@R12@z00

    ske1 = skew(z01)
    ske2 = skew(z02)
    # check = ske1@(rc200-r100)
    # print('size is', check.size)
    # print(check)

    Jc2   = jnp.block([
        [ske1@(rc200-r100), jnp.zeros((3,1))],
        [z01,               jnp.zeros((3,1))]
        ])
    Jc3   = jnp.block([
        [ske1@(rc300-r100), ske2@(rc300-r200)],
        [z01,               z02]
        ])

    # Jc2 = jnp.asmatrix(Jc2)
    # Jc3 = jnp.asmatrix(Jc3)
    # print(Jc2)

    # Mass Matrix
    R02 = A02[0:3,0:3]
    R03 = A03[0:3,0:3]
    # R02 = R01@R12
    # R03 = R02@R23
    M2 = Jc2.T@jnp.block([
        [jnp.multiply(m2,jnp.eye(3,3)), jnp.zeros((3,3))],
        [jnp.zeros((3,3)),            R02.T@I2@R02 ]
    ])@Jc2
    M3 = Jc3.T@jnp.block([
        [jnp.multiply(m3,jnp.eye(3,3)), jnp.zeros((3,3))],
        [jnp.zeros((3,3)),            R03.T@I3@R03 ]
    ])@Jc3


    # print(A01)
    Mq = M2 + M3
    return Mq
    


# a = 2
# b = 3

q1 = 0.1
q2 = 0.1
# p0 = jnp.matrix([[0],[0]])
# q0 = jnp.array([[q1],[q2]])
q0 = jnp.array([q1,q2])



def dMq(q0):
    # q0 = jnp.array([[q1],[q2]])
    Mq = massMatrix(q0)
    # Mq =jnp.ravel(massMatrix(q0), order= 'F')
    # Mq = jnp.reshape(massMatrix(q1,q2),(4,1), order='F')
    Mprime = jnp.zeros([Mq.size])
    Mprime = Mprime.at[0:2].set(Mq.at[0:2,0].get())
    Mprime = Mprime.at[2:4].set(Mq.at[0:2,1].get())
    return Mprime

# result = massMatrix(q1,q2)

result = massMatrix(q0)

# result = massMatrix(q1,q2)
print(result)
dMq(q0)
# dMq(q1,q2)
gradtest = jacfwd(dMq)

result = gradtest(q0)

print('result is',result)

