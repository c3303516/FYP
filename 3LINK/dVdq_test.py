from params import *
from homogeneousTransforms import *
from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.numpy import pi, sin, cos, linalg
from jax import grad, jacobian, jacfwd
from effectorFKM import FKM
from massMatrix import massMatrix



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
    return V.at[0].get()


def derivePotEng(q):
    q1 = q.at[0].get()
    q2 = q.at[1].get()

    dVdq1 = - g*m3*(c3*(cos(q1)*sin(q2) + cos(q2)*sin(q1)) + l2*sin(q1)) - c2*g*m2*sin(q1)

    dVdq2 = -c3*g*m3*(cos(q1)*sin(q2) + cos(q2)*sin(q1))

    result = jnp.array([dVdq1, dVdq2])

    return result


q_vector = jnp.arange(-2,2,0.1)

m = jnp.size(q_vector)
# print(m)
for i in range(m):
    q1 = q_vector.at[i].get()
    q2 = 0.4
    # p0 = jnp.array([0,0])
    # q0 = jnp.array([[q1],[q2]])
    q0 = jnp.array([q1,q2])

    p0 = jnp.array([0.,0.])

    x0 = jnp.block([[q0,p0]])
    x0 = jnp.transpose(x0)

    jnp.set_printoptions(precision=15)

    V = Vq(q0)
    # print('V',V)
    dV = jacfwd(Vq)
    dVMatlab = derivePotEng(q0)
    print('dV', dV(q0), 'dV Matlab', dVMatlab)

    difference = dV(q0) - dVMatlab
    print('Difference', difference)


