
import autograd.numpy as anp
from autograd.numpy import pi, cos, sin
from autograd.numpy.linalg import matrix_power
from autograd import grad

def skew(u):
    ans = anp.block([[0., -u[2], u[1]],
                    [u[2], 0., -u[0]],
                    [-u[1], u[0], 0.]])
    return ans

def hatSE3(x):
    A = skew(x[3:5])
    return A


def rotx(mu):
    A = anp.block([[1., 0., 0., 0.],
                   [0., anp.cos(mu), -anp.sin(mu), 0.],
                   [0., anp.sin(mu), anp.cos(mu), 0.],
                   [0., 0., 0., 1.]])
    return A

def roty(mu):
    A = anp.block([[anp.cos(mu), 0., anp.sin(mu), 0.],
                   [0., 1., 0., 0.],
                   [-anp.sin(mu), 0., anp.cos(mu), 0.],
                   [0., 0., 0., 1.]])
    return A

def rotz(mu):
    A = anp.block([[anp.cos(mu), -anp.sin(mu), 0., 0.],
                   [anp.sin(mu), anp.cos(mu), 0., 0.],
                   [0., 0., 1., 0.],
                   [0., 0., 0., 1.]])
    return A

def tranx(mu):
    A = anp.block([[1., 0., 0., mu],
                   [0., 1., 0., 0.],
                   [0., 0., 1., 0.],
                   [0., 0., 0., 1.]])
    return A

def trany(mu):
    A = anp.block([[1., 0., 0., 0.],
                   [0., 1., 0., mu],
                   [0., 0., 1., 0.],
                   [0., 0., 0., 1.]])
    return A

def tranz(mu):
    A = anp.block([[1., 0., 0., 0.],
                   [0., 1., 0., 0.],
                   [0., 0., 1., mu],
                   [0., 0., 0., 1.]])
    return A

def hatSE3(x):
    S = anp.block([
        skew(x[3:6]), x[0:3], anp.zeros((4,1))
    ])
    return S