import jax.numpy as jnp
from jax.numpy import pi, sin, cos, linalg
import jax
from copy import deepcopy
from functools import partial
# from main import dynamics

# @partial(jax.jit, static_argnames=['s'])
def rk4(xt,func,dt,Mq,Tq,dMdq_values,dTqdq_values,dVdq,gC,x_tilde,Kp,Kd,alpha):
# def rk4(xt,func,dt,*arguments):
    t = 1
    #check what arguments are being sent. Massive purg incoming
    k1 = func(xt,Tq,dTqdq_values,dVdq,gC,x_tilde,Kp,Kd,alpha)
    # print('A071', s.A07)
    k2 = func(xt + (k1 * dt) / 2,Tq,dTqdq_values,dVdq,gC,x_tilde,Kp,Kd,alpha)
    # print('A072', s.A07)
    k3 = func(xt + (k2 * dt) / 2,Tq,dTqdq_values,dVdq,gC,x_tilde,Kp,Kd,alpha)
    # print('A073', s.A07)
    k4 = func(xt + (k3 * dt),Tq,dTqdq_values,dVdq,gC,x_tilde,Kp,Kd,alpha)
    # print(x_tilde)


    # #old. check what arguments are being sent. Massive purg happend
    # k1 = func(xt,Mq,Tq,dMdq_values,dTqdq_values,dVdq,gC,x_tilde, s)
    # # print('A071', s.A07)
    # k2 = func(xt + (k1 * dt) / 2,Mq,Tq,dMdq_values,dTqdq_values,dVdq,gC,x_tilde, s)
    # # print('A072', s.A07)
    # k3 = func(xt + (k2 * dt) / 2,Mq,Tq,dMdq_values,dTqdq_values,dVdq,gC,x_tilde, s)
    # # print('A073', s.A07)
    # k4 = func(xt + (k3 * dt),Mq,Tq,dMdq_values,dTqdq_values,dVdq,gC,x_tilde, s)
    # # print(x_tilde)

    # k1 = func(xt,arguments)
    # # print('A071', s.A07)
    # k2 = func(xt + (k1 * dt) / 2,arguments)
    # # print('A072', s.A07)
    # k3 = func(xt + (k2 * dt) / 2,arguments)
    # # print('A073', s.A07)
    # k4 = func(xt + (k3 * dt),arguments)

    return xt + (k1/6 + k2/3 + k3/3 + k4/6) * dt