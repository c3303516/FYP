from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.numpy import pi, sin, cos, linalg


def robotParams(s):
    s.l1 = 156.4e-3
    s.l2 = 128.4e-3
    s.l3 = 210.4e-3
    s.l4 = 210.4e-3
    s.l5 = 208.4e-3
    s.l6 = 105.9e-3
    s.l7 = 105.9e-3
    s.l8 = 61.5e-3      #this is without gripper
    s.lGripper = 120e-3

    s.d1 = 5.4e-3
    s.d2 = 6.4e-3


    s.c1 = jnp.transpose(jnp.array([-6.48e-4, -1.66e-4, 8.4487e-2]))       #frame 0, link 1
    s.c2 = jnp.transpose(jnp.array([-2.3e-5, -1.0364e-2, -7.336e-2]))   #frame 1, link 2
    s.c3 = jnp.transpose(jnp.array([-4.4e-5, -9.958e-2, -1.3278e-2]))  #frame 2, link 3
    s.c4 = jnp.transpose(jnp.array([-4.4e-5, -6.641e-3, -1.17892e-1]))  #frame 3, link 4
    s.c5 = jnp.transpose(jnp.array([-1.8e-5, -7.5478e-2, -1.5006e-2]))  #frame 4, link 5
    s.c6 = jnp.transpose(jnp.array([1e-6, -9.432e-3, -6.3883e-2]))  #frame 5, link 6
    s.c7 = jnp.transpose(jnp.array([1e-6, -4.5483e-2, -9.650e-3]))  #frame 6, link 7
    s.c8 = jnp.transpose(jnp.array([-2.81e-4, -1.1402e-2, -2.9798e-2]))  #frame 7, link E
    s.cGripper = jnp.transpose(jnp.array([0.,0.,5.8e-2]))

    s.g = 9.81
    s.m1 = 1.697
    s.m2 = 1.377
    s.m3 = 1.1636
    s.m4 = 1.1636
    s.m5 = 0.930
    s.m6 = 0.678
    s.m7 = 0.678
    s.m8 = 0.5
    s.mGripper = 0.925

    #joint limits
    #these limits are joint limits but are NOT the limits on robot pose
    angleupper = (pi/180)*jnp.transpose(jnp.array([360, 128.9, 360, 147.8, 360,120.3,360])) #really hope this wrap around does not cause issues later


    # angleupper = jnp.array([pipipi]
    # torquelim = jnp.array([10101010101010] #placeholder value

    # s.qupper = jnp.array([torquelim  angleupper]
    s.qupper = angleupper
    s.qlower = -s.qupper


    s.I1 = jnp.array([[0.4622e-2, 0.0009e-2,0.006e-2],  #base
                [0.0009e-2, 0.4495e-2, 0.0009e-2],
                [0.006e-2, 0.0009e-2,0.2079e-2]])

    s.I2 = jnp.array([[0.457e-2, 0.0001e-2,0.0002e-2],
                [0.0001e-2, 0.4831e-2, 0.0448e-2],
                [0.0002e-2, 0.0448e-2,0.1409e-2]])

    s.I3 = jnp.array([[1.1088e-2, 0.0005e-2,0],
                [0.0005e-2, 0.1072e-2, -0.0691e-2],
                [0, -0.0691e-2,1.1255e-2]])

    s.I4 = jnp.array([[1.0932e-2, 0,-0.0007e-2],
                [0, 1.1127e-2, 0.0606e-2],
                [-0.0007e-2, 0.0606e-2,0.1043e-2]])

    s.I5 = jnp.array([[0.8147e-2, -0.0001e-2,0,],
                [-0.0001e-2, 0.0631e-2, -0.05e-2],
                [0, -0.05e-2,0.8316e-2]])

    s.I6 = jnp.array([[0.1596e-2, 0, 0],
                [0, 0.1607e-2, 0.0256e-2],
                [0, 0.0256e-2,0.0399e-2]])

    s.I7 = jnp.array([[0.1641e-2, 0,0],
                [0, 0.0410e-2, -0.0278e-2],
                [0, -0.0278e-2,0.0399e-2]])

    s.I8 = jnp.array([[0.0587e-2, 0.0003e-2,0.0003e-2],
                [0.0003e-2, 0.0369e-2, 0.0118e-2],
                [0.0003e-2, 0.0118e-2,0.0609e-2]])
    
    #this is placeholder with ml^2/12 on Ixx and Iyy
    s.IG = jnp.array([[s.mGripper*s.lGripper**2/12, 0, 0],
            [0, s.mGripper*s.lGripper**2/12, 0],
            [0, 0, 0]])

    return s