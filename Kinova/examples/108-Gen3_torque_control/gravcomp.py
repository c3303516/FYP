import numpy as jnp
from numpy import pi, sin, cos, linalg
from params import *
from homogeneousTransforms import *


# @partial(jax.jit, static_argnames=['s'])
def gq(q0):
    q2 = q0[1]      #pass the joint variable to the specific joints 
    q4 = q0[3]
    q6 = q0[5]
    
    q1 = q0[0]              #pass out to joints. Values of holonomic constraint
    q3 = q0[2]
    q5 = q0[4]
    q7 = q0[6]


    l1 = 156.4e-3
    l2 = 128.4e-3
    l3 = 210.4e-3
    l4 = 210.4e-3
    l5 = 208.4e-3
    l6 = 105.9e-3
    l7 = 105.9e-3
    l8 = 61.5e-3

    d1 = 5.4e-3
    d2 = 6.4e-3



    g = 9.81
    m1 = 1.697
    m2 = 1.377
    m3 = 1.1636
    m4 = 1.1636
    m5 = 0.930
    m6 = 0.678
    m7 = 0.678
    m8 = 0.5

    # I1 = jnp.array([[0.4622e-2, 0.0009e-2,0.006e-2],  #base
    #             [0.0009e-2, 0.4495e-2, 0.0009e-2],
    #             [0.006e-2, 0.0009e-2,0.2079e-2]])

    # I2 = jnp.array([[0.457e-2, 0.0001e-2,0.0002e-2],
    #             [0.0001e-2, 0.4831e-2, 0.0448e-2],
    #             [0.0002e-2, 0.0448e-2,0.1409e-2]])

    # I3 = jnp.array([[1.1088e-2, 0.0005e-2,0],
    #             [0.0005e-2, 0.1072e-2, -0.0691e-2],
    #             [0, -0.0691e-2,1.1255e-2]])

    # I4 = jnp.array([[1.0932e-2, 0,-0.0007e-2],
    #             [0, 1.1127e-2, 0.0606e-2],
    #             [-0.0007e-2, 0.0606e-2,0.1043e-2]])

    # I5 = jnp.array([[0.8147e-2, -0.0001e-2,0,],
    #             [-0.0001e-2, 0.0631e-2, -0.05e-2],
    #             [0, -0.05e-2,0.8316e-2]])

    # I6 = jnp.array([[0.1596e-2, 0, 0],
    #             [0, 0.1607e-2, 0.0256e-2],
    #             [0, 0.0256e-2,0.0399e-2]])

    # I7 = jnp.array([[0.1641e-2, 0,0],
    #             [0, 0.0410e-2, -0.0278e-2],
    #             [0, -0.0278e-2,0.1641e-2]])

    # I8 = jnp.array([[0.0587e-2, 0.0003e-2,0.0003e-2],
    #             [0.0003e-2, 0.0369e-2, 0.0118e-2],
    #             [0.0003e-2, 0.0118e-2,0.0609e-2]])
    

    ## Effector FKM
    # A01old = tranz(l1)@rotx(pi)@rotz(q1)          
    # A12old = rotx(pi/2)@tranz(-d1)@trany(-l2)@rotz(q2)
    # A23old = rotx(-pi/2)@trany(d2)@tranz(-l3)@rotz(q3)
    # A34old = rotx(pi/2)@tranz(-d2)@trany(-l4)@rotz(q4)
    # A45old = rotx(-pi/2)@trany(d2)@tranz(-l5)@rotz(q5)
    # A56old = rotx(pi/2)@trany(-l6)@rotz(q6)
    # A67old = rotx(-pi/2)@tranz(-l7)@rotz(q7)
    # A7Eold = rotx(pi)@tranz(l8)    
    # AEG = tranz(lGripper)


    A01 = jnp.array([[1., 0., 0., 0.],
                     [0.,-1., 0., 0.],
                     [0., 0.,-1., 0.1564],
                     [0., 0., 0., 1.]])@rotz(q1) 
    
    A12 = jnp.array([[1., 0., 0., 0.],
                     [0., 0.,-1., 0.0054],
                     [0., 1., 0.,-0.1284],
                     [0., 0., 0., 1.]])@rotz(q2)
    
    A23 = jnp.array([[1., 0., 0., 0.],
                     [0., 0., 1., -0.2104],
                     [0.,-1., 0., -0.0064],
                     [0., 0., 0., 1.]])@rotz(q3)
    
    A34 = jnp.array([[1., 0., 0., 0.],
                     [0., 0.,-1., 0.0064],
                     [0., 1., 0.,-0.2104],
                     [0., 0., 0., 1.]])@rotz(q4)
    
    A45 = jnp.array([[1., 0., 0., 0.],
                     [0., 0., 1.,-0.2084],
                     [0.,-1., 0.,-0.0064],
                     [0., 0., 0., 1.]])@rotz(q5)
    
    A56 = jnp.array([[1., 0., 0., 0.],
                     [0., 0.,-1., 0.],
                     [0., 1., 0.,-0.1059],
                     [0., 0., 0., 1.]])@rotz(q6)
    
    A67 = jnp.array([[1., 0., 0., 0.],
                     [0., 0., 1.,-0.1059],
                     [0.,-1., 0., 0.],
                     [0., 0., 0., 1.]])@rotz(q7)
    
    A7E = jnp.array([[1., 0., 0., 0.],
                     [0.,-1., 0., 0.],
                     [0., 0.,-1.,-0.0615],
                     [0., 0., 0., 1.]])

    A02 = A01@A12
    A03 = A02@A23
    A04 = A03@A34
    A05 = A04@A45
    A06 = A05@A56
    A07 = A06@A67
    # A0E = A07@A7E
    # A0G = A0E@AEG

    # print(A01-A01old)
    # print(A12-A12old)
    # print(A23-A23old)
    # print(A34-A34old)
    # print(A45-A45old)
    # print(A56-A56old)
    # print(A67-A67old)
    # print(A7E-A7Eold)

    # c1 = jnp.transpose(jnp.array([-6.48e-4, -1.66e-4, 8.4487e-2]))       #frame 0, link 1
    # c2 = jnp.transpose(jnp.array([-2.3e-5, -1.0364e-2, -7.336e-2]))   #frame 1, link 2
    # c3 = jnp.transpose(jnp.array([-4.4e-5, -9.958e-2, -1.3278e-2]))  #frame 2, link 3
    # c4 = jnp.transpose(jnp.array([-4.4e-5, -6.641e-3, -1.17892e-1]))  #frame 3, link 4
    # c5 = jnp.transpose(jnp.array([-1.8e-5, -7.5478e-2, -1.5006e-2]))  #frame 4, link 5
    # c6 = jnp.transpose(jnp.array([1e-6, -9.432e-3, -6.3883e-2]))  #frame 5, link 6
    # c7 = jnp.transpose(jnp.array([1e-6, -4.5483e-2, -9.650e-3]))  #frame 6, link 7
    # c8 = jnp.transpose(jnp.array([-2.81e-4, -1.1402e-2, -2.9798e-2]))  #frame 7, link E
    # cGripper = jnp.transpose(jnp.array([0.,0.,5.8e-2]))
    ## Mass Matrix
    c1x = -6.48e-4
    c1y = -1.66e-4
    c1z =  8.4487e-2
    
    c2x = -2.3e-5
    c2y = -1.0364e-2
    c2z = -7.336e-2
    
    c3x = -4.4e-5
    c3y = -9.958e-2
    c3z = -1.3278e-2
    
    c4x = -4.4e-5
    c4y = -6.641e-3
    c4z = -1.17892e-1
    
    c5x = -1.8e-5
    c5y = -7.5478e-2
    c5z = -1.5006e-2
    
    c6x = 1e-6
    c6y = -9.432e-3
    c6z = -6.3883e-2
    
    c7x = 1e-6
    c7y = -4.5483e-2
    c7z = -9.650e-3
    
    c8x = -2.81e-4
    c8y = -1.1402e-2
    c8z = -2.9798e-2

    # A0c1 = tranx(c1x)@trany(c1y)@tranz(c1z)
    A0c2 = A01@tranx(c2x)@trany(c2y)@tranz(c2z)
    A0c3 = A02@tranx(c3x)@trany(c3y)@tranz(c3z)
    A0c4 = A03@tranx(c4x)@trany(c4y)@tranz(c4z)
    A0c5 = A04@tranx(c5x)@trany(c5y)@tranz(c5z)
    A0c6 = A05@tranx(c6x)@trany(c6y)@tranz(c6z)
    A0c7 = A06@tranx(c7x)@trany(c7y)@tranz(c7z)
    A0c8 = A07@tranx(c8x)@trany(c8y)@tranz(c8z)
    # A0cG = A0E@tranz(cGz)

            # Geometric Jacobians
    R01 = A01[0:3,0:3]     #rotation matrices
    R12 = A12[0:3,0:3]
    R23 = A23[0:3,0:3]
    R34 = A34[0:3,0:3]
    R45 = A45[0:3,0:3]
    R56 = A56[0:3,0:3]
    R67 = A67[0:3,0:3]
    R7E = A7E[0:3,0:3]
    # REG = AEG[0:3,0:3]

    r100   = A01[0:3,[3]]
    r200   = A02[0:3,[3]]
    r300   = A03[0:3,[3]]
    r400   = A04[0:3,[3]]
    r500   = A05[0:3,[3]]
    r600   = A06[0:3,[3]]
    r700   = A07[0:3,[3]]
    # r800   = A0E[0:3,[3]]
    # rG00   = A0G[0:3,[3]]


    # rc100   = A0c1[0:3,[3]]
    rc200   = A0c2[0:3,[3]]
    rc300   = A0c3[0:3,[3]]
    rc400   = A0c4[0:3,[3]]
    rc500   = A0c5[0:3,[3]]
    rc600   = A0c6[0:3,[3]]
    rc700   = A0c7[0:3,[3]]
    rc800   = A0c8[0:3,[3]]
    # rcG00   = A0cG[0:3,[3]]

    z00 = jnp.array([[0.], [0.], [1.]])
    z01 = R01@z00
    z02 = R01@R12@z00
    z03 = R01@R12@R23@z00
    z04 = R01@R12@R23@R34@z00
    z05 = R01@R12@R23@R34@R45@z00
    z06 = R01@R12@R23@R34@R45@R56@z00
    z07 = R01@R12@R23@R34@R45@R56@R67@z00
    z08 = R01@R12@R23@R34@R45@R56@R67@R7E@z00
    # z0G = R01@R12@R23@R34@R45@R56@R67@R7E@REG@z00

    ske1 = skew(z01)
    ske2 = skew(z02)
    ske3 = skew(z03)
    ske4 = skew(z04)
    ske5 = skew(z05)
    ske6 = skew(z06)
    ske7 = skew(z07)
    # ske8 = skew(z08)
    # skeG = skew(z0G)


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

    # # Mass Matrix
    # R02 = A02[0:3,0:3]
    # R03 = A03[0:3,0:3]
    # R04 = A04[0:3,0:3]
    # R05 = A05[0:3,0:3]
    # R06 = A06[0:3,0:3]
    # R07 = A07[0:3,0:3]
    # R08 = A0E[0:3,0:3]
    # R0G = A0G[0:3,0:3]


    # M2 = Jc2.T@jnp.block([
    #     [jnp.multiply(m2,jnp.eye(3,3)), jnp.zeros((3,3))],
    #     [jnp.zeros((3,3)),            R02.T@I2@R02 ]
    # ])@Jc2
    # M3 = Jc3.T@jnp.block([
    #     [jnp.multiply(m3,jnp.eye(3,3)), jnp.zeros((3,3))],
    #     [jnp.zeros((3,3)),            R03.T@I3@R03 ]
    # ])@Jc3
    # M4 = Jc4.T@jnp.block([
    #     [jnp.multiply(m4,jnp.eye(3,3)), jnp.zeros((3,3))],
    #     [jnp.zeros((3,3)),            R04.T@I4@R04 ]
    # ])@Jc4
    # M5 = Jc5.T@jnp.block([
    #     [jnp.multiply(m5,jnp.eye(3,3)), jnp.zeros((3,3))],
    #     [jnp.zeros((3,3)),            R05.T@I5@R05 ]
    # ])@Jc5
    # M6 = Jc6.T@jnp.block([
    #     [jnp.multiply(m6,jnp.eye(3,3)), jnp.zeros((3,3))],
    #     [jnp.zeros((3,3)),            R06.T@I6@R06 ]
    # ])@Jc6
    # M7 = Jc7.T@jnp.block([
    #     [jnp.multiply(m7,jnp.eye(3,3)), jnp.zeros((3,3))],
    #     [jnp.zeros((3,3)),            R07.T@I7@R07 ]
    # ])@Jc7
    # M8 = Jc8.T@jnp.block([
    #     [jnp.multiply(m8,jnp.eye(3,3)), jnp.zeros((3,3))],
    #     [jnp.zeros((3,3)),            R08.T@I8@R08 ]
    # ])@Jc8

    # Mq7 = M2 + M3 + M4 + M5 + M6 + M7 + M8

    #holonomic contraints
    # holonomicTransform = jnp.array([
    #     [0.,0.,0.],
    #     [1.,0.,0.],
    #     [0.,0.,0.],
    #     [0.,1.,0.],
    #     [0.,0.,0.],
    #     [0.,0.,1.],
    #     [0.,0.,0.],
    # ])

    g0 = jnp.array([[0],[0],[-g]])
    tauc2 = jnp.block([[m2*g0],[jnp.zeros((3,1))]])
    tauc3 = jnp.block([[m3*g0],[jnp.zeros((3,1))]])
    tauc4 = jnp.block([[m4*g0],[jnp.zeros((3,1))]])
    tauc5 = jnp.block([[m5*g0],[jnp.zeros((3,1))]])
    tauc6 = jnp.block([[m6*g0],[jnp.zeros((3,1))]])
    tauc7 = jnp.block([[m7*g0],[jnp.zeros((3,1))]])
    tauc8 = jnp.block([[m8*g0],[jnp.zeros((3,1))]])
    gq      = -jnp.transpose(( + jnp.transpose(tauc2)@Jc2 + jnp.transpose(tauc3)@Jc3 + jnp.transpose(tauc4)@Jc4 + jnp.transpose(tauc5)@Jc5 + jnp.transpose(tauc6)@Jc6 + jnp.transpose(tauc7)@Jc7 + jnp.transpose(tauc8)@Jc8))


    # Mq = jnp.transpose(holonomicTransform)@Mq7@holonomicTransform  #transform needed to produce Mq_hat

    return gq

    