import jax.numpy as jnp
from jax.numpy import pi, sin, cos, linalg
import jax
from functools import partial
from params import *
from homogeneousTransforms import *
# from main import massMatrix
from massMatrix import massMatrix
from effectorFKM import FKM
from copy import deepcopy
from jax import lax
import csv

@partial(jax.jit, static_argnames=['s'])
def dynamics_test(x,dMdq_block,dVdq, controlAction,s):
    # sprime = deepcopy(s)
    # sprime = s
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
    p = jnp.transpose(p)
    q = jnp.transpose(q)

    ## Effector FKM
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

    ## Mass Matrix
    c1 = s.c1
    c1x = c1.at[0].get()
    c1y = c1.at[1].get()
    c1z = c1.at[2].get()
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

    Mq = M2 + M3 + M4 + M5 + M6 + M7 + M8
    


    # Gravitation torque
    g0 = jnp.array([[0],[0],[-s.g]])

    
    dVdq1 = dVdq.at[0].get()
    dVdq2 = dVdq.at[1].get()
    dVdq3 = dVdq.at[2].get()
    dVdq4 = dVdq.at[3].get()
    dVdq5 = dVdq.at[4].get()
    dVdq6 = dVdq.at[5].get()
    dVdq7 = dVdq.at[6].get()


    # Mass matrix inverse

    dMdq1 = dMdq_block.at[0].get()
    dMdq2 = dMdq_block.at[1].get()
    dMdq3 = dMdq_block.at[2].get()
    dMdq4 = dMdq_block.at[3].get()
    dMdq5 = dMdq_block.at[4].get()
    dMdq6 = dMdq_block.at[5].get()
    dMdq7 = dMdq_block.at[6].get()


    temp1 = jnp.transpose(linalg.solve(jnp.transpose(Mq), jnp.transpose(dMdq1)))
    temp2 = jnp.transpose(linalg.solve(jnp.transpose(Mq), jnp.transpose(dMdq2)))
    temp3 = jnp.transpose(linalg.solve(jnp.transpose(Mq), jnp.transpose(dMdq3)))
    temp4 = jnp.transpose(linalg.solve(jnp.transpose(Mq), jnp.transpose(dMdq4)))
    temp5 = jnp.transpose(linalg.solve(jnp.transpose(Mq), jnp.transpose(dMdq5)))
    temp6 = jnp.transpose(linalg.solve(jnp.transpose(Mq), jnp.transpose(dMdq6)))
    temp7 = jnp.transpose(linalg.solve(jnp.transpose(Mq), jnp.transpose(dMdq7)))
    
    dMinvdq1 = -linalg.solve(Mq, temp1)     #m1 and 2 ae messed up for some reason.
    dMinvdq2 = -linalg.solve(Mq, temp2)
    dMinvdq3 = -linalg.solve(Mq, temp3)
    dMinvdq4 = -linalg.solve(Mq, temp4)
    dMinvdq5 = -linalg.solve(Mq, temp5)
    dMinvdq6 = -linalg.solve(Mq, temp6)
    dMinvdq7 = -linalg.solve(Mq, temp7)


    # jnp.set_printoptions(precision=15)
    # print('Temp1',temp1)
    # print('temp2',temp2)

    # print('dMdq1',dMdq1)
    # print('dMdq2',dMdq2)
    # print('dMdq3',dMdq3)
    # print('dMdq4',dMdq4)
    # print('dMdq5',dMdq5)
    # print('dMdq6',dMdq6)
    # print('dMdq7',dMdq7)
    # print('dMinvdq1',dMinvdq1)
    # print('dMinvdq2',dMinvdq2)
    # print('dMinvdq3',dMinvdq3)
    # print('dMinvdq4',dMinvdq4)
    # print('dMinvdq5',dMinvdq5)
    # print('dMinvdq6',dMinvdq6)
    # print('dMinvdq7',dMinvdq7)

    # with open('/root/FYP/7LINK/M_data', 'w', newline='') as f:

    #     writer = csv.writer(f)

    #     data = [
    #             ['dMdq1:', dMdq1],
    #             ['dMdq2:', dMdq2],
    #             ['dMdq3:', dMdq3],
    #             ['dMdq4:', dMdq4],
    #             ['dMdq5:', dMdq5],
    #             ['dMdq6:', dMdq6],
    #             ['dMdq7:', dMdq7],
    #             ['dMinvdq1',dMinvdq1],
    #             ['dMinvdq2',dMinvdq2],
    #             ['dMinvdq3',dMinvdq3],
    #             ['dMinvdq4',dMinvdq4],
    #             ['dMinvdq5',dMinvdq5],
    #             ['dMinvdq6',dMinvdq6],
    #             ['dMinvdq7',dMinvdq7],
    #     ]
            
    #     writer.writerows(data)



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
    [jnp.transpose(p)@linalg.solve(Mq,jnp.array([[1.],[0.],[0.],[0.],[0.],[0.],[0.]])) + (jnp.array([1.,0.,0.,0.,0.,0.,0.])@linalg.solve(Mq,p))],
    [jnp.transpose(p)@linalg.solve(Mq,jnp.array([[0.],[1.],[0.],[0.],[0.],[0.],[0.]])) + (jnp.array([0.,1.,0.,0.,0.,0.,0.])@linalg.solve(Mq,p))],
    [jnp.transpose(p)@linalg.solve(Mq,jnp.array([[0.],[0.],[1.],[0.],[0.],[0.],[0.]])) + (jnp.array([0.,0.,1.,0.,0.,0.,0.])@linalg.solve(Mq,p))],
    [jnp.transpose(p)@linalg.solve(Mq,jnp.array([[0.],[0.],[0.],[1.],[0.],[0.],[0.]])) + (jnp.array([0.,0.,0.,1.,0.,0.,0.])@linalg.solve(Mq,p))],
    [jnp.transpose(p)@linalg.solve(Mq,jnp.array([[0.],[0.],[0.],[0.],[1.],[0.],[0.]])) + (jnp.array([0.,0.,0.,0.,1.,0.,0.])@linalg.solve(Mq,p))],
    [jnp.transpose(p)@linalg.solve(Mq,jnp.array([[0.],[0.],[0.],[0.],[0.],[1.],[0.]])) + (jnp.array([0.,0.,0.,0.,0.,1.,0.])@linalg.solve(Mq,p))],
    [jnp.transpose(p)@linalg.solve(Mq,jnp.array([[0.],[0.],[0.],[0.],[0.],[0.],[1.]])) + (jnp.array([0.,0.,0.,0.,0.,0.,1.])@linalg.solve(Mq,p))]
    ]) 

    # print('dHdp', dHdp)

    # if s.gravityCompensation == 1:
    # gq = gravTorque(s)
        # print('GRAV COMPE IS ON')
    # else:
    gq = jnp.zeros((7,1))
        # print('shouldnt go here')
        
    # jnp.set_printoptions(precision=15)
    # t = s.time
    # print('t',t)
    # tau = inputTorque(q,gq,s)
    u = controlAction + gq
    tau = jnp.block([[jnp.zeros((7,1))],[u]])
    # D = 0*jnp.eye(7)
    D = jnp.zeros((7,7))
    xdot = jnp.block([
        [jnp.zeros((7,7)), jnp.eye(7)],
        [-jnp.eye(7),      -D ]
    ])@jnp.block([[dHdq],[dHdp]])  + tau     #CHECK YOU ACTUALLY PUT THE GRAVITY IN 

    # print('xdot', xdot)

    return xdot

def gravTorque(s):
    Jc2 = s.Jc2
    Jc3 = s.Jc3
    Jc4 = s.Jc4
    Jc5 = s.Jc5
    Jc6 = s.Jc6
    Jc7 = s.Jc7
    Jc8 = s.Jc8

    # g0 = jnp.array([[0],[0],[-g]])
    g0 = s.g0
    tauc2 = jnp.block([
        [jnp.multiply(s.m2,g0)],
        [jnp.zeros((3,1))]
    ])
    tauc3 = jnp.block([
        [jnp.multiply(s.m3,g0)],
        [jnp.zeros((3,1))]
    ])
    tauc4 = jnp.block([
        [jnp.multiply(s.m4,g0)],
        [jnp.zeros((3,1))]
    ])
    tauc5 = jnp.block([
        [jnp.multiply(s.m5,g0)],
        [jnp.zeros((3,1))]
    ])
    tauc6 = jnp.block([
        [jnp.multiply(s.m6,g0)],
        [jnp.zeros((3,1))]
    ])
    tauc7 = jnp.block([
        [jnp.multiply(s.m7,g0)],
        [jnp.zeros((3,1))]
    ])
    tauc8 = jnp.block([
        [jnp.multiply(s.m8,g0)],
        [jnp.zeros((3,1))]
    ])

    gq = jnp.transpose(-((jnp.transpose(tauc2))@Jc2 + (jnp.transpose(tauc3))@Jc3 + (jnp.transpose(tauc4))@Jc4 + (jnp.transpose(tauc5))@Jc5 + (jnp.transpose(tauc6))@Jc6 + (jnp.transpose(tauc7))@Jc7 + (jnp.transpose(tauc8))@Jc8))
    return gq

def inputTorque(q,gq, s):

    # gq = gravTorque(s)

    # if s.controlActive == 0:
    # controlAction = jnp.zeros((7,1))
 
    # controlAction = jnp.array([[0.],[0.],[0.],[5.],[0.],[0.],[0.]])
        # print('cA',controlAction)

    # operand = jnp.array([0.])
    # controlAction = lax.cond(t<0, lambda x: jnp.array([[0.],[0.],[0.],[5.],[0.],[0.],[0.]]), lambda x: jnp.zeros((7,1)), operand)

    u = s.controlAction + gq
    # print('u',u,'gq',gq)
    # u = gq
    tau = jnp.block([[jnp.zeros((7,1))],[u]])
    return tau
