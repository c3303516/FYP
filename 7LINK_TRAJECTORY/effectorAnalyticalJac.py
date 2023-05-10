import jax.numpy as jnp
from jax.numpy import pi, sin, cos, linalg
import jax
from functools import partial
from params import *
from homogeneousTransforms import *
from scipy.linalg import solve_continuous_lyapunov
from jax.scipy.linalg import sqrtm
import csv


def effectorAnalyticalJac(q):
    #EFFECTOR IKM
    q2 = q.at[(0,0)].get()      #make the full q vector
    q4 = q.at[(1,0)].get()
    q6 = q.at[(2,0)].get()

    q0 = jnp.array([     #this is qhat. q0 denotes before momentum transform
        q2, q4, q6
        ])

    q0 = jnp.transpose(q0)
    qconstants = s.constants
    q0_bold = qconstants.at[0].get()     #constrained variabless
    
    q1 = q0_bold.at[0].get()
    q3 = q0_bold.at[1].get()
    q5 = q0_bold.at[2].get()
    q7 = q0_bold.at[3].get()

    A01 = tranz(s.l1)@rotx(pi)@rotz(q1)          
    A12 = rotx(pi/2)@tranz(-s.d1)@trany(-s.l2)@rotz(q2)
    A23 = rotx(-pi/2)@trany(s.d2)@tranz(-s.l3)@rotz(q3)
    A34 = rotx(pi/2)@tranz(-s.d2)@trany(-s.l4)@rotz(q4)
    A45 = rotx(-pi/2)@trany(s.d2)@tranz(-s.l5)@rotz(q5)
    A56 = rotx(pi/2)@trany(-s.l6)@rotz(q6)
    A67 = rotx(-pi/2)@tranz(-s.l7)@rotz(q7)
    A7E = rotx(pi)@tranz(s.l8)    

    A02 = A01@A12
    A03 = A02@A23
    A04 = A03@A34
    A05 = A04@A45
    A06 = A05@A56
    A07 = A06@A67
    A0E = A07@A7E
    #end effector pose
    r0E0 = A0E[0:3,[3]]


    # Derivative of matrix exponential of one-parameter subgroup
    dA01dq1 = hatSE3(jnp.array([[0],[0],[0],[0],[0],[1]]))@A01     # Derivative of A01 w.r.t. q1
    dA12dq2 = hatSE3(jnp.array([[0],[0],[0],[0],[0],[1]]))@A12     # Derivative of A12 w.r.t. q2
    dA23dq3 = hatSE3(jnp.array([[0],[0],[0],[0],[0],[1]]))@A23     # Derivative of A23 w.r.t. q3
    dA34dq4 = hatSE3(jnp.array([[0],[0],[0],[0],[0],[1]]))@A34     # Derivative of A34 w.r.t. q4
    dA45dq5 = hatSE3(jnp.array([[0],[0],[0],[0],[0],[1]]))@A23     # Derivative of A23 w.r.t. q3
    dA56dq6 = hatSE3(jnp.array([[0],[0],[0],[0],[0],[1]]))@A34     # Derivative of A34 w.r.t. q4
    dA67dq7 = hatSE3(jnp.array([[0],[0],[0],[0],[0],[1]]))@A34     # Derivative of A34 w.r.t. q4

   # Use product rule for these
    dA07dq1 = dA01dq1@A12@A23@A34@A45@A56@A67
    dA07dq2 = A01@dA12dq2@A23@A34@A45@A56@A67
    dA07dq3 = A01@A12@dA23dq3@A34@A45@A56@A67
    dA07dq4 = A01@A12@A23@dA34dq4@A45@A56@A67
    dA07dq5 = A01@A12@A23@A34@dA45dq5@A56@A67
    dA07dq6 = A01@A12@A23@A34@A45@dA56dq6@A67
    dA07dq7 = A01@A12@A23@A34@A45@A56@dA67dq7

    # dr700dq = [dA07dq1(1:3,4),dA07dq2(1:3,4),dA07dq3(1:3,4),dA07dq4(1:3,4),dA07dq5(1:3,4),dA07dq6(1:3,4),dA07dq7(1:3,4)];
    dr700dqhat = jnp.block([dA07dq2[0:3,[3]],dA07dq4[0:3,[3]],dA07dq6[0:3,[3]]]);
    
    # dpsidq  = [A04(2,1)*dA04dq1(1,1)-A04(1,1)*dA04dq1(2,1);
    #         A04(2,1)*dA04dq2(1,1)-A04(1,1)*dA04dq2(2,1);
    #         A04(2,1)*dA04dq3(1,1)-A04(1,1)*dA04dq3(2,1);
    #         A04(2,1)*dA04dq4(1,1)-A04(1,1)*dA04dq4(2,1)];

    # Analytical Jacobian
    # JA = [dr400dq; dpsidq.'];
    JA = dr700dqhat

    # # Update structure
    # s.dA01dq1 = dA01dq1;
    # s.dA12dq2 = dA12dq2;
    # s.dA23dq3 = dA23dq3;
    # s.dA34dq4 = dA34dq4;
    # s.JA = JA;
    return JA