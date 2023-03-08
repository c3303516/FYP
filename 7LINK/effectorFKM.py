import jax.numpy as jnp
from jax.numpy import pi, sin, cos, linalg
from params import *

from params import *
from homogeneousTransforms import *


def FKM(q,s):
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

    s.A01 = A01
    s.A02 = A02
    s.A03 = A03
    s.A04 = A04
    s.A05 = A05
    s.A06 = A06
    s.A07 = A07
    s.A0E = A0E
    s.A0G = A0G

    s.A01 = A01
    s.A12 = A12
    s.A23 = A23
    s.A34 = A34
    s.A45 = A45
    s.A56 = A56
    s.A67 = A67
    s.A7E = A7E
    s.AEG = AEG
    #end effector pose
    r0E0 = A0E[0:3,[3]]

    a = jnp.sqrt(A0E[2,1]*A0E[2,1] + A0E[2,2]*A0E[2,2])
    psi  = jnp.arctan2(A0E[1,0],A0E[0,0])
    theta = jnp.arctan2(-A0E[2,0], a)
    phi = jnp.arctan2(A0E[2,1],A0E[2,2])

    # s.xe = jnp.block([[r0E0], [phi], [theta], [psi]])       # Shouldn't output this here - compute from x data afterwards
    return s



def endEffector(q,s):
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
    A7E = rotx(pi)@tranz(s.l8)    
    # AEG = tranz(s.lGripper)

    A02 = A01@A12
    A03 = A02@A23
    A04 = A03@A34
    A05 = A04@A45
    A06 = A05@A56
    A07 = A06@A67
    A0E = A07@A7E
    # A0G = A0E@AEG

    r0E0 = A0E[0:3,[3]]
    a = jnp.sqrt(A0E[2,1]*A0E[2,1] + A0E[2,2]*A0E[2,2])
    psi  = jnp.arctan2(A0E[1,0],A0E[0,0])
    theta = jnp.arctan2(-A0E[2,0], a)
    phi = jnp.arctan2(A0E[2,1],A0E[2,2])

    xe = jnp.block([[r0E0], [phi], [theta], [psi]]) 
    return xe