import jax
from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.numpy import pi, sin, cos, linalg
from jax import grad, jacobian, jacfwd
# from effectorFKM import FKM, endEffector
from massMatrix import massMatrix
from dynamics import dynamics_test
from dynamics_transform import dynamics_transform
# from errorIKM import errorIKM #now in main!
from rk4 import rk4
from params import robotParams
from copy import deepcopy
from scipy.linalg import solve_continuous_lyapunov
from jax.scipy.linalg import sqrtm
from scipy.optimize import least_squares
from functools import partial
import sys
import csv

class self:
    def __init___(self):
        return 'initialised'


def skew(u):
    ans = jnp.block([[0., -u[2], u[1]],
                    [u[2], 0., -u[0]],
                    [-u[1], u[0], 0.]])
    return ans

def hatSE3(x):
    A = skew(x[3:5])
    return A


def rotx(mu):
    A = jnp.block([[1., 0., 0., 0.],
                   [0., jnp.cos(mu), -jnp.sin(mu), 0.],
                   [0., jnp.sin(mu), jnp.cos(mu), 0.],
                   [0., 0., 0., 1.]])
    return A

def roty(mu):
    A = jnp.block([[jnp.cos(mu), 0., jnp.sin(mu), 0.],
                   [0., 1., 0., 0.],
                   [-jnp.sin(mu), 0., jnp.cos(mu), 0.],
                   [0., 0., 0., 1.]])
    return A

def rotz(mu):
    A = jnp.block([[jnp.cos(mu), -jnp.sin(mu), 0., 0.],
                   [jnp.sin(mu), jnp.cos(mu), 0., 0.],
                   [0., 0., 1., 0.],
                   [0., 0., 0., 1.]])
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
    q1 = q0.at[0,0].get()
    q2 = q0.at[1,0].get()

    A01 = rotx(-pi/2)@trany(-s.l1)@rotz(q1)
    A12 = trany(-s.l2)@rotz(q2)
    A23 = trany(-s.l3)@rotx(pi/2)
    A02 = A01@A12
    A03 = A02@A23

    # Geometric Jacobians
    R01 = A01[0:3,0:3]     #rotation matrices
    R12 = A12[0:3,0:3]

    A0c1 = rotx(-pi/2)@trany(-s.c1)@rotz(q1)
    A0c2 = A01@trany(-s.c2)@rotz(q2)
    A0c3 = A02@trany(-s.c3)@rotx(pi/2)

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
    I2 = s.I2
    I3 = s.I3

    M2 = Jc2.T@jnp.block([
        [jnp.multiply(s.m2,jnp.eye(3,3)), jnp.zeros((3,3))],
        [jnp.zeros((3,3)),            R02.T@I2@R02 ]
    ])@Jc2
    M3 = Jc3.T@jnp.block([
        [jnp.multiply(s.m3,jnp.eye(3,3)), jnp.zeros((3,3))],
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


def unravel(dMdq_temp, s):
    # could probably generalise this for any array
    (m,n,l) = jnp.shape(dMdq_temp)
    dMdq1 = jnp.zeros((n,n))
    dMdq2 = jnp.zeros((n,n))
    # print('dmdqshpae',jnp.shape(dMdq1))

    for i in range(n):
        dMdq1 = dMdq1.at[0:n,i].set(dMdq_temp.at[n*i:n*(i+1),0,0].get())
        dMdq2 = dMdq2.at[0:n,i].set(dMdq_temp.at[n*i:n*(i+1),1,0].get())
    # print(dMdq2)
    return dMdq1, dMdq2

def Vq(q):
    #Function has to do FKM again to enable autograd to work
    q1 = q.at[0].get()
    q2 = q.at[1].get()

    A01 = rotx(-pi/2)@trany(-s.l1)@rotz(q1)
    A12 = trany(-s.l2)@rotz(q2)

    A02 = A01@A12
    c1 = s.c1
    c2 = s.c2
    c3 = s.c3

    A0c1 = rotx(-pi/2)@trany(-c1)@rotz(q1)
    A0c2 = A01@trany(-c2)@rotz(q2)
    A0c3 = A02@trany(-c3)@rotx(pi/2)

    rc100   = A0c1[0:3,3]
    rc200   = A0c2[0:3,3]
    rc300   = A0c3[0:3,3]
    g0 = jnp.array([[0.],[0.],[-s.g]])
    gprime = jnp.transpose(g0)
    
    V = -s.m1*gprime@rc100 -s.m2*gprime@rc200 -s.m3*gprime@rc300
    # s.V = V
    return V.at[0].get()


# def dynamics(x, s):
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
    s.gq = gq.at[0].get()
    # print('gq',s.gq)
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
    ])@jnp.block([[dHdq],[dHdp]]) + jnp.block([[jnp.zeros((2,1))],[gq]])

    # +  [zeros(2,1);(u)]   #need to implement this grav torque control action

    return xdot

# def ode_solve(dt,substep_no,xt,xpt,p_h,v,Cqph,D,Tq_val,dTqinv_block,dVdq,phi):  #LEGACY FUNCTION - NO LONGER CALLAED
    # x_step = jnp.zeros(m,1)
    x_nextstep = xt
    xp_nextstep = xpt
    substep = dt/substep_no
    args = (v,D,Tq_val,dTqinv_block,dVdq)
    obs_args = (p_h,phi,v,Cqph,D,dVdq,Tq_val)

    for i in range(substep_no):
        x_step= rk4(x_nextstep,dynamics_transform,substep,*args)
        x_nextstep = x_step

        xp_step = rk4(xp_nextstep,observer_dynamics,substep,*obs_args)              # PASS PHAT INTO OBSERVER - PUT THIS IN 7DOF
        # xp_step = jnp.zeros((3,1))       #just put this here to test controller works
        xp_nextstep = xp_step


    x_finalstep = x_nextstep
    xp_finalstep = xp_nextstep
    return x_finalstep,xp_finalstep

################################# OBSERVER #######################################

def C_SYS(p_sym,p_sym2,Tq,dTqinvdq_val):
    #This function is specifically used in the creation of \bar{C}
    dTqinvdq1 = dTqinvdq_val.at[0].get()
    dTqinvdq2 = dTqinvdq_val.at[1].get()
    # print('dTqinvdq1',dTqinvdq1)
    dTqinv_phatdq1 = dTqinvdq1@p_sym
    dTqinv_phatdq2 = dTqinvdq2@p_sym
    # print('dTqinv',jnp.shape(dTqinv_phatdq1))
    temphat = jnp.block([dTqinv_phatdq1, dTqinv_phatdq2])
    temphatT = jnp.transpose(temphat)
    # print('temp',temp)
    Ctemp = temphatT - temphat
    # print('shape Ctemp', jnp.shape(Ctemp))
    # print('Ctemp',Ctemp)
    Cq_phat = Tq@Ctemp@Tq
    # print('shape Cqphat', jnp.shape(Cq_phat))

    # print(jnp.shape(Cq_phat@p_sym2))
    return Cq_phat@p_sym2

# @jax.jit
def Cqp(p,Tq,dTqinvdq_val):
    #Calculation of the Coriolis Damping matrix (Check if this is correct name)
    dTqinvdq1 = dTqinvdq_val.at[0].get()
    dTqinvdq2 = dTqinvdq_val.at[1].get()
    # print('dTidq1', dTqinvdq1)
    # print('p',p)
    dTqinv_phatdq1 = dTqinvdq1@p
    dTqinv_phatdq2 = dTqinvdq2@p
    # print('dTqinv_phat',dTqinv_phatdq1)
    temphat = jnp.block([dTqinv_phatdq1, dTqinv_phatdq2])
    temphatT = jnp.transpose(temphat)
    # print('temp',temphat)
    Ctemp = temphatT - temphat
    # print(Ctemp)
    Cq = Tq@Ctemp@Tq

    return Cq

# @jax.jit
def observer_dynamics(xp,phat,phi,u,Cq_phat,Da,dVq,Tqv):            #xp doesn't get used - passed in only for rk4
    Dqa = Tqv@Da@Tqv
    # CbSYM(jnp.array([[0.],[0.],[0.]]),phat,Tq,dTqinvdq_values)
    Gq = Tqv #previous result - confirm this
    u0 = jnp.zeros((2,1))
    # print('dVdq', jnp.shape(dVq))
    # print('Tq',jnp.shape(Tq))
    # print('phiTq', jnp.shape(phi*Tq))
    # print('phat',jnp.shape(phat))
    # print('Cqphat', jnp.shape(Cq_phat))
    # # print('Dq', jnp.shape(Dq))
    # print('phat',phat)
    # # print(Tq,phi*Tq)
    # print(Tqv@dVq)
    # print((Cq_phat - Dqa - phi*Tq)@phat)
    xp_dot = (Cq_phat - Dqa - phi*Tqv)@phat - Tqv@dVq + Gq@(u+u0)

    # print('xpdot',xp_dot)

    return  xp_dot


#rewrite to bring switch out of switch condtition
def switchCond(phat,kappa,phi,Tq,dTqinvdq_values):
    m,n = jnp.shape(Tq)
    # print('phat',phat)
    Cbar_large = CbSYM(jnp.zeros((2,1)),phat,Tq,dTqinvdq_values)  
    
    # Cbar_test = CbSYM(jnp.ones((2,1)),phat,Tq,dTqinvdq_values)  

    # print('diff', Cbar_large - Cbar_test)     #no differene=ce

    # print('CbarL', Cbar_large)
    #process Cbar to the correct size, shape and order. Removes the columns, and transpose reorders columns back to how they should be with jac function
    # print(jnp.transpose(Cbar_large.at[:,0,:,0].get()))
    # print('SHape Cbar', jnp.shape(Cbar))
    # Cbar = jnp.transpose(Cbar_large.at[:,0,:,0].get())
    Cbar = Cbar_large.at[:,0,:,0].get()
    # print('Cbar', Cbar)
    min = jnp.amin(jnp.real(linalg.eigvals(phi*Tq - 0.5*(Cbar + jnp.transpose(Cbar)))))-kappa
    return min

def observerSwitch(q,phi,xp,kappa):
    phinew = phi + kappa
    xpnew = xp - kappa*q
    print('Switch Occurred')

    return phinew,xpnew

######################################## MAIN CODE STARTS HERE #################################
######################################## MAIN CODE STARTS HERE #################################
######################################## MAIN CODE STARTS HERE #################################
######################################## MAIN CODE STARTS HERE #################################
######################################## MAIN CODE STARTS HERE #################################
jnp.set_printoptions(precision=15)
#INITIAL VALUES
q0_1 = 4*pi/6.
q0_2 = 0.


q_0 = jnp.array([[q0_1,q0_2]])
p0 = jnp.array([0.5,0.])

x0 = jnp.block([[q_0,p0]])
x0 = jnp.transpose(x0)
print('Initial States', x0)


q_0 = jnp.transpose(q_0)
s = self()
s = robotParams(s)


massMatrixJac = jacfwd(MqPrime)


# # result = massMatrix(q1,q2)
jnp.set_printoptions(precision=15)
print('Mq', massMatrix_continuous(q_0))
# MqPrime(q0)

massMatrixJac = jacfwd(MqPrime)
dMdq = massMatrixJac(q_0)
unravel(dMdq, s)
V = Vq(q_0)
print('V',V)
dV_func = jacfwd(Vq, argnums = 0)


#compute \barC matrix
CbSYM = jacfwd(C_SYS,argnums=0)


##attempting new way of defining dTq


################################## SIMULATION/PLOT############################################

## SIMULATION/PLOT
(n,hold) = q_0.shape
(m,hold) = x0.shape

print('m,n', m,n)

#Initialise Simulation Parameters
dt = 0.01
substeps = 1
dt_sub = dt/substeps
T = 5.
updatetime = 0.
controlActive = 0     #CONTROL
gravComp = 0.       #1 HAS GRAVITY COMP. Must be a float to maintain precision
# #Define tuning parameters
alpha = 0.
Kp = 1000.*jnp.eye(3)
Kd = 1000.*jnp.eye(3)
ContRate = 100 #Hz: Controller refresh rate

t = jnp.arange(0,T,dt)
l = t.size

#Define Friction
# D = jnp.zeros((n,n))
D = 0.5*jnp.eye(n)          #check this imple


t = jnp.arange(0,T,dt)
l = jnp.size(t)

#Define Storage
xHist = jnp.zeros((m,l+1))
xeHist = jnp.zeros((m,l))
hamHist = jnp.zeros(l)
kinHist = jnp.zeros(l)
potHist = jnp.zeros(l)
H0Hist = jnp.zeros(l)
xpHist = jnp.zeros((n,l))
phiHist = jnp.zeros(l)
phatHist = jnp.zeros((n,l+1))
switchHist = jnp.zeros(l)

# OBSERVER PARAMETER
kappa = 1.     #low value to test switches
phi = kappa #phi(0) = k
phat0 = jnp.array([[0.],[0.]])           #initial momentum estimate
xp0 = phat0 - phi*q_0     #inital xp 
ObsRate = 100.   #Hz: refresh rate of observer
obscounter = 0

dt_obs = 1/ObsRate
print('Obs dt',dt_obs)

Mqh0, Tq0, Tq0inv = massMatrix(q_0,s)   #Get Mq, Tq and Tqinv for function to get dTqdq
dMdq0 = massMatrixJac(q_0)
dMdq10, dMdq20 = unravel(dMdq0, s)
dTq0invdq1 = solve_continuous_lyapunov(Tq0inv,dMdq10)
dTq0invdq2 = solve_continuous_lyapunov(Tq0inv,dMdq20)
dTqinv0 = jnp.array([dTq0invdq1,dTq0invdq2])

while switchCond(phat0,kappa,phi,Tq0,dTqinv0) <= 0:         #Find initial phi
    phitemp, xptmp = observerSwitch(q_0,phi,xp0,kappa)
    phi = phitemp
    xp0 = xptmp
    print('xp0', xp0)
    print(phi)

#Setting Initial Values
xHist = xHist.at[:,[0]].set(x0)
phatHist = phatHist.at[:,[0]].set(phat0)
xpHist = xpHist.at[:,[0]].set(xp0)


for k in range(l):
    time = t.at[k].get()
    print('Time',time)

    x = xHist.at[:,[k]].get()
    q = jnp.array([[x.at[0,0].get()],
                   [x.at[1,0].get()]])
    p = jnp.array([[x.at[2,0].get()],        #This is currently returning p, not p0
                   [x.at[3,0].get()]])
    print('x',x)

    xp = xpHist.at[:,[k]].get()
    
    phat = xp + phi*q           #find phat for this timestep

    
    Mq_hat, Tq, Tqinv = massMatrix(q,s)   #Get Mq, Tq and Tqinv for function to get dTqdq

    dMdq = massMatrixJac(q)
    dMdq1, dMdq2 = unravel(dMdq, s)

    dTqinvdq1 = solve_continuous_lyapunov(Tqinv,dMdq1)
    dTqinvdq2 = solve_continuous_lyapunov(Tqinv,dMdq2)

    dTqinv_block = jnp.array([dTqinvdq1,dTqinvdq2])
    dMdq_block = jnp.array([dMdq1, dMdq2])
    dVdq = dV_func(q)

    # print(dVdq)

    Cqph = Cqp(phat,Tq,dTqinv_block)     #Calculate value of C(q,phat) Matrix.


    Cbar_L_actual = CbSYM(jnp.zeros((2,1)),p,Tq,dTqinv_block)      #for actual cbar
    Cbar_actual = Cbar_L_actual.at[:,0,:,0].get()
    # print('True Cbar', Cbar_actual)

    # print('Difference', Cqph@p - Cbar_actual@phat) #this test indicates its correct. Problem must be in xp


    # result = CbSYM(jnp.zeros((3,1)),phat,Tq,dTqinv_block)

    # print('Cbar', result)
    # print('size Cbar', jnp.shape(result))

    cond = switchCond(phat,kappa,phi,Tq,dTqinv_block)   #check if jump is necessary
    print(cond)
    switchHist = switchHist.at[k].set(cond)
    if cond <= 0:           #This is the correct signage, change 7dof
        phiplus, xpplus = observerSwitch(q,phi,xp,kappa)     #switch to xp+,phi+ values
        phi = phiplus
        xp = xpplus          #update phi and xp with new values

    ptilde = phat - p       #observer error for k timestep
    print('ptilde',ptilde)


    # if controlActive == 1:
    #     p_d = Tqinv@dq_d.at[:,k].get()                      #as p0 = Mq*qdot, and p = Tq*p0
    #     x_d = jnp.block([[q_d.at[:,k].get(), p_d]])
    #     err = jnp.block([[q], [p]]) - jnp.transpose(x_d)     #define error
    #     #Find Control Input for current x, xtilde
    #     v = control(err,Tq,Cqph,dVdq,Kp,Kd,alpha,gravComp)
    # else:
    v = jnp.zeros((2,1))            #ensure this works 
        # print('controlzero')

    #OBSERVER ODE SOLVE 
    timeelapsed = round((time - updatetime),3)      #dealing with this float time issue
    # print('Time Elapsed', timeelapsed)
    if timeelapsed >= dt_obs:    #update observer
        print('Time Elapsed', timeelapsed)
        updatetime = time
        print('Observer Updating')
        obs_args = (phat,phi,v,Cqph,D,dVdq,Tq)      #THIS NOW HAS PHAT IN IT
        xp_update = rk4(xp,observer_dynamics,dt_obs,*obs_args)          #call rk4 solver to update ode
        # xp_step = jnp.zeros((3,1))       #just put this here to test controller works
        xp_k  = xp_update
    else:
        xp_k = xp           #hold xp value from previous iteration


    #SYSTEM ODE SOLVE
    args = (v,D,Tq,dTqinv_block,dVdq)
    x_nextstep = x
    for i in range(substeps):
        x_step= rk4(x_nextstep,dynamics_transform,dt_sub,*args)
        x_nextstep = x_step

    x_k = x_nextstep           #extract final rk values from ODE solve

    if jnp.isnan(x_k.any()):          #check if simiulation messes up
        print(x_k)
        print('NAN found, exiting loop')
        break

    if jnp.isnan(xp_k.any()):
        print(xp_k)
        print('NAN found, exiting loop')
        break


    #Store Variables for next time step
    xHist = xHist.at[:,[k+1]].set(x_k)            #x for next timestep       
    xpHist = xpHist.at[:,[k+1]].set(xp_k)



    #store current timestep variables
    phatHist = phatHist.at[:,[k]].set(phat)
    #Check Observer dynamics
    H0Hist = H0Hist.at[k].set(0.5*(linalg.norm(ptilde.at[:,0].get()))**2)
    print('H obs', 0.5*(jnp.transpose(ptilde.at[:,0].get())@ptilde.at[:,0].get()))
    phiHist = phiHist.at[k].set(phi)

    kinTemp = 0.5*(jnp.transpose(p.at[:,0].get())@p.at[:,0].get())
    potTemp = Vq(q)
    hamTemp = kinTemp + potTemp   

    hamHist = hamHist.at[k].set(hamTemp)
    kinHist = kinHist.at[k].set(kinTemp)        #CHANGE THIS BACK
    potHist = potHist.at[k].set(potTemp)
    

print('xHist',xHist)    
# print('xeHist',xeHist)

#outputting to csv file

details = [ 'dT', dt, 'Substep Number', substeps]
header = ['Time', 'State History']
with open('/root/FYP/3LINK/data/freeswing_observer', 'w', newline='') as f:

    writer = csv.writer(f)
    writer.writerow(details)
    writer.writerow(header)

    # writer.writerow(['Time', t])
    for i in range(l):
        q1 = xHist.at[0,i].get()
        q2 = xHist.at[1,i].get()
        p1 = xHist.at[2,i].get()
        p2 = xHist.at[3,i].get()
        ham = hamHist.at[i].get()
        phat1 = phatHist.at[0,i].get()
        phat2 = phatHist.at[1,i].get()
        Hobs = H0Hist.at[i].get()
        ph = phiHist.at[i].get()
        xp1 = xpHist.at[0,i].get()
        xp2 = xpHist.at[1,i].get()
        sc = switchHist.at[i].get()

        timestamp = t.at[i].get()
        data = ['Time:', timestamp  , 'x:   ', q1,q2,p1,p2,ham, phat1,phat2,ph,Hobs,sc,xp1,xp2]
        # data = ['State',i,':', xHist[k,:]] #xHist.at[k,:].get()]# 'End Effector Pose', xeHist.at[k,:].get()]
        
        writer.writerow(data)

# fig, ax = plt.subplots(4,1)
# # ax = fig.subplots()
# ax[0].plot(t, xHist.at[0,:].get())

# ax[1].plot(t, xHist.at[1,:].get())

# ax[2].plot(t, xHist.at[2,:].get())

# ax[3].plot(t, xHist.at[3,:].get())
# fig.savefig('test.png')


# # ax = plt.figure().add_subplot(projection = '3d')