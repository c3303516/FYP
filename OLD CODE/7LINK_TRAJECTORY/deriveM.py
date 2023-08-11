from cmath import pi, sin
import autograd.numpy as np
from autograd import grad



def M2(q1,q2,m,lc,Izz):
    mq = np.matrix([[m*lc*lc + Izz*((np.cos(q1)*np.cos(q2) - np.sin(q1)*np.sin(q2))**2), 0], [0,0]])
    # p = np.matrix([[p1],[p2]])
    # hold = np.matmul(mq,p)
    # ans = np.matmul(p.transpose(),hold)
    ans = mq
    return ans

def M3(q1,q2,m,lc,Izz):
    mq = np.matrix([[m*lc*lc + Izz*((np.cos(q1)*np.cos(q2) - np.sin(q1)*np.sin(q2))**2), 0]
                    [0,0]])
    # p = np.matrix([[p1],[p2]])
    # hold = np.matmul(mq,p)
    # ans = np.matmul(p.transpose(),hold)
    ans = mq
    return ans

def dMinvdq(q):
    q1 = q(0)
    mq = M2(q1)
              


l2 = 0.8
lc2 = 0.4
l3 = 0.5
lc3 = 0.25
m2 = 1
Izz2 = m2*l2*l2/12
              

print("function stuff ")
print(M2(0,0,m2,lc2,Izz2))

x = np.matrix([1,2,3,4,5])
print(x)

indices = [3,4]
y = np.take(x, indices)

print(y)


if __name__ == '__main__':
    

    ...