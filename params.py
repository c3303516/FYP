import jax.numpy as np

# class Params:
    
#     def __init__(self):
g = 9.81
m1 = 1.5
m2 = 1
m3 = 0.8

l1 = 1.5
l2 = 0.8
l3 = 0.5
c1 = 0.6
c2 = 0.4
c3 = 0.25

Iz1 = m1*l1*l1/12
Iz2 = m2*l2*l2/12
Iz3 = m3*l3*l3/12

I1 = np.array([[0., 0., 0.],
          [0., 0., 0.],
          [0., 0., Iz1]])
I2 = np.array([[0., 0., 0.],
          [0., 0., 0.],
          [0., 0., Iz2]])
I3 = np.array([[0., 0., 0.],
                [0., Iz3, 0.],
                [0., 0., 0.]])

# a = 2
# b = 3