This repository holds all the code I used during my FYP.

Each folder holds all the files for a different type of simulation or stage in the project. 

7 Link details simulations with a dynamic model where all Kinova joints are free.

3 Link details simulations with a dynamic model were joints 1,3,5 and 7 are constrained to zero. This has the affect of turning the robot arm into a 3 DOF model.

'Transformed' means the simulation is being calculated in the new momentum coordinates such that the kinetic energy is independent of 'q'.

'Implementation' means that both the energy based controller and observer are being simulated.

Code to actually physically control the arm through python scripts (based on the examples found in Kinova's resources) are within the Kinova/examples/108-Gen3_torque_control.

