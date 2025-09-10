import pybullet as p
from load_humanoid import humanoid

def move_to_target(hold_pos, end_effector_link=15):
    joint_pos = p.calculateInverseKinematics(
        humanoid, end_effector_link, hold_pos,
        maxNumIterations=100,
        residualThreshold=0.001
    )
    p.setJointMotorControlArray(
        humanoid, 
        range(p.getNumJoints(humanoid)), 
        p.POSITION_CONTROL, 
        targetPositions=joint_pos
    )