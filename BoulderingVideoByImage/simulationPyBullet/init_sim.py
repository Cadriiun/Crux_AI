import pybullet as p

def init_sim():
    p.connect(p.GUI)
    p.setGravity(0, 0, -9.8)
    wall_id = p.loadURDF("wall.urdf")
    return wall_id

