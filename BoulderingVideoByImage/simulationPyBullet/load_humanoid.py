import pybullet as p

'''
    loads humanoid URDF into pybullet
'''


def humanoid():
    p.connect(p.GUI)
    humanoid = p.loadURDF(
        "BoulderingVideoByImage\simulationPyBullet\humanoid.urdf", 
        basePosition=[0, 0, 1],
        baseOrientation=p.getQuaternionFromEuler([0,0,0]),
        useFixedBase=False,
        flags=p.URDF_MAINTAIN_LINK_ORDER | p.URDF_USE_SELF_COLLISION
    )
    return humanoid