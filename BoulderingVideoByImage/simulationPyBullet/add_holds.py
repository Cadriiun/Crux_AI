import pybullet as p

def add_holds(holds_3d):
    hold_ids = []
    for x, y, z in holds_3d:
        col_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=0.05)
        hold_ids.append(p.createMultiBody(0.1, col_shape, basePosition=[x, y, z]))
    return hold_ids

