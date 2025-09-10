def generate_urdf_from_obj(obj_path: str, urdf_path: str, scale=(1, 1, 1), mass=1.0):
    """
    Generate a simple URDF from a given .obj file.

    Args:
        obj_path (str): Path to the .obj mesh file.
        urdf_path (str): Path to save the output URDF file.
        scale (tuple): Scale factor for the mesh (x, y, z).
        mass (float): Mass of the object for physics simulation.
    """
    urdf_template = f"""<?xml version="1.0" ?>
<robot name="obj_robot">
  <link name="base_link">
    <visual>
      <geometry>
        <mesh filename="{obj_path}" scale="{scale[0]} {scale[1]} {scale[2]}"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <mesh filename="{obj_path}" scale="{scale[0]} {scale[1]} {scale[2]}"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="{mass}"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="0.1" iyy="0.1" izz="0.1" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>
</robot>
"""

    with open(urdf_path, 'w') as f:
        f.write(urdf_template)
    print(f"URDF saved to: {urdf_path}")

generate_urdf_from_obj(
    r"BoulderingVideoByImage\simulationPyBullet\assets\climbing_wall.obj",
    r"BoulderingVideoByImage\simulationPyBullet\assets\climbing_wall.urdf",
    scale=(1, 1, 1),
    mass=2.0
)
