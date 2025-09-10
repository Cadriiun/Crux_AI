import numpy as np
from simulationPyBullet.load_humanoid import humanoid

class RewardCalculator:
    def __init__(self, hold_positions, wall_height = 3.0):
        self.hold_positions = hold_positions
        self.current_hold = 0
        self.max_holds = len(hold_positions)
        self.wall_height = wall_height
    
    def calculate(self, humanoid_pos, effector_positions, joint_angles):
        # 1. Progress reward
        target_pos = self.hold_positions[min(self.current_hold + 1, self.max_holds-1)]
        distance = np.linalg.norm(humanoid_pos - target_pos)
        progress_reward = -distance * 0.2
        height_reward = humanoid_pos[2] / self.wall_height * 0.5
        # 2. Grip reward
        grip_reward = 0
        if self._check_grip(effector_positions, target_pos):
            grip_reward = 15.0 * (1 + self.current_hold * 0.3)
            self.current_hold += 1
        
        # 3. Stability penalty
        joint_penalty = np.sum(np.abs(joint_angles) * 0.01)
        
        return progress_reward + grip_reward - joint_penalty + height_reward
    
    def _check_grip(self, effector_pos, target_pos, threshold=0.15):
        return any(np.linalg.norm(pos - target_pos) < threshold 
                  for pos in effector_pos.values())
    
    