import gym
import numpy as np
import pybullet as p
import pybullet_data
import cv2
import torch
import os
from typing import Dict, List, Tuple, Optional, Union
from collections import OrderedDict, deque
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

# ====================== COMBINED POSE PROCESSOR ======================

class PoseProcessor:
    """Handles pose estimation using YOLOv8 pose model"""
    def __init__(self, model_path):
        try:
            # First try loading from local path
            if os.path.exists(model_path):
                from ultralytics import YOLO
                self.model = YOLO(model_path)
            else:
                # Fallback to downloading the model
                from ultralytics import YOLO
                self.model = YOLO('yolo11n.pt') 
        except Exception as e:
            print(f"Failed to load pose model: {e}")
            self.model = None

    def __call__(self, image):
        if self.model is None:
            return []
        return self.model(image)

# ====================== COMBINED REWARD CALCULATOR ======================

class RewardCalculator:
    def __init__(self, hold_positions):
        self.hold_positions = hold_positions
        self.current_hold = 0
        self.max_holds = len(hold_positions)
    
    def calculate(self, humanoid_pos, effector_positions, joint_angles):
        # 1. Progress reward
        target_pos = self.hold_positions[min(self.current_hold + 1, self.max_holds-1)]
        distance = np.linalg.norm(humanoid_pos - target_pos)
        progress_reward = -distance * 0.2
        
        # 2. Grip reward
        grip_reward = 0
        if self._check_grip(effector_positions, target_pos):
            grip_reward = 15.0 * (1 + self.current_hold * 0.3)
            self.current_hold += 1
        
        # 3. Stability penalty
        joint_penalty = np.sum(np.abs(joint_angles) * 0.01)
        
        return {
            'total': progress_reward + grip_reward - joint_penalty,
            'reached_top': self.current_hold >= self.max_holds - 1,
            'effector_hold_contact': grip_reward > 0
        }
    
    def _check_grip(self, effector_pos, target_pos, threshold=0.15):
        return any(np.linalg.norm(pos - target_pos) < threshold 
                  for pos in effector_pos.values())

# ====================== COMBINED VIDEO DATASET ======================

class VideoDataset:
    """Processes climbing videos into demonstration trajectories."""
    def __init__(self, video_dir: str, pose_model: str, max_frames: int = 1000):
        self.video_dir = video_dir
        self.pose_processor = PoseProcessor(model_path=pose_model)
        self.max_frames = max_frames
        self.trajectories = []
    
    def process_videos(self):
        video_files = [f for f in os.listdir(self.video_dir) 
                      if f.endswith(('.mp4', '.avi', '.mov'))]
        
        for video_file in video_files:
            self._process_video(os.path.join(self.video_dir, video_file))
            
    def _process_video(self, video_path: str):
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        observations = []
        actions = []
        
        while cap.isOpened() and frame_count < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            results = self.pose_processor.model(frame)
            
            if len(results[0].keypoints) > 0:
                obs = self._pose_to_obs(results[0].keypoints.xy[0].cpu().numpy())
                observations.append(obs)
                
                if len(observations) >= 2:
                    action = self._estimate_action(observations[-2], observations[-1])
                    actions.append(action)
                    
            frame_count += 1
            
        cap.release()
        
        if len(observations) > 1:
            observations = observations[:len(actions)+1]
            traj = {
                'obs': np.array(observations[:-1]),
                'acts': np.array(actions),
                'infos': None,
                'terminal': True
            }
            self.trajectories.append(traj)
                
    def _pose_to_obs(self, keypoints: np.ndarray) -> np.ndarray:
        normalized_kps = (keypoints - keypoints.mean(axis=0)) / (keypoints.std(axis=0) + 1e-8)
        return normalized_kps.flatten()
    
    def _estimate_action(self, prev_obs: np.ndarray, curr_obs: np.ndarray) -> np.ndarray:
        return curr_obs - prev_obs

# ====================== MAIN BOULDERING ENVIRONMENT ======================

class BoulderingEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, render=False, camera_view=True, use_imitation=True,
                 wall_urdf="simulationPyBullet\assets\climbing_wall.urdf", 
                 humanoid_urdf="simulationPyBullet\assets\humanoid.urdf",
                 pose_model="yolo11n-pose.pt",
                 video_dir="videos"):
        
        super(BoulderingEnv, self).__init__()
        self._init_spaces()
        
        # Physics setup
        self.physicsClient = p.connect(p.GUI if render else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setTimeStep(1/240)
        p.setGravity(0, 0, -9.8)
        
        # Configuration
        self.camera_view = camera_view
        self.wall_urdf = wall_urdf
        self.humanoid_urdf = humanoid_urdf
        self.use_imitation = use_imitation
        
        # Initialize components
        self.pose_processor = PoseProcessor(model_path=pose_model)
        self.holds = []
        self.hold_positions = []
        self.joint_info = OrderedDict()
        self.effector_links = {
            'left_hand': -1, 'right_hand': -1, 
            'left_foot': -1, 'right_foot': -1
        }
        
        # Training stats
        self.episode_count = 0
        self.success_count = 0
        self.episode_rewards = deque(maxlen=100)
        self.current_reward = 0
        
        # Initialize world
        self._setup_world()

    def _init_spaces(self):
        """Initialize Gym spaces with more realistic dimensions."""
        # Action space: Normalized joint targets [-1, 1]
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(24,), dtype=np.float32)
        
        # Observation space (153-D):
        # [base_pos(3), base_orn(4), joint_angles(24), 
        #  effector_pos(4*3), hold_dists(8*3), keypoints(17*3)]
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(153,), dtype=np.float32)

    def _setup_world(self):
        """Initialize simulation world with enhanced features."""
        p.resetSimulation()
        
        # Load climbing wall with texture
        self.wall = p.loadURDF(self.wall_urdf, basePosition=[0, 0, 0])
        
        # Generate holds with realistic distribution
        self._generate_holds(num_holds=8, wall_height=3.0)
        
        # Load humanoid with better physical properties
        self.humanoid = p.loadURDF(
            self.humanoid_urdf,
            basePosition=[0, 0, 1.0],  # Start slightly lower
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=False,
            flags=p.URDF_USE_SELF_COLLISION | 
                  p.URDF_MAINTAIN_LINK_ORDER |
                  p.URDF_USE_INERTIA_FROM_FILE
        )
        
        # Cache joint information and find effectors
        self._cache_joint_info()
        self._identify_effector_links()
        
        # Setup reward calculator with hold information
        self.reward_calculator = RewardCalculator(
            hold_positions=self.hold_positions,
        )
        
        # Setup camera if enabled
        if self.camera_view:
            self._setup_camera(width=640, height=480)
            
        # Setup TensorBoard logging
        log_dir = f"logs/bouldering_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.writer = SummaryWriter(log_dir)

    def _generate_holds(self, num_holds: int, wall_height: float):
        """Generate climbing holds with realistic distribution."""
        self.holds = []
        self.hold_positions = []
        
        # Create starting holds near the bottom
        for i in range(2):
            x = np.random.uniform(-0.3, 0.3)
            y = np.random.uniform(-0.4, 0.4)
            z = 0.5 + i * 0.3
            self._create_hold(x, y, z)
        
        # Create intermediate holds with increasing difficulty
        for i in range(2, num_holds-2):
            x = np.random.uniform(-0.5, 0.5)
            y = np.random.uniform(-0.6, 0.6)
            z = 1.0 + (i-2) * (wall_height-1.5)/(num_holds-4)
            self._create_hold(x, y, z)
        
        # Create top holds
        for i in range(num_holds-2, num_holds):
            x = np.random.uniform(-0.2, 0.2)
            y = np.random.uniform(-0.3, 0.3)
            z = wall_height - 0.3 + (i-(num_holds-2)) * 0.2
            self._create_hold(x, y, z)
            
    def _create_hold(self, x: float, y: float, z: float):
        """Create a single climbing hold."""
        hold_shape = p.createCollisionShape(
            p.GEOM_SPHERE, 
            radius=0.08,
            collisionFramePosition=[x, y, z]
        )
        
        visual_shape = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=0.1,
            rgbaColor=[0.8, 0.2, 0.2, 1]
        )
        
        hold_id = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=hold_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=[x, y, z]
        )
        
        self.holds.append(hold_id)
        self.hold_positions.append([x, y, z])

    def _cache_joint_info(self):
        """Cache joint information with enhanced data."""
        self.joint_info = OrderedDict()
        
        for i in range(p.getNumJoints(self.humanoid)):
            info = p.getJointInfo(self.humanoid, i)
            joint_name = info[1].decode('utf-8')
            
            self.joint_info[joint_name] = {
                'index': i,
                'type': info[2],
                'limits': (info[8], info[9]),
                'damping': info[6],
                'friction': info[7],
                'max_force': info[10],
                'link_name': info[12].decode('utf-8')
            }

    def _identify_effector_links(self):
        """Identify link indices for hands and feet."""
        for joint_name, joint_data in self.joint_info.items():
            for effector in self.effector_links.keys():
                if effector in joint_name.lower():
                    self.effector_links[effector] = joint_data['index']

    def _setup_camera(self, width: int = 640, height: int = 480):
        """Configure simulation camera with better positioning."""
        self.camera_width = width
        self.camera_height = height
        self.fov = 60
        self.aspect = width / height
        self.near = 0.1
        self.far = 10.0
        
        # Camera follows climber
        self.cam_distance = 2.5
        self.cam_yaw = 45
        self.cam_pitch = -20
        
        self._update_camera_matrices()

    def _update_camera_matrices(self):
        """Update camera to follow climber."""
        base_pos = p.getBasePositionAndOrientation(self.humanoid)[0]
        
        self.cam_pos = [
            base_pos[0] + self.cam_distance * np.cos(np.radians(self.cam_yaw)) * np.cos(np.radians(self.cam_pitch)),
            base_pos[1] + self.cam_distance * np.sin(np.radians(self.cam_yaw)) * np.cos(np.radians(self.cam_pitch)),
            base_pos[2] + self.cam_distance * np.sin(np.radians(self.cam_pitch))
        ]
        
        self.view_matrix = p.computeViewMatrix(
            cameraEyePosition=self.cam_pos,
            cameraTargetPosition=base_pos,
            cameraUpVector=[0, 0, 1]
        )
        
        self.proj_matrix = p.computeProjectionMatrixFOV(
            fov=self.fov,
            aspect=self.aspect,
            nearVal=self.near,
            farVal=self.far
        )

    def _get_camera_image(self):
        """Capture and process camera image with better quality."""
        self._update_camera_matrices()
        
        _, _, rgb, _, _ = p.getCameraImage(
            width=self.camera_width,
            height=self.camera_height,
            viewMatrix=self.view_matrix,
            projectionMatrix=self.proj_matrix,
            renderer=p.ER_TINY_RENDERER
        )
        
        rgb = np.reshape(rgb, (self.camera_height, self.camera_width, 4))[:, :, :3]
        
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
        
        return cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    def _get_obs(self) -> np.ndarray:
        """Enhanced observation with more relevant features."""
        # 1. Base state
        base_pos, base_orn = p.getBasePositionAndOrientation(self.humanoid)
        
        # 2. Joint states
        joint_angles = []
        for joint_data in self.joint_info.values():
            state = p.getJointState(self.humanoid, joint_data['index'])
            
            if joint_data['type'] == p.JOINT_SPHERICAL:
                if isinstance(state[0], (list, tuple)):
                    joint_angles.extend(state[0])
                else:
                    joint_angles.append(state[0]) 
            else:
                joint_angles.append(state[0])
        
        # 3. Effector positions
        effector_pos = []
        for link_idx in self.effector_links.values():
            if link_idx != -1:
                state = p.getLinkState(self.humanoid, link_idx)
                effector_pos.extend(state[0])
            else:
                effector_pos.extend([0, 0, 0])
        
        # 4. Distance to holds
        hold_dists = []
        for hold_pos in self.hold_positions:
            # Get closest effector distance to each hold
            min_dist = min([
                np.linalg.norm(np.array(p.getLinkState(self.humanoid, link_idx)[0]) - hold_pos)
                for link_idx in self.effector_links.values() if link_idx != -1
            ])
            hold_dists.append(min_dist)
        
        # 5. Visual keypoints
        if self.camera_view:
            img = self._get_camera_image()
            try:
                results = self.pose_processor.model(img)
                if results and len(results[0].keypoints) > 0:
                    keypoints = results[0].keypoints.xy[0].cpu().numpy()
                    # Normalize keypoints
                    kp_mean = keypoints.mean(axis=0)
                    kp_std = keypoints.std(axis=0) + 1e-8
                    normalized_kps = (keypoints - kp_mean) / kp_std
                    kp_vector = normalized_kps.flatten()
                else:
                    kp_vector = np.zeros(17*3)
            except Exception as e:
                print(f"Pose detection error: {e}")
                kp_vector = np.zeros(17*3)
        else:
            kp_vector = np.zeros(17*3)
        
        return np.concatenate([
            np.array(base_pos),
            np.array(base_orn),
            np.array(joint_angles),
            np.array(effector_pos),
            np.array(hold_dists),
            kp_vector
        ], dtype=np.float32)

    def _apply_actions(self, actions: np.ndarray):
        """Apply actions with smoother control and force limits."""
        action_idx = 0
        
        for joint_name, joint_data in self.joint_info.items():
            if action_idx >= len(actions):
                break
                
            if joint_data['type'] == p.JOINT_SPHERICAL:
                # Spherical joint (3DOF)
                target_pos = [
                    actions[action_idx] * np.pi,
                    actions[action_idx+1] * (np.pi/2),
                    actions[action_idx+2] * np.pi
                ]
                action_idx += 3
                
                p.setJointMotorControlMultiDof(
                    bodyUniqueId=self.humanoid,
                    jointIndex=joint_data['index'],
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=target_pos,
                    force=[200, 200, 200],
                    positionGain=[0.3, 0.3, 0.3],
                    velocityGain=[0.1, 0.1, 0.1]
                )
                    
            elif joint_data['type'] == p.JOINT_REVOLUTE:
                # Revolute joint (1DOF)
                low, high = joint_data['limits']
                target_pos = low + (actions[action_idx] + 1) * (high - low) / 2
                action_idx += 1
                
                p.setJointMotorControl2(
                    bodyUniqueId=self.humanoid,
                    jointIndex=joint_data['index'],
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=target_pos,
                    force=300,
                    positionGain=0.5,
                    velocityGain=0.2
                )

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """Run one timestep with enhanced reward and termination."""
        self._apply_actions(action)
        p.stepSimulation()
        
        obs = self._get_obs()
        
        # Get state information for reward calculation
        base_pos = np.array(p.getBasePositionAndOrientation(self.humanoid)[0])
        effector_pos = {
            name: np.array(p.getLinkState(self.humanoid, link_idx)[0])
            for name, link_idx in self.effector_links.items() if link_idx != -1
        }
        joint_angles = self._get_joint_angles()
        
        # Calculate reward components
        reward_info = self.reward_calculator.calculate(
            humanoid_pos=base_pos,
            effector_positions=effector_pos,
            joint_angles=joint_angles
        )
        
        reward = reward_info['total']
        self.current_reward += reward
        
        # Check termination
        done = self._check_termination(base_pos[2])
        
        # Log metrics
        info = {
            'reward': reward,
            'height': base_pos[2],
            'success': reward_info['reached_top'],
            'effector_hold_contact': reward_info['effector_hold_contact']
        }
        
        if done:
            self._log_episode(reward_info['reached_top'])
            
        return obs, reward, done, info

    def _get_joint_angles(self) -> np.ndarray:
        """Get current joint angles."""
        angles = []
        
        for joint_data in self.joint_info.values():
            state = p.getJointState(self.humanoid, joint_data['index'])
            
            if isinstance(state[0], (list, tuple)):
                angles.extend(state[0])
            else:
                angles.append(state[0])
                
        return np.array(angles)

    def _check_termination(self, current_height: float) -> bool:
        """Enhanced termination conditions."""
        # Fell below minimum height
        if current_height < 0.3:
            return True
            
        # Reached top hold
        top_hold = self.hold_positions[-1]
        for link_idx in self.effector_links.values():
            if link_idx != -1:
                effector_pos = p.getLinkState(self.humanoid, link_idx)[0]
                if np.linalg.norm(np.array(effector_pos) - top_hold) < 0.1:
                    return True
                    
        # Exceeded max steps (optional)
        if hasattr(self, 'step_count') and self.step_count >= 1000:
            return True
            
        return False

    def _log_episode(self, success: bool):
        """Log episode statistics."""
        self.episode_count += 1
        if success:
            self.success_count += 1
            
        self.episode_rewards.append(self.current_reward)
        self.current_reward = 0
        
        # Log to TensorBoard
        if hasattr(self, 'writer'):
            self.writer.add_scalar('Episode/Reward', self.episode_rewards[-1], self.episode_count)
            self.writer.add_scalar('Episode/Success', float(success), self.episode_count)
            self.writer.add_scalar('Stats/Success_Rate', 
                                  self.success_count / self.episode_count, 
                                  self.episode_count)

    def reset(self) -> np.ndarray:
        """Reset environment with optional curriculum."""
        p.resetSimulation()
        self._setup_world()
        self.step_count = 0
        self.current_reward = 0
        return self._get_obs()

    def render(self, mode='human'):
        """Render environment with better visualization."""
        if mode == 'rgb_array':
            return self._get_camera_image()
        return None

    def close(self):
        """Clean up resources."""
        p.disconnect()
        if hasattr(self, 'writer'):
            self.writer.close()
        if hasattr(self, 'pose_processor'):
            del self.pose_processor
            
    def get_expert_trajectories(self) -> List[dict]:
        """Get demonstration trajectories from videos."""
        if hasattr(self, 'video_dataset') and self.video_dataset:
            return self.video_dataset.trajectories
        return []

# ====================== MAIN EXECUTION ======================

if __name__ == "__main__":
    env = BoulderingEnv(render=True, camera_view=True, use_imitation=True)
    
    # Simple test loop
    obs = env.reset()
    done = False
    while not done:
        action = np.random.uniform(-1, 1, size=env.action_space.shape)
        obs, reward, done, info = env.step(action)
        
    env.close()