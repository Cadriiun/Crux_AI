from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
import BoulderingEnv
from utils.reward_utils import RewardCalculator

'''
ABOUT:
    Initialize the gym environment
    Train the model with PPO aka Multi-layer perceptron
    Then evalulate the environment mostly rewards length or success rate
    Train the data by learning from and save the best model
    Save the model
'''

def train():
    env = BoulderingEnv(render=False, camera_view=True)
    
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./logs/",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        policy_kwargs={
            "net_arch": dict(pi=[256, 256], vf=[256, 256])
        }
    )
    
    eval_callback = EvalCallback(
        BoulderingEnv(render=False),
        best_model_save_path="./best/",
        eval_freq=5000,
        deterministic=True
    )
    
    model.learn(
        total_timesteps=1_000_000,
        callback=eval_callback,
        tb_log_name="ppo_pose_enhanced"
    )
    model.save("climber_ppo_enhanced")

if __name__ == "__main__":
    train()