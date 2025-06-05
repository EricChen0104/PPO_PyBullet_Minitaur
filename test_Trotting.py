import gym
import pybullet as p
import pybullet_envs
import time
import os
import pybullet_data
import numpy as np
from PPO import Agent

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

np.bool8 = np.bool_


def custom_render(env):
    """使用環境的 PyBullet 客戶端進行渲染"""
    try:
        width, height = 320, 240
        view_matrix = env._pybullet_client.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0, 0, 0.5],
            distance=1.5,
            yaw=30,
            pitch=-30,
            roll=0,
            upAxisIndex=2
        )
        proj_matrix = env._pybullet_client.computeProjectionMatrixFOV(
            fov=60, aspect=float(width) / height, nearVal=0.1, farVal=100.0
        )
        _, _, rgb, _, _ = env._pybullet_client.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=env._pybullet_client.ER_BULLET_HARDWARE_OPENGL
        )
        rgb_array = np.array(rgb).reshape(height, width, 4)[:, :, :3]
        return rgb_array
    except Exception as e:
        print(f"Custom render error: {e}")
        return None

def update_stacked_obs(stacked_obs, new_obs, obs_shape):
    return np.concatenate([stacked_obs[obs_shape:], new_obs])

def test_agent():
    # 初始化环境
    env = gym.make("MinitaurBulletEnv-v0", render="human")
    obs = env.reset()
    obs = np.array(obs, dtype=np.float32)
    
    # 获取状态和动作空间
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    print('action_dim: ', state_dim)
    
    stack_size = 8  # 疊 4 步觀察
    obs_shape = env.observation_space.shape[0]
    stacked_obs = np.zeros(stack_size * obs_shape, dtype=np.float32)

    # PPO参数设置
    agent = Agent(n_actions=action_dim, 
                 batch_size=128,
                 alpha=3e-5, 
                 n_epochs=10,
                 input_dims=obs_shape * stack_size,
                 action_low=env.action_space.low,  
                 action_high=env.action_space.high,
                 actor_model_path='./tmp/ppo/v5',
                 critic_model_path='./tmp/ppo/v5'
                )  # 使用实际状态维度

    
    agent.load_models()
    
    best_score = -np.inf
    score_history = []
    n_steps = 0
    n_episodes = 5
    
    # 训练循环
    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        score = 0
        
        while not done:
            # 获取动作
            action_env, action_store, prob, val = agent.choose_action(stacked_obs)
            # 执行动作
            new_obs, reward, done, info = env.step(action_env)
            new_obs = np.array(new_obs, dtype=np.float32)

            next_stacked_obs = update_stacked_obs(stacked_obs, new_obs, obs_shape)
            
            # 存储经验
            # agent.remember(obs, action, prob, val, reward, done)
            
            # 更新状态
            score += reward
            stacked_obs = next_stacked_obs
            n_steps += 1
        
        # 记录和保存
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        
        if score > best_score:
            best_score = score
            
        print(f'Episode {episode}, Score: {score:.1f}, Avg: {avg_score:.1f}, Steps: {n_steps}')

    plt.ioff()  # 關閉互動模式
    plt.show()
    env.close()

def train(load_existing_model=False, model_path="./tmp/ppo/v5"):
    # 初始化环境
    env = gym.make("MinitaurBulletEnv-v0", render=None)
    obs = env.reset()
    obs = np.array(obs, dtype=np.float32)
    
    # 获取状态和动作空间
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    print("Action low:", env.action_space.low)
    print("Action high:", env.action_space.high)

    print('action_dim: ', state_dim)

    stack_size = 8  # 疊 4 步觀察
    obs_shape = env.observation_space.shape[0]
    stacked_obs = np.zeros(stack_size * obs_shape, dtype=np.float32)

    # PPO参数设置
    agent = Agent(n_actions=action_dim, 
                 batch_size=128,
                 alpha=3e-5, 
                 n_epochs=10,
                 input_dims=obs_shape * stack_size,
                 action_low=env.action_space.low,  
                 action_high=env.action_space.high,
                 actor_model_path='./tmp/ppo/v6',
                 critic_model_path='./tmp/ppo/v6',
                )  # 使用实际状态维度

    if load_existing_model:
        actor_model_file = os.path.join(model_path, 'actor_torch_ppo_best_3')
        critic_model_file = os.path.join(model_path, 'critic_torch_ppo_best_3')
        if os.path.exists(actor_model_file) and os.path.exists(critic_model_file):
            try:
                agent.load_models() # 這會調用 actor.load_checkpoint() 和 critic.load_checkpoint()
            except Exception as e:
                print(f"Error loading models: {e}. Starting from scratch.")
                load_existing_model = False # 如果加載失敗，則從頭開始
        else:
            print(f"Model files not found in {model_path}. Starting from scratch.")
            load_existing_model = False
    
    best_score = 24.4
    score_history = []
    n_steps = 0
    n_episodes = 700

    update_timestep = 4096

    plt.ion()  # 開啟互動模式
    fig, ax = plt.subplots()

    # 训练循环
    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        score = 0
        
        while not done:
            # 获取动作
            action_env, action_store, prob, val = agent.choose_action(stacked_obs)
            # 执行动作
            new_obs, reward, done, info = env.step(action_env)
            new_obs = np.array(new_obs, dtype=np.float32)

            next_stacked_obs = update_stacked_obs(stacked_obs, new_obs, obs_shape)
            
            # 存储经验
            agent.remember(stacked_obs, action_store, prob, val, reward, done)
            
            # 更新状态
            score += reward
            stacked_obs = next_stacked_obs
            n_steps += 1
        
        if len(agent.memory.states) >= agent.memory.batch_size: # 確保有足夠數據
            agent.learn()
        
        # 记录和保存
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        # 清除前一張圖
        ax.clear()
        ax.plot(score_history, label='score')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Score')
        ax.set_title(f"Episode {episode}, Score: {score:.1f}, Best Score: {best_score:.1f}")
        ax.legend()
        
        plt.pause(0.01)  # 暫停一下讓圖更新
        if score > best_score:
            best_score = score
            agent.save_models()
            
        print(f'Episode {episode}, Score: {score:.2f}, Best score: {best_score:.2f}, Avg: {avg_score:.2f}, Steps: {n_steps}')

    plt.ioff()  # 關閉互動模式（可加可不加）
    plt.savefig("./plot/ppo_training_curve_4.png")
    env.close()

if __name__ == '__main__':
    test_agent()
    # train(load_existing_model=True)