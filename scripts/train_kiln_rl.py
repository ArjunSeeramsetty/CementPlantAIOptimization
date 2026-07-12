"""
Training script for the Kiln-Cooler Sintering Zone RL agent.
Runs training loops in the physics-informed custom Gym environment.
"""

import os
import sys
import numpy as np

# Add src to python path to resolve local imports cleanly
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from cement_ai_platform.simulation.kiln_gym_env import KilnCoolerGymEnv
from cement_ai_platform.control.kiln_rl_agent import PPOAgent


def train_agent():
    print("[INFO] Starting Kiln-Cooler PPO Agent training...")
    
    # Hyperparameters
    episodes = 250
    steps_per_episode = 50
    update_interval = 100  # Update PPO every 100 steps (2 episodes)
    
    env = KilnCoolerGymEnv(max_steps=steps_per_episode)
    agent = PPOAgent(state_dim=7, action_dim=3)
    
    # Metrics tracking
    episode_rewards = []
    step_counter = 0
    
    for ep in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0
        
        for step in range(steps_per_episode):
            step_counter += 1
            
            # Select action under Gaussian policy
            action, log_prob, val = agent.select_action(state)
            
            # Execute step in simulator
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            
            # Store in agent memory
            agent.memory.states.append(state)
            agent.memory.actions.append(action)
            agent.memory.rewards.append(reward)
            agent.memory.log_probs.append(log_prob)
            agent.memory.values.append(val)
            agent.memory.is_terminals.append(done)
            
            state = next_state
            
            # Periodically update policy network weights
            if step_counter % update_interval == 0:
                actor_loss, critic_loss = agent.update()
                
            if done:
                break
                
        episode_rewards.append(total_reward)
        
        # Log progress every 20 episodes
        if ep % 20 == 0 or ep == 1:
            avg_rew = np.mean(episode_rewards[-20:])
            print(f"Episode {ep:03d}/{episodes} | Average Reward (last 20): {avg_rew:.2f} | Total Steps: {step_counter}")
            
    # Save model weights to standard path
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "kiln_rl_actor.pt")
    
    agent.save(model_path)
    print("[INFO] Training completed successfully!")


if __name__ == "__main__":
    train_agent()
