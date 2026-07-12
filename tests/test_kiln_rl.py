"""
Unit tests for Kiln-Cooler Sintering Zone RL Optimization module.
Tests the Gym environment, the PyTorch PPO Agent policy loops, and controller fallbacks.
"""

import os
import sys
import numpy as np
import pytest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cement_ai_platform.simulation.kiln_gym_env import KilnCoolerGymEnv
from cement_ai_platform.control.kiln_rl_agent import PPOAgent
from cement_ai_platform.models.agents.unified_kiln_cooler_controller import UnifiedKilnCoolerController


def test_kiln_gym_env_dynamics():
    """Test environment initialization, steps, clamping, and reward computation."""
    env = KilnCoolerGymEnv(max_steps=50)
    
    # 1. Reset Env
    obs = env.reset()
    assert len(obs) == 7
    assert obs.dtype == np.float32
    
    # Check nominal state values
    assert env.burning_zone_temp == 1450.0
    assert env.fuel_rate == 15.0
    assert env.kiln_speed == 3.0
    assert env.feed_rate == 200.0
    
    # 2. Step Env
    action = np.array([0.1, -0.2, 1.5], dtype=np.float32)
    next_obs, reward, done, info = env.step(action)
    
    assert len(next_obs) == 7
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)
    
    # Ensure setpoints updated correctly
    assert env.kiln_speed == pytest.approx(3.1)
    assert env.fuel_rate == pytest.approx(14.8)
    assert env.feed_rate == pytest.approx(201.5)
    
    # 3. Action Clamping Test
    # Try action adjustments exceeding limits
    action_extreme = np.array([5.0, 5.0, 5.0], dtype=np.float32)
    next_obs2, reward2, done2, info2 = env.step(action_extreme)
    
    # Speed is clamped to environment's max speed boundary (4.5 RPM)
    assert env.kiln_speed <= 4.5
    assert env.fuel_rate <= 25.0
    assert env.feed_rate <= 250.0


def test_ppo_agent_policy_execution():
    """Test PPO Agent initialization, selection, memory, and checkpoint save/load."""
    agent = PPOAgent(state_dim=7, action_dim=3)
    state = np.array([1450.0, 15.0, 3.0, 200.0, 1.2, 500.0, 550.0], dtype=np.float32)
    
    # 1. Selection pass
    action, log_prob, val = agent.select_action(state)
    assert len(action) == 3
    assert isinstance(log_prob, float)
    assert isinstance(val, float)
    
    # 2. Save and Load Test
    temp_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(temp_dir, exist_ok=True)
    temp_filepath = os.path.join(temp_dir, 'temp_test_kiln_actor.pt')
    
    try:
        # Save model
        agent.save(temp_filepath)
        assert os.path.exists(temp_filepath)
        
        # Load model to a new agent instance
        new_agent = PPOAgent(state_dim=7, action_dim=3)
        load_success = new_agent.load(temp_filepath)
        assert load_success is True
        
    finally:
        # Clean up temporary test weights file
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)


def test_controller_fallback_logic():
    """Test that controller works with and without RL option, falling back gracefully."""
    controller = UnifiedKilnCoolerController()
    
    # Create sample sensor reading
    sensor_data = {
        'burning_zone_temp_c': 1445.0,
        'fuel_rate_tph': 15.2,
        'kiln_speed_rpm': 3.1,
        'feed_rate_tph': 200.5,
        'free_lime_percent': 1.15,
        'nox_mg_nm3': 490.0,
        'cooler_outlet_temp_c': 90.0,
        'cooler_air_flow_nm3_h': 150000.0,
        'gas_flow_nm3_h': 200000.0
    }
    
    # Call with use_rl=False
    output_no_rl = controller.compute_setpoints(sensor_data, use_rl=False)
    assert 'kiln_setpoints' in output_no_rl
    assert 'preheater_setpoints' in output_no_rl
    assert 'cooler_setpoints' in output_no_rl
    
    # Call with use_rl=True
    # Verify that it processes setpoints without throwing error, whether model file is present or not
    output_rl = controller.compute_setpoints(sensor_data, use_rl=True)
    assert 'kiln_setpoints' in output_rl
    assert 'preheater_setpoints' in output_rl
    assert 'cooler_setpoints' in output_rl
    
    # Ensure setpoint keys contain numeric types
    assert isinstance(output_rl['kiln_setpoints']['kiln_speed_rpm'], float)
    assert isinstance(output_rl['kiln_setpoints']['fuel_rate_tph'], float)
    assert isinstance(output_rl['kiln_setpoints']['feed_rate_tph'], float)
