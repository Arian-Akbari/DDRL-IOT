#!/usr/bin/env python3
"""
DDRL System Integration Test
Tests the complete DDRL system to ensure it works correctly before deployment.
"""

import time

import numpy as np

import action_utils
from Config import Config
from D3QN import DuelingDoubleDeepQNetwork
from MEC_Env import MEC


def test_basic_functionality():
    """Test basic functionality of all components"""
    print("=" * 60)
    print("TESTING BASIC FUNCTIONALITY")
    print("=" * 60)

    # Test 1: Config parameters
    print("1. Testing Config parameters...")
    assert hasattr(Config, "NUM_SUBCHANNELS"), "NUM_SUBCHANNELS missing"
    assert hasattr(Config, "POWER_LEVELS"), "POWER_LEVELS missing"
    assert hasattr(Config, "LYAPUNOV_V"), "LYAPUNOV_V missing"
    print(
        f"   ✓ DDRL params loaded: k={Config.NUM_SUBCHANNELS}, powers={len(Config.POWER_LEVELS)}"
    )

    # Test 2: Action utilities
    print("2. Testing action utilities...")
    local_action = action_utils.encode_action(local=True)
    offload_action = action_utils.encode_action(local=False, edge=0, subch=1, p_idx=2)
    decoded = action_utils.decode_action(offload_action, n_edge=2)

    assert local_action == 0, "Local action should be 0"
    assert decoded["edge"] == 0 and decoded["subchannel"] == 1, "Action decoding failed"
    print(
        f"   ✓ Action encoding/decoding works: local={local_action}, offload={offload_action}"
    )

    # Test 3: Environment initialization
    print("3. Testing environment initialization...")
    env = MEC(num_ue=5, num_edge=2, num_time=10, num_component=1, max_delay=5)
    expected_actions = 1 + 2 * 4 * 5  # 1 + n_edge * k * n_power
    assert (
        env.n_actions == expected_actions
    ), f"Expected {expected_actions} actions, got {env.n_actions}"
    print(
        f"   ✓ Environment created: n_actions={env.n_actions}, n_features={env.n_features}"
    )

    # Test 4: DQN initialization
    print("4. Testing DQN initialization...")
    agent = DuelingDoubleDeepQNetwork(
        n_actions=env.n_actions,
        n_features=env.n_features,
        n_lstm_features=env.n_lstm_state,
        n_time=10,
        learning_rate=Config.DDRL_LEARNING_RATE,
        reward_decay=Config.DDRL_GAMMA,
        memory_size=100,  # Small for testing
        batch_size=5,
        replace_target_iter=5,
    )
    assert agent.N_L1 == 120, f"Expected N_L1=120, got {agent.N_L1}"
    assert agent.N_lstm == 80, f"Expected N_lstm=80, got {agent.N_lstm}"
    print(f"   ✓ DQN created: N_L1={agent.N_L1}, N_lstm={agent.N_lstm}")

    print("\n✅ All basic functionality tests PASSED!\n")
    return env, agent


def test_environment_step():
    """Test environment step function with DDRL reward"""
    print("=" * 60)
    print("TESTING ENVIRONMENT STEP FUNCTION")
    print("=" * 60)

    # Setup small environment for testing
    env = MEC(num_ue=3, num_edge=2, num_time=20, num_component=1, max_delay=5)

    # Create task arrivals
    arrive_task_size = np.random.uniform(1, 3, (20, 3))  # Small tasks
    arrive_task_dens = np.random.choice([0.197, 0.297, 0.397], (20, 3))

    # Reset environment
    print("1. Testing environment reset...")
    obs, lstm_state = env.reset(arrive_task_size, arrive_task_dens)
    assert obs.shape == (3, env.n_features), f"Wrong obs shape: {obs.shape}"
    print(f"   ✓ Reset successful: obs_shape={obs.shape}")

    # Test different action combinations
    test_cases = [
        ([0, 0, 0], "All local"),  # All local
        ([1, 5, 10], "Mixed actions"),  # Mixed actions
        ([20, 25, 30], "All offload"),  # All offload
    ]

    for actions, description in test_cases:
        print(f"2. Testing {description}: {actions}")

        # Take step
        obs_next, reward, done, info = env.step(actions)

        # Check return values
        assert obs_next.shape == (
            3,
            env.n_features,
        ), f"Wrong next obs shape: {obs_next.shape}"
        assert isinstance(
            reward, (int, float)
        ), f"Reward should be numeric, got {type(reward)}"
        assert isinstance(done, bool), f"Done should be boolean, got {type(done)}"
        assert isinstance(info, dict), f"Info should be dict, got {type(info)}"

        # Check info contents
        required_keys = ["ee", "drift", "conflicts", "slot_bits", "slot_power"]
        for key in required_keys:
            assert key in info, f"Missing key '{key}' in info"

        print(
            f"   ✓ Step successful: reward={reward:.4f}, conflicts={len(info['conflicts'])}"
        )
        print(
            f"     Energy efficiency: {info['ee']:.4f}, Lyapunov drift: {info['drift']:.4f}"
        )

        obs = obs_next
        if done:
            break

    print("\n✅ Environment step tests PASSED!\n")


def test_agent_interaction():
    """Test agent choosing actions and learning"""
    print("=" * 60)
    print("TESTING AGENT-ENVIRONMENT INTERACTION")
    print("=" * 60)

    # Setup
    env = MEC(num_ue=3, num_edge=2, num_time=15, num_component=1, max_delay=3)
    agent = DuelingDoubleDeepQNetwork(
        n_actions=env.n_actions,
        n_features=env.n_features,
        n_lstm_features=env.n_lstm_state,
        n_time=15,
        learning_rate=0.001,
        memory_size=50,
        batch_size=3,
        replace_target_iter=5,
    )

    # Create task arrivals
    arrive_task_size = np.random.uniform(0.5, 2, (15, 3))
    arrive_task_dens = np.random.choice([0.197, 0.297, 0.397], (15, 3))

    print("1. Testing agent action selection...")
    obs, lstm_state = env.reset(arrive_task_size, arrive_task_dens)

    # Test action selection for each UE
    actions = []
    for ue in range(env.n_ue):
        if np.sum(obs[ue]) > 0:  # Only if UE has task
            action = agent.choose_action(obs[ue])
            assert 0 <= action < env.n_actions, f"Invalid action {action} for UE {ue}"
            actions.append(action)
        else:
            actions.append(0)  # Local if no task

    print(f"   ✓ Actions selected: {actions}")

    print("2. Testing learning process...")
    total_reward = 0
    step_count = 0

    for step in range(5):  # Short episode
        # Choose actions for all UEs
        actions = []
        for ue in range(env.n_ue):
            if np.sum(obs[ue]) > 0:
                action = agent.choose_action(obs[ue])
                actions.append(action)
                agent.update_lstm(lstm_state[ue])
            else:
                actions.append(0)

        # Take environment step
        obs_next, reward, done, info = env.step(actions)
        total_reward += reward
        step_count += 1

        # Store experiences
        for ue in range(env.n_ue):
            if np.sum(obs[ue]) > 0:  # Only store if UE had task
                agent.store_transition(
                    obs[ue],
                    lstm_state[ue],
                    actions[ue],
                    reward,
                    obs_next[ue],
                    lstm_state[ue],
                )

        # Learn (if enough memory)
        if agent.memory_counter > agent.batch_size:
            cost = agent.learn()
            print(
                f"   Step {step}: reward={reward:.4f}, cost={cost:.4f}, conflicts={len(info['conflicts'])}"
            )
        else:
            print(
                f"   Step {step}: reward={reward:.4f}, conflicts={len(info['conflicts'])} (building memory)"
            )

        obs = obs_next
        lstm_state = np.zeros((env.n_ue, env.n_lstm_state))  # Reset LSTM for simplicity

        if done:
            break

    avg_reward = total_reward / step_count if step_count > 0 else 0
    print(
        f"   ✓ Learning completed: avg_reward={avg_reward:.4f}, memory_size={agent.memory_counter}"
    )

    print("\n✅ Agent interaction tests PASSED!\n")


def test_action_mask():
    """Test optional action masking functionality"""
    print("=" * 60)
    print("TESTING ACTION MASKING (OPTIONAL)")
    print("=" * 60)

    env = MEC(num_ue=2, num_edge=2, num_time=5, num_component=1, max_delay=2)
    agent = DuelingDoubleDeepQNetwork(
        n_actions=env.n_actions,
        n_features=env.n_features,
        n_lstm_features=env.n_lstm_state,
        n_time=5,
        learning_rate=0.001,
        memory_size=20,
        batch_size=2,
    )

    # Create dummy observation
    obs = np.random.random(env.n_features)

    print("1. Testing action selection without mask...")
    action_normal = agent.choose_action(obs)
    print(f"   ✓ Normal action: {action_normal}")

    print("2. Testing action selection with mask...")
    # Create mask that disallows all but first 3 actions
    action_mask = np.zeros(env.n_actions)
    action_mask[:3] = 1.0

    # Force deterministic behavior for testing
    agent.epsilon = 1.0  # Always exploit
    action_masked = agent.choose_action(obs, action_mask=action_mask)
    assert action_masked < 3, f"Masked action {action_masked} should be < 3"
    print(f"   ✓ Masked action: {action_masked} (allowed: 0-2)")

    print("\n✅ Action masking tests PASSED!\n")


def test_performance():
    """Test system performance for longer runs"""
    print("=" * 60)
    print("TESTING SYSTEM PERFORMANCE")
    print("=" * 60)

    # Larger test scenario
    env = MEC(num_ue=10, num_edge=2, num_time=50, num_component=1, max_delay=5)
    agent = DuelingDoubleDeepQNetwork(
        n_actions=env.n_actions,
        n_features=env.n_features,
        n_lstm_features=env.n_lstm_state,
        n_time=50,
        learning_rate=Config.DDRL_LEARNING_RATE,
        reward_decay=Config.DDRL_GAMMA,
        memory_size=200,
        batch_size=10,
    )

    # Create realistic task arrivals
    arrive_task_size = np.zeros((50, 10))
    arrive_task_dens = np.zeros((50, 10))

    for t in range(50):
        for ue in range(10):
            if np.random.random() < Config.TASK_ARRIVE_PROB:  # Task arrives
                arrive_task_size[t, ue] = np.random.uniform(
                    Config.TASK_MIN_SIZE, Config.TASK_MAX_SIZE
                )
                arrive_task_dens[t, ue] = np.random.choice(Config.TASK_COMP_DENS)

    print(f"1. Running {50} steps with {10} UEs...")
    start_time = time.time()

    obs, lstm_state = env.reset(arrive_task_size, arrive_task_dens)
    total_reward = 0
    total_ee = 0
    step_count = 0

    for step in range(20):  # Test 20 steps
        # Choose actions
        actions = []
        for ue in range(env.n_ue):
            if np.sum(obs[ue]) > 0:
                action = agent.choose_action(obs[ue])
                actions.append(action)
                agent.update_lstm(lstm_state[ue])
            else:
                actions.append(0)

        # Environment step
        obs_next, reward, done, info = env.step(actions)
        total_reward += reward
        total_ee += info["ee"]
        step_count += 1

        # Store and learn
        for ue in range(env.n_ue):
            if np.sum(obs[ue]) > 0:
                agent.store_transition(
                    obs[ue],
                    lstm_state[ue],
                    actions[ue],
                    reward,
                    obs_next[ue],
                    lstm_state[ue],
                )

        if agent.memory_counter > agent.batch_size:
            agent.learn()

        if step % 5 == 0:
            print(
                f"   Step {step:2d}: reward={reward:6.3f}, EE={info['ee']:6.3f}, conflicts={len(info['conflicts'])}"
            )

        obs = obs_next
        lstm_state = np.zeros((env.n_ue, env.n_lstm_state))

        if done:
            break

    elapsed_time = time.time() - start_time
    avg_reward = total_reward / step_count
    avg_ee = total_ee / step_count

    print(f"\n   ✓ Performance test completed:")
    print(f"     Time: {elapsed_time:.2f}s ({elapsed_time/step_count:.3f}s per step)")
    print(f"     Average reward: {avg_reward:.4f}")
    print(f"     Average energy efficiency: {avg_ee:.4f}")
    print(f"     Memory utilization: {agent.memory_counter}/{agent.memory_size}")

    print("\n✅ Performance tests PASSED!\n")


def main():
    """Run all tests"""
    print("\n🚀 DDRL SYSTEM INTEGRATION TEST")
    print("Testing system before server deployment...\n")

    try:
        # Basic functionality
        env, agent = test_basic_functionality()

        # Environment step function
        test_environment_step()

        # Agent-environment interaction
        test_agent_interaction()

        # Action masking
        test_action_mask()

        # Performance test
        test_performance()

        print("🎉 ALL TESTS PASSED! 🎉")
        print("✅ System is ready for server deployment!")
        print("\nNext steps:")
        print("1. Copy this codebase to your server")
        print("2. Run full training with your desired parameters")
        print("3. Monitor the energy efficiency and Lyapunov drift metrics")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        print("🔧 Fix the issue before deploying to server!")
        raise


if __name__ == "__main__":
    main()
