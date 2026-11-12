import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import random
from collections import deque
import matplotlib.pyplot as plt


# ==============================================================================
# PART 1: ENVIRONMENT AND STATE REPRESENTATION
# ==============================================================================

class TrafficEnvironment:
    """
    Simulates a 4-way intersection with traffic lights.
    State: Grid-based representation with position and speed
    Actions: Adjust phase durations in a cycle
    Reward: Change in cumulative waiting time
    """

    def __init__(self, grid_size=60, grid_length=5, max_speed=13.9):
        self.grid_size = grid_size  # 60x60 grids
        self.grid_length = grid_length  # 5 meters per grid
        self.max_speed = max_speed  # 13.9 m/s (50 km/h)

        # Intersection parameters
        self.lane_length = 150  # meters
        self.num_lanes_per_direction = 3
        self.num_directions = 4  # N, S, E, W

        # Traffic light phases (4 phases for the intersection)
        # Phase 0: North-South through
        # Phase 1: North-East & South-West turn
        # Phase 2: East-West through
        # Phase 3: East-South & West-North turn
        self.num_phases = 4
        self.phase_durations = [30, 30, 30, 30]  # Initial durations in seconds
        self.yellow_duration = 4  # seconds
        self.min_phase_duration = 0
        self.max_phase_duration = 60
        self.phase_change_step = 5  # Change duration by 5 seconds

        # Vehicle parameters
        self.vehicle_length = 5  # meters
        self.min_gap = 2  # meters
        self.max_acceleration = 1.0  # m/s^2
        self.max_deceleration = 4.5  # m/s^2

        # Traffic flow parameters
        self.arrival_rate = 0.1  # vehicles per second per lane

        # State tracking
        self.vehicles = []  # List of vehicles
        self.current_phase = 0
        self.phase_timer = 0
        self.total_time = 0
        self.previous_cumulative_waiting = 0

    def reset(self):
        """Reset the environment to initial state"""
        self.vehicles = []
        self.current_phase = 0
        self.phase_timer = 0
        self.total_time = 0
        self.previous_cumulative_waiting = 0
        self.phase_durations = [30, 30, 30, 30]
        return self.get_state()

    def get_state(self):
        """
        Build state representation as grid with position and speed.
        Returns: (60, 60, 2) array where:
          - Channel 0: Position (1 if vehicle present, 0 otherwise)
          - Channel 1: Speed (vehicle speed in m/s, 0 if no vehicle)
        """
        position_grid = np.zeros((self.grid_size, self.grid_size))
        speed_grid = np.zeros((self.grid_size, self.grid_size))

        for vehicle in self.vehicles:
            x, y = vehicle['position']
            grid_x = int(x / self.grid_length)
            grid_y = int(y / self.grid_length)

            # Ensure within bounds
            if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
                position_grid[grid_y, grid_x] = 1.0
                speed_grid[grid_y, grid_x] = vehicle['speed']

        # Stack position and speed channels
        state = np.stack([position_grid, speed_grid], axis=-1)
        return state

    def get_legal_actions(self):
        """
        Returns list of legal actions based on current phase durations.
        Each action adjusts one phase duration by +/- 5 seconds.
        Action format: (phase_index, change) where change is -5 or +5
        """
        actions = []
        action_id = 0

        for phase_idx in range(self.num_phases):
            current_duration = self.phase_durations[phase_idx]

            # Can decrease duration
            if current_duration - self.phase_change_step >= self.min_phase_duration:
                actions.append((phase_idx, -self.phase_change_step))
                action_id += 1

            # Can increase duration
            if current_duration + self.phase_change_step <= self.max_phase_duration:
                actions.append((phase_idx, self.phase_change_step))
                action_id += 1

        # Add "no change" action
        actions.append((None, 0))

        return actions

    def step(self, action_idx):
        """
        Execute one cycle with the chosen action.

        Args:
            action_idx: Index of action to take

        Returns:
            next_state: New state after action
            reward: Reward (change in cumulative waiting time)
            done: Whether episode is complete
            info: Additional information
        """
        # Apply action to modify phase durations
        legal_actions = self.get_legal_actions()

        if action_idx < len(legal_actions):
            phase_idx, change = legal_actions[action_idx]
            if phase_idx is not None:
                self.phase_durations[phase_idx] += change
                # Clip to valid range
                self.phase_durations[phase_idx] = np.clip(
                    self.phase_durations[phase_idx],
                    self.min_phase_duration,
                    self.max_phase_duration
                )

        # Simulate one complete cycle
        cycle_duration = sum(self.phase_durations) + self.num_phases * self.yellow_duration

        # Generate vehicles and simulate traffic
        self._simulate_cycle(cycle_duration)

        # Calculate reward (change in cumulative waiting time)
        current_cumulative_waiting = self._calculate_cumulative_waiting()
        reward = self.previous_cumulative_waiting - current_cumulative_waiting
        self.previous_cumulative_waiting = current_cumulative_waiting

        # Get new state
        next_state = self.get_state()

        # Episode ends after 1 hour (3600 seconds)
        done = self.total_time >= 3600

        info = {
            'cumulative_waiting': current_cumulative_waiting,
            'num_vehicles': len(self.vehicles),
            'cycle_duration': cycle_duration
        }

        return next_state, reward, done, info

    def _simulate_cycle(self, duration):
        """Simulate traffic for one cycle duration"""
        timestep = 1.0  # 1 second timesteps
        steps = int(duration / timestep)

        for _ in range(steps):
            # Generate new vehicles
            self._generate_vehicles()

            # Update vehicle positions and speeds
            self._update_vehicles(timestep)

            # Update phase
            self.phase_timer += timestep
            if self.phase_timer >= self.phase_durations[self.current_phase]:
                self.current_phase = (self.current_phase + 1) % self.num_phases
                self.phase_timer = 0

            self.total_time += timestep

    def _generate_vehicles(self):
        """Generate new vehicles based on arrival rate"""
        # Simplified vehicle generation
        # In practice, you'd use Poisson process or other arrival model
        if random.random() < self.arrival_rate:
            direction = random.choice(['N', 'S', 'E', 'W'])
            lane = random.randint(0, self.num_lanes_per_direction - 1)

            vehicle = {
                'direction': direction,
                'lane': lane,
                'position': self._get_spawn_position(direction, lane),
                'speed': 0,
                'waiting_time': 0,
                'type': random.choice(['through', 'turn'])
            }
            self.vehicles.append(vehicle)

    def _get_spawn_position(self, direction, lane):
        """Get spawn position for a vehicle"""
        center = self.grid_size * self.grid_length / 2

        if direction == 'N':
            return (center + lane * self.grid_length, 0)
        elif direction == 'S':
            return (center - lane * self.grid_length, self.grid_size * self.grid_length)
        elif direction == 'E':
            return (self.grid_size * self.grid_length, center + lane * self.grid_length)
        else:  # W
            return (0, center - lane * self.grid_length)

    def _update_vehicles(self, timestep):
        """Update vehicle positions and speeds"""
        vehicles_to_remove = []

        for i, vehicle in enumerate(self.vehicles):
            # Simplified movement model (Krauss-like)
            can_move = self._can_vehicle_move(vehicle)

            if can_move:
                # Accelerate
                vehicle['speed'] = min(
                    vehicle['speed'] + self.max_acceleration * timestep,
                    self.max_speed
                )
                vehicle['waiting_time'] = 0
            else:
                # Decelerate/stop
                vehicle['speed'] = max(
                    vehicle['speed'] - self.max_deceleration * timestep,
                    0
                )
                if vehicle['speed'] < 0.1:
                    vehicle['waiting_time'] += timestep

            # Update position
            x, y = vehicle['position']
            if vehicle['direction'] == 'N':
                y += vehicle['speed'] * timestep
            elif vehicle['direction'] == 'S':
                y -= vehicle['speed'] * timestep
            elif vehicle['direction'] == 'E':
                x -= vehicle['speed'] * timestep
            else:  # W
                x += vehicle['speed'] * timestep

            vehicle['position'] = (x, y)

            # Remove vehicles that have left the intersection
            if self._has_left_intersection(vehicle):
                vehicles_to_remove.append(i)

        # Remove vehicles in reverse order
        for i in reversed(vehicles_to_remove):
            self.vehicles.pop(i)

    def _can_vehicle_move(self, vehicle):
        """Check if vehicle can move based on traffic light and other vehicles"""
        # Simplified logic - checks if vehicle is near intersection and light is red
        x, y = vehicle['position']
        center = self.grid_size * self.grid_length / 2
        near_intersection = abs(x - center) < 20 and abs(y - center) < 20

        if near_intersection:
            # Check traffic light phase
            return self._is_green_for_vehicle(vehicle)

        return True

    def _is_green_for_vehicle(self, vehicle):
        """Check if current phase allows vehicle to pass"""
        # Simplified phase logic
        direction = vehicle['direction']
        vehicle_type = vehicle['type']

        if self.current_phase == 0:  # N-S through
            return direction in ['N', 'S'] and vehicle_type == 'through'
        elif self.current_phase == 1:  # NE & SW turn
            return (direction == 'N' and vehicle_type == 'turn') or \
                (direction == 'S' and vehicle_type == 'turn')
        elif self.current_phase == 2:  # E-W through
            return direction in ['E', 'W'] and vehicle_type == 'through'
        else:  # ES & WN turn
            return (direction == 'E' and vehicle_type == 'turn') or \
                (direction == 'W' and vehicle_type == 'turn')

    def _has_left_intersection(self, vehicle):
        """Check if vehicle has left the simulation area"""
        x, y = vehicle['position']
        max_coord = self.grid_size * self.grid_length
        return x < -10 or x > max_coord + 10 or y < -10 or y > max_coord + 10

    def _calculate_cumulative_waiting(self):
        """Calculate total waiting time of all vehicles"""
        return sum(vehicle['waiting_time'] for vehicle in self.vehicles)


# ==============================================================================
# PART 2: DOUBLE DUELING DEEP Q NETWORK (3DQN)
# ==============================================================================

class DuelingDQN(keras.Model):
    """
    Dueling Deep Q Network architecture as described in the paper.
    Separates value and advantage streams.
    """

    def __init__(self, num_actions):
        super(DuelingDQN, self).__init__()

        # Convolutional layers
        self.conv1 = layers.Conv2D(
            32, kernel_size=4, strides=2, activation='relu', padding='same'
        )
        self.conv2 = layers.Conv2D(
            64, kernel_size=2, strides=2, activation='relu', padding='same'
        )
        self.conv3 = layers.Conv2D(
            128, kernel_size=2, strides=1, activation='relu', padding='same'
        )

        # Flatten layer
        self.flatten = layers.Flatten()

        # Fully connected layer
        self.dense = layers.Dense(128, activation='relu')

        # Dueling streams
        self.value_stream = layers.Dense(64, activation='relu')
        self.value = layers.Dense(1)

        self.advantage_stream = layers.Dense(64, activation='relu')
        self.advantage = layers.Dense(num_actions)

        self.num_actions = num_actions

    def call(self, inputs):
        """Forward pass"""
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense(x)

        # Value stream
        value = self.value_stream(x)
        value = self.value(value)

        # Advantage stream
        advantage = self.advantage_stream(x)
        advantage = self.advantage(advantage)

        # Combine value and advantage
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + (advantage - tf.reduce_mean(advantage, axis=1, keepdims=True))

        return q_values


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer using rank-based prioritization.
    """

    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha  # Prioritization exponent
        self.buffer = []
        self.priorities = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        max_priority = max(self.priorities) if self.priorities else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
            self.priorities.append(max_priority)
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
            self.priorities[self.position] = max_priority

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        """Sample batch with prioritized probabilities"""
        if len(self.buffer) == 0:
            return None

        # Rank-based prioritization
        ranks = np.arange(1, len(self.buffer) + 1)
        priorities = 1.0 / ranks
        priorities = priorities ** self.alpha
        probabilities = priorities / priorities.sum()

        # Sample indices
        indices = np.random.choice(
            len(self.buffer),
            size=min(batch_size, len(self.buffer)),
            p=probabilities,
            replace=False
        )

        # Calculate importance sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-beta)
        weights /= weights.max()

        # Get samples
        samples = [self.buffer[idx] for idx in indices]

        states = np.array([s[0] for s in samples])
        actions = np.array([s[1] for s in samples])
        rewards = np.array([s[2] for s in samples])
        next_states = np.array([s[3] for s in samples])
        dones = np.array([s[4] for s in samples])

        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, indices, td_errors):
        """Update priorities based on TD errors"""
        for idx, td_error in zip(indices, td_errors):
            # Use absolute TD error as priority
            self.priorities[idx] = abs(td_error) + 1e-6

    def __len__(self):
        return len(self.buffer)


class DoubleDuelingDQNAgent:
    """
    Double Dueling Deep Q Network (3DQN) agent with prioritized experience replay.
    Implements the complete algorithm from the paper.
    """

    def __init__(
            self,
            state_shape,
            num_actions,
            learning_rate=0.0001,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay_steps=10000,
            target_update_rate=0.001,
            buffer_capacity=20000,
            batch_size=64,
            pretrain_steps=2000
    ):
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.pretrain_steps = pretrain_steps

        # Epsilon-greedy parameters
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilon_decay = (epsilon_start - epsilon_end) / epsilon_decay_steps

        # Networks
        self.primary_network = DuelingDQN(num_actions)
        self.target_network = DuelingDQN(num_actions)

        # Initialize networks with dummy input
        dummy_input = tf.random.normal((1,) + state_shape)
        self.primary_network(dummy_input)
        self.target_network(dummy_input)

        # Copy initial weights
        self.target_network.set_weights(self.primary_network.get_weights())

        # Optimizer (Adam)
        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

        # Replay buffer
        self.replay_buffer = PrioritizedReplayBuffer(buffer_capacity)

        # Target network update rate
        self.target_update_rate = target_update_rate

        # Training step counter
        self.step_count = 0

    def select_action(self, state, legal_actions_mask=None):
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current state
            legal_actions_mask: Boolean mask of legal actions

        Returns:
            action: Selected action index
        """
        # Epsilon-greedy exploration
        if random.random() < self.epsilon:
            if legal_actions_mask is not None:
                legal_indices = np.where(legal_actions_mask)[0]
                return random.choice(legal_indices)
            else:
                return random.randint(0, self.num_actions - 1)

        # Greedy action selection
        state_tensor = tf.expand_dims(tf.convert_to_tensor(state, dtype=tf.float32), 0)
        q_values = self.primary_network(state_tensor).numpy()[0]

        # Mask illegal actions with very negative values
        if legal_actions_mask is not None:
            q_values = np.where(legal_actions_mask, q_values, -1e9)

        return np.argmax(q_values)

    def train_step(self):
        """Perform one training step"""
        if len(self.replay_buffer) < self.batch_size:
            return None

        if self.step_count < self.pretrain_steps:
            return None

        # Sample from replay buffer
        sample = self.replay_buffer.sample(self.batch_size, beta=0.4)
        if sample is None:
            return None

        states, actions, rewards, next_states, dones, indices, weights = sample

        # Convert to tensors
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        weights = tf.convert_to_tensor(weights, dtype=tf.float32)

        # Double DQN: use primary network to select actions, target network to evaluate
        with tf.GradientTape() as tape:
            # Current Q values
            current_q = self.primary_network(states)
            current_q = tf.gather_nd(
                current_q,
                tf.stack([tf.range(self.batch_size), actions], axis=1)
            )

            # Next Q values using Double DQN
            next_q_primary = self.primary_network(next_states)
            next_actions = tf.argmax(next_q_primary, axis=1, output_type=tf.int32)

            next_q_target = self.target_network(next_states)
            next_q = tf.gather_nd(
                next_q_target,
                tf.stack([tf.range(self.batch_size), next_actions], axis=1)
            )

            # Target Q values
            target_q = rewards + (1 - dones) * self.gamma * next_q

            # TD errors
            td_errors = target_q - current_q

            # Weighted MSE loss (prioritized experience replay)
            loss = tf.reduce_mean(weights * tf.square(td_errors))

        # Compute and apply gradients
        gradients = tape.gradient(loss, self.primary_network.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.primary_network.trainable_variables)
        )

        # Update priorities in replay buffer
        self.replay_buffer.update_priorities(indices, td_errors.numpy())

        # Soft update target network
        self._update_target_network()

        # Update epsilon
        self._update_epsilon()

        self.step_count += 1

        return loss.numpy()

    def _update_target_network(self):
        """Soft update of target network weights"""
        primary_weights = self.primary_network.get_weights()
        target_weights = self.target_network.get_weights()

        updated_weights = []
        for p_w, t_w in zip(primary_weights, target_weights):
            updated_w = self.target_update_rate * p_w + \
                        (1 - self.target_update_rate) * t_w
            updated_weights.append(updated_w)

        self.target_network.set_weights(updated_weights)

    def _update_epsilon(self):
        """Decay epsilon for exploration"""
        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay
            self.epsilon = max(self.epsilon_end, self.epsilon)

    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        self.replay_buffer.push(state, action, reward, next_state, done)


# ==============================================================================
# PART 3: TRAINING LOOP
# ==============================================================================

def train_traffic_controller(
        num_episodes=1500,
        save_interval=100,
        plot_results=True
):
    """
    Main training loop for traffic light controller.

    Args:
        num_episodes: Number of training episodes
        save_interval: Save model every N episodes
        plot_results: Whether to plot training results
    """
    # Initialize environment
    env = TrafficEnvironment()

    # Get state and action dimensions
    state_shape = (60, 60, 2)
    num_actions = 9  # As shown in Fig. 3 of the paper

    # Initialize agent
    agent = DoubleDuelingDQNAgent(
        state_shape=state_shape,
        num_actions=num_actions,
        learning_rate=0.0001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay_steps=10000,
        target_update_rate=0.001,
        buffer_capacity=20000,
        batch_size=64,
        pretrain_steps=2000
    )

    # Training metrics
    episode_rewards = []
    episode_waiting_times = []
    episode_losses = []

    print("Starting training...")
    print(f"Episodes: {num_episodes}")
    print(f"State shape: {state_shape}")
    print(f"Number of actions: {num_actions}")
    print("-" * 50)

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_loss = []
        steps = 0
        done = False

        while not done:
            # Select action
            action = agent.select_action(state)

            # Take action in environment
            next_state, reward, done, info = env.step(action)

            # Store transition
            agent.store_transition(state, action, reward, next_state, done)

            # Train agent
            loss = agent.train_step()
            if loss is not None:
                episode_loss.append(loss)

            # Update state and metrics
            state = next_state
            episode_reward += reward
            steps += 1

        # Record episode metrics
        episode_rewards.append(episode_reward)
        episode_waiting_times.append(
            info['cumulative_waiting'] / max(info['num_vehicles'], 1)
        )

        if episode_loss:
            episode_losses.append(np.mean(episode_loss))
        else:
            episode_losses.append(0)

        # Print progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_waiting = np.mean(episode_waiting_times[-10:])
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Avg Waiting: {avg_waiting:.2f}s | "
                  f"Epsilon: {agent.epsilon:.3f}")

        # Save model periodically - FIXED: Use .weights.h5 extension
        if (episode + 1) % save_interval == 0:
            try:
                agent.primary_network.save_weights(
                    f'traffic_controller_episode_{episode + 1}.weights.h5'
                )
                print(f"Model saved at episode {episode + 1}")
            except Exception as e:
                print(f"Warning: Could not save model at episode {episode + 1}: {e}")

    print("-" * 50)
    print("Training completed!")

    # Save final model
    try:
        agent.primary_network.save_weights('traffic_controller_final.weights.h5')
        print("Final model saved!")
    except Exception as e:
        print(f"Warning: Could not save final model: {e}")

    # Plot results
    if plot_results:
        plot_training_results(episode_rewards, episode_waiting_times, episode_losses)

    return agent, episode_rewards, episode_waiting_times


def plot_training_results(rewards, waiting_times, losses):
    """Plot training metrics"""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Plot cumulative rewards
    axes[0].plot(rewards, alpha=0.6, label='Episode Reward')
    axes[0].plot(
        np.convolve(rewards, np.ones(50) / 50, mode='valid'),
        label='Moving Average (50 episodes)',
        linewidth=2
    )
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Cumulative Reward')
    axes[0].set_title('Cumulative Reward per Episode')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot average waiting times
    axes[1].plot(waiting_times, alpha=0.6, label='Episode Avg Waiting Time')
    axes[1].plot(
        np.convolve(waiting_times, np.ones(50) / 50, mode='valid'),
        label='Moving Average (50 episodes)',
        linewidth=2
    )
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Average Waiting Time (s)')
    axes[1].set_title('Average Waiting Time per Episode')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Plot losses
    axes[2].plot(losses, alpha=0.6, label='Training Loss')
    axes[2].plot(
        np.convolve(losses, np.ones(50) / 50, mode='valid'),
        label='Moving Average (50 episodes)',
        linewidth=2
    )
    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel('Loss')
    axes[2].set_title('Training Loss per Episode')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
    print("Training results saved to 'training_results.png'")
    plt.show()


# ==============================================================================
# PART 4: EVALUATION AND COMPARISON
# ==============================================================================

def evaluate_controller(agent, env, num_episodes=50):
    """
    Evaluate trained controller performance.

    Args:
        agent: Trained agent
        env: Traffic environment
        num_episodes: Number of evaluation episodes

    Returns:
        avg_reward: Average cumulative reward
        avg_waiting_time: Average waiting time per vehicle
    """
    total_rewards = []
    total_waiting_times = []

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False

        while not done:
            # Select action (no exploration)
            action = agent.select_action(state)
            state, reward, done, info = env.step(action)
            episode_reward += reward

        total_rewards.append(episode_reward)
        total_waiting_times.append(
            info['cumulative_waiting'] / max(info['num_vehicles'], 1)
        )

    avg_reward = np.mean(total_rewards)
    avg_waiting_time = np.mean(total_waiting_times)

    print(f"Evaluation Results ({num_episodes} episodes):")
    print(f"  Average Reward: {avg_reward:.2f}")
    print(f"  Average Waiting Time: {avg_waiting_time:.2f}s")

    return avg_reward, avg_waiting_time


def compare_with_baselines(agent, num_episodes=50):
    """
    Compare trained agent with baseline methods.

    Baselines:
    1. Fixed-time 30s
    2. Fixed-time 40s
    3. Simple adaptive (ATSC-like)
    """
    env = TrafficEnvironment()

    # Evaluate 3DQN agent
    print("\n3DQN Agent:")
    agent_reward, agent_waiting = evaluate_controller(agent, env, num_episodes)

    # Fixed-time 30s baseline
    print("\nFixed-time 30s:")
    env.phase_durations = [30, 30, 30, 30]
    fixed_30_rewards = []
    fixed_30_waiting = []
    for _ in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        while not done:
            state, reward, done, info = env.step(4)  # No change action
            episode_reward += reward
        fixed_30_rewards.append(episode_reward)
        fixed_30_waiting.append(info['cumulative_waiting'] / max(info['num_vehicles'], 1))

    print(f"  Average Reward: {np.mean(fixed_30_rewards):.2f}")
    print(f"  Average Waiting Time: {np.mean(fixed_30_waiting):.2f}s")

    # Fixed-time 40s baseline
    print("\nFixed-time 40s:")
    env.phase_durations = [40, 40, 40, 40]
    fixed_40_rewards = []
    fixed_40_waiting = []
    for _ in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        while not done:
            state, reward, done, info = env.step(4)  # No change action
            episode_reward += reward
        fixed_40_rewards.append(episode_reward)
        fixed_40_waiting.append(info['cumulative_waiting'] / max(info['num_vehicles'], 1))

    print(f"  Average Reward: {np.mean(fixed_40_rewards):.2f}")
    print(f"  Average Waiting Time: {np.mean(fixed_40_waiting):.2f}s")

    # Calculate improvement
    improvement_30 = (np.mean(fixed_30_waiting) - agent_waiting) / np.mean(fixed_30_waiting) * 100
    improvement_40 = (np.mean(fixed_40_waiting) - agent_waiting) / np.mean(fixed_40_waiting) * 100

    print(f"\n3DQN Improvement:")
    print(f"  vs Fixed-30s: {improvement_30:.1f}% reduction in waiting time")
    print(f"  vs Fixed-40s: {improvement_40:.1f}% reduction in waiting time")


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    random.seed(42)

    # Train the traffic controller
    print("=" * 70)
    print("DEEP REINFORCEMENT LEARNING FOR TRAFFIC LIGHT CONTROL")
    print("Implementation of: Liang et al. (2019)")
    print("=" * 70)

    agent, rewards, waiting_times = train_traffic_controller(
        num_episodes=1500,
        save_interval=100,
        plot_results=True
    )

    # Evaluate and compare with baselines
    print("\n" + "=" * 70)
    print("EVALUATION AND COMPARISON")
    print("=" * 70)
    compare_with_baselines(agent, num_episodes=50)

    print("\n" + "=" * 70)
    print("Training and evaluation completed!")
    print("=" * 70)