import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
import os
from typing import Dict, Any, Tuple, List


class MultiAgentTrainer:
    def __init__(
        self,
        env: Any,
        agent_ids: List[str],
        state_size: int,
        action_size: int,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        max_states_per_agent: int = 50000,
    ):
        """
        Multi-agent Q-learning trainer with memory management and comprehensive evaluation.

        Args:
            env: Environment implementing gym-like interface
            agent_ids: List of agent identifiers
            state_size: Dimension of state space
            action_size: Number of discrete actions per agent
            alpha: Learning rate
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_decay: Epsilon decay factor per episode
            max_states_per_agent: Maximum Q-table size per agent
        """
        self.env = env
        self.agent_ids = agent_ids
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.max_states_per_agent = max_states_per_agent

        # Q-learning components
        self.q_tables = {agent_id: {} for agent_id in agent_ids}
        self.visit_counts = {agent_id: {} for agent_id in agent_ids}
        
        # Training tracking
        self.training_history = {
            'episode_rewards': [],
            'epsilon_history': []
        }

        # Validate parameters
        self._validate_hyperparameters()

    def _validate_hyperparameters(self):
        """Validate hyperparameter ranges"""
        if not 0 < self.alpha <= 1:
            raise ValueError(f"Learning rate must be in (0,1], got {self.alpha}")
        if not 0 <= self.gamma <= 1:
            raise ValueError(f"Discount factor must be in [0,1], got {self.gamma}")
        if not 0 <= self.epsilon <= 1:
            raise ValueError(f"Epsilon must be in [0,1], got {self.epsilon}")
        if not 0 < self.epsilon_decay <= 1:
            raise ValueError(f"Epsilon decay must be in (0,1], got {self.epsilon_decay}")

    def _hash_state(self, state: Any) -> Tuple[float, ...]:
        """Convert state to hashable tuple for Q-table indexing"""
        arr = np.round(np.asarray(state, dtype=np.float64).flatten(), 3)
        return tuple(float(x) for x in arr)

    def _manage_memory(self, agent_id: str):
        """Remove least-visited states if Q-table exceeds memory limit"""
        if len(self.q_tables[agent_id]) > self.max_states_per_agent:
            visit_counts = self.visit_counts[agent_id]
            sorted_states = sorted(visit_counts.items(), key=lambda x: x[1])
            states_to_remove = [state for state, _ in sorted_states[:len(sorted_states)//10]]
            
            for state in states_to_remove:
                self.q_tables[agent_id].pop(state, None)
                self.visit_counts[agent_id].pop(state, None)

    def _check_convergence(self, window: int = 100, threshold: float = 1.0) -> bool:
        """Check if training has converged based on reward stability"""
        if len(self.training_history['episode_rewards']) < window:
            return False
        
        recent_totals = [sum(ep.values()) for ep in self.training_history['episode_rewards'][-window:]]
        return np.std(recent_totals) < threshold

    def train(self, episodes: int = 100, log_interval: int = 10, early_stopping: bool = False):
        """
        Train agents using Q-learning with epsilon-greedy exploration.

        Args:
            episodes: Number of training episodes
            log_interval: Episodes between progress logs
            early_stopping: Whether to stop early if converged
        """
        for episode in range(1, episodes + 1):
            state_dict = self.env.reset()
            done = False
            episode_rewards = {agent_id: 0.0 for agent_id in self.agent_ids}

            while not done:
                actions = {}
                
                # Select actions for all agents
                for agent_id in self.agent_ids:
                    state_key = self._hash_state(state_dict[agent_id])
                    
                    if random.random() < self.epsilon:
                        # Explore: random action
                        action = random.randint(0, self.action_size - 1)
                    else:
                        # Exploit: best known action
                        q_values = self.q_tables[agent_id].get(state_key, np.zeros(self.action_size))
                        action = int(np.argmax(q_values))
                    
                    actions[agent_id] = action

                # Take environment step
                next_states, reward_dict, done, _ = self.env.step(actions)

                # Update Q-values for all agents
                for agent_id in self.agent_ids:
                    state_key = self._hash_state(state_dict[agent_id])
                    next_key = self._hash_state(next_states[agent_id])

                    # Track state visits
                    self.visit_counts[agent_id][state_key] = self.visit_counts[agent_id].get(state_key, 0) + 1

                    # Get current and next Q-values
                    q_values = self.q_tables[agent_id].get(state_key, np.zeros(self.action_size))
                    q_next = self.q_tables[agent_id].get(next_key, np.zeros(self.action_size))

                    # Q-learning update
                    old_q = q_values[actions[agent_id]]
                    td_target = reward_dict[agent_id] + self.gamma * np.max(q_next)
                    td_error = td_target - old_q
                    q_values[actions[agent_id]] = old_q + self.alpha * td_error

                    # Store updated Q-values
                    self.q_tables[agent_id][state_key] = q_values
                    episode_rewards[agent_id] += reward_dict[agent_id]

                state_dict = next_states

            # Episode completed - update tracking
            self.training_history['episode_rewards'].append(episode_rewards.copy())
            self.training_history['epsilon_history'].append(self.epsilon)
            self.epsilon *= self.epsilon_decay

            # Periodic logging
            if episode % log_interval == 0:
                avg_reward = np.mean([sum(ep.values()) for ep in self.training_history['episode_rewards'][-log_interval:]])
                print(f"Episode {episode}: Avg Reward (last {log_interval}): {avg_reward:.2f}, ε: {self.epsilon:.3f}")

            # Periodic memory management
            if episode % 100 == 0:
                for agent_id in self.agent_ids:
                    self._manage_memory(agent_id)

            # Check for early stopping
            if early_stopping and episode > 200 and self._check_convergence():
                print(f"Training converged at episode {episode}")
                break

        return self.training_history

    def evaluate(self, episodes: int = 10) -> Dict[str, float]:
        """
        Evaluate trained agents without exploration.

        Args:
            episodes: Number of evaluation episodes

        Returns:
            Dictionary of average rewards per agent
        """
        old_epsilon = self.epsilon
        self.epsilon = 0.0  # No exploration during evaluation

        eval_rewards = {agent_id: [] for agent_id in self.agent_ids}

        for _ in range(episodes):
            state_dict = self.env.reset()
            done = False
            episode_rewards = {agent_id: 0.0 for agent_id in self.agent_ids}

            while not done:
                actions = {}
                for agent_id in self.agent_ids:
                    state_key = self._hash_state(state_dict[agent_id])
                    q_values = self.q_tables[agent_id].get(state_key, np.zeros(self.action_size))
                    actions[agent_id] = int(np.argmax(q_values))

                next_states, rewards, done, _ = self.env.step(actions)

                for agent_id in self.agent_ids:
                    episode_rewards[agent_id] += rewards[agent_id]
                state_dict = next_states

            for agent_id in self.agent_ids:
                eval_rewards[agent_id].append(episode_rewards[agent_id])

        self.epsilon = old_epsilon  # Restore original epsilon
        return {agent_id: np.mean(rewards) for agent_id, rewards in eval_rewards.items()}

    def evaluate_detailed(self, episodes: int = 10) -> Dict[str, Any]:
        """
        Comprehensive evaluation with grid metrics and action analysis.

        Args:
            episodes: Number of evaluation episodes

        Returns:
            Dictionary containing detailed performance metrics
        """
        old_epsilon = self.epsilon
        self.epsilon = 0.0

        eval_data = {
            'episode_rewards': [],
            'grid_balance_scores': [],
            'storage_utilization': [],
            'action_distributions': {agent: [0, 0, 0] for agent in self.agent_ids}
        }

        for episode in range(episodes):
            state_dict = self.env.reset()
            done = False
            ep_rewards = {agent: 0.0 for agent in self.agent_ids}
            balance_errors = []

            while not done:
                actions = {}
                for agent_id in self.agent_ids:
                    state_key = self._hash_state(state_dict[agent_id])
                    q_values = self.q_tables[agent_id].get(state_key, np.zeros(self.action_size))
                    action = int(np.argmax(q_values))
                    actions[agent_id] = action
                    eval_data['action_distributions'][agent_id][action] += 1

                next_states, rewards, done, _ = self.env.step(actions)

                # Track grid performance metrics
                if hasattr(self.env, '_last_imbalance'):
                    balance_errors.append(abs(self.env._last_imbalance))

                for agent_id in self.agent_ids:
                    ep_rewards[agent_id] += rewards[agent_id]
                state_dict = next_states

            eval_data['episode_rewards'].append(ep_rewards)
            if balance_errors:
                eval_data['grid_balance_scores'].append(np.mean(balance_errors))
            if hasattr(self.env, 'storage_level'):
                eval_data['storage_utilization'].append(
                    self.env.storage_level / getattr(self.env, 'max_storage_capacity', 1000)
                )

        self.epsilon = old_epsilon

        # Calculate summary statistics
        return {
            'avg_rewards': {agent: np.mean([ep[agent] for ep in eval_data['episode_rewards']]) 
                           for agent in self.agent_ids},
            'avg_grid_balance': np.mean(eval_data['grid_balance_scores']) if eval_data['grid_balance_scores'] else 0.0,
            'avg_storage_util': np.mean(eval_data['storage_utilization']) if eval_data['storage_utilization'] else 0.0,
            'action_preferences': {agent: np.array(dist) / (sum(dist) if sum(dist) > 0 else 1) 
                                 for agent, dist in eval_data['action_distributions'].items()},
            'detailed_data': eval_data
        }

    def get_training_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive training statistics.

        Returns:
            Dictionary containing training metrics and performance data
        """
        if not self.training_history['episode_rewards']:
            return {"status": "No training data available"}

        total_rewards = [sum(ep.values()) for ep in self.training_history['episode_rewards']]
        
        stats = {
            'total_episodes': len(self.training_history['episode_rewards']),
            'final_epsilon': self.epsilon,
            'q_table_sizes': {agent: len(qtable) for agent, qtable in self.q_tables.items()},
            'total_states_explored': sum(len(qtable) for qtable in self.q_tables.values()),
            'reward_statistics': {
                'mean': np.mean(total_rewards),
                'std': np.std(total_rewards),
                'min': np.min(total_rewards),
                'max': np.max(total_rewards)
            }
        }

        # Calculate improvement if enough episodes
        if len(total_rewards) >= 100:
            initial_avg = np.mean(total_rewards[:50])
            final_avg = np.mean(total_rewards[-50:])
            stats['reward_statistics']['improvement'] = final_avg - initial_avg

        return stats

    def save_model(self, filepath: str):
        """
        Save trained model to file.

        Args:
            filepath: Path where to save the model
        """
        model_data = {
            'q_tables': self.q_tables,
            'training_history': self.training_history,
            'agent_ids': self.agent_ids,
            'hyperparameters': {
                'alpha': self.alpha,
                'gamma': self.gamma,
                'epsilon': self.epsilon,
                'epsilon_decay': self.epsilon_decay
            }
        }
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

    def load_model(self, filepath: str):
        """
        Load trained model from file.

        Args:
            filepath: Path to the saved model
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.q_tables = model_data['q_tables']
        self.training_history = model_data.get('training_history', {})
        print(f"Loaded model with {len(self.q_tables)} agents")

    def analyze_policy(self, test_states: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Analyze learned policies for different grid conditions.

        Args:
            test_states: Dictionary of scenario names to state arrays

        Returns:
            Policy analysis for each scenario and agent
        """
        if test_states is None:
            # Default test scenarios for wind grid
            test_states = {
                'high_wind_low_load': np.array([800, 50, 0.5, 0]),
                'low_wind_high_load': np.array([200, 100, 0.5, 0]),
                'storage_full': np.array([500, 75, 0.9, 0]),
                'storage_empty': np.array([500, 75, 0.1, 0])
            }

        analysis = {}
        action_names = ['Decrease/Discharge', 'Maintain/Hold', 'Increase/Charge']

        for scenario, state in test_states.items():
            analysis[scenario] = {}
            for agent in self.agent_ids:
                state_key = self._hash_state(state)
                q_values = self.q_tables[agent].get(state_key, np.zeros(self.action_size))
                best_action = int(np.argmax(q_values))
                
                analysis[scenario][agent] = {
                    'action': action_names[best_action],
                    'q_values': q_values.tolist(),
                    'confidence': float(np.max(q_values) - np.mean(q_values))
                }

        return analysis


def plot_training_results(training_history, save_path="results/training_plots.png"):
    """
    Create comprehensive training visualization.

    Args:
        training_history: Dictionary containing episode rewards and epsilon history
        save_path: Path where to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Extract data
    episode_rewards = training_history.get('episode_rewards', [])
    epsilon_history = training_history.get('epsilon_history', [])
    
    # Total rewards over time
    total_rewards = [sum(ep.values()) for ep in episode_rewards]
    axes[0, 0].plot(total_rewards, alpha=0.3, color='blue', label='Episode Rewards')
    
    # Moving average
    window = min(50, len(total_rewards) // 10)
    if window > 1:
        moving_avg = np.convolve(total_rewards, np.ones(window) / window, mode='valid')
        axes[0, 0].plot(range(window - 1, len(total_rewards)), moving_avg,
                       color='red', linewidth=2, label=f'{window}-Episode Moving Avg')
    
    axes[0, 0].set_title('Total Rewards Over Time')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Agent-specific rewards
    if episode_rewards:
        agent_names = list(episode_rewards[0].keys())
        colors = ['blue', 'green', 'orange']
        
        for i, agent in enumerate(agent_names):
            agent_rewards = [ep[agent] for ep in episode_rewards]
            window = min(20, len(agent_rewards) // 10)
            if window > 1:
                smoothed = np.convolve(agent_rewards, np.ones(window) / window, mode='valid')
                axes[0, 1].plot(range(window - 1, len(agent_rewards)), smoothed,
                               label=agent.replace('_', ' ').title(),
                               color=colors[i % len(colors)], linewidth=2)
            else:
                axes[0, 1].plot(agent_rewards,
                               label=agent.replace('_', ' ').title(),
                               color=colors[i % len(colors)], alpha=0.7)
        
        axes[0, 1].legend()
        axes[0, 1].set_title('Agent-Specific Performance')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Reward')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Epsilon decay plot
    axes[1, 0].plot(epsilon_history, color='purple', linewidth=2)
    axes[1, 0].set_title('Exploration Rate (Epsilon)')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Epsilon')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Reward histogram
    if total_rewards:
        axes[1, 1].hist(total_rewards, bins=30, alpha=0.7, edgecolor='black', color='skyblue')
        mean_reward = np.mean(total_rewards)
        axes[1, 1].axvline(mean_reward, color='red', linestyle='--', linewidth=2,
                          label=f'Mean: {mean_reward:.2f}')
        axes[1, 1].axvline(np.median(total_rewards), color='green', linestyle='--', linewidth=2,
                          label=f'Median: {np.median(total_rewards):.2f}')
        axes[1, 1].legend()
        axes[1, 1].set_title('Reward Distribution')
        axes[1, 1].set_xlabel('Total Reward')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3)
    
    # Save figure
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[✓] Training plots saved to: {save_path}")