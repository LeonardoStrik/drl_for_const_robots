import gymnasium as gym
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from Env import make_env
import os


def train_dqn_pathfinding():
    """
    Train a DQN agent in the simulation environment
    """

    # Create directories for saving models and logs
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    env = make_env()
    env = Monitor(env)

    # Create evaluation environment
    eval_env = make_env()
    eval_env = Monitor(eval_env)

    # Create callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/best_model",
        log_path="./logs/",
        eval_freq=5000,
        deterministic=True,
        render=False,
        n_eval_episodes=10,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./models/checkpoints/",
        name_prefix="dqn_pathfinding",
    )

    # Create the DQN agent TODO: Validate most of these values
    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=5e-4,
        buffer_size=100000,
        learning_starts=2000,
        batch_size=64,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=2000,
        exploration_fraction=0.4,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        verbose=1,
        tensorboard_log="./logs/tensorboard/",
    )

    print("Starting training...")
    print(f"Total timesteps: 500,000")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Train the agent
    model.learn(
        total_timesteps=500000,
        callback=[eval_callback, checkpoint_callback],
        log_interval=100,
        progress_bar=True,
    )

    # Save the final model
    model.save("models/dqn_pathfinding_final")
    print("Training complete! Model saved to 'models/dqn_pathfinding_final'")

    return model


def train_ppo_pathfinding():
    """
    Train a PPO agent in the simulation environment
    """

    # Create directories for saving models and logs
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    env = make_env()
    env = Monitor(env)

    # Create evaluation environment
    eval_env = make_env()
    eval_env = Monitor(eval_env)

    # Create callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/best_model",
        log_path="./logs/",
        eval_freq=5000,
        deterministic=True,
        render=False,
        n_eval_episodes=10,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./models/checkpoints/",
        name_prefix="ppo_pathfinding",
    )

    print("Starting training...")
    print(f"Total timesteps: 500,000")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Train the agent
    model.learn(
        total_timesteps=500000 * 2,
        callback=[eval_callback, checkpoint_callback],
        log_interval=100,
        progress_bar=True,
    )

    # Save the final model
    model.save("models/ppo_pathfinding_final")
    print("Training complete! Model saved to 'models/ppo_pathfinding_final'")

    return model


_TRAIN_FNS = {"dqn": train_dqn_pathfinding, "ppo": train_ppo_pathfinding}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a pathfinding agent")
    parser.add_argument(
        "--algo",
        type=str,
        default="dqn",
        choices=sorted(_TRAIN_FNS.keys()),
        help="Which SB3 algorithm to train with",
    )
    args = parser.parse_args()

    model = _TRAIN_FNS[args.algo]()
