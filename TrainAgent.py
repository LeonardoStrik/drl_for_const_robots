import gymnasium as gym
from stable_baselines3 import DQN, PPO, SAC
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from Env import make_env, make_continuous_env
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

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=1,
        tensorboard_log="./logs/tensorboard/",
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


def train_ppo_continuous():
    """
    Train a PPO agent with continuous [v, omega] actions on ContinuousSimEnv.
    """

    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    env = DummyVecEnv([lambda: Monitor(make_continuous_env())])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_env = DummyVecEnv([lambda: Monitor(make_continuous_env())])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/best_model_continuous",
        log_path="./logs/",
        eval_freq=5000,
        deterministic=True,
        render=False,
        n_eval_episodes=10,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./models/checkpoints/",
        name_prefix="ppo_continuous",
        save_vecnormalize=True,
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        target_kl=0.02,
        verbose=1,
        tensorboard_log="./logs/tensorboard/",
        use_sde=True,
        policy_kwargs=dict(squash_output=True),
    )

    print("Starting training...")
    print(f"Total timesteps: 2,000,000")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    model.learn(
        total_timesteps=500000 * 4,
        callback=[eval_callback, checkpoint_callback],
        log_interval=100,
        progress_bar=True,
    )

    model.save("models/ppo_continuous_final")
    env.save("models/ppo_continuous_vecnormalize.pkl")
    print("Training complete! Model saved to 'models/ppo_continuous_final'")

    return model


def train_sac_continuous():
    """
    Train a SAC agent with continuous [v, omega] actions on ContinuousSimEnv.
    SAC's actor squashes its output with tanh by construction, so it doesn't
    need the use_sde/squash_output/target_kl workarounds the PPO variant did.
    """

    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    env = DummyVecEnv([lambda: Monitor(make_continuous_env())])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_env = DummyVecEnv([lambda: Monitor(make_continuous_env())])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/best_model_sac_continuous",
        log_path="./logs/",
        eval_freq=5000,
        deterministic=True,
        render=False,
        n_eval_episodes=10,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./models/checkpoints/",
        name_prefix="sac_continuous",
        save_vecnormalize=True,
    )

    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        buffer_size=200000,
        learning_starts=5000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        verbose=1,
        tensorboard_log="./logs/tensorboard/",
    )

    print("Starting training...")
    print(f"Total timesteps: 500,000")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    model.learn(
        total_timesteps=500000,
        callback=[eval_callback, checkpoint_callback],
        log_interval=100,
        progress_bar=True,
    )

    model.save("models/sac_continuous_final")
    env.save("models/sac_continuous_vecnormalize.pkl")
    print("Training complete! Model saved to 'models/sac_continuous_final'")

    return model


_TRAIN_FNS = {
    "dqn": train_dqn_pathfinding,
    "ppo": train_ppo_pathfinding,
    "ppo_continuous": train_ppo_continuous,
    "sac_continuous": train_sac_continuous,
}


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
