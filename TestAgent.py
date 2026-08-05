import argparse
import os
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from stable_baselines3 import DQN, PPO

from Env import SimEnv, make_env

_ALGOS = {"dqn": DQN, "ppo": PPO}
_DEFAULT_MODEL_PATHS = {
    "dqn": "models/dqn_pathfinding_final",
    "ppo": "models/ppo_pathfinding_final",
}


def run_episode(env: SimEnv, model, seed=None, deterministic: bool = True) -> dict:
    obs, info = env.reset(seed=seed)

    agent_trajectory = [info["agent_position"].copy()]
    object_trajectory = [info["object_position"].copy()]
    obj_dims = np.array(info["object_dimensions"], dtype=np.float32)
    obj_target = info["object_target_position"].copy()

    total_reward = 0.0
    terminated = truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        agent_trajectory.append(info["agent_position"].copy())
        object_trajectory.append(info["object_position"].copy())
        total_reward += reward

    return {
        "agent_trajectory": np.array(agent_trajectory, dtype=np.float32),
        "object_trajectory": np.array(object_trajectory, dtype=np.float32),
        "obj_dims": obj_dims,
        "obj_target": obj_target.astype(np.float32),
        "total_reward": total_reward,
        "steps": len(agent_trajectory) - 1,
        "terminated": terminated,
        "truncated": truncated,
    }


def plot_episode(
    env: SimEnv,
    result: dict,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
):
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(projection="3d")

    ax.voxels(env.obstacle_matrix, facecolors=[0, 0, 0, 0.5], edgecolors=[1, 1, 1])

    # Cell-center offset so paths/markers sit inside cells
    agent_path = result["agent_trajectory"] + 0.5
    ax.plot3D(
        agent_path[:, 0],
        agent_path[:, 1],
        agent_path[:, 2],
        color="tab:blue",
        linewidth=2,
        marker="o",
        markersize=3,
        label="Agent path",
    )
    ax.scatter(*agent_path[0], color="green", s=90, marker="^", label="Agent start")
    ax.scatter(*agent_path[-1], color="red", s=90, marker="X", label="Agent end")

    obj_center_offset = result["obj_dims"] / 2.0
    obj_start = result["object_trajectory"][0] + obj_center_offset
    obj_end = result["object_trajectory"][-1] + obj_center_offset
    obj_target = result["obj_target"] + obj_center_offset

    ax.scatter(*obj_start, color="darkorange", s=110, marker="s", label="Object start")
    ax.scatter(*obj_end, color="gold", s=110, marker="D", label="Object end")
    ax.scatter(*obj_target, color="magenta", s=130, marker="*", label="Object target")
    ax.plot3D(
        [obj_end[0], obj_target[0]],
        [obj_end[1], obj_target[1]],
        [obj_end[2], obj_target[2]],
        color="magenta",
        linestyle="--",
        linewidth=1,
        alpha=0.6,
    )

    xlim, ylim, zlim = env.matrix.shape
    ax.set(
        xlabel="x",
        ylabel="y",
        zlabel="z",
        xlim=[0, xlim],
        ylim=[0, ylim],
        zlim=[0, zlim],
        xticks=range(xlim + 1),
        yticks=range(ylim + 1),
        zticks=range(zlim + 1),
        xticklabels=[],
        yticklabels=[],
        zticklabels=[],
    )
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=8)
    if title:
        ax.set_title(title)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    return fig


def animate_episode(
    env: SimEnv,
    result: dict,
    save_path: str,
    fps: int = 10,
    dpi: int = 150,
    title: Optional[str] = None,
):
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(projection="3d")

    ax.voxels(env.obstacle_matrix, facecolors=[0, 0, 0, 0.5], edgecolors=[1, 1, 1])

    agent_path = result["agent_trajectory"] + 0.5
    obj_center_offset = result["obj_dims"] / 2.0
    object_path = result["object_trajectory"] + obj_center_offset
    obj_target = result["obj_target"] + obj_center_offset

    xlim, ylim, zlim = env.matrix.shape
    ax.set(
        xlabel="x",
        ylabel="y",
        zlabel="z",
        xlim=[0, xlim],
        ylim=[0, ylim],
        zlim=[0, zlim],
        xticks=range(xlim + 1),
        yticks=range(ylim + 1),
        zticks=range(zlim + 1),
        xticklabels=[],
        yticklabels=[],
        zticklabels=[],
    )
    ax.set_aspect("equal")
    if title:
        ax.set_title(title)

    ax.scatter(*obj_target, color="magenta", s=130, marker="D", label="Object target")
    (agent_line,) = ax.plot3D(
        [], [], [], color="tab:blue", linewidth=2, label="Agent path"
    )
    agent_point = ax.scatter(
        [], [], [], color="green", s=90, marker="o", label="Agent"  # type: ignore
    )
    object_point = ax.scatter(
        [], [], [], color="gold", s=110, marker="D", label="Object"  # type: ignore
    )
    ax.legend(loc="upper left", fontsize=8)

    n_frames = len(agent_path)

    def update(frame):
        agent_line.set_data_3d(  # type: ignore[attr-defined]
            agent_path[: frame + 1, 0],
            agent_path[: frame + 1, 1],
            agent_path[: frame + 1, 2],
        )
        agent_point._offsets3d = (  # type: ignore[attr-defined]
            agent_path[frame : frame + 1, 0],
            agent_path[frame : frame + 1, 1],
            agent_path[frame : frame + 1, 2],
        )
        object_point._offsets3d = (  # type: ignore[attr-defined]
            object_path[frame : frame + 1, 0],
            object_path[frame : frame + 1, 1],
            object_path[frame : frame + 1, 2],
        )
        return agent_line, agent_point, object_point

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps, blit=False)

    try:
        anim.save(save_path, writer=FFMpegWriter(fps=fps), dpi=dpi)
    except FileNotFoundError as exc:
        raise RuntimeError("Couldn't find FFMPEG on PATH") from exc
    finally:
        plt.close(fig)

    print(f"Saved animation to {save_path}")


def test_and_visualize(
    model_path: str,
    algo: str = "ppo",
    num_episodes: int = 5,
    out_dir: str = "test_plots",
    seed: Optional[int] = None,
    show: bool = True,
    animate: bool = False,
    fps: int = 10,
):
    os.makedirs(out_dir, exist_ok=True)
    model_cls = _ALGOS[algo]
    model = model_cls.load(model_path)

    for ep in range(num_episodes):
        env = make_env()
        ep_seed = None if seed is None else seed + ep
        result = run_episode(env, model, seed=ep_seed)

        print(
            f"Episode {ep + 1}: steps={result['steps']} "
            f"reward={result['total_reward']:.2f} "
            f"terminated={result['terminated']} truncated={result['truncated']}"
        )

        title = (
            f"Episode {ep + 1} | reward={result['total_reward']:.1f} | "
            f"steps={result['steps']} | delivered={result['terminated']}"
        )
        save_path = os.path.join(out_dir, f"episode_{ep + 1}.png")
        plot_episode(env, result, title=title, save_path=save_path)

        if animate:
            video_path = os.path.join(out_dir, f"episode_{ep + 1}.mp4")
            animate_episode(env, result, video_path, fps=fps, title=title)

        env.close()

    if show:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test and visualise a trained agent")
    parser.add_argument(
        "--algo",
        type=str,
        default="dqn",
        choices=sorted(_ALGOS.keys()),
        help="Which SB3 algorithm class the model was trained/saved with",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to the saved model (defaults to models/<algo>_pathfinding_final)",
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--out-dir", type=str, default="test_plots")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument(
        "--animate", action="store_true", help="Also save an mp4 of each episode"
    )
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    model_path = args.model_path or _DEFAULT_MODEL_PATHS[args.algo]

    test_and_visualize(
        model_path,
        algo=args.algo,
        num_episodes=args.episodes,
        out_dir=args.out_dir,
        seed=args.seed,
        show=not args.no_show,
        animate=args.animate,
        fps=args.fps,
    )
