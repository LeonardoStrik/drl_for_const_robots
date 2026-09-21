import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from typing import Any, List, Optional, Sequence, Tuple, Self
from copy import deepcopy
from enum import Enum
from collections import deque

_RANDOM_TRY_TIMEOUT = 100
_POSITION_HISTORY_LEN = 3
_N_CLOSEST_OBTACLES = 10
_OBS_DIMS = 18 + 3 * _POSITION_HISTORY_LEN + 4 * _N_CLOSEST_OBTACLES
_MIN_TARGET_DIST = 10
_USE_BFS_REWARD = False  # False: Euclidean-distance reward instead of BFS shaping


class _ConstructionItem:
    """
    base class for obstacles and construction objects
    """

    def __init__(
        self, shape: Tuple[int, int, int], origin: Tuple[int, int, int]
    ) -> None:
        self.matrix = np.ones((shape), dtype=np.uint8)
        self.pos = np.array(origin, dtype=np.uint8)

    def copy(self) -> Self:
        return deepcopy(self)


class ConstructionObject(_ConstructionItem):
    """
    class to represent the objects to  be placed
    """

    def __init__(
        self,
        shape: Tuple[int, int, int],
        origin: Tuple[int, int, int],
        target_pos: Tuple[int, int, int],
    ) -> None:
        super().__init__(shape, origin)
        self.target_pos = np.array(target_pos, dtype=np.uint8)
        self.is_carried = False

    def is_at_target(self, threshold=1.0) -> bool:
        """Check if object center is within threshold of its target."""
        dist = float(
            np.linalg.norm(
                self.pos.astype(np.float32) - self.target_pos.astype(np.float32)
            )
        )
        return dist < threshold

    def is_adjacent(self, pos, pickup_range=1.0) -> bool:
        """Check if agent is close enough to pick up the object."""
        dist = float(
            np.linalg.norm(pos.astype(np.float32) - self.pos.astype(np.float32))
        )
        return dist <= pickup_range


class EnvObstacle(_ConstructionItem):
    """
    class to represent obstacles in the area
    """

    def __init__(
        self, shape: Tuple[int, int, int], origin: Tuple[int, int, int]
    ) -> None:
        super().__init__(shape, origin)


class SimEnv(gym.Env):
    """
    class for the simulation environment, contains a list of objects and obstacles
    """

    class Phase(Enum):
        TO_OBJECT = 0
        TO_TARGET = 1

    def __init__(
        self,
        shape: Tuple[int, int, int],
        construction_objects: List[ConstructionObject],
        obstacles: Optional[List[EnvObstacle]] = None,
        agent_pos: Optional[np.ndarray] = None,
        max_steps=200,
    ) -> None:
        super().__init__()

        if obstacles is None:
            obstacles = []
        self.matrix = np.zeros(shape, dtype=np.uint8)

        construction_objects = [obj.copy() for obj in construction_objects]
        obstacles = [obs.copy() for obs in obstacles]

        # Check that all objects and obstacles are within bounds
        for obj in construction_objects:
            self._check_collision(obj)
            for axis in range(3):
                if (
                    obj.target_pos[axis] < 0
                    or obj.target_pos[axis] >= self.matrix.shape[axis]
                ):
                    raise ValueError(
                        f"target_pos {obj.target_pos} out of bounds on axis {axis} "
                        f"vs shape {self.matrix.shape}"
                    )
        for obs in obstacles:
            self._check_collision(obs)
        self._construction_objects: List[ConstructionObject] = construction_objects
        self._obstacles = obstacles
        self._obstacle_matrix = np.clip(
            self.place_objects(obstacles), a_min=None, a_max=1
        )
        # precompute obstacle voxel array for later closest obstacle computation
        self._obstacle_voxels = np.argwhere(self._obstacle_matrix > 0)

        combined = self._obstacle_matrix + self.place_objects(
            self._construction_objects
        )
        if np.any(combined > 1):
            raise ValueError("objects/obstacles overlap")

        self._recompute_distance_fields()

        if agent_pos is not None:
            self.agent_pos = agent_pos
        else:
            self._randomise_agent_pos()
        self._reset_position_history()

        # Action space: 6 actions (no flying!)
        # 0: +x, 1: -x, 2: +y, 3: -y, 4: PICK_UP, 5: DROP
        self.action_space = spaces.Discrete(4)

        # Observation space: 18 base dimensions + 3 per remembered past position
        # + 4 per closest-obstacle slot
        self._max_dist = max(self.matrix.shape)
        self.observation_space = spaces.Box(
            low=-self._max_dist,
            high=self._max_dist,
            shape=(_OBS_DIMS,),
            dtype=np.float32,
        )

        self.max_steps = max_steps
        self.current_step = 0
        self.phase = self.Phase.TO_OBJECT

    def _check_collision(self, obj: _ConstructionItem):
        """
        Checks a given object for if it collides.
        """
        # TODO: implement clean failure without exceptions
        for axis in range(3):
            lo = obj.pos[axis]
            hi = lo + obj.matrix.shape[axis]
            if lo < 0 or hi > self.matrix.shape[axis]:
                raise ValueError(
                    f"Object {obj} out of bounds on axis {axis}: [{lo}, {hi}) vs {self.matrix.shape[axis]}"
                )

    def _is_position_valid(self, position):
        # TODO: implement complex objects
        if np.any(position < 0) or np.any(position == self.matrix.shape):
            return False
        matrix = self.place_objects(self._construction_objects, self._obstacle_matrix)
        x, y, z = position[0], position[1], position[2]
        return matrix[x, y, z] == 0

    def render(self, top_down: bool = False):
        """
        plots the obstacles in black and objects in white in a 3D grid. If
        top_down=True, instead renders a 2D bird's-eye view of the z=0 plane with
        the BFS shaped-reward field for the agent's current phase (distance to
        object, or distance to target once carried) drawn as a heatmap.
        """
        if top_down:
            self._render_top_down()
            return

        ax = plt.figure().add_subplot(projection="3d")
        ax.voxels(
            self._obstacle_matrix,
            facecolors=[0, 0, 0],
            edgecolors=[1, 1, 1],
        )
        ax.voxels(
            self.place_objects(),
            facecolors=[1, 1, 1],
            edgecolors=[0, 0, 0],
        )
        temp_matrix = self.matrix.copy()
        x0, x1 = self.agent_pos[0], self.agent_pos[0] + 1
        y0, y1 = self.agent_pos[1], self.agent_pos[1] + 1
        z0, z1 = self.agent_pos[2], self.agent_pos[2] + 1
        temp_matrix[x0:x1, y0:y1, z0:z1] += 1
        ax.voxels(
            temp_matrix,
            facecolors=[1, 0, 0],
            edgecolors=[1, 1, 1],
        )

        xlim, ylim, zlim = self.matrix.shape  # type: ignore
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

    def _render_top_down(self):
        """
        Bird's-eye view of the z=0 plane: the BFS shaped-reward field currently
        driving the agent (distance-to-object, or distance-to-target once carried)
        as a heatmap, obstacles overlaid in black, and markers for the agent,
        object, and target.
        """
        obj = self._construction_objects[0]
        dist_field = (
            self._dist_to_object
            if self.phase == self.Phase.TO_OBJECT
            else self._dist_to_target
        )
        reward_field = -dist_field

        free = self._obstacle_matrix[:, :, 0] == 0
        size_x, size_y = free.shape
        # NaN out obstacle cells so they don't get colored by the reward colormap;
        # they're drawn separately below instead.
        heatmap = np.where(free, reward_field, np.nan)

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(
            heatmap.T,
            origin="lower",
            cmap="viridis",
            extent=(0, size_x, 0, size_y),
        )
        fig.colorbar(im, ax=ax, label="shaped reward (-BFS distance)")

        obstacle_overlay = np.where(free, np.nan, 1.0)
        ax.imshow(
            obstacle_overlay.T,
            origin="lower",
            cmap="Greys",
            vmin=0,
            vmax=1,
            extent=(0, size_x, 0, size_y),
        )

        agent_xy = self.agent_pos[:2].astype(np.float32) + 0.5
        obj_xy = obj.pos[:2].astype(np.float32) + 0.5
        target_xy = obj.target_pos[:2].astype(np.float32) + 0.5

        ax.scatter(
            *agent_xy,
            color="red",
            s=140,
            marker="o",
            edgecolors="white",
            label="Agent",
            zorder=5,
        )
        ax.scatter(
            *obj_xy,
            color="cyan",
            s=140,
            marker="s",
            edgecolors="black",
            label="Object",
            zorder=5,
        )
        ax.scatter(
            *target_xy,
            color="magenta",
            s=180,
            marker="*",
            edgecolors="black",
            label="Target",
            zorder=5,
        )

        ax.set(
            xlabel="x",
            ylabel="y",
            xticks=range(size_x + 1),
            yticks=range(size_y + 1),
            xticklabels=[],
            yticklabels=[],
            title=f"Top-down view - phase={self.phase.name}",
        )
        ax.set_aspect("equal")
        ax.grid(True, color="white", linewidth=0.5, alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)

    def place_objects(
        self,
        objects: Optional[Sequence[_ConstructionItem]] = None,
        matrix: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        places the construction objects (default) or obstacles in a matrix (default is normal env matrix) and returns the matrix representing this
        """

        if objects is None:
            objects = self._construction_objects
        if matrix is None:
            matrix = self.matrix.copy()
        else:
            matrix = matrix.copy()

        for obj in objects:

            x0, x1 = obj.pos[0], obj.matrix.shape[0] + obj.pos[0]
            y0, y1 = obj.pos[1], obj.matrix.shape[1] + obj.pos[1]
            z0, z1 = obj.pos[2], obj.matrix.shape[2] + obj.pos[2]
            matrix[x0:x1, y0:y1, z0:z1] += obj.matrix
        return matrix

    def find_empty_cell(
        self, matrix, fix_axes=[None, None, None]
    ) -> tuple[np.int64, np.int64, np.int64]:
        """Find an empty cell in the matrix"""
        for _ in range(_RANDOM_TRY_TIMEOUT):
            x = self.np_random.integers(
                0,
                matrix.shape[0],
            )
            y = self.np_random.integers(
                0,
                matrix.shape[1],
            )
            z = self.np_random.integers(
                0,
                matrix.shape[2],
            )
            if fix_axes[0] is not None:
                x = fix_axes[0]
            if fix_axes[1] is not None:
                y = fix_axes[1]
            if fix_axes[2] is not None:
                z = fix_axes[2]
            if matrix[x, y, z] == 0:
                return (x, y, z)
        raise RuntimeError("Couldn't find an empty spot in the matrix")

    def _bfs_distance_field(self, source_xy: Tuple[int, int]) -> np.ndarray:
        """
        4-connected BFS shortest-path distance (in grid steps) from source_xy to every
        free cell in the agent's z=0 plane, respecting obstacles.

        Unreachable cells get the grid's total cell count as a large-but-finite
        penalty, so a stray episode where BFS can't reach a cell degrades gracefully
        instead of blowing up the reward with inf.
        """
        free = self._obstacle_matrix[:, :, 0] == 0
        size_x, size_y = free.shape
        unreachable = size_x * size_y
        dist = np.full((size_x, size_y), unreachable, dtype=np.float32)

        sx, sy = int(source_xy[0]), int(source_xy[1])
        if not free[sx, sy]:
            return dist  # source sits on an obstacle cell; shouldn't happen, fail safe

        dist[sx, sy] = 0
        queue = deque([(sx, sy)])
        while queue:
            x, y = queue.popleft()
            d = dist[x, y]
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if (
                    0 <= nx < size_x
                    and 0 <= ny < size_y
                    and free[nx, ny]
                    and dist[nx, ny] == unreachable
                ):
                    dist[nx, ny] = d + 1
                    queue.append((nx, ny))
        return dist

    def _recompute_distance_fields(self) -> None:
        """
        Rebuilds the BFS distance fields used for reward shaping. Call this whenever
        the object's spawn position or target position changes (construction and
        every reset()).
        """
        obj = self._construction_objects[0]
        self._dist_to_object = self._bfs_distance_field((obj.pos[0], obj.pos[1]))
        self._dist_to_target = self._bfs_distance_field(
            (obj.target_pos[0], obj.target_pos[1])
        )

    def _reset_position_history(self) -> None:
        """
        (Re)fills the agent's position history with copies of its current position,
        so the very first observation of an episode has a well-defined (zero-delta)
        history instead of stale data from a previous episode.
        """
        self._position_history = deque(
            [self.agent_pos.copy() for _ in range(_POSITION_HISTORY_LEN)],
            maxlen=_POSITION_HISTORY_LEN,
        )

    def _randomise_agent_pos(
        self,
    ) -> None:
        matrix = self.place_objects(self._construction_objects, self._obstacle_matrix)
        pos = self.find_empty_cell(matrix, [None, None, 0])
        self.agent_pos = np.array((pos), dtype=np.uint8)

    def _get_closest_obstacle_info(self, agent_pos):
        """
        Get array of distances and directions to closest obstacles from agent.
        """
        # TODO: include objects and other robots in this logic?
        # calculate vectors and distances to each occupied voxel
        diffs = self._obstacle_voxels - agent_pos
        dists = np.linalg.norm(diffs, axis=1)
        # Handle fewer occupied voxels than _N_CLOSEST_OBSTACLES
        n_found = min(len(dists), _N_CLOSEST_OBTACLES)
        if n_found > 0:
            # find the n_found closest occupied voxels
            idxs = np.argpartition(dists, n_found - 1)[:n_found]
            closest_diffs = diffs[idxs]
            closest_dists = dists[idxs]

            # sort them correctly
            order = np.argsort(closest_dists)
            closest_diffs = closest_diffs[order]
            closest_dists = closest_dists[order]

            # compute unit directions
            directions = np.divide(
                closest_diffs,
                closest_dists[:, None],
                out=np.zeros_like(closest_diffs, dtype=np.float64),
                where=closest_dists[:, None] > 0,
            )
            found = np.column_stack([closest_dists, directions]).ravel()
        else:
            found = np.empty(0, dtype=np.float32)

        # Pad missing slots with max_dist/zero-direction so the output always has a fixed size.
        n_missing = _N_CLOSEST_OBTACLES - n_found
        if n_missing > 0:
            padding = np.zeros((n_missing, 4), dtype=np.float32)
            padding[:, 0] = self._max_dist
            found = np.concatenate([found, padding.ravel()])

        return found

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> Tuple[Any, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self.phase = self.Phase.TO_OBJECT
        self.current_step = 0
        obj = self._construction_objects[0]
        obj.is_carried = False
        obj.pos = np.array(
            self.find_empty_cell(self._obstacle_matrix, [None, None, 0]), dtype=np.uint8
        )
        for _ in range(_RANDOM_TRY_TIMEOUT):
            obj.target_pos = np.array(
                self.find_empty_cell(self._obstacle_matrix, [None, None, 1]),
                dtype=np.uint8,
            )
            if (
                float(
                    np.linalg.norm(
                        obj.pos.astype(np.float32) - obj.target_pos.astype(np.float32)
                    )
                )
                >= _MIN_TARGET_DIST
            ):
                break

        self._randomise_agent_pos()
        self._recompute_distance_fields()
        self._reset_position_history()
        return self._get_obs(), self._get_info()

    def _get_obs(self):
        """
        Get current observation (_OBS_DIMS dimensions).
        """

        agent_pos = self.agent_pos.astype(np.float32)
        obj = self._construction_objects[0]
        obj_pos = obj.pos.astype(np.float32)
        obj_target_pos = obj.target_pos.astype(np.float32)

        dist_agent_to_obj = float(np.linalg.norm(agent_pos - obj_pos))
        dist_obj_to_target = float(np.linalg.norm(obj_pos - obj_target_pos))
        is_carrying = 1.0 if obj.is_carried else 0.0

        obstacles = self._get_closest_obstacle_info(agent_pos)

        # Normalized direction from agent to object target
        diff_to_obj_target = obj_target_pos - agent_pos
        norm = float(np.linalg.norm(diff_to_obj_target))
        dir_to_obj_target = (
            diff_to_obj_target / norm if norm > 0 else np.zeros(3, dtype=np.float32)
        )
        # TODO: fix calculation of distance to object to not use origin
        obs = np.concatenate(
            [
                agent_pos,  # 0-2
                obj_pos,  # 3-5
                obj_target_pos,  # 6-8
                np.array([dist_agent_to_obj], dtype=np.float32),  # 9
                np.array([dist_obj_to_target], dtype=np.float32),  # 10
                np.array([is_carrying], dtype=np.float32),  # 11
                obj.matrix.shape,  # 12-14
                dir_to_obj_target,  # 15-17
                obstacles,  # 18-57
            ]
        ).astype(np.float32)

        # Optionally add position history if this is configured
        if _POSITION_HISTORY_LEN != 0:
            position_history = np.concatenate(
                [
                    agent_pos - hist_pos.astype(np.float32)
                    for hist_pos in self._position_history
                ]
            )
            obs = np.concatenate(
                [
                    obs,
                    position_history,
                ]  # 22 to 22+3*_POSITION_HISTORY_LEN-1
            )
        return obs

    def _get_info(self):
        """
        Get additional info complementing the observation from _get_obs().
        """
        obj = self._construction_objects[
            0
        ]  # TODO: add support for multiple objects/agents
        return {
            "agent_position": self.agent_pos.copy(),
            "object_position": obj.pos.copy(),
            "object_target_position": obj.target_pos.copy(),
            "object_dimensions": deepcopy(obj.matrix.shape),
            "is_carrying": obj.is_carried,
            "dist_agent_to_object": float(
                np.linalg.norm(
                    self.agent_pos.astype(np.float32) - obj.pos.astype(np.float32)
                )
            ),
            "dist_object_to_target": float(
                np.linalg.norm(
                    obj.pos.astype(np.float32) - obj.target_pos.astype(np.float32)
                )
            ),
            "steps": self.current_step,
            "obstacles": [o.copy() for o in self._obstacles],
            "num_obstacles": len(self._obstacles),
        }

    def step(self, action):
        """
        Function that performs each step of the simulation. Also checks whether the step is valid.
        """
        # TODO: multiple agents/objects
        self.current_step += 1
        old_pos = self.agent_pos.copy()
        obj = self._construction_objects[0]

        collision = False
        invalid_action = False
        reward = 0.0
        terminated = False

        # MOVEMENT  (actions 0-3)
        if action in (0, 1, 2, 3):
            movement = np.zeros(3, dtype=np.int32)
            if action == 0:
                movement[0] = 1
            elif action == 1:
                movement[0] = -1
            elif action == 2:
                movement[1] = 1
            elif action == 3:
                movement[1] = -1
            new_pos = np.clip(self.agent_pos + movement, 0, self.matrix.shape).astype(
                np.uint8
            )

            if not self._is_position_valid(new_pos):
                collision = True
                self.agent_pos = old_pos
            else:
                self.agent_pos = new_pos

        self._position_history.append(old_pos)

        # for later, first implement basics
        # # PICK UP (action 4)
        # elif action == 4:
        #     if obj.is_carried:
        #         invalid_action = True
        #     elif not obj.is_agent_adjacent(self.agent_pos, self.pickup_range):
        #         invalid_action = True
        #     else:
        #         # Check if carrying the object here would cause cuboid collision
        #         if self._is_colliding_cuboid(self.agent_pos, obj.dimensions):
        #             collision = True
        #         else:
        #             obj.is_carried = True
        #             obj.snap_to_agent(self.agent_pos)
        #             self.phase = 1
        #             reward += 20.0

        # # --- DROP (action 5) ---
        # elif action == 5:
        #     if not obj.is_carried:
        #         invalid_action = True
        #     else:
        #         obj.is_carried = False
        #         obj.position = self.agent_pos.copy()

        #         if obj.is_at_target(self.drop_threshold):
        #             terminated = True
        #             reward += 100.0
        #         else:
        #             reward -= 20.0
        #             self.phase = 0

        # auto-pickup object - TEMPORARY TODO: remove this
        if obj.is_adjacent(self.agent_pos, 2.1):
            obj.is_carried = True
            self.phase = self.Phase.TO_TARGET
        if obj.is_carried:
            obj.pos = (self.agent_pos + [0, 0, 1]).astype(np.uint8)

        # auto-dropoff - TEMPORARY TODO: remove this
        if obj.is_adjacent(obj.target_pos, 1.9) and obj.is_carried:
            terminated = True
            reward += 100

        # REWARD: BFS-shaped distance (obstacle-aware, only 2D) or plain Euclidean
        # distance, depending on _USE_BFS_REWARD. The BFS path also gets a small
        # bump to cells that are directly adjacent rather than diagonally adjacent.
        agent_x, agent_y = int(self.agent_pos[0]), int(self.agent_pos[1])
        if self.phase == self.Phase.TO_OBJECT:
            goal_x, goal_y = float(obj.pos[0]), float(obj.pos[1])
            bfs_dist = float(self._dist_to_object[agent_x, agent_y])
        else:
            goal_x, goal_y = float(obj.target_pos[0]), float(obj.target_pos[1])
            bfs_dist = float(self._dist_to_target[agent_x, agent_y])
        euclidean_dist = float(np.hypot(agent_x - goal_x, agent_y - goal_y))

        if _USE_BFS_REWARD:
            reward -= bfs_dist
            reward -= 0.01 * euclidean_dist
        else:
            reward -= euclidean_dist

        if collision:
            reward -= 10.0

        if invalid_action:
            reward -= 2.0

        reward -= 0.1

        # Penalize immediately reversing the last move/wiggling.
        # only if not collision, otherwise not actually a wiggle
        if (
            not collision
            and len(self._position_history) >= 2
            and np.array_equal(self.agent_pos, self._position_history[-2])
        ):
            reward -= 1.0

        truncated = bool(self.current_step >= self.max_steps)

        observation = self._get_obs()
        info = self._get_info()
        info["collision"] = collision
        info["invalid_action"] = invalid_action
        info["phase"] = self.phase
        info["is_carrying"] = obj.is_carried

        return observation, float(reward), terminated, truncated, info


def make_env() -> SimEnv:
    """
    Make an env with a fixed set of obstacles TODO: add support for obstacle randomisation, add scene validation
    """
    const_objs = [
        ConstructionObject((1, 1, 1), (0, 0, 0), (40, 28, 1)),
    ]
    obstacles = [
        EnvObstacle((6, 2, 2), (0, 16, 0)),
        EnvObstacle((2, 12, 2), (16, 0, 0)),
        EnvObstacle((2, 12, 2), (26, 17, 0)),
        EnvObstacle((2, 12, 2), (14, 17, 0)),
    ]
    # obstacles = [
    #     EnvObstacle((1, 1, 1), (0, 8, 0)),
    #     EnvObstacle((1, 1, 1), (8, 0, 0)),
    #     EnvObstacle((1, 1, 1), (13, 6, 0)),
    #     EnvObstacle((1, 1, 1), (7, 9, 0)),
    #     EnvObstacle((1, 1, 1), (2, 2, 0)),
    #     EnvObstacle((1, 1, 1), (2, 12, 0)),
    #     EnvObstacle((1, 1, 1), (5, 4, 0)),
    #     EnvObstacle((1, 1, 1), (5, 13, 0)),
    #     EnvObstacle((1, 1, 1), (10, 3, 0)),
    #     EnvObstacle((1, 1, 1), (10, 10, 0)),
    #     EnvObstacle((1, 1, 1), (15, 2, 0)),
    #     EnvObstacle((1, 1, 1), (15, 9, 0)),
    #     EnvObstacle((1, 1, 1), (17, 13, 0)),
    #     EnvObstacle((1, 1, 1), (3, 6, 0)),
    # ]

    return SimEnv((41, 29, 5), const_objs, obstacles)
