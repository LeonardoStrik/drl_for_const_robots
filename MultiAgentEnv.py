from copy import deepcopy
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding
from pettingzoo.utils.env import ParallelEnv

from Env import (
    ConstructionAgent,
    ConstructionObject,
    EnvObstacle,
    SimEnv,
    _RANDOM_TRY_TIMEOUT,
    _POSITION_HISTORY_LEN,
    _N_CLOSEST_OBTACLES,
    _OBS_DIMS,
    _MIN_TARGET_DIST,
    _USE_BFS_REWARD,
    _V_MAX,
    _OMEGA_MAX,
    _DT,
    _place_objects_in,
    _find_empty_cell,
    _bfs_distance_field_2d,
    _check_bounds,
    _k_nearest_info,
)

_N_CLOSEST_AGENTS_CAP = 8
_AGENT_COLLISION_RADIUS = 1.0


class MultiAgentSimEnv(ParallelEnv):
    """Non-holonomic multi-agent environment; each agent has its own object/target."""

    metadata = {"name": "multi_agent_sim_env_v0"}

    def __init__(
        self,
        shape: Tuple[int, int, int],
        construction_objects: List[ConstructionObject],
        agents: List[ConstructionAgent],
        obstacles: Optional[List[EnvObstacle]] = None,
        max_steps: int = 200,
    ) -> None:
        if len(construction_objects) != len(agents):
            raise ValueError("MultiAgentSimEnv needs one construction object per agent")
        if obstacles is None:
            obstacles = []

        self.render_mode = None
        self.matrix = np.zeros(shape, dtype=np.uint8)
        self.possible_agents = [f"agent_{i}" for i in range(len(agents))]
        self.agents = list(self.possible_agents)

        self._robots: Dict[str, ConstructionAgent] = {
            aid: ag.copy() for aid, ag in zip(self.possible_agents, agents)
        }
        self._objects: Dict[str, ConstructionObject] = {
            aid: obj.copy()
            for aid, obj in zip(self.possible_agents, construction_objects)
        }
        self._obstacles = [o.copy() for o in obstacles]

        for obj in self._objects.values():
            _check_bounds(obj, self.matrix.shape)
            for axis in range(3):
                if (
                    obj.target_pos[axis] < 0
                    or obj.target_pos[axis] >= self.matrix.shape[axis]
                ):
                    raise ValueError(
                        f"target_pos {obj.target_pos} out of bounds on axis {axis} "
                        f"vs shape {self.matrix.shape}"
                    )
        for obs in self._obstacles:
            _check_bounds(obs, self.matrix.shape)

        self._obstacle_matrix = np.clip(
            _place_objects_in(self.matrix, self._obstacles), a_min=None, a_max=1
        )
        self._obstacle_voxels = np.argwhere(self._obstacle_matrix > 0)

        combined = self._obstacle_matrix + _place_objects_in(
            self.matrix, list(self._objects.values())
        )
        if np.any(combined > 1):
            raise ValueError("objects/obstacles overlap")

        self._max_dist = max(self.matrix.shape)
        self._n_closest_agents = min(
            max(len(self.possible_agents) - 1, 1), _N_CLOSEST_AGENTS_CAP
        )
        agent_obs_dims = _OBS_DIMS + 2 + 4 * self._n_closest_agents  # +2: heading

        obs_space = spaces.Box(
            low=-self._max_dist,
            high=self._max_dist,
            shape=(agent_obs_dims,),
            dtype=np.float32,
        )
        act_space = spaces.Box(  # Box, not Discrete: SAC requires continuous actions
            low=np.array([-_V_MAX, -_OMEGA_MAX], dtype=np.float32),
            high=np.array([_V_MAX, _OMEGA_MAX], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_spaces = {aid: obs_space for aid in self.possible_agents}
        self.action_spaces = {aid: act_space for aid in self.possible_agents}

        self.max_steps = max_steps
        self.current_step = 0
        self._np_random = None

    def observation_space(self, agent: str) -> spaces.Space:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Space:
        return self.action_spaces[agent]

    @property
    def np_random(self):
        if self._np_random is None:
            self._np_random, _ = seeding.np_random()
        return self._np_random

    def _recompute_distance_fields(self) -> None:
        free = self._obstacle_matrix[:, :, 0] == 0
        self._dist_to_object: Dict[str, np.ndarray] = {}
        self._dist_to_target: Dict[str, np.ndarray] = {}
        for aid, obj in self._objects.items():
            center = obj.center
            self._dist_to_object[aid] = _bfs_distance_field_2d(
                free, (center[0], center[1])
            )
            self._dist_to_target[aid] = _bfs_distance_field_2d(
                free, (obj.target_pos[0], obj.target_pos[1])
            )

    def _is_cell_blocked(self, x: int, y: int, z: int) -> bool:
        """Whether (x, y, z) is occupied by an obstacle or a construction object."""
        if self._obstacle_matrix[x, y, z] != 0:
            return True
        for obj in self._objects.values():
            ox, oy, oz = int(obj.pos[0]), int(obj.pos[1]), int(obj.pos[2])
            sx, sy, sz = obj.matrix.shape
            if ox <= x < ox + sx and oy <= y < oy + sy and oz <= z < oz + sz:
                return True
        return False

    def _is_path_clear(
        self, start: np.ndarray, end: np.ndarray, exclude_agent: str
    ) -> bool:
        """Samples the segment start->end against obstacles/objects and other agents."""
        distance = float(np.hypot(end[0] - start[0], end[1] - start[1]))
        n_substeps = max(1, int(np.ceil(distance / 0.5)))
        size_x, size_y = self.matrix.shape[0], self.matrix.shape[1]
        for i in range(1, n_substeps + 1):
            t = i / n_substeps
            x = start[0] + t * (end[0] - start[0])
            y = start[1] + t * (end[1] - start[1])
            if x < 0 or y < 0 or x >= size_x or y >= size_y:
                return False
            if self._is_cell_blocked(int(x), int(y), 0):
                return False
            for oid, robot in self._robots.items():
                if oid == exclude_agent:
                    continue
                dx, dy = x - float(robot.pos[0]), y - float(robot.pos[1])
                if dx * dx + dy * dy < _AGENT_COLLISION_RADIUS**2:
                    return False
        return True

    def _interpolate_field(self, field: np.ndarray, x: float, y: float) -> float:
        """Bilinear interpolation, so the BFS field has a gradient between cell centers."""
        size_x, size_y = field.shape
        x = float(np.clip(x, 0, size_x - 1))
        y = float(np.clip(y, 0, size_y - 1))
        x0, y0 = int(x), int(y)
        x1, y1 = min(x0 + 1, size_x - 1), min(y0 + 1, size_y - 1)
        fx, fy = x - x0, y - y0
        return float(
            field[x0, y0] * (1 - fx) * (1 - fy)
            + field[x1, y0] * fx * (1 - fy)
            + field[x0, y1] * (1 - fx) * fy
            + field[x1, y1] * fx * fy
        )

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, dict]]:
        self._np_random, _ = seeding.np_random(seed)
        self.agents = list(self.possible_agents)
        self.current_step = 0
        self._delivered = {aid: False for aid in self.possible_agents}
        self._phases = {aid: SimEnv.Phase.TO_OBJECT for aid in self.possible_agents}

        for obj in self._objects.values():
            obj.is_carried = False
            obj.pos = np.array(
                _find_empty_cell(
                    self.np_random, self._obstacle_matrix, [None, None, 0]
                ),
                dtype=np.uint8,
            )
            for _ in range(_RANDOM_TRY_TIMEOUT):
                obj.target_pos = np.array(
                    _find_empty_cell(
                        self.np_random, self._obstacle_matrix, [None, None, 1]
                    ),
                    dtype=np.uint8,
                )
                if (
                    float(
                        np.linalg.norm(
                            obj.pos.astype(np.float32)
                            - obj.target_pos.astype(np.float32)
                        )
                    )
                    >= _MIN_TARGET_DIST
                ):
                    break

        # place agents one at a time so later spawns avoid earlier ones this reset
        occ_base = np.clip(
            self._obstacle_matrix
            + _place_objects_in(self.matrix, list(self._objects.values())),
            a_min=None,
            a_max=1,
        )
        placed = []
        for aid in self.possible_agents:
            occ = occ_base.copy()
            for px, py, pz in placed:
                occ[px, py, pz] = 1
            cell = _find_empty_cell(self.np_random, occ, [None, None, 0])
            self._robots[aid].pos = np.array(
                [cell[0] + 0.5, cell[1] + 0.5, 0.0], dtype=np.float32
            )
            angle = self.np_random.uniform(-np.pi, np.pi)
            self._robots[aid].heading = np.array(
                [np.cos(angle), np.sin(angle)], dtype=np.float32
            )
            placed.append(cell)

        self._recompute_distance_fields()
        self._position_histories = {
            aid: deque(
                [self._robots[aid].pos.copy()] * _POSITION_HISTORY_LEN,
                maxlen=_POSITION_HISTORY_LEN,
            )
            for aid in self.possible_agents
        }

        obs = {aid: self._get_obs(aid) for aid in self.agents}
        infos = {aid: self._get_info(aid) for aid in self.agents}
        return obs, infos

    def _get_obs(self, aid: str) -> np.ndarray:
        robot = self._robots[aid]
        obj = self._objects[aid]
        agent_pos = robot.pos.astype(np.float32)
        obj_pos = obj.pos.astype(np.float32)
        obj_target_pos = obj.target_pos.astype(np.float32)

        dist_agent_to_obj = float(np.linalg.norm(agent_pos - obj.center))
        dist_obj_to_target = float(np.linalg.norm(obj_pos - obj_target_pos))
        is_carrying = 1.0 if obj.is_carried else 0.0

        obstacles = _k_nearest_info(
            agent_pos, self._obstacle_voxels, _N_CLOSEST_OBTACLES, self._max_dist
        )

        if len(self._robots) > 1:
            other_positions = np.array(
                [
                    r.pos.astype(np.float32)
                    for oid, r in self._robots.items()
                    if oid != aid
                ],
                dtype=np.float32,
            )
        else:
            other_positions = np.empty((0, 3), dtype=np.float32)
        nearby_agents = _k_nearest_info(
            agent_pos, other_positions, self._n_closest_agents, self._max_dist
        )

        diff_to_obj_target = obj_target_pos - agent_pos
        norm = float(np.linalg.norm(diff_to_obj_target))
        dir_to_obj_target = (
            diff_to_obj_target / norm if norm > 0 else np.zeros(3, dtype=np.float32)
        )

        obs = np.concatenate(
            [
                agent_pos,
                obj_pos,
                obj_target_pos,
                np.array([dist_agent_to_obj], dtype=np.float32),
                np.array([dist_obj_to_target], dtype=np.float32),
                np.array([is_carrying], dtype=np.float32),
                np.array(obj.matrix.shape, dtype=np.float32),
                dir_to_obj_target,
                obstacles,
                nearby_agents,
            ]
        ).astype(np.float32)

        if _POSITION_HISTORY_LEN != 0:
            history = self._position_histories[aid]
            position_history = np.concatenate(
                [agent_pos - h.astype(np.float32) for h in history]
            )
            obs = np.concatenate([obs, position_history])

        return np.concatenate([obs, robot.heading.astype(np.float32)])

    def _get_info(self, aid: str) -> dict:
        robot = self._robots[aid]
        obj = self._objects[aid]
        return {
            "agent_position": robot.pos.copy(),
            "agent_heading": robot.heading.copy(),
            "object_position": obj.pos.copy(),
            "object_target_position": obj.target_pos.copy(),
            "object_dimensions": deepcopy(obj.matrix.shape),
            "is_carrying": obj.is_carried,
            "phase": self._phases[aid],
            "delivered": self._delivered[aid],
            "steps": self.current_step,
        }

    def step(self, actions: Dict[str, np.ndarray]) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, dict],
    ]:
        self.current_step += 1
        truncate_all = self.current_step >= self.max_steps

        obs: Dict[str, np.ndarray] = {}
        rewards: Dict[str, float] = {}
        terminations: Dict[str, bool] = {}
        truncations: Dict[str, bool] = {}
        infos: Dict[str, dict] = {}

        for aid in self.possible_agents:
            robot = self._robots[aid]
            obj = self._objects[aid]

            if self._delivered[aid]:
                # Frozen, not dropped (supersuit needs agents == possible_agents every step).
                # terminated only pulses True on delivery, else eval logging sees a spurious episode each frozen step.
                rewards[aid] = 0.0
                terminations[aid] = False
                truncations[aid] = truncate_all
                obs[aid] = self._get_obs(aid)
                infos[aid] = self._get_info(aid)
                infos[aid]["collision"] = False
                continue

            action = actions[aid]
            old_pos = robot.pos.copy()

            v = float(np.clip(action[0], -_V_MAX, _V_MAX))
            omega = float(np.clip(action[1], -_OMEGA_MAX, _OMEGA_MAX))

            cos_o, sin_o = np.cos(omega * _DT), np.sin(omega * _DT)
            hx, hy = float(robot.heading[0]), float(robot.heading[1])
            heading = np.array(
                [cos_o * hx - sin_o * hy, sin_o * hx + cos_o * hy], dtype=np.float32
            )
            robot.heading = heading / np.linalg.norm(heading)

            new_pos = old_pos.copy()
            new_pos[0] = np.clip(
                old_pos[0] + v * _DT * robot.heading[0], 0, self.matrix.shape[0] - 1e-3
            )
            new_pos[1] = np.clip(
                old_pos[1] + v * _DT * robot.heading[1], 0, self.matrix.shape[1] - 1e-3
            )

            collision = not self._is_path_clear(old_pos, new_pos, exclude_agent=aid)
            robot.pos = old_pos if collision else new_pos
            self._position_histories[aid].append(old_pos)

            terminated = False
            reward = 0.0

            if obj.is_adjacent(robot.pos, 2.1):
                obj.is_carried = True
                self._phases[aid] = SimEnv.Phase.TO_TARGET
            if obj.is_carried:
                obj.pos = robot.pos + np.array([0, 0, 1], dtype=np.float32)  # type: ignore[assignment]
            if obj.is_adjacent(obj.target_pos, 1.9) and obj.is_carried:
                terminated = True
                self._delivered[aid] = True
                reward += 100.0

            agent_x, agent_y = float(robot.pos[0]), float(robot.pos[1])
            if self._phases[aid] == SimEnv.Phase.TO_OBJECT:
                goal_x, goal_y = float(obj.center[0]), float(obj.center[1])
                bfs_dist = self._interpolate_field(
                    self._dist_to_object[aid], agent_x, agent_y
                )
            else:
                goal_x, goal_y = float(obj.target_pos[0]), float(obj.target_pos[1])
                bfs_dist = self._interpolate_field(
                    self._dist_to_target[aid], agent_x, agent_y
                )
            euclidean_dist = float(np.hypot(agent_x - goal_x, agent_y - goal_y))

            if _USE_BFS_REWARD:
                reward -= bfs_dist
                reward -= 0.01 * euclidean_dist
            else:
                reward -= euclidean_dist

            if collision:
                reward -= 10.0
            reward -= 0.1

            rewards[aid] = float(reward)
            terminations[aid] = terminated
            truncations[aid] = truncate_all
            obs[aid] = self._get_obs(aid)
            infos[aid] = self._get_info(aid)
            infos[aid]["collision"] = collision

        return obs, rewards, terminations, truncations, infos


def make_multi_agent_env(n_agents: int = 3) -> MultiAgentSimEnv:
    """Same obstacle scene as make_continuous_env(), with n_agents independent agent/object pairs."""
    const_objs = [
        ConstructionObject((1, 1, 1), (i, 0, 0), (40, 28, 1)) for i in range(n_agents)
    ]
    obstacles = [
        EnvObstacle((6, 2, 2), (0, 16, 0)),
        EnvObstacle((2, 12, 2), (16, 0, 0)),
        EnvObstacle((2, 12, 2), (26, 17, 0)),
        EnvObstacle((2, 12, 2), (14, 17, 0)),
    ]
    agents = [
        ConstructionAgent((1, 1, 1), (0, 0, 0), (1.0, 0.0)) for _ in range(n_agents)
    ]
    return MultiAgentSimEnv((41, 29, 5), const_objs, agents, obstacles)
