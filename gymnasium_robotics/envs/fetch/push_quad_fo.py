import os

import numpy as np

from gymnasium import spaces
from gymnasium.utils.ezpickle import EzPickle

from gymnasium_robotics.envs.fetch import MujocoFetchEnv

# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join("fetch", "push_quad.xml")


class MujocoFetchPushQuadFOEnv(MujocoFetchEnv, EzPickle):
    def __init__(self, reward_type="sparse", **kwargs):
        initial_qpos = {
            "robot0:slide0": 0.405,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.0,
            "object0:joint": [1.25, 0.53, 0.4, 1.0, 0.0, 0.0, 0.0],
        }
        MujocoFetchEnv.__init__(
            self,
            model_path=MODEL_XML_PATH,
            has_object=True,
            block_gripper=True,
            n_substeps=20,
            gripper_extra_height=0.0,
            target_in_the_air=False,
            target_offset=0.0,
            obj_range=0.03,
            target_range=0.15,
            distance_threshold=0.05,
            initial_qpos=initial_qpos,
            reward_type=reward_type,
            **kwargs,
        )
        # consists of images and proprioception.
        _obs_space = {}
        _obs_space["observation"] = spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32)  # object x,y,z & gripper x,y,z
        self.observation_space = spaces.Dict(_obs_space)
        EzPickle.__init__(self, image_size=32, reward_type=reward_type, **kwargs)

    def _sample_goal(self):
        goal = np.array([1.1 - 0.1, 0.95 + 0.1, 0.42])
        return goal.copy()

    def _reset_sim(self):
        self.data.time = self.initial_time
        self.data.qpos[:] = np.copy(self.initial_qpos)
        self.data.qvel[:] = np.copy(self.initial_qvel)
        if self.model.na != 0:
            self.data.act[:] = None

        # Randomize start position of object.
        if self.has_object:
            object_xpos = [1.2, 0.85] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
            object_qpos = self._utils.get_joint_qpos(
                self.model, self.data, "object0:joint"
            )
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
            self._utils.set_joint_qpos(
                self.model, self.data, "object0:joint", object_qpos
            )

        self._mujoco.mj_forward(self.model, self.data)
        return True

    def _get_obs(self): 
        obs = {}
        if hasattr(self, "mujoco_renderer"):
            obj_qpos = self._utils.get_site_xpos(self.model, self.data, "object0")
            gripper_qpos = self._utils.get_site_xpos(self.model, self.data, "robot0:grip")
            obs["observation"] = np.concatenate((obj_qpos, gripper_qpos))
        else:
            obs["achieved_goal"] = obs["observation"] = np.zeros((6,))
        return obs

    def step(self, action):
        if np.array(action).shape != self.action_space.shape:
            raise ValueError("Action dimension mismatch")

        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)

        self._mujoco_step(action)

        self._step_callback()

        if self.render_mode == "human":
            self.render()
        obs = self._get_obs()

        obj0_pos = self._utils.get_site_xpos(self.model, self.data, "object0")
        info = {
            "is_success": self._is_success(obj0_pos, self.goal),
        }

        terminated = self.compute_terminated(obj0_pos, self.goal, info)
        truncated = self.compute_truncated(obj0_pos, self.goal, info)

        reward = self.compute_reward(obj0_pos, self.goal, info)

        return obs, reward, terminated, truncated, info

    def reset(
        self,
        *,
        seed = None,
        options = None,
    ):
        # removed super.reset call
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()
        self.goal = self._sample_goal().copy()
        obs = self._get_obs()
        if self.render_mode == "human":
            self.render()

        return obs, {}