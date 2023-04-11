import os

import numpy as np

from gymnasium import spaces
from gymnasium.utils.ezpickle import EzPickle

from gymnasium_robotics.envs.fetch import MujocoFetchEnv, goal_distance
from gymnasium_robotics.utils import rotations

# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join("fetch", "push_quad.xml")


class MujocoFetchPushQuadStateEnv(MujocoFetchEnv, EzPickle):
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
        self.observation_space = spaces.Box(low=float("-inf"), high=float("inf"), shape=(18,))  # object & gripper position, linear vel, rotational vel
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
        obs = None
        if hasattr(self, "mujoco_renderer"):
            self._render_callback()
            dt = self.n_substeps * self.model.opt.timestep
            # Object positions and velocities
            obj_qpos = self._utils.get_site_xpos(self.model, self.data, "object0")
            obj_qvelp = self._utils.get_site_xvelp(self.model, self.data, "object0") * dt
            obj_qvelr = self._utils.get_site_xvelr(self.model, self.data, "object0") * dt
            # Gripper positions and velocities
            gripper_qpos = self._utils.get_site_xpos(self.model, self.data, "robot0:grip")
            gripper_qvelp = self._utils.get_site_xvelp(self.model, self.data, "robot0:grip") * dt
            gripper_qvelr = self._utils.get_site_xvelr(self.model, self.data, "robot0:grip") * dt
            obs = np.concatenate((obj_qpos, obj_qvelp, obj_qvelr, gripper_qpos, gripper_qvelp, gripper_qvelr), dtype=np.float32)
        else:
            obs = {}
            obs["achieved_goal"] = obs["observation"] = np.zeros((18,))
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
    
    def render(self):
        """Render a frame of the MuJoCo simulation.
        Returns:
            rgb image (np.ndarray): if render_mode is "rgb_array", return a 3D image array.
        """
        self._render_callback()
        img = self.mujoco_renderer.render(self.render_mode, camera_name="camera_under")
        if np.sum(img) == 0:
            import ipdb; ipdb.set_trace()
        return img

    def close(self):
        """Close contains the code necessary to "clean up" the environment.
        Terminates any existing WindowViewer instances in the Gymnaisum MujocoRenderer.
        """
        pass

class MujocoFetchPushQuadStateHardEnv(MujocoFetchEnv, EzPickle):
    def __init__(self, full_joint_state=True, reward_type="sparse", action_space_type="object", **kwargs):
        initial_qpos = {
            "robot0:slide0": 0.405,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.0,
            "object0:joint": [1.25, 0.53, 0.4, 1.0, 0.0, 0.0, 0.0],
        }
        self.prev_goal_dist = None
        self.full_joint_state = full_joint_state
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
        self.action_space_type = action_space_type
        if self.full_joint_state:
            self.joints = self._utils.extract_mj_names(self.model, self._utils.mjtObj.mjOBJ_JOINT)[0]
        else:
            self.joints = []
        self.sites = ("robot0:grip", "object0")  # only use gripper
        obs_shape = self.get_joint_state().shape[0] + self.get_site_state().shape[0] + 3  # +3 for goal TODO(James)
        self.observation_space = spaces.Box(low=float("-inf"), high=float("inf"), shape=(obs_shape,))  # position, velocity, and angular velocity for each site
        EzPickle.__init__(self, reward_type=reward_type, action_space_type=action_space_type, **kwargs)
        
        self.goal_idx = 0
        q1_goal = np.array([1.1, 0.95, 0.42])
        q2_goal = np.array([1.1, 0.55, 0.42])
        q3_goal = np.array([1.5, 0.55, 0.42])
        q4_goal = np.array([1.5, 0.95, 0.42])
        self.goals = [q1_goal, q2_goal, q3_goal, q4_goal]

    def _sample_goal(self):
        goal = self.goals[self.goal_idx]
        return goal.copy()

    def _reset_sim(self):
        self.data.time = self.initial_time
        self.data.qpos[:] = np.copy(self.initial_qpos)
        self.data.qvel[:] = np.copy(self.initial_qvel)
        if self.model.na != 0:
            self.data.act[:] = None

        # Randomize start position of object.
        if self.has_object:
            # object_xpos = [1.2, 0.85] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
            object_xpos = [1.2, 0.85]
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
    
    def get_joint_state(self):
        if not self.joints:
            return np.array([])
        dt = self.n_substeps * self.model.opt.timestep
        qpos = np.concatenate([self._utils.get_joint_qpos(self.model, self.data, name) for name in self.joints])
        qvel = np.concatenate([(self._utils.get_joint_qvel(self.model, self.data, name) * dt) for name in self.joints])
        return np.concatenate((qpos, qvel), dtype=np.float32)
    
    def get_site_state(self):
        dt = self.n_substeps * self.model.opt.timestep
        xpos = np.concatenate([self._utils.get_site_xpos(self.model, self.data, name) for name in self.sites])
        xrot = np.concatenate([rotations.mat2euler(self._utils.get_site_xmat(self.model, self.data, name)) for name in self.sites])
        xvelp = np.concatenate([(self._utils.get_site_xvelp(self.model, self.data, name) * dt) for name in self.sites])
        xvelr = np.concatenate([(self._utils.get_site_xvelr(self.model, self.data, name) * dt) for name in self.sites])
        return np.concatenate((xpos, xrot, xvelp, xvelr), dtype=np.float32)

    def _get_obs(self):
        obs = None
        if hasattr(self, "mujoco_renderer"):
            self._render_callback()
            site_states = self.get_site_state()
            joint_states = self.get_joint_state()
            goal = self.goal
            obs = np.concatenate((site_states, joint_states, goal), dtype=np.float32)
        else:
            # BaseRobotEnv has called _get_obs to determine observation space dims but mujoco renderer has not been initialized yet.
            # in this case, return an obs dict with arbitrary values for each ey
            # since observation space will be overwritten later.
            obs = {}
            obs["achieved_goal"] = obs["observation"] = np.zeros((1,))
        return obs

    def step(self, action):
        if np.array(action).shape != self.action_space.shape:
            raise ValueError("Action dimension mismatch")

        if self.action_space_type == "object":
            # todo: make this xy control, and rescale to 5cm increments.
            action = np.clip(action, -1, 1)
            action = action * 0.05
            obj0_pos = self._utils.get_site_xpos(self.model, self.data, "object0")
            obj0_new_pos = obj0_pos + np.array([*action[:2], 0])
            obj0_qpos = np.array(self._utils.get_joint_qpos(self.model, self.data, "object0:joint"))
            obj0_qpos[:2] = obj0_new_pos[:2]
            self._utils.set_joint_qpos(
                self.model, self.data, "object0:joint", obj0_qpos
            )
            self._mujoco.mj_forward(self.model, self.data)
            self._mujoco_step(action)
            self._step_callback()
        else:
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

        if self.reward_type == "dense":
            curr_goal_dist = goal_distance(obj0_pos, self.goal)
            reward = self.compute_reward(curr_goal_dist, self.prev_goal_dist, info)
            if info["is_success"] and self.goal_idx < len(self.goals) - 1:
                self.goal_idx += 1
                self.goal = self.goals[self.goal_idx]
            self.prev_goal_dist = goal_distance(obj0_pos, self.goal)
            
        else:
            curr_goal_dist = goal_distance(obj0_pos, self.goal)
            reward = 1.0 * (curr_goal_dist < 0.05)
            if info["is_success"] and self.goal_idx < len(self.goals) - 1:
                self.goal_idx += 1
                self.goal = self.goals[self.goal_idx]

        return obs, reward, terminated, truncated, info
    
    def compute_reward(self, curr_goal_dist, prev_goal_dist, info):
        # we want prev_goal_dist > curr_goal_dist.
        reward = self.prev_goal_dist - curr_goal_dist
        return reward

    def reset(
        self,
        *,
        seed = None,
        options = None,
    ):
        # removed super.reset call
        self.goal_idx = 0
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()
        self.goal = self._sample_goal().copy()
        obs = self._get_obs()
        if self.render_mode == "human":
            self.render()
        # initialize block distance
        obj0_pos = self._utils.get_site_xpos(self.model, self.data, "object0")
        self.prev_goal_dist = goal_distance(obj0_pos, self.goal)
        return obs, {}

    def render(self):
        """Render a frame of the MuJoCo simulation.

        Returns:
            rgb image (np.ndarray): if render_mode is "rgb_array", return a 3D image array.
        """
        self._render_callback()
        img = self.mujoco_renderer.render(self.render_mode, camera_name="camera_under")
        if np.sum(img) == 0:
            import ipdb; ipdb.set_trace()
        return img

    def close(self):
        """Close contains the code necessary to "clean up" the environment.

        Terminates any existing WindowViewer instances in the Gymnaisum MujocoRenderer.
        """
        pass
