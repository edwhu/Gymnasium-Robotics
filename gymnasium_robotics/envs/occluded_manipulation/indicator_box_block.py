import gymnasium as gym
import numpy as np
from gymnasium import spaces
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.transform_utils import convert_quat

from gymnasium_robotics.envs.occluded_manipulation.blocked_table import (
    BlockedTableArena as TableArena,
)


class IndicatorBoxBlock(SingleArmEnv):
    """
    This class corresponds to the lifting task for a single robot arm.

    Args:
        robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)
            Note: Must be a single single-arm robot!

        env_configuration (str): Specifies how to position the robots within the environment (default is "default").
            For most single arm environments, this argument has no impact on the robot setup.

        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param

        gripper_types (str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory. Default is "default", which is the default grippers(s) associated
            with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model
            overrides the default gripper. Should either be single str if same gripper type is to be used for all
            robots or else it should be a list of the same length as "robots" param

        initialization_noise (dict or list of dict): Dict containing the initialization noise parameters.
            The expected keys and corresponding value types are specified below:

            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to `None` or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"

            Should either be single dict if same noise value is to be used for all robots or else it should be a
            list of the same length as "robots" param

            :Note: Specifying "default" will automatically use the default noise settings.
                Specifying None will automatically create the required dict with "magnitude" set to 0.0.

        table_full_size (3-tuple): x, y, and z dimensions of the table.

        table_friction (3-tuple): the three mujoco friction parameters for
            the table.

        use_camera_obs (bool): if True, every observation includes rendered image(s)

        use_object_obs (bool): if True, include object (cube) information in
            the observation.

        reward_scale (None or float): Scales the normalized reward function by the amount specified.
            If None, environment reward remains unnormalized

        reward_shaping (bool): if True, use dense rewards.

        placement_initializer (ObjectPositionSampler): if provided, will
            be used to place objects on every reset, else a UniformRandomSampler
            is used by default.

        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.

        has_offscreen_renderer (bool): True if using off-screen rendering

        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse

        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.

        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.

        render_gpu_device_id (int): corresponds to the GPU device id to use for offscreen rendering.
            Defaults to -1, in which case the device will be inferred from environment variables
            (GPUS or CUDA_VISIBLE_DEVICES).

        control_freq (float): how many control signals to receive in every second. This sets the amount of
            simulation time that passes between every action input.

        horizon (int): Every episode lasts for exactly @horizon timesteps.

        ignore_done (bool): True if never terminating the environment (ignore @horizon).

        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables

        camera_names (str or list of str): name of camera to be rendered. Should either be single str if
            same name is to be used for all cameras' rendering or else it should be a list of cameras to render.

            :Note: At least one camera must be specified if @use_camera_obs is True.

            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                robot's camera list).

        camera_heights (int or list of int): height of camera frame. Should either be single int if
            same height is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_widths (int or list of int): width of camera frame. Should either be single int if
            same width is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
            bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
            "camera names" param.

    Raises:
        AssertionError: [Invalid number of robots specified]
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1., 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=False,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=True,
    ):

        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
        )

    def reward(self, action=None):
        """
        Reward function for the task.

        Sparse un-normalized reward:

            - a discrete reward of 2.25 is provided if the cube is lifted

        Un-normalized summed components if using reward shaping:

            - Reaching: in [0, 1], to encourage the arm to reach the cube
            - Grasping: in {0, 0.25}, non-zero if arm is grasping the cube
            - Lifting: in {0, 1}, non-zero if arm has lifted the cube

        The sparse reward only consists of the lifting component.

        Note that the final reward is normalized and scaled by
        reward_scale / 2.25 as well so that the max score is equal to reward_scale

        Args:
            action (np array): [NOT USED]

        Returns:
            float: reward value
        """
        reward = 0.

        # sparse completion reward
        if self._check_success():
            reward = 2.25

        # use a shaping reward
        elif self.reward_shaping:

            # reaching reward
            cube_pos = self.sim.data.body_xpos[self.cube_body_id]
            gripper_site_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
            dist = np.linalg.norm(gripper_site_pos - cube_pos)
            reaching_reward = 1 - np.tanh(10.0 * dist)
            reward += reaching_reward

            # grasping reward
            if self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cube):
                reward += 0.25

        # Scale reward if requested
        if self.reward_scale is not None:
            reward *= self.reward_scale / 2.25

        return reward #consider returning "done"

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # initialize objects of interest
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1",
        }
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="redwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        self.cube = BoxObject(
            name="cube",
            size_min=[0.020, 0.020, 0.020],  # [0.015, 0.015, 0.015],
            size_max=[0.020, 0.020, 0.020],  # [0.018, 0.018, 0.018])
            rgba=[1, 0, 0, 1],
            material=redwood,
            density=2000, #used to be 2000
            friction=5 #change back to 5
        )

        self.indicator = BoxObject(
            name="indicator",
            size_min=[0.020, 0.1, 0.1],  # [0.015, 0.015, 0.015],
            size_max=[0.020, 0.1, 0.1],  # [0.018, 0.018, 0.018])
            rgba=[0, 0, 1, 0.5],
            material=redwood,
        )

        # Create placement initializer
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.cube)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.cube,
                x_range=[0, 0], #used to be 0.005
                # y_range=[-0.17, 0.17], #used to be 0.18
                y_range=[-0.1, 0.1], #used to be 0.18
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            )

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.cube, self.indicator],
        )

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.cube_body_id = self.sim.model.body_name2id(self.cube.root_body)

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        # low-level object information
        if self.use_object_obs:
            # Get robot prefix and define observables modality
            pf = self.robots[0].robot_model.naming_prefix
            modality = "object"

            # cube-related observables
            @sensor(modality=modality)
            def cube_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.cube_body_id])

            @sensor(modality=modality)
            def cube_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.cube_body_id]), to="xyzw")

            @sensor(modality=modality)
            def gripper_to_cube_pos(obs_cache):
                return obs_cache[f"{pf}eef_pos"] - obs_cache["cube_pos"] if \
                    f"{pf}eef_pos" in obs_cache and "cube_pos" in obs_cache else np.zeros(3)

            sensors = [cube_pos, cube_quat, gripper_to_cube_pos]
            names = [s.__name__ for s in sensors]

            # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )
        @sensor(modality = modality)
        def gripper_force(obs_cache):
            return self.robots[0].get_sensor_measurement("gripper0_force_ee")/20#hardcoded for now
        observables["gripper_force"] = Observable(name = "gripper_force", sensor = gripper_force, sampling_rate = self.control_freq)
        #
        # @sensor(modality = modality)
        # def gripper_torque(obs_cache):
        #     return self.robots[0].get_sensor_measurement("gripper0_torque_ee")/20#hardcoded for now
        # observables["gripper_torque"] = Observable(name = "gripper_torque", sensor = gripper_torque, sampling_rate = self.control_freq)
        #
        # @sensor(modality = modality)
        # def gripper_tip_force(obs_cache):
        #     return self.robots[0].get_sensor_measurement("gripper0_force_ee_tip")
        # observables["gripper_tip_force"] = Observable(name = "gripper_tip_force", sensor = gripper_tip_force, sampling_rate = self.control_freq)

        @sensor(modality = modality)
        def object_sound(obs_cache):
            sound = np.zeros((6,))
            if self.sim.data.body_xpos[self.cube_body_id][2] < 0.84:
                sound = self.sim.data.cfrc_ext[self.cube_body_id]
            return sound
        observables["object_sound"] = Observable(name = "object_sound", sensor = object_sound, sampling_rate = self.control_freq)

        #
        # @sensor(modality=modality)
        # def gripper_joint_force(obs_cache):
        #     return np.array([self.sim.data.efc_force[x] / 10 for x in self.robots[0]._ref_gripper_joint_vel_indexes])  # divide by 10 to normalize somewhat
        #
        # observables["robot0_gripper_joint_force"] = Observable(
        #         name=gripper_joint_force,
        #         sensor=gripper_joint_force,
        #         sampling_rate=self.control_freq,
        #     )

        return observables

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
#                 print(obj_pos)
#                 print(obj.joints)
#                 print("-----")
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

                if obj_pos[1] < -0.1:
                    indicator_pos = (0.25, -0.15, obj_pos[2])
                elif obj_pos[1] < 0:
                    indicator_pos = (0.25, -0.05, obj_pos[2])
                elif obj_pos[1] < 0.1:
                    indicator_pos = (0.25, 0.05, obj_pos[2])
                else:
                    indicator_pos = (0.25, 0.15, obj_pos[2])

                self.sim.data.set_joint_qpos(self.indicator.joints[0], np.concatenate([np.array(indicator_pos), np.array([1, 0, 0, 0])]))

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the cube.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        # Color the gripper visualization site according to its distance to the cube
        if vis_settings["grippers"]:
            self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=self.cube)

    def _check_success(self):
        """
        Check if cube has been lifted.

        Returns:
            bool: True if cube has been lifted
        """
        cube_height = self.sim.data.body_xpos[self.cube_body_id][2]
        table_height = self.model.mujoco_arena.table_offset[2]

        # cube is higher than the table top above a margin
        return cube_height > table_height + 0.04



class GymIndicatorBoxBlock(IndicatorBoxBlock, gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    render_mode = "rgb_array"

    def __init__(self, views, width, height):
        import robosuite.macros as macros
        from robosuite.controllers import load_controller_config
        macros.IMAGE_CONVENTION = "opencv"

        super().__init__(
            robots=["Panda"], #the robot
            controller_configs=load_controller_config(default_controller="OSC_POSITION"), #this decides which controller settings to use!
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            use_object_obs = True,
            reward_shaping=True,
            control_freq=20,
            horizon = 100,
            camera_names=views,
            camera_heights=width,
            camera_widths=height,
            camera_depths=True,
        )
        # observables we want to keep
        self.observation_keys = {"robot0_eef_pos","robot0_gripper_qpos","gripper_force", "object_sound"}
        for v in views:
            self.observation_keys.add(f"{v}_depth")
            self.observation_keys.add(f"{v}_image")

        for active_obs in self.active_observables:
            if active_obs not in self.observation_keys:
                self.modify_observable(active_obs, attribute='enabled', modifier=False)

        # Create name for gym
        self.name = "GymIndicatorBoxBlock"

        # Gym specific attributes
        self.spec = None

        # set up observation and action spaces
        obs = self.reset()[0]
        self.modality_dims = {key: obs[key].shape for key in self.observation_keys}

        self.observation_space = spaces.Dict(
            {
                key: spaces.Box(0 if "_image" in key else -np.inf, 255 if "_image" in key else np.inf, shape=shape, dtype=np.uint8 if "_image" in key else np.float32)
                for key, shape in self.modality_dims.items()
            }
        )

        low, high = self.action_spec
        self.action_space = spaces.Box(low, high)

    def reset(self, seed=None, options=None):
        """
        Extends env reset method to return flattened observation instead of normal OrderedDict and optionally resets seed

        Returns:
            np.array: Flattened environment observation space after reset occurs
        """
        if seed is not None:
            if isinstance(seed, int):
                np.random.seed(seed)
            else:
                raise TypeError("Seed must be an integer type!")
        ob_dict = super().reset()
        output = {}
        for k in ob_dict.keys():
            if k in self.observation_keys:
                dtype = np.float32 if ob_dict[k].dtype == np.float64 else ob_dict[k].dtype
                output[k] = ob_dict[k].astype(dtype)
        return output, {}

    def step(self, action):
        """
        Extends vanilla step() function call to return flattened observation instead of normal OrderedDict.

        Args:
            action (np.array): Action to take in environment

        Returns:
            4-tuple:

                - (np.array) flattened observations from the environment
                - (float) reward from the environment
                - (bool) episode ending after reaching an env terminal state
                - (bool) episode ending after an externally defined condition
                - (dict) misc information
        """
        ob_dict, reward, terminated, info = super().step(action)
        output = {}
        for k in ob_dict.keys():
            if k in self.observation_keys:
                dtype = np.float32 if ob_dict[k].dtype == np.float64 else ob_dict[k].dtype
                output[k] = ob_dict[k].astype(dtype)
        return output, reward, terminated, False, info
    def render(self):
        return super().render()
    
    def close(self):
        return super().close()


if __name__ == "__main__":
    import gymnasium

    env = gymnasium.make("FOIndicatorBoxBlock-v0")
    imgs = []
    obs = env.reset()[0]
    depth = obs["sideview_depth"]
    # convert depth to rgb. it's already normalized to [0, 1]
    depth = np.repeat(depth, 3, axis=2)
    # convert depth to 0-255 uint8
    depth = (depth * 255).astype(np.uint8)
    import imageio
    imageio.imwrite("test.png", depth)


    # import ipdb; ipdb.set_trace()
    # imgs.append(np.concatenate([obs["sideview_image"], obs["agentview_image"]], axis=1))
    # for i in range(15):
    #     obs, *_ = env.step(np.array([1, 0, 0, 0]))
    #     print(i, obs["object-state"][:3][0], obs["robot0_eef_pos"][0])
    #     imgs.append(np.concatenate([obs["sideview_image"], obs["agentview_image"]], axis=1))
    # import imageio
    # imageio.mimwrite("test.gif", imgs, fps=10)
    # env = GymIndicatorBoxBlock()
    # import ipdb; ipdb.set_trace()
    # import robosuite.macros as macros
    # from robosuite.controllers import load_controller_config
    # macros.IMAGE_CONVENTION = "opencv"

    # views = ["sideview"]
    # env = IndicatorBoxBlock(
    #     robots=["Panda"], #the robot
    #     controller_configs=load_controller_config(default_controller="OSC_POSITION"), #this decides which controller settings to use!
    #     has_renderer=False,
    #     has_offscreen_renderer=True,
    #     use_camera_obs=True,
    #     use_object_obs = True,
    #     reward_shaping=True,
    #     control_freq=20,
    #     horizon = 100,
    #     camera_names=views,
    #     camera_heights=64,
    #     camera_widths=64
    # )
    # # observables we want to keep
    # observable_whitelist = {"robot0_eef_pos","robot0_gripper_qpos","gripper_force", "object_sound", "sideview_image"}
    # for active_obs in env.active_observables:
    #     if active_obs not in observable_whitelist:
    #         env.modify_observable(active_obs, attribute='enabled', modifier=False)
    # obs = env.reset()
    # for k,v in obs.items():
    #     print(k, v.shape)
    # import ipdb; ipdb.set_trace()