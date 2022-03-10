from collections import OrderedDict
import numpy as np
from gym.spaces import  Dict , Box
import math
import os
from metaworld.envs.mujoco.sawyer_xyz.base import SawyerXYZEnv
from PIL import Image
import cv2
import imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
import time


class LorlTabletop(SawyerXYZEnv):
    def __init__(
            self,
            obj_low=None,
            obj_high=None,
            goal_low=None,
            goal_high=None,
            hand_init_pos=(0, 0.4, 0.2),
            liftThresh=0.04,
            rewMode='orig',
            rotMode='fixed',
            problem="rand",
            xml='updated_new',
            filepath="test",
            max_path_length=20,
            verbose=0,
            log_freq=100, # in terms of episode num
            **kwargs
    ):
        self.max_path_length = max_path_length
        self.cur_path_length = 0
        self.xml = xml

        hand_low=(-0.3, 0.4, 0.0)
        hand_high=(0.3, 0.8, 0.15)
        obj_low=(-0.3, 0.4, 0.1)
        obj_high=(0.3, 0.8, 0.3)
        SawyerXYZEnv.__init__(
            self,
            frame_skip=5,
            action_scale=1./5,
            hand_low=hand_low,
            hand_high=hand_high,
            model_name=self.model_name,
            **kwargs
        )

        self.liftThresh = liftThresh
        self.rewMode = rewMode
        self.rotMode = rotMode
        self.hand_init_pos = np.array(hand_init_pos)
        self.action_rot_scale = 1./10
        self.action_space = Box(
                np.array([-1, -1, -1, -np.pi, -1]),
                np.array([1, 1, 1, np.pi, 1]),
            )
        self.hand_and_obj_space = Box(
            np.hstack((self.hand_low, obj_low)),
            np.hstack((self.hand_high, obj_high)),
        )

        self.imsize = 64
        self.imsize_x = 64
        self.observation_space = Box(0, 1.0, (self.imsize_x*self.imsize*3, ))


    @property
    def model_name(self):
        dirname = os.path.dirname(__file__)
        file = "../assets_updated/sawyer_xyz/" + self.xml + ".xml"
        filename = os.path.join(dirname, file)
        return filename

    def step(self, action):
        action[3] = 0 ### Rotation always 0
        self.set_xyz_action_rotz(action[:4])
        self.do_simulation([action[-1], -action[-1]])

        ob = self._get_obs()
        if self.cur_path_length == self.max_path_length - 1:
            done = True
        else:
            done = False
        self.cur_path_length +=1
        return ob, 0, done, {}

    def _get_obs(self):
        obs = self.sim.render(self.imsize_x, self.imsize, camera_name="cam0") / 255.
        obs = np.flip(obs, 0).copy()
        return obs

    def reset_model(self):
        ''' For logging '''
        self.cur_path_length = 0

        ### Reset gripper
        self._reset_hand()
        for _ in range(100):
            try:
                self.do_simulation([0.0, 0.0])
            except:
                print("Got Mujoco Unstable Simulation Warning")
                continue
        self.cur_path_length = 0

        # Set inital pos for mugs
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        rd = np.random.uniform(-0.1, 0.1, (2,))
        qpos[9:11] = np.array([-0.2, 0.65]) + rd
        rd = np.random.uniform(-0.1, 0.1, (2,))
        qpos[11:13] = np.array([-0.2, 0.65]) + rd

        # Set initial pos for drawer, faucet, button
        qpos[13] = np.random.uniform(-np.pi/4, np.pi/4)
        qpos[14] = np.random.uniform(-0.09, 0.0)
        self.set_state(qpos, qvel)
        for _ in range(100):
          self.sim.step()
        o = self._get_obs()
        return o, {}


    def _reset_hand(self, pos=None):
        if pos is None:
            if np.random.uniform() < 0.5:
              pos = [-0.0, 0.5, 0.07]
            else:
              pos = [-0.2, 0.65, 0.07] + np.random.uniform(-0.05, 0.05, (3,))
              pos[2] = 0.07
        for _ in range(100):
            self.data.set_mocap_pos('mocap', pos)
            self.data.set_mocap_quat('mocap', np.array([0.707, 0.0, 0.707, 0.0]))
            try:
                self.do_simulation([-1,1], self.frame_skip)
            except:
                print("Got Mujoco Unstable Simulation Warning")
                continue
        rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')
        self.init_fingerCOM  =  (rightFinger + leftFinger)/2
        self.pickCompleted = False

    def get_site_pos(self, siteName):
        _id = self.model.site_names.index(siteName)
        return self.data.site_xpos[_id].copy()

    def compute_reward(self):
        return 0.0

