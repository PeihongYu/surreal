import airsim
import time
import copy
import numpy as np
from PIL import Image
import cv2
import gym
from gym.spaces import Box
from collections import OrderedDict

goal_threshold = 3
np.set_printoptions(precision=3, suppress=True)
IMAGE_VIEW = True


class drone_env(gym.Env):
    def __init__(self, start=[0, 0, -5], aim=[32, 38, -4]):
        #simGetObjectPose

        self.start = np.array(start)
        self.aim = np.array(aim)
        #self.client = AirSimClient.MultirotorClient()
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.threshold = goal_threshold
        self.observation_space = Box(low=0, high=255, shape=(64,64,1))
        self.action_space = Box(low=-1, high=1, shape=(1,))

    @property
    def dof(self):
        return 3

    def reset(self):
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.moveToPositionAsync(self.start.tolist()[0], self.start.tolist()[1], self.start.tolist()[2], 5)
        time.sleep(2)

    def isDone(self):
        pos = self.getCurPosition()
        if self.distance(self.aim, pos) < self.threshold:
            return True
        return False

    def getState(self):
        return OrderedDict()

    def observation_spec(self):
        observation = self.getState()
        return observation

    def action_spec(self):
        low = np.ones(self.dof) * -1.
        high = np.ones(self.dof) * 1.
        return low, high

    def moveByDist(self, diff, forward=False):
        temp = airsim.YawMode()
        temp.is_rate = not forward
        self.client.moveByVelocityAsync(diff[0], diff[1], diff[2], 1, drivetrain=airsim.DrivetrainType.ForwardOnly, yaw_mode=temp)
        time.sleep(0.5)
        return 0

    def render(self, extra1="", extra2=""):
        pos = self.v2t(self.getCurPosition())
        goal = self.distance(self.aim, pos)
        print(extra1, "distance:", int(goal), "position:", pos.astype("int"), extra2)

    def help(self):
        print("drone simulation environment")

    def getCurPosition(self):
        return self.v2t(self.client.getMultirotorState().kinematics_estimated.position)

    def getCurVelocity(self):
        return self.v2t(self.client.getMultirotorState().kinematics_estimated.linear_velocity)

    def v2t(self, vect):
        if isinstance(vect, airsim.Vector3r):
            res = np.array([vect.x_val, vect.y_val, vect.z_val])
        else:
            res = np.array(vect)
        return res

    def distance(self, pos1, pos2):
        pos1 = self.v2t(pos1)
        pos2 = self.v2t(pos2)
        dist = np.linalg.norm(pos1 - pos2)
        return dist

# -------------------------------------------------------
# height control
# continuous control

class drone_env_heightcontrol(drone_env):
    def __init__(self, start=[-23, 0, -10], aim=[-23, 60, -10], scaling_factor=2, img_size=[64, 64]):
        drone_env.__init__(self, start, aim)
        self.scaling_factor = scaling_factor
        self.aim = np.array(aim)
        self.height_limit = -30
        self.rand = False
        self.state = self.getState()
        if aim == None:
            self.rand = True
            self.start = np.array([0, 0, -10])
        else:
            self.aim_height = self.aim[2]

    def reset_aim(self):
        self.aim = (np.random.rand(3) * 100).astype("int") - 50
        self.aim[2] = -np.random.randint(10) - 5
        print("Our aim is: {}".format(self.aim).ljust(80, " "), end='\r')
        self.aim_height = self.aim[2]

    def reset(self):
        if self.rand:
            self.reset_aim()
        drone_env.reset(self)
        self.state = self.getState()
        return self.state

    def getState(self):
        state = super().getState()

        state["depth"] = self.getImg('depth')
        state["image"] = self.getImg('rgb')
        state["robot-state"] = self.getCurPosition() - self.aim
        state["robot-position"] = self.getCurPosition()
        state["robot-velocity"] = self.getCurVelocity()

        return state

    def step(self, action):

        # temp = np.sqrt(dpos[0] ** 2 + dpos[1] ** 2)
        # dx = - dpos[0] / temp * self.scaling_factor
        # dy = - dpos[1] / temp * self.scaling_factor
        # dz = - action[2] * self.scaling_factor
        # drone_env.moveByDist(self, [dx, dy, dz], forward=True)

        drone_env.moveByDist(self, action, forward=True)
        state_ = self.getState()

        info = None
        done = False
        reward = self.rewardf(self.state, state_)

        if self.isDone():
            done = True
            reward = 50
            info = "success"

        if self.client.simGetCollisionInfo().has_collided:
            reward = -50
            done = True
            info = "collision"

        if self.distance(self.state["robot-position"], state_["robot-position"]) < 1e-3:
            done = True
            info = "freeze"
            reward = -50

        self.state = state_
        infor = {}
        infor['info'] = info

        print("aim:", self.aim, "pos:", self.state["robot-position"], "action:", action, "  reward: ", reward, "info: ", info)

        return state_, reward, done, infor

    def isDone(self):
        # pos = v2t(self.client.getPosition())
        pos = self.v2t(self.client.getMultirotorState().kinematics_estimated.position)
        pos[2] = self.aim[2]
        if self.distance(self.aim, pos) < self.threshold:
            return True
        return False

    def rewardf(self, state_old, state_cur):
        dis_old = self.distance(state_old["robot-position"], self.aim)
        dis_cur = self.distance(state_cur["robot-position"], self.aim)
        reward = dis_old - dis_cur
        return reward

    def getImg(self, type):
        image_size = 84

        if type == 'depth':
            imageType = airsim.ImageType.DepthPerspective
            pixels_as_float = True
        elif type == 'rgb':
            imageType = airsim.ImageType.Scene
            pixels_as_float = False
        else:
            imageType = airsim.ImageType.Segmentation
            pixels_as_float = False

        responses = self.client.simGetImages([airsim.ImageRequest(0, imageType, pixels_as_float, False)])
        while responses[0].height == 0:
            responses = self.client.simGetImages([airsim.ImageRequest(0, imageType, pixels_as_float, False)])

        if pixels_as_float:
            img1d = np.array(responses[0].image_data_float, dtype=np.float)
        else:
            img1d = np.fromstring(responses[0].image_data_uint8, dtype=np.uint8)

        if type == 'rgb':
            img2d = np.reshape(img1d, (responses[0].height, responses[0].width, 3))
        else:
            img2d = np.reshape(img1d, (responses[0].height, responses[0].width))

        if type == 'depth':
            image = Image.fromarray(img2d)
            im_final = np.array(image.resize((image_size, image_size)).convert('L'), dtype=np.float) / 255
            im_final.resize((image_size, image_size, 1))
            # if IMAGE_VIEW:
            #     cv2.imshow("view", im_final)
            #     key = cv2.waitKey(1) & 0xFF
        else:
            im_final = img2d

        return im_final