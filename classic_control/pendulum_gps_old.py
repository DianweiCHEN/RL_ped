import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from numpy import pi
from os import path
import math
from gym.envs.classic_control import rendering
import logging
import numpy as np
# import gym
# from gym import spaces
from gym.utils import seeding
from numpy import pi
from numpy import sin
from numpy import cos
from numpy import linalg as LA
import gym
from gym.envs.classic_control import rendering
from matplotlib import animation
import matplotlib.pyplot as plt


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)  # <<<<<<<<<<<<<<<<<<<<


class PendulumEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(self, g=10.0):
        #initialize the physical environment
        n_towers=5
        self.n_towers=n_towers
        rx = 500*np.random.rand()
        ry = 500*np.random.rand()
        vx = 10*cos(pi*np.random.rand())
        vy = 10*sin(pi*np.random.rand())
        self.x=np.transpose(np.array([rx, ry, vx, vy]))

        x_min=-1000
        x_max=1000
        y_min = -100
        y_max=1000
        vx_min=-15
        vx_max=15
        vy_min=-15
        vy_max=15
        state_low=np.transpose(np.array([x_min,y_min, vx_min, vy_min]))
        state_high = np.transpose(np.array([x_max, y_max, vx_max, vy_max]))


        # dynamics model: 1= turn -1= straight
        self.dynamics=1
        # define two dynamics models
        # turning

        omega=0.1
        s_w = 0.01

        ts=0.01

        self.q_mx_circle=s_w*np.array([[2*(omega * ts - sin(omega * ts)) / omega ** 3, 0, (1 - cos(omega * ts))/omega**2, (omega * ts - sin(omega * ts))/omega ** 2],
                    [0, 2 * (omega * ts - sin(omega * ts)) / omega ** 3, -(omega*ts-sin(omega*ts))/omega**2, (1-cos(omega*ts))/omega**2],
                    [(1-cos(omega * ts))/omega**2, -(omega*ts-sin(omega*ts))/omega**2, ts,0],
                    [(omega*ts-sin(omega*ts))/omega**2, (1-cos(omega*ts))/omega**2,0, ts]])
        self.f_mx_circle=[[1,0, sin(omega*ts)/omega, (cos(omega*ts)-1)/omega],
           [0,1,(1-cos(omega*ts))/omega,sin(omega*ts)/omega],
           [0,0, cos(omega*ts),        -sin(omega*ts)],
           [0,0,sin(omega*ts),        cos(omega*ts)]]

        #straight line
        q=0.1
        self.q_mx_straight= q*np.array([[1/3*ts**3, 0, 0.5*ts**2, 0],
                                 [0, 1/3*ts**3, 0, 0.5*ts**2],
                                 [0.5*ts**2, 0, ts,  0],
                                 [0,  0.5*ts**2, 0,  ts]])
        self.f_mx_straight=np.array([[1, 0, ts, 0],
                                     [0, 1, 0, ts],
                                     [0, 0, 1, 0],
                                     [0, 0, 0, 1]]

        )



        sigma2_sm = 100

        self.r_mx = sigma2_sm*np.eye(n_towers, dtype = np.float)


        #initialize EKF
        self.p_mx = np.diag([10, 10, 1, 1])
        self.xhat=self.x+np.dot(np.sqrt(self.p_mx), np.random.randn(4))

        # Define action range
        self.max_bias = 10
        self.min_bias = -10

        self.dt = ts


        self.viewer = None

        # self.xcar_start = 0
        # self.ycar_start = 0
        # self.xped_start = 50#+np.random.normal(-20,20)
        # self.yped_start = -5
        # self.lcar = 1  # actually half the car's length
        # self.wcar = 0.5  # actually half the car's width
        # self.initangle = pi / 2
        # self.tau = 0.1  # seconds between state updates
        # self.vcar = 7
        # self.vped = 3
        # self.min_action = -2*pi
        # self.max_action = 2*pi
        # Distance at which to fail the episode
        # self.x_distance_threshold_min = 1
        # self.y_distance_threshold_min = 1
        # self.x_distance_threshold_max = 200
        # self.y_distance_threshold_max = 200
        # self.viewer = rendering.Viewer(600, 400)

        # self.action_space = spaces.Box(
        #     low=-pi, high=pi, shape=(1,), dtype=np.float32
        # )
        # self.observation_space = spaces.Box()
        # high = np.array([1000, 1000, 1000, 5, np.finfo(np.float32).max, 5], dtype=np.float32)
        # low = np.array([-50, 1000, -50, -5, -np.finfo(np.float32).max, 0], dtype=np.float32)
        # high = np.array([1.0, 1.0, self.max_speed], dtype=np.float32)
        self.action_space = spaces.Box(
            low=-self.max_bias*np.ones((n_towers,1 )), high=self.max_bias*np.ones((n_towers,1)), shape=(n_towers,1), dtype=np.float32
        )
        self.observation_space = spaces.Box(low=state_low, high=state_high, dtype=np.float32)
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        n_states=4
        x = self.x
        dynamics=self.dynamics
        # dynamics change probability
        p=0.1

        if np.random.rand()<p:
            dynamics=-dynamics

        if dynamics==1:
            q_mx=self.q_mx_circle
            f_mx=self.f_mx_circle
        else:
            q_mx=self.q_mx_straight
            f_mx=self.f_mx_straight

        r_mx=self.r_mx

        # system state propagation
        x_next=f_mx.dot(x)+np.random.multivariate_normal(np.zeros([1,n_states]), q_mx)
        self.x=x_next

        self.state=x_next
        reward = 1
        # return self._get_obs(f_mx), reward, done, {}


        #
        # xcar, ycar, xped, yped, theta, vcar_x = self.state
        # dt = self.dt
        # # self.last_u = u
        # self.last_xcar = xcar
        # self.last_ycar = ycar
        # self.last_xped = xped
        # self.last_yped = yped
        #
        # u = u#+theta#- pi/4- pi/16
        # # print('u=', u)
        # costheta = np.cos(theta)
        # sintheta = np.sin(theta)
        # in_vcar = self.vcar
        # stopsignal = False
        #
        #
        # self.vcar_x = in_vcar
        #
        # if np.sqrt(abs(xcar-xped)**2+abs(ycar-yped)**2)>10 :#and not self.away
        #     self.vcar_x = in_vcar
        #     if stopsignal:
        #         self.vcar_x = 0
        # elif np.sqrt(abs(xcar-xped)**2+abs(ycar-yped)**2)<=10 and -2.5<yped<2.5:
        #     # self.away = True
        #     self.vcar_x = self.vcar_x-7*dt
        # else:
        #     self.vcar_x = in_vcar
        # if self.vcar_x<0:
        #     stopsignal = True
        #     self.vcar_x = 0
        #
        #
        #
        # vcar_y = 0
        # vped_x = self.vped * costheta
        # vped_y = self.vped * sintheta
        # vcar_x = self.vcar_x
        # last_distance = np.sqrt((xcar - xped) ** 2 + (ycar - yped) ** 2)
        #
        # # u = np.clip(u, self.min_action, self.max_action)[0]
        # self.last_u = u  # for rendering
        # # costs = angle_normalize(th) ** 2 + 0.1 * thdot ** 2 + 0.001 * (u ** 2)
        # newxcar = xcar + self.vcar_x * dt
        # newycar = ycar + vcar_y * dt
        # newxped = xped + vped_x * dt
        # newyped = yped + vped_y * dt
        # newtheta = u#% (2*pi)
        # new_distance = np.sqrt((newxcar - newxped) ** 2 + (newycar - newyped) ** 2)
        #
        # # print('newtheta=', newtheta)
        # # newthdot = thdot + (3 * g / (2 * l) * np.sin(th) + 3.0 / (m * l ** 2) * u) * dt
        # # newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        # # newth = th + newthdot * dt
        # self.state = np.array([newxcar, newycar, newxped, newyped, newtheta, vcar_x])
        # # self.state = np.array([newxcar, newycar, newxped, newyped])
        # # print(self.state,1)
        # var_xcp = newxcar - newxped
        # var_ycp = newycar - newyped
        # act_theta = np.arctan2(var_ycp, var_xcp)
        # dis = np.sqrt(var_xcp**2+var_ycp**2)
        # distance = dis.squeeze()
        # # print(act_theta)
        # # print(distance)
        # # print(abs(act_theta - newtheta)*180/pi)
        # if np.sqrt(abs(newxcar-newxped)**2+abs(newycar-newyped)**2) < self.x_distance_threshold_min: #and < self.y_distance_threshold_min:  # if the car and pedestrian collision
        #     done = True
        # elif abs(newxcar - newxped) > self.x_distance_threshold_max or abs(
        #         newycar - newyped) > self.y_distance_threshold_max:  # if the car and pedestrian are too far away
        #
        #     done = True
        # else:
        #     done = False
        #
        # done = bool(done)
        # # print(self.vcar_x)
        # if not done:
        #     # print(act_theta, newtheta)
        #     # print(var_xcp,var_ycp)
        #     # if yped<-2.5 or yped>2.5:
        #     #     reward = -pi+0/(abs(act_theta - newtheta)+1)
        #     if abs(act_theta - newtheta) < pi / 2: # if the difference of pedestrain's velocity angle and car&ped's true angle < pi/8
        #         # reward = 1/(abs(act_theta - newtheta)+0.05)+100/last_distance
        #         reward = 1/(abs(act_theta - newtheta)%(2*pi)+0.05)
        #         # reward = -10/(sqrt((newxcar-newxped)**2+(newycar-newyped)**2)+0.1)
        #         # reward = 0
        #         #500/distance # means the pedestrian run towards the car
        #     else:
        #         # reward = -(abs(act_theta - newtheta)+0)*0-100 / last_distance
        #         # reward = -1/(abs(abs(act_theta - newtheta)-2*pi/2)+0.05)-100/last_distance
        #         reward = -(abs(act_theta - newtheta)%(2*pi)+0)-1
        #         # reward = -10 / (np.sqrt((newxcar - newxped) ** 2 + (newycar - newyped) ** 2) + 0.1)-1
        #         # reward = -2/(pi-abs(act_theta - newtheta)+0.5)-1
        #         # reward = 0
        #         # reward = -2
        #         # print(reward.size)
        #         #-500/distance*1.5
        # elif done and self.steps_beyond_done is None:
        #     self.steps_beyond_done = 0
        #     if np.sqrt(abs(newxcar - newxped) ** 2 + abs(newycar - newyped) ** 2) < self.x_distance_threshold_min:
        #     # if abs(newxcar - newxped) < self.x_distance_threshold_min and abs(
        #     #         newycar - newyped) < self.y_distance_threshold_min:  # if the car and pedestrian collision
        #     #     reward = [[500]] #1000*self.vcar_x+
        #     #     reward = [[3000]]
        #         reward = np.sqrt((vped_x-((50-1000)*vped_x+2*1000*vcar_x)/1050)**2+(vped_y-((50-1000)*vped_y+2*1000*vcar_y)/1050)**2)*50*5
        #
        #     else:
        #
        #             # abs(newxcar - newxped) > self.x_distance_threshold_max and abs(
        #             # newycar - newyped) > self.y_distance_threshold_max:  # if the car and pedestrian are too far away
        #         # reward = [[-0]]
        #         reward = [[0]]
        #
        # else:
        #     if self.steps_beyond_done == 0:
        #         logger.warn("""
        #         You are calling 'step()' even though this environment has already returned
        #         done = True. You should always call 'reset()' once you receive 'done = True'
        #         Any further steps are undefined behavior.
        #                         """)
        #         self.steps_beyond_done += 1
        #         reward = [[0.0]]
        # print(done)
        # print(1)
        # print(reward)
        # print((self._get_obs()))
        return self._get_obs(f_mx, q_mx), reward, done, {}


    def reset(self):

        # self.state = self.np_random.uniform(low=-high, high=high)
        rx = 500 * np.random.rand()
        ry = 500 * np.random.rand()
        vx = 10 * cos(pi * np.random.rand())
        vy = 10 * sin(pi * np.random.rand())
        self.x = np.transpose(np.array([rx, ry, vx, vy]))

        self.last_u = None
        # self.last_xcar = 0#+np.random.normal(0,5)
        # self.last_ycar = 0
        # self.last_xped = 50+np.random.randint(-50,50)
        # # self.xped_start
        # self.last_yped = -5
        self.steps_beyond_done = None
        self.viewer = None
        self.done = False
        self.away = False
        self.j = 0
        # origin = np.array([self.xcar_start, self.ycar_start, self.last_xped, self.yped_start, np.pi / 4 * 3, self.vcar])
        # self.state = origin
        return self._get_obs()

    def _get_obs(self, f_mx,q_mx):
        n_towers=self.n_towers


        #tower postions
        rs1_0 = np.array([-110, 240]).transpose()
        rs2_0 = np.array([-150, 340]).transpose()
        rs3_0 = np.array([-215, -60]).transpose()
        rs4_0 = np.array([-250, 105]).transpose()
        rs5_0 = np.array([30, 80]).transpose()
        rs = np.column_stack((rs1_0, rs2_0, rs3_0, rs4_0, rs5_0))


        p_mx=self.p_mx

        r_mx=self.r_mx

        # observation is the state estimates from EKF
        x=self.x
        n_states=x.size

        bias=self.get_bias
        # mesasurement
        z= measFunc(x,rs)+bias+np.random.multivariate_normal(np.zeros(n_towers), r_mx)
        xhat_current=self.xhat

        x_pred = f_mx.dot(xhat_current)
        p_mx_pred=f_mx.dot(p_mx).dot(f_mx.transpose())+q_mx
        y_hat=measFunc(x_pred)
        y_tilde=z-y_hat
        h_mx=measJacobian(x_pred, rs)
        s_mx=h_mx.dot(p_mx).dot(h_mx.transpose())+r_mx
        k_mx=p_mx.dot(h_mx.transpose()).dot(LA.inv(s_mx))
        xhat=x_pred+k_mx.dot(y_tilde)
        p_mx_upd=(np.eye(n_states) -k_mx.dot(h_mx)).dot(p_mx_pred).dot(np.transpose(np.eye(n_states) -k_mx.dot(h_mx)))+k_mx.dot(r_mx).dot(k_mx.traanspose())

        self.p_mx=p_mx_upd


        return xhat


    
    def render(self, mode="human", close=False):
        # close()
        # if self.viewer is None:
        carwidth = 100
        carlength = 200
        # cary = 300
        screen_width = 4050
        screen_height = 1550
        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)
            # linelength1 = rendering.line((self.last_xcar
            l, r, t, b = -carlength / 2, carlength / 2, carwidth / 2, -carwidth / 2
            car = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            ped = rendering.make_circle(25)
            self.cartrans = rendering.Transform()
            self.pedtrans = rendering.Transform()
            car.add_attr(self.cartrans)
            ped.add_attr(self.pedtrans)
            car.set_color(255, 0, 0)
            ped.set_color(0, 0, 255)
            self.viewer.add_geom(car)
            self.viewer.add_geom(ped)

            # carx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        # 设置平移属性
        
        if self.state is None: return None

        s = self.state
        carx = (s[0]-475) * 100 + 50  # MIDDLE OF CART
        cary = s[1] * 100 + 1050

        pedx = (s[2]-475) * 100 + 50
        pedy = s[3] * 100 + 550

        # 设置平移属性
        # self.trackc = rendering.Line((50, 1050), (carx, cary))
        # self.trackc.set_color(0, 0, 0)
        # self.viewer.add_geom(self.trackc)
        # self.trackp = rendering.Line((2050, 550), (pedx, pedy))  # Rl
        # self.trackp.set_color(0, 0, 0)
        # self.viewer.add_geom(self.trackp)
        self.cartrans.set_translation(carx, cary)
        self.pedtrans.set_translation(pedx, pedy)
        # self.poletrans.set_rotation(-x[2])
        # self.carttrans.set_translation(cartx, carty)
        # self.poletrans.set_rotation(-x[2])
        
        return self.viewer.render(return_rgb_array=mode == "rgb_array")
        #

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):

        # Mess with this to change frame size
        plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

        patch = plt.imshow(frames[0])
        plt.axis('off')

        def animate(i):
            patch.set_data(frames[i])

        anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
        anim.save(path + filename, writer='imagemagick', fps=60)


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi


def measFunc(x,rs):

    n_towers=rs.shape[1]
    h=LA.norm(rs-np.tile(x[0:2], (n_towers,1)).transpose(), axis = 0)
    return h

def measJacobian(x,rs):
    n_states=x.size
    n_towers = rs.shape[1]
    h_mx=np.zeros([n_towers, n_states])

    h_mx[:,0]=np.divide(x[0]-rs[0,:],  LA.norm(rs-np.tile(x[0:2], (n_towers,1)).transpose(), axis = 0))
    h_mx[:, 1] = np.divide(x[1] - rs[1, :], LA.norm(rs - np.tile(x[0:2], (n_towers, 1)).transpose(), axis=0))


    return h_mx

def get_bias(self):
    return bias



