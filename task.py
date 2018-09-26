import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])

        self.prev_pos = np.array([0.0, 0.0, 0.0]) if init_pose is None else np.copy(init_pose[:3])
        self.prev_velocities = np.array([0.0, 0.0, 0.0]) if init_velocities is None else np.copy(init_velocities)
        self.prev_angle_velocities = np.array([0.0, 0.0, 0.0]) if init_angle_velocities is None else np.copy(init_angle_velocities)


    def get_reward(self):
        """Uses current pose of sim to return reward."""
        reward_pos = abs(self.sim.pose[:3] - self.target_pos)
        # reward_pos = np.array([1. if x > 0. else 0. for x in reward_pos])

        # reward_pos_2 = abs(self.sim.pose[:3] - self.prev_pos)
        # reward_pos_2 = np.array([1 if x > 5 else 0 for x in reward_pos_2])
        # self.prev_pos = self.sim.pose[:3]

        reward_vel = abs(self.sim.v - self.prev_velocities)
        reward_vel = np.array([1. if x > 3. else 0. for x in reward_vel])
        self.prev_velocities = self.sim.v

        reward_ang_vel = abs(self.sim.angular_v - self.prev_angle_velocities)
        reward_ang_vel = np.array([1. if x > 3. else 0. for x in reward_ang_vel])
        self.prev_angle_velocities = self.sim.angular_v

        reward = 1. - (0.7 * reward_pos).sum() - \
            (0.1 * reward_vel).sum() - (0.1 * reward_ang_vel).sum() + 0.5 * self.sim.runtime
        return 1 / (1 + np.exp(-reward))

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat)
        return state
