import sys
import time
PEN_OFFSET = 0.055
# # center position of sphere in arm's coordinate frame.
# C = (0.21, 0.0, 0.1)
# # radius of sphere in meters.
# R = 0.19 #0.115
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
import numpy as np
def main():
    bot = InterbotixManipulatorXS(        
        robot_model='wx250',
        group_name='arm',
        gripper_name='gripper'
        )
        
    if (bot.arm.group_info.num_joints < 5):
        bot.core.get_logger().fatal('This demo requires the robot to have at least 5 joints!')
        bot.shutdown()
        sys.exit()

    # a little bit of clearance so it doesn't initially draw
    RANDOM_UPPER_OFFSET = 0.015
    bot.arm.set_ee_pose_components(x=0.4, y= 0.1, z=0.0935+RANDOM_UPPER_OFFSET, moving_time=2)
    time.sleep(1)

    center = np.array([0.4, 0.1, 0.0935])
    radius = 0.05
    flag = True
    num_points = 200
    for i in range(num_points+20): # +20 since it doesn't complete the circle at +0
        theta = 2 * np.pi * i / num_points
        x = center[0] + radius * np.cos(theta)
        y = center[1] + radius * np.sin(theta)
        z = center[2]
        if (flag):
            bot.arm.set_ee_pose_components(x=x, y=y, z=z+RANDOM_UPPER_OFFSET, moving_time=1)
            flag = False
        bot.arm.set_ee_pose_components(x=x, y=y, z=z, moving_time=0.05)

    bot.arm.set_ee_pose_components(x=x, y=y, z=z, moving_time=2)  


    # pos = [0.35, 0.0, 0.08, 0.0, 0.0] # base of main arc
    # bot.arm.set_ee_pose_components(x=pos[0], y=pos[1], z=pos[2], roll=pos[3], pitch=pos[4])
    # bottom edge.
    # dx = 0.0; dz = 0.0; dp = 0.0; dr = 0.5
    # bot.arm.pos = [0.35, 0.0, 0.08, 0.0, 0.0] # base of main arc
    # bot.arm.set_ee_pose_components(x=pos[0], y=pos[1], z=pos[2], roll=pos[3], pitch=pos[4])
    # # bottom edge.
    # dx = 0.0; dz = 0.0; dp = 0.0; dr = 0.5
    # bot.arm.pos = [0.35, 0.0, 0.08, 0.0, 0.0] # base of main arc
    # bot.arm.set_ee_pose_components(x=pos[0], y=pos[1], z=pos[2], roll=pos[3], pitch=pos[4])
    # # bottom edge.
    # dx = 0.0; dz = 0.0; dp = 0.0; dr = 0.5
    # bot.arm.set_ee_cartesian_trajectory(dx, dz, dr, dp, PEN_OFFSET)
    # # right edge.
    # dx = 0.1; dz = 0.0; dp = 0.01; dr = 0.0
    # bot.arm.set_ee_cartesian_trajectory(dx, dz, dr, dp, PEN_OFFSET)
    # # top edge.
    # dx = 0.0; dz = 0.0; dp = 0.0; dr = -1
    # bot.arm.set_ee_cartesian_trajectory(dx, dz, dr, dp, PEN_OFFSET)
    # # left edge.
    # dx = -0.1; dz = 0.0; dp = -0.01; dr = 0.0
    # bot.arm.set_ee_cartesian_trajectory(dx, dz, dr, dp, PEN_OFFSET)
    # # return to home position.(dx, dz, dr, dp, PEN_OFFSET)
    # # right edge.
    # dx = 0.1; dz = 0.0; dp = 0.01; dr = 0.0
    # bot.arm.set_ee_cartesian_trajectory(dx, dz, dr, dp, PEN_OFFSET)
    # # top edge.
    # dx = 0.0; dz = 0.0; dp = 0.0; dr = -1
    # bot.arm.set_ee_cartesian_trajectory(dx, dz, dr, dp, PEN_OFFSET)
    # # left edge.
    # dx = -0.1; dz = 0.0; dp = -0.01; dr = 0.0
    # bot.arm.set_ee_cartesian_trajectory(dx, dz, dr, dp, PEN_OFFSET)
    # # return to home position.(dx, dz, dr, dp, PEN_OFFSET)
    # # right edge.
    # dx = 0.1; dz = 0.0; dp = 0.01; dr = 0.0
    # bot.arm.set_ee_cartesian_trajectory(dx, dz, dr, dp, PEN_OFFSET)
    # # top edge.
    # dx = 0.0; dz = 0.0; dp = 0.0; dr = -1
    # bot.arm.set_ee_cartesian_trajectory(dx, dz, dr, dp, PEN_OFFSET)
    # # left edge.
    # dx = -0.1; dz = 0.0; dp = -0.01; dr = 0.0
    # bot.arm.set_ee_cartesian_trajectory(dx, dz, dr, dp, PEN_OFFSET)
    # return to home position.  
    # pos = [0.4, 0.0, 0.11, 0.0, 0.7] # base of main arc
    # bot.arm.set_ee_pose_components(x=pos[0], y=pos[1], z=pos[2], roll=pos[3], pitch=pos[4])
    # # first arc.
    # dx = -0.18; dz = 0.0756; dr = -0.5236; dp = -0.8
    # dr = -0.7
    # bot.arm.set_ee_arc_trajectory(dx, dz, dr, dp, PEN_OFFSET)
    # # across.
    # dx = 0.0; dz = 0.0; dr = 1.05; dp = 0
    # dr = 1.4
    # bot.arm.set_ee_arc_trajectory(dx, dz, dr, dp, PEN_OFFSET)
    # # back down.
    # dx = 0.16; dz = -0.0756; dr = -0.8; dp = 0.8
    # bot.arm.set_ee_arc_trajectory(dx, dz, dr, dp, PEN_OFFSET)
    bot.arm.go_to_home_pose()
    bot.arm.go_to_sleep_pose()
    bot.shutdown()


if __name__ == '__main__':
    main()
