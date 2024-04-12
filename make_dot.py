from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
import numpy as np
import control_arm as ca
import math


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


    # start_x, start_y = (x + dx/2) + (dx*i), (y + dy/2) + (dy*j)
    # print(start_x, start_y)
    ## multiply dx and dy (in reference from the center square) and multiply by some constant 
    # a little bit of clearance so it doesn't initially draw
    # RANDOM_UPPER_OFFSET = 0.015
    # bot.arm.set_ee_pose_components(x=start_x, y=start_y, z=.0915+RANDOM_UPPER_OFFSET, moving_time=1)
    # time.sleep(1)

    # center = np.array([.2, .1, 0.0915])
    # # radius = 0.05
    # radius = 0.025
    # flag = True
    # num_points = 200
    # for i in range(num_points+20): # +20 since it doesn't complete the circle at +0
    #     theta = 2 * np.pi * i / num_points
    #     x = center[0] + radius * np.cos(theta)
    #     y = center[1] + radius * np.sin(theta)
    #     z = center[2]
    #     if (flag):
    #         bot.arm.set_ee_pose_components(x=x, y=y, z=z+RANDOM_UPPER_OFFSET, moving_time=1)
    #         flag = False
    #     bot.arm.set_ee_pose_components(x=x, y=y, z=z, moving_time=0.05)

    # bot.arm.set_ee_pose_components(x=x, y=y, z=z+0.01, moving_time=2)  
    
    
    # bot.arm.go_to_home_pose()
    # bot.arm.go_to_sleep_pose()
    # bot.shutdown()
    
    # 117, 200
    ## makes dots on the board to calibrate the pixel space to robot frame space
    ## uses linear transforamtion to convert 
    new_x, new_y = ca.pixel_space_to_robot_frame(328, 250)  
    print(new_x, new_y)  
    bot.arm.set_ee_pose_components(x=new_x, y=new_y, z=0.2, moving_time=2)
    bot.arm.set_ee_pose_components(x=new_x, y=new_y, z=0.07, moving_time=2)
    bot.arm.set_ee_pose_components(x=new_x, y=new_y, z=0.2, moving_time=2)

    # bot.arm.set_ee_pose_components(x=0.5, z=0.2, moving_time=2)
    # bot.arm.set_ee_pose_components(x=0.5, z=0.09, moving_time=2)
    # bot.arm.set_ee_pose_components(x=0.5, z=0.2, moving_time=2)

    # bot.arm.set_ee_pose_components(x=0.3, y = 0.2, z=0.2, moving_time=2)
    # bot.arm.set_ee_pose_components(x=0.3, y = 0.2, z=0.07, moving_time=2)
    # bot.arm.set_ee_pose_components(x=0.3, y = 0.2, z=0.2, moving_time=2)
    # bot.arm.go_to_home_pose()

    # bot.arm.set_ee_pose_components(x=0.4, y=-0.1, z=0.2, moving_time=2)
    # bot.arm.set_ee_pose_components(x=0.4, y=-0.1, z=0.07, moving_time=2)
    # bot.arm.set_ee_pose_components(x=0.4, y=-0.1, z=0.2, moving_time=2)

    # bot.arm.set_ee_pose_components(x=0.25, y = 0.1, z=0.2, moving_time=2)
    # bot.arm.set_ee_pose_components(x=0.25, y = 0.1, z=0.07, moving_time=2)
    # bot.arm.set_ee_pose_components(x=0.25, y = 0.1, z=0.2, moving_time=2)

    bot.arm.go_to_home_pose()
    bot.arm.go_to_sleep_pose()
    bot.shutdown()


if __name__ == '__main__':
    main()
