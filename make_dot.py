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

    ## makes dots on the board to calibrate the pixel space to robot frame space
    ## uses linear transforamtion to convert 
    new_x, new_y = ca.pixel_space_to_robot_frame(393, 209)  
    print(new_x, new_y)  
    bot.arm.set_ee_pose_components(x=new_x, y=new_y, z=0.2, moving_time=2)
    bot.arm.set_ee_pose_components(x=new_x, y=new_y, z=0.09, moving_time=2)
    bot.arm.set_ee_pose_components(x=new_x, y=new_y, z=0.2, moving_time=2)

    # bot.arm.set_ee_pose_components(x=0.5, z=0.2, moving_time=2)
    # bot.arm.set_ee_pose_components(x=0.5, z=0.09, moving_time=2)
    # bot.arm.set_ee_pose_components(x=0.5, z=0.2, moving_time=2)

    # bot.arm.set_ee_pose_components(x=0.3, y = 0.2, z=0.2, moving_time=2)
    # bot.arm.set_ee_pose_components(x=0.3, y = 0.2, z=0.09, moving_time=2)
    # bot.arm.set_ee_pose_components(x=0.3, y = 0.2, z=0.2, moving_time=2)


    bot.arm.go_to_home_pose()
    bot.arm.go_to_sleep_pose()
    bot.shutdown()


if __name__ == '__main__':
    main()
