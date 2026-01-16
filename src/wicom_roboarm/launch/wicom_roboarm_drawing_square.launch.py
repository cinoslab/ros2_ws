from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory("wicom_roboarm")
    servo_yaml = os.path.join(pkg_share, "config", "servos.yaml")

    unified = Node(
        package="wicom_roboarm",
        executable="wicom_roboarm_unified_node.py",
        name="wicom_roboarm_unified",
        output="screen",
        parameters=[
            servo_yaml,
            {
                "i2c_bus": 1,
                "i2c_address": "0x40",
                "use_mux": True,
                "mux_address": "0x70",
                "mux_channel": 2,
            },
        ],
        remappings=[
            ("joint_states", "/pca9685_servo/joint_states"),
            ("enable", "/pca9685_servo/enable"),
            ("disable", "/pca9685_servo/disable"),
            ("home", "/pca9685_servo/home"),
            ("command", "/pca9685_servo/command"),
        ],
    )

    drawing = Node(
        package="wicom_roboarm",
        executable="wicom_roboarm_drawing_ik_node.py",
        name="wicom_roboarm_drawing_ik",
        output="screen",
        parameters=[
            {
                "servo_command_topic": "/pca9685_servo/command",
                "vl53_long_topic": "/vl53/long_range",

                "elbow_up": True,
                "zPlane_cm_default": 15.0,

                "auto_draw": True,
                "auto_loop": True,
                "auto_point_interval": 0.1,
                "auto_square_side_uv": 0.25,
                "auto_points_per_side": 50,

                "send_interval_sec": 0.5,
            }
        ],
    )

    return LaunchDescription([unified, drawing])