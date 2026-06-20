#!/usr/bin/env python3
"""Launch RDK X5 DNN detection without starting another camera node."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package_share = get_package_share_directory('origin_competition_auto')
    config_path = os.path.join(package_share, 'config', 'rdk_yolov8workconfig.json')

    image_topic_arg = DeclareLaunchArgument(
        'image_topic',
        default_value='/image',
        description='CompressedImage topic produced by the existing camera service.',
    )
    hbmem_topic_arg = DeclareLaunchArgument(
        'hbmem_topic',
        default_value='/hbmem_img',
        description='Shared-memory image topic consumed by dnn_node_example.',
    )
    detection_topic_arg = DeclareLaunchArgument(
        'detection_topic',
        default_value='hobot_dnn_detection',
        description='ai_msgs/PerceptionTargets output topic.',
    )
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=config_path,
        description='DNN work config file.',
    )
    dump_render_arg = DeclareLaunchArgument(
        'dump_render_img',
        default_value='0',
        description='Whether dnn_node_example should dump rendered debug images.',
    )

    shared_mem_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('hobot_shm'),
                'launch/hobot_shm.launch.py',
            )
        )
    )

    jpeg_to_nv12_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('hobot_codec'),
                'launch/hobot_codec_decode.launch.py',
            )
        ),
        launch_arguments={
            'codec_in_mode': 'ros',
            'codec_out_mode': 'shared_mem',
            'codec_sub_topic': LaunchConfiguration('image_topic'),
            'codec_pub_topic': LaunchConfiguration('hbmem_topic'),
        }.items(),
    )

    dnn_node = Node(
        package='dnn_node_example',
        executable='example',
        output='screen',
        parameters=[
            {'config_file': LaunchConfiguration('config_file')},
            {'dump_render_img': LaunchConfiguration('dump_render_img')},
            {'feed_type': 1},
            {'is_shared_mem_sub': 1},
            {'msg_pub_topic_name': LaunchConfiguration('detection_topic')},
        ],
        arguments=['--ros-args', '--log-level', 'warn'],
    )

    return LaunchDescription([
        image_topic_arg,
        hbmem_topic_arg,
        detection_topic_arg,
        config_file_arg,
        dump_render_arg,
        shared_mem_node,
        jpeg_to_nv12_node,
        dnn_node,
    ])
