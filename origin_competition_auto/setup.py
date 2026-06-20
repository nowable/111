from glob import glob

from setuptools import find_packages, setup

package_name = 'origin_competition_auto'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.json')),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='origin',
    maintainer_email='origin@example.com',
    description='OriginCar competition automation and calibration tools.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'motion_calibration = origin_competition_auto.motion_calibration:main',
            'square_drive_test = origin_competition_auto.square_drive_test:main',
            'qr_parse_debug = origin_competition_auto.decision_parser:main',
            'qr_capture = origin_competition_auto.qr_capture:main',
            'auto_mission = origin_competition_auto.auto_mission:main',
            'competition_run = origin_competition_auto.competition_run:main',
            'vision_debug = origin_competition_auto.vision_detector:main',
            'motion_state_debug = origin_competition_auto.motion_state:main',
            'lane_debug = origin_competition_auto.lane_follow:main',
            'llm_debug = origin_competition_auto.llm_client:main',
            'mission_replay = origin_competition_auto.mission_replay:main',
            'dataset_capture = origin_competition_auto.dataset_capture:main',
            'dataset_audit = origin_competition_auto.dataset_audit:main',
            'system_check = origin_competition_auto.system_check:main',
            'vision_tune = origin_competition_auto.vision_tune:main',
            'field_data_review = origin_competition_auto.field_data_review:main',
            'apply_review_recommendations = origin_competition_auto.apply_review_recommendations:main',
            'field_session = origin_competition_auto.field_session:main',
            'yolo_pipeline = origin_competition_auto.yolo_pipeline:main',
            'solution_audit = origin_competition_auto.solution_audit:main',
            'route_plan_from_map = origin_competition_auto.route_plan_from_map:main',
            'handoff_bundle = origin_competition_auto.handoff_bundle:main',
            'yolo_overlay_view = origin_competition_auto.yolo_overlay_view:main',
            'yolo_overlay_server = origin_competition_auto.yolo_overlay_server:main',
        ],
    },
)
