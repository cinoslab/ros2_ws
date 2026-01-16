cmake_minimum_required(VERSION 3.8)
project(wicom_roboarm)

find_package(ament_cmake REQUIRED)

find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_srvs REQUIRED)

install(PROGRAMS
  src/wicom_roboarm_unified_node.py
  src/wicom_roboarm_drawing_ik_node.py
  scripts/pca9685_kill_all.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

install(FILES resource/${PROJECT_NAME}
  DESTINATION share/ament_index/resource_index/packages
)

ament_package()