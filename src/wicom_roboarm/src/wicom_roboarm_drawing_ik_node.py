#!/usr/bin/env python3
import math

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Point
from sensor_msgs.msg import JointState, Range


class RoboArmDrawingIKNode(Node):
    """
    ROS2 node that mirrors Unity client's mapping + IK:
      - Input: UV (0..1) + penDown via geometry_msgs/Point (x=u, y=v, z=penDown 0/1)
      - zPlane_cm:
          * Fixed mode (default): zPlane_cm_default (+ z_offset_cm)
          * Sensor mode: from VL53 long range topic (m -> cm) + offset
            controlled by use_vl53_for_z
      - Solve IK and publish JointState to servo controller command topic
    """

    def __init__(self):
        super().__init__("wicom_roboarm_drawing_ik")

        # ---------- Params ----------
        self.declare_parameter("L1_cm", 6.0)
        self.declare_parameter("L2_cm", 5.5)
        self.declare_parameter("L3_cm", 5.5)

        self.declare_parameter("elbow_up", False)
        self.declare_parameter("z_offset_cm", 2.0)
        self.declare_parameter("zPlane_cm_default", 15.0)
        self.declare_parameter("use_yspan_as_draw_area", True)

        # ONLY CHANGE: toggle using sensor Z (default OFF => fixed z)
        self.declare_parameter("use_vl53_for_z", False)

        self.declare_parameter("send_interval_sec", 0.5)

        self.declare_parameter("base_scale", 1.5)
        self.declare_parameter("shoulder_scale", 1.25)
        self.declare_parameter("elbow_scale", 1.25)
        self.declare_parameter("wrist_scale", 1.25)

        self.declare_parameter("base_offset_deg", 90.0)
        self.declare_parameter("shoulder_offset_deg", 90.0)
        self.declare_parameter("elbow_offset_deg", 180.0)
        self.declare_parameter("wrist_offset_deg", 100.0)

        self.declare_parameter("joint_name_base", "base")
        self.declare_parameter("joint_name_shoulder", "shoulder")
        self.declare_parameter("joint_name_elbow", "elbow")
        self.declare_parameter("joint_name_wrist", "wrist_pitch")

        self.declare_parameter("servo_command_topic", "/pca9685_servo/command")

        self.declare_parameter("input_uv_topic", "input_uv")
        self.declare_parameter("vl53_long_topic", "/vl53/long_range")

        self.declare_parameter("auto_draw", False)
        self.declare_parameter("auto_loop", True)
        self.declare_parameter("auto_point_interval", 0.2)
        self.declare_parameter("auto_square_side_uv", 0.25)
        self.declare_parameter("auto_points_per_side", 20)

        # read params
        self.L1_cm = float(self.get_parameter("L1_cm").value)
        self.L2_cm = float(self.get_parameter("L2_cm").value)
        self.L3_cm = float(self.get_parameter("L3_cm").value)

        self.elbow_up = bool(self.get_parameter("elbow_up").value)
        self.z_offset_cm = float(self.get_parameter("z_offset_cm").value)
        self.zPlane_cm_default = float(self.get_parameter("zPlane_cm_default").value)
        self.use_yspan_as_draw_area = bool(self.get_parameter("use_yspan_as_draw_area").value)

        self.use_vl53_for_z = bool(self.get_parameter("use_vl53_for_z").value)

        self.send_interval = float(self.get_parameter("send_interval_sec").value)

        self.base_scale = float(self.get_parameter("base_scale").value)
        self.shoulder_scale = float(self.get_parameter("shoulder_scale").value)
        self.elbow_scale = float(self.get_parameter("elbow_scale").value)
        self.wrist_scale = float(self.get_parameter("wrist_scale").value)

        self.base_offset_deg = float(self.get_parameter("base_offset_deg").value)
        self.shoulder_offset_deg = float(self.get_parameter("shoulder_offset_deg").value)
        self.elbow_offset_deg = float(self.get_parameter("elbow_offset_deg").value)
        self.wrist_offset_deg = float(self.get_parameter("wrist_offset_deg").value)

        self.joint_name_base = str(self.get_parameter("joint_name_base").value)
        self.joint_name_shoulder = str(self.get_parameter("joint_name_shoulder").value)
        self.joint_name_elbow = str(self.get_parameter("joint_name_elbow").value)
        self.joint_name_wrist = str(self.get_parameter("joint_name_wrist").value)

        self.servo_command_topic = str(self.get_parameter("servo_command_topic").value)
        self.uv_topic = str(self.get_parameter("input_uv_topic").value)
        self.vl53_long_topic = str(self.get_parameter("vl53_long_topic").value)

        self.auto_draw = bool(self.get_parameter("auto_draw").value)
        self.auto_loop = bool(self.get_parameter("auto_loop").value)
        self.auto_point_interval = float(self.get_parameter("auto_point_interval").value)
        self.auto_square_side_uv = float(self.get_parameter("auto_square_side_uv").value)
        self.auto_points_per_side = int(self.get_parameter("auto_points_per_side").value)

        # ---------- State ----------
        self._zPlane_cm_fixed = self.zPlane_cm_default + self.z_offset_cm
        self._zPlane_cm = self._zPlane_cm_fixed

        self._have_uv = False
        self._last_u = 0.5
        self._last_v = 0.5
        self._last_pen_down = False

        self._last_send_time = 0.0

        # Auto path
        self._auto_path = []
        self._auto_index = 0
        self._auto_next_time = self.get_clock().now().nanoseconds / 1e9

        # ---------- Pub/Sub ----------
        self.pub_debug_target = self.create_publisher(Point, "debug_target_cm", 10)
        self.pub_debug_angles = self.create_publisher(JointState, "debug_angles", 10)

        self.pub_servo_cmd = self.create_publisher(JointState, self.servo_command_topic, 10)

        self.sub_uv = self.create_subscription(Point, self.uv_topic, self._on_uv, 10)
        self.sub_vl53 = self.create_subscription(Range, self.vl53_long_topic, self._on_vl53_long, 10)

        # Timer 50 Hz internal loop
        self.timer = self.create_timer(0.02, self._tick)

        if self.auto_draw:
            self._build_auto_square_path()

        self.get_logger().info(
            f"DrawingIK node started. input_uv={self.uv_topic} vl53_long={self.vl53_long_topic} "
            f"servo_command_topic={self.servo_command_topic} auto_draw={self.auto_draw} "
            f"use_vl53_for_z={self.use_vl53_for_z} z_fixed={self._zPlane_cm_fixed:.2f}cm"
        )

    # ---------- Subscribers ----------
    def _on_vl53_long(self, msg: Range):
        # only update z when explicitly enabled
        if not self.use_vl53_for_z:
            return
        if msg.range is None or math.isnan(msg.range) or msg.range <= 0.0:
            return
        self._zPlane_cm = (msg.range * 100.0) + self.z_offset_cm

    def _on_uv(self, msg: Point):
        self._last_u = float(msg.x)
        self._last_v = float(msg.y)
        self._last_pen_down = (float(msg.z) >= 0.5)
        self._have_uv = True

    # ---------- Workspace & mapping ----------
    def calculate_workspace(self, zDistanceCm: float):
        r_max_total = self.L1_cm + self.L2_cm + self.L3_cm
        r_max_total_sq = r_max_total * r_max_total
        z_dist_sq = zDistanceCm * zDistanceCm

        if z_dist_sq >= r_max_total_sq:
            xHalfSpan = 0.0
        else:
            xHalfSpan = math.sqrt(r_max_total_sq - z_dist_sq)

        r_max_L1_L2 = self.L1_cm + self.L2_cm
        r_max_L1_L2_sq = r_max_L1_L2 * r_max_L1_L2

        r_wrist_at_X_zero = zDistanceCm - self.L3_cm
        r_wrist_sq = r_wrist_at_X_zero * r_wrist_at_X_zero

        if r_wrist_sq >= r_max_L1_L2_sq:
            yHalfSpan = 0.0
        else:
            yHalfSpan = math.sqrt(r_max_L1_L2_sq - r_wrist_sq)

        return xHalfSpan, yHalfSpan

    def map_to_robot_space_3d(self, u01: float, v01: float, penDown: bool):
        # lock Z when not using sensor
        if not self.use_vl53_for_z:
            self._zPlane_cm = self._zPlane_cm_fixed

        xHalf, yHalf = self.calculate_workspace(self._zPlane_cm)
        draw_area = (2.0 * yHalf) if self.use_yspan_as_draw_area else (2.0 * xHalf)

        Xr_cm = (2.0 * u01 - 1.0) * draw_area
        Yr_cm = (2.0 * v01 - 1.0) * draw_area
        Zr_cm = self._zPlane_cm
        return Xr_cm, Yr_cm, Zr_cm, draw_area

    # ---------- IK ----------
    def solve_ik_3d_with_base(self, Xr_cm, Yr_cm, Zr_cm, elbow_up_mode):
        theta0 = math.atan2(Xr_cm, Zr_cm)

        r_target = math.sqrt(Xr_cm * Xr_cm + Zr_cm * Zr_cm)
        y_target = Yr_cm

        r_wrist = r_target - self.L3_cm
        y_wrist = y_target

        d_sq = r_wrist * r_wrist + y_wrist * y_wrist
        L1_sq = self.L1_cm * self.L1_cm
        L2_sq = self.L2_cm * self.L2_cm

        cosTheta2 = (d_sq - L1_sq - L2_sq) / (2.0 * self.L1_cm * self.L2_cm)
        cosTheta2 = max(-1.0, min(1.0, cosTheta2))

        theta2 = math.acos(cosTheta2)
        if elbow_up_mode:
            theta2 = -theta2

        k1 = self.L1_cm + self.L2_cm * math.cos(theta2)
        k2 = self.L2_cm * math.sin(theta2)

        theta1 = math.atan2(y_wrist, r_wrist) - math.atan2(k2, k1)
        theta3 = -(theta1 + theta2)

        deg0 = math.degrees(theta0)
        deg1 = math.degrees(theta1)
        deg2 = math.degrees(theta2)
        deg3 = math.degrees(theta3)

        degBase = self.base_offset_deg + (deg0 * self.base_scale)
        degShoulder = self.shoulder_offset_deg - (deg1 * self.shoulder_scale)
        degElbow = self.elbow_offset_deg + (deg2 * self.elbow_scale)
        degWrist = self.wrist_offset_deg - (deg3 * self.wrist_scale)

        return degBase, degShoulder, degElbow, degWrist

    # ---------- Auto draw square ----------
    def _build_auto_square_path(self):
        self._auto_path = []
        half = max(0.0, min(1.0, self.auto_square_side_uv)) * 0.5
        cx, cy = 0.5, 0.5

        n = max(2, int(self.auto_points_per_side))

        def lerp(a, b, t):
            return a + (b - a) * t

        # bottom
        for i in range(n):
            t = i / float(n - 1)
            self._auto_path.append((lerp(cx - half, cx + half, t), cy - half, True))
        # right
        for i in range(1, n):
            t = i / float(n - 1)
            self._auto_path.append((cx + half, lerp(cy - half, cy + half, t), True))
        # top
        for i in range(1, n):
            t = i / float(n - 1)
            self._auto_path.append((lerp(cx + half, cx - half, t), cy + half, True))
        # left
        for i in range(1, n - 1):
            t = i / float(n - 1)
            self._auto_path.append((cx - half, lerp(cy + half, cy - half, t), True))

        self._auto_index = 0
        self._auto_next_time = self.get_clock().now().nanoseconds / 1e9

    def _maybe_advance_auto(self, now_s: float):
        if not self._auto_path:
            return

        if now_s < self._auto_next_time:
            return

        u, v, pen = self._auto_path[self._auto_index]
        self._auto_index += 1
        if self._auto_index >= len(self._auto_path):
            if self.auto_loop:
                self._auto_index = 0
            else:
                self.auto_draw = False
                return

        self._last_u = u
        self._last_v = v
        self._last_pen_down = pen
        self._have_uv = True

        self._auto_next_time = now_s + self.auto_point_interval

    # ---------- Main loop ----------
    def _tick(self):
        now = self.get_clock().now().nanoseconds / 1e9

        if self.auto_draw:
            self._maybe_advance_auto(now)

        if not self._have_uv:
            return

        # throttle send
        if (now - self._last_send_time) < self.send_interval:
            return

        u = max(0.0, min(1.0, self._last_u))
        v = max(0.0, min(1.0, self._last_v))
        pen = bool(self._last_pen_down)

        Xr_cm, Yr_cm, Zr_cm, draw_area = self.map_to_robot_space_3d(u, v, pen)

        dbg = Point()
        dbg.x = float(Xr_cm)
        dbg.y = float(Yr_cm)
        dbg.z = float(Zr_cm)
        self.pub_debug_target.publish(dbg)

        if not pen:
            return

        degBase, degShoulder, degElbow, degWrist = self.solve_ik_3d_with_base(
            Xr_cm, Yr_cm, Zr_cm, self.elbow_up
        )

        dbg_js = JointState()
        dbg_js.name = [self.joint_name_base, self.joint_name_shoulder, self.joint_name_elbow, self.joint_name_wrist]
        dbg_js.position = [float(degBase), float(degShoulder), float(degElbow), float(degWrist)]
        self.pub_debug_angles.publish(dbg_js)

        cmd = JointState()
        cmd.name = [self.joint_name_base, self.joint_name_shoulder, self.joint_name_elbow, self.joint_name_wrist]
        cmd.position = [float(degBase), float(degShoulder), float(degElbow), float(degWrist)]
        self.pub_servo_cmd.publish(cmd)

        self._last_send_time = now


def main():
    rclpy.init()
    node = RoboArmDrawingIKNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()