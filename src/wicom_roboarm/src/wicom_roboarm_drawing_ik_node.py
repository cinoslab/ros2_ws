#!/usr/bin/env python3
import math
import time

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Point
from sensor_msgs.msg import JointState


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def wrap_pi(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def rotz(a):
    ca, sa = math.cos(a), math.sin(a)
    return [[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]]


def roty(a):
    ca, sa = math.cos(a), math.sin(a)
    return [[ca, 0.0, sa], [0.0, 1.0, 0.0], [-sa, 0.0, ca]]


def matmul(A, B):
    return [
        [A[i][0] * B[0][j] + A[i][1] * B[1][j] + A[i][2] * B[2][j] for j in range(3)]
        for i in range(3)
    ]


def matT(A):
    return [[A[0][0], A[1][0], A[2][0]], [A[0][1], A[1][1], A[2][1]], [A[0][2], A[1][2], A[2][2]]]


class RoboArmIKXYZNode(Node):
    """
    Strategy (so you can calibrate from real results):
      1) solve IK -> q_geo (deg) (can be negative)
      2) map to servo cmd with per-joint sign and home:
            cmd = home + sign * q_geo
         (NO other assumptions)
      3) print both IK and CMD
      4) publish CMD to /pca9685_servo/command

    You can change sign_* params at runtime to fix direction without touching IK math.
    """

    def __init__(self):
        super().__init__("wicom_roboarm_ik_xyz")

        # Geometry (cm)
        self.declare_parameter("d1_base_to_shoulder_cm", 4.0)
        self.declare_parameter("L2_shoulder_to_elbow_cm", 12.0)
        self.declare_parameter("L3_elbow_to_wristpitch_cm", 14.0)  # 6 + 8
        self.declare_parameter("L_tool_from_wristpitch_to_tip_cm", 13.0)  # 6.5 + 6.5

        # Topics
        self.declare_parameter("xyz_input_topic", "/target_xyz_cm")
        self.declare_parameter("servo_command_topic", "/pca9685_servo/command")

        # Joint names
        self.declare_parameter("joint_name_base", "base")
        self.declare_parameter("joint_name_shoulder", "shoulder")
        self.declare_parameter("joint_name_elbow", "elbow")
        self.declare_parameter("joint_name_wrist_roll", "wrist_roll")
        self.declare_parameter("joint_name_wrist_pitch", "wrist_pitch")
        self.declare_parameter("joint_name_pen", "pen")

        # Home and sign (calibration knobs)
        self.declare_parameter("home_deg", 90.0)
        self.declare_parameter("sign_base", 1.0)
        self.declare_parameter("sign_shoulder", 1.0)
        self.declare_parameter("sign_elbow", 1.0)
        self.declare_parameter("sign_wrist_roll", 1.0)
        self.declare_parameter("sign_wrist_pitch", 1.0)
        self.declare_parameter("sign_pen", 1.0)

        # branch selection / stability
        self.declare_parameter("elbow_up", False)
        self.declare_parameter("pen_roll_deg_default", 0.0)

        # publish rate limit (avoid spamming the driver)
        self.declare_parameter("max_publish_hz", 5.0)

        # Read params
        self.d1 = float(self.get_parameter("d1_base_to_shoulder_cm").value)
        self.L2 = float(self.get_parameter("L2_shoulder_to_elbow_cm").value)
        self.L3 = float(self.get_parameter("L3_elbow_to_wristpitch_cm").value)
        self.Ltool = float(self.get_parameter("L_tool_from_wristpitch_to_tip_cm").value)

        self.xyz_topic = str(self.get_parameter("xyz_input_topic").value)
        self.cmd_topic = str(self.get_parameter("servo_command_topic").value)

        self.j_base = str(self.get_parameter("joint_name_base").value)
        self.j_shoulder = str(self.get_parameter("joint_name_shoulder").value)
        self.j_elbow = str(self.get_parameter("joint_name_elbow").value)
        self.j_wroll = str(self.get_parameter("joint_name_wrist_roll").value)
        self.j_wpitch = str(self.get_parameter("joint_name_wrist_pitch").value)
        self.j_pen = str(self.get_parameter("joint_name_pen").value)

        self.home = float(self.get_parameter("home_deg").value)
        self.sign_base = float(self.get_parameter("sign_base").value)
        self.sign_sh = float(self.get_parameter("sign_shoulder").value)
        self.sign_el = float(self.get_parameter("sign_elbow").value)
        self.sign_wr = float(self.get_parameter("sign_wrist_roll").value)
        self.sign_wp = float(self.get_parameter("sign_wrist_pitch").value)
        self.sign_pen = float(self.get_parameter("sign_pen").value)

        self.elbow_up = bool(self.get_parameter("elbow_up").value)
        self.pen_roll_default = math.radians(float(self.get_parameter("pen_roll_deg_default").value))

        self.max_hz = float(self.get_parameter("max_publish_hz").value)
        self._min_period = 1.0 / self.max_hz if self.max_hz > 0 else 0.0
        self._last_pub_t = 0.0

        self._last_q = None

        self.pub_cmd = self.create_publisher(JointState, self.cmd_topic, 10)
        self.sub_xyz = self.create_subscription(Point, self.xyz_topic, self._on_xyz, 10)

        self.get_logger().info(
            f"IK XYZ node started. xyz_topic={self.xyz_topic} cmd_topic={self.cmd_topic} "
            f"home={self.home} signs=[{self.sign_base},{self.sign_sh},{self.sign_el},{self.sign_wr},{self.sign_wp},{self.sign_pen}]"
        )

    def _on_xyz(self, msg: Point):
        now = time.monotonic()
        if self._min_period > 0 and (now - self._last_pub_t) < self._min_period:
            return

        x = float(msg.x)
        y = float(msg.y)
        z = float(msg.z)

        sol = self.solve_ik(x, y, z)
        if sol is None:
            self.get_logger().warn(f"IK failed for target (cm)=({x:.2f},{y:.2f},{z:.2f})")
            return

        self._last_q = sol
        ik_deg = [math.degrees(a) for a in sol]
        cmd_deg = self.map_geo_deg_to_cmd_deg(ik_deg)

        self.get_logger().info(
            f"Target(cm)=({x:.2f},{y:.2f},{z:.2f}) | "
            f"IK(deg) base={ik_deg[0]:.2f} sh={ik_deg[1]:.2f} el={ik_deg[2]:.2f} wr={ik_deg[3]:.2f} wp={ik_deg[4]:.2f} pen={ik_deg[5]:.2f} | "
            f"CMD(deg) base={cmd_deg[0]:.2f} sh={cmd_deg[1]:.2f} el={cmd_deg[2]:.2f} wr={cmd_deg[3]:.2f} wp={cmd_deg[4]:.2f} pen={cmd_deg[5]:.2f}"
        )

        self.publish_servo_cmd(cmd_deg)
        self._last_pub_t = now

    def map_geo_deg_to_cmd_deg(self, ik_deg):
        base, sh, el, wr, wp, pen = ik_deg
        return [
            self.home + self.sign_base * base,
            self.home + self.sign_sh * sh,
            self.home + self.sign_el * el,
            self.home + self.sign_wr * wr,
            self.home + self.sign_wp * wp,
            self.home + self.sign_pen * pen,
        ]

    def publish_servo_cmd(self, cmd_deg):
        cmd = JointState()
        cmd.name = [self.j_base, self.j_shoulder, self.j_elbow, self.j_wroll, self.j_wpitch, self.j_pen]
        cmd.position = [float(v) for v in cmd_deg]
        self.pub_cmd.publish(cmd)

    def solve_ik(self, x_tip, y_tip, z_tip):
        # Tool constraint (current model): tool_z -> +X0
        # Wrist_pitch point = tip - Ltool along +X0
        xw = x_tip - self.Ltool
        yw = y_tip
        zw = z_tip

        # Base yaw (geometry)
        q1 = math.atan2(yw, xw)

        # Shoulder/elbow planar
        r = math.sqrt(xw * xw + yw * yw)
        zp = zw - self.d1

        D = math.sqrt(r * r + zp * zp)
        if D < 1e-6:
            return None
        if D > (self.L2 + self.L3) + 1e-6 or D < abs(self.L2 - self.L3) - 1e-6:
            return None

        c3 = (D * D - self.L2 * self.L2 - self.L3 * self.L3) / (2.0 * self.L2 * self.L3)
        c3 = clamp(c3, -1.0, 1.0)

        q3a = math.acos(c3)
        q3b = -q3a

        def q2_for(q3):
            phi = math.atan2(zp, r)
            psi = math.atan2(self.L3 * math.sin(q3), self.L2 + self.L3 * math.cos(q3))
            return phi - psi

        q2a = q2_for(q3a)
        q2b = q2_for(q3b)

        cands = [(q1, q2a, q3a), (q1, q2b, q3b)]
        if self._last_q is None:
            q1, q2, q3 = cands[1] if self.elbow_up else cands[0]
        else:
            best = None
            best_cost = 1e18
            for (a1, a2, a3) in cands:
                cost = abs(wrap_pi(a2 - self._last_q[1])) + abs(wrap_pi(a3 - self._last_q[2]))
                if cost < best_cost:
                    best_cost = cost
                    best = (a1, a2, a3)
            q1, q2, q3 = best

        # Wrist orientation (kept as before; you will validate with real robot)
        R03 = matmul(rotz(q1), roty(q2 + q3))
        R30 = matT(R03)

        # pick a consistent R06 with z_tool=+X0 (only constraint we know)
        R06 = [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ]
        R36 = matmul(R30, R06)

        # solve Rx(q4)Ry(q5)Rz(q6)
        s5 = clamp(R36[0][2], -1.0, 1.0)
        q5_1 = math.asin(s5)
        q5_2 = math.pi - q5_1

        def q4_q6_for(q5):
            c5 = math.cos(q5)
            if abs(c5) < 1e-6:
                q6 = self._last_q[5] if self._last_q is not None else self.pen_roll_default
                q4 = 0.0
                return q4, q6
            q4 = math.atan2(-R36[1][2], R36[2][2])
            q6 = math.atan2(-R36[0][1], R36[0][0])
            return q4, q6

        wrist = []
        for q5c in (q5_1, q5_2):
            q4c, q6c = q4_q6_for(q5c)
            wrist.append((q4c, q5c, q6c))

        if self._last_q is None:
            # bias q6 to default
            q4, q5, q6 = min(wrist, key=lambda t: abs(wrap_pi(t[2] - self.pen_roll_default)))
        else:
            q4, q5, q6 = min(
                wrist,
                key=lambda t: abs(wrap_pi(t[0] - self._last_q[3])) + abs(wrap_pi(t[1] - self._last_q[4])) + abs(wrap_pi(t[2] - self._last_q[5])),
            )

        return (q1, q2, q3, q4, q5, q6)


def main():
    rclpy.init()
    node = RoboArmIKXYZNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()