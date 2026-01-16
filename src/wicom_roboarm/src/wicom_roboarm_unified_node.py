#!/usr/bin/env python3
import math
import time
import threading

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor, ParameterType

from sensor_msgs.msg import Range, JointState
from std_srvs.srv import Trigger

from smbus2 import SMBus

import board
import busio
import adafruit_tca9548a
import adafruit_vl53l0x
import adafruit_vl53l1x


def _parse_i2c_addr(val, default):
    if val is None:
        return default
    if isinstance(val, int):
        return val
    if isinstance(val, str):
        try:
            return int(val.strip(), 0)
        except ValueError:
            return default
    return default


def _parse_int(val, default):
    if val is None:
        return default
    if isinstance(val, int):
        return val
    if isinstance(val, str):
        try:
            return int(val.strip(), 0)
        except ValueError:
            return default
    try:
        return int(val)
    except Exception:
        return default


MODE1      = 0x00
MODE2      = 0x01
PRESCALE   = 0xFE
LED0_ON_L  = 0x06
LED0_ON_H  = 0x07
LED0_OFF_L = 0x08
LED0_OFF_H = 0x09
ALL_LED_ON_L  = 0xFA
ALL_LED_ON_H  = 0xFB
ALL_LED_OFF_L = 0xFC
ALL_LED_OFF_H = 0xFD

RESTART = 0x80
SLEEP   = 0x10
ALLCALL = 0x01
OUTDRV  = 0x04


class PCA9685:
    def __init__(self, bus: SMBus, address: int, oscillator_hz: int = 25_000_000,
                 select_mux_fn=None, mux_channel=None):
        self._bus = bus
        self._addr = address
        self._osc = oscillator_hz
        self._select_mux = select_mux_fn
        self._mux_channel = mux_channel
        self._init_device()

    def _sel(self):
        if self._select_mux and self._mux_channel is not None:
            self._select_mux(self._mux_channel)

    def _write8(self, reg, val):
        self._sel()
        self._bus.write_byte_data(self._addr, reg, val & 0xFF)

    def _read8(self, reg):
        self._sel()
        return self._bus.read_byte_data(self._addr, reg)

    def _init_device(self):
        self._write8(MODE2, OUTDRV)
        self._write8(MODE1, ALLCALL)
        time.sleep(0.005)
        mode1 = self._read8(MODE1) & ~SLEEP
        self._write8(MODE1, mode1)
        time.sleep(0.005)

    def set_pwm_freq(self, freq_hz: float):
        prescaleval = float(self._osc) / (4096.0 * float(freq_hz)) - 1.0
        prescale = int(math.floor(prescaleval + 0.5))
        oldmode = self._read8(MODE1)
        newmode = (oldmode & 0x7F) | SLEEP
        self._write8(MODE1, newmode)
        self._write8(PRESCALE, prescale)
        self._write8(MODE1, oldmode)
        time.sleep(0.005)
        self._write8(MODE1, oldmode | RESTART)

    def set_pwm(self, channel: int, on: int, off: int):
        base = LED0_ON_L + 4 * channel
        for attempt in range(3):
            try:
                self._write8(base + 0, on & 0xFF)
                self._write8(base + 1, (on >> 8) & 0x0F)
                self._write8(base + 2, off & 0xFF)
                self._write8(base + 3, (off >> 8) & 0x0F)
                return
            except OSError:
                if attempt == 2:
                    raise
                time.sleep(0.001)
                self._sel()

    def set_off(self, channel: int):
        base = LED0_ON_L + 4 * channel
        self._write8(base + 0, 0x00)
        self._write8(base + 1, 0x00)
        self._write8(base + 2, 0x00)
        self._write8(base + 3, 0x10)

    def set_all_off(self):
        self._write8(ALL_LED_ON_L, 0)
        self._write8(ALL_LED_ON_H, 0)
        self._write8(ALL_LED_OFF_L, 0)
        self._write8(ALL_LED_OFF_H, 0x10)


class UnifiedRoboArmNode(Node):
    def __init__(self):
        super().__init__("wicom_roboarm_unified")

        # ---- typed descriptors (avoid [] => BYTE_ARRAY) ----
        desc_str_arr = ParameterDescriptor(type=ParameterType.PARAMETER_STRING_ARRAY)
        desc_int_arr = ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER_ARRAY)
        desc_dbl_arr = ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE_ARRAY)

        # -------- Parameters (servo) ----------
        self.declare_parameter("i2c_bus", 1)
        self.declare_parameter("mux_address", "0x70")
        self.declare_parameter("mux_channel", 2)
        self.declare_parameter("i2c_address", "0x40")
        self.declare_parameter("oscillator_hz", 25_000_000)
        self.declare_parameter("pwm_frequency_hz", 50.0)
        self.declare_parameter("enable_on_start", False)
        self.declare_parameter("use_mux", True)

        self.declare_parameter("pulse_us_min", 600.0)
        self.declare_parameter("pulse_us_max", 2400.0)

        self.declare_parameter("neutral_deg", 30.0)

        # arrays (typed)
        self.declare_parameter("joint_names", ["__dummy__"], desc_str_arr)
        self.declare_parameter("channels", [0], desc_int_arr)

        # optional arrays aligned with joint_names
        self.declare_parameter("neutral_deg_by_joint", [0.0], desc_dbl_arr)
        self.declare_parameter("limits_min_deg_by_joint", [0.0], desc_dbl_arr)
        self.declare_parameter("limits_max_deg_by_joint", [180.0], desc_dbl_arr)

        self.declare_parameter("publish_rate_hz", 10.0)
        self.declare_parameter("command_timeout_sec", 1.0)
        self.declare_parameter("timeout_behavior", "hold")     # hold|neutral|off
        self.declare_parameter("shutdown_behavior", "neutral") # hold|neutral|off

        # -------- Parameters (VL53 sensors) ----------
        self.declare_parameter("vl53_publish_rate_hz", 15.0)
        self.declare_parameter("channel_short", 0)
        self.declare_parameter("channel_long", 1)
        self.declare_parameter("frame_id_short", "vl53_short")
        self.declare_parameter("frame_id_long", "vl53_long")
        self.declare_parameter("max_range_short", 2.0)
        self.declare_parameter("max_range_long", 4.0)

        # -------- Read Parameters (servo) ----------
        self.busnum          = _parse_int(self.get_parameter("i2c_bus").value, 1)
        self.mux_address     = _parse_i2c_addr(self.get_parameter("mux_address").value, 0x70)
        self.servo_mux_chan  = _parse_int(self.get_parameter("mux_channel").value, 2)
        self.pca_address     = _parse_i2c_addr(self.get_parameter("i2c_address").value, 0x40)
        self.oscillator_hz   = _parse_int(self.get_parameter("oscillator_hz").value, 25_000_000)
        self.pwm_freq        = float(self.get_parameter("pwm_frequency_hz").value)
        self.enable_on_start = bool(self.get_parameter("enable_on_start").value)
        self.use_mux         = bool(self.get_parameter("use_mux").value)

        self.pulse_us_min    = float(self.get_parameter("pulse_us_min").value)
        self.pulse_us_max    = float(self.get_parameter("pulse_us_max").value)
        self.neutral_deg     = float(self.get_parameter("neutral_deg").value)

        self.publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self.command_timeout_sec = float(self.get_parameter("command_timeout_sec").value)
        self.timeout_behavior    = str(self.get_parameter("timeout_behavior").value)
        self.shutdown_behavior   = str(self.get_parameter("shutdown_behavior").value)

        joint_names = list(self.get_parameter("joint_names").value)
        channels = list(self.get_parameter("channels").value)

        # drop dummy defaults if YAML didn't override
        if joint_names == ["__dummy__"]:
            joint_names = []
        if channels == [0] and not joint_names:
            channels = []

        if (not joint_names) or (not channels) or len(joint_names) != len(channels):
            raise RuntimeError("joint_names and channels must be set and same length (ROS2 typed arrays).")

        self.joint_names = joint_names
        self.channels = channels

        self.num_joints = len(self.joint_names)
        self.name_to_idx = {n: i for i, n in enumerate(self.joint_names)}
        self.channel_by_idx = {i: ch for i, ch in enumerate(self.channels)}

        # optional per-joint arrays
        neutral_list = list(self.get_parameter("neutral_deg_by_joint").value)
        if neutral_list == [0.0] and self.num_joints != 1:
            neutral_list = []
        if len(neutral_list) == self.num_joints:
            self.neutral_deg_by_idx = [float(x) for x in neutral_list]
        else:
            self.neutral_deg_by_idx = [self.neutral_deg] * self.num_joints

        min_list = list(self.get_parameter("limits_min_deg_by_joint").value)
        max_list = list(self.get_parameter("limits_max_deg_by_joint").value)
        if min_list == [0.0] and self.num_joints != 1:
            min_list = []
        if max_list == [180.0] and self.num_joints != 1:
            max_list = []
        if len(min_list) == self.num_joints and len(max_list) == self.num_joints:
            self.limits_min_by_idx = [float(x) for x in min_list]
            self.limits_max_by_idx = [float(x) for x in max_list]
        else:
            self.limits_min_by_idx = [0.0] * self.num_joints
            self.limits_max_by_idx = [180.0] * self.num_joints

        self.current_deg = list(self.neutral_deg_by_idx)
        self.last_cmd_time = [self._now_s()] * self.num_joints
        self.enabled = self.enable_on_start

        # -------- Read Parameters (VL53) ----------
        self.vl53_rate_hz   = float(self.get_parameter("vl53_publish_rate_hz").value)
        self.ch_short       = _parse_int(self.get_parameter("channel_short").value, 0)
        self.ch_long        = _parse_int(self.get_parameter("channel_long").value, 1)
        self.frame_short    = str(self.get_parameter("frame_id_short").value)
        self.frame_long     = str(self.get_parameter("frame_id_long").value)
        self.vl53_max_short = float(self.get_parameter("max_range_short").value)
        self.vl53_max_long  = float(self.get_parameter("max_range_long").value)

        # -------- I2C & Lock ----------
        self.lock = threading.Lock()
        self.bus_smbus = SMBus(self.busnum)

        self.bus_busio = busio.I2C(board.SCL, board.SDA)
        self.tca = adafruit_tca9548a.TCA9548A(self.bus_busio, address=self.mux_address)

        def select_mux(channel: int):
            self.bus_smbus.write_byte(self.mux_address, 1 << channel)
            time.sleep(0.001)

        self._select_mux = select_mux

        # -------- Init PCA9685 ----------
        self.pca = PCA9685(
            self.bus_smbus,
            self.pca_address,
            oscillator_hz=self.oscillator_hz,
            select_mux_fn=self._select_mux if self.use_mux else None,
            mux_channel=self.servo_mux_chan if self.use_mux else None,
        )
        self.pca.set_pwm_freq(self.pwm_freq)

        if self.enabled:
            self._apply_all(self.current_deg)
        else:
            self._apply_behavior_all("off")

        # -------- Init VL53 sensors ----------
        self.sensor_long = None
        self.sensor_short = None
        with self.lock:
            try:
                self._select_mux(self.ch_long)
                self.sensor_long = adafruit_vl53l1x.VL53L1X(self.tca[self.ch_long])
                self.sensor_long.start_ranging()
                self.get_logger().info(f"VL53L1X (long) init channel {self.ch_long} OK")
            except Exception as e:
                self.get_logger().warn(f"VL53L1X init error: {e}")

            try:
                self._select_mux(self.ch_short)
                self.sensor_short = adafruit_vl53l0x.VL53L0X(self.tca[self.ch_short])
                self.get_logger().info(f"VL53L0X (short) init channel {self.ch_short} OK")
            except Exception as e:
                self.get_logger().warn(f"VL53L0X init error: {e}")

        # -------- Pub/Sub / Services --------
        self.pub_joint = self.create_publisher(JointState, "joint_states", 10)
        self.pub_short = self.create_publisher(Range, "/vl53/short_range", 10)
        self.pub_long  = self.create_publisher(Range, "/vl53/long_range", 10)

        self.sub_command = self.create_subscription(JointState, "command", self._on_command, 10)

        self.srv_enable    = self.create_service(Trigger, "enable", self.handle_enable)
        self.srv_disable   = self.create_service(Trigger, "disable", self.handle_disable)
        self.srv_home      = self.create_service(Trigger, "home", self.handle_home)

        self.pub_timer = self.create_timer(1.0 / max(self.publish_rate_hz, 1.0), self._publish_joint_state)
        self.watchdog_timer = self.create_timer(0.05, self._watchdog_tick)

        self._vl53_stop = threading.Event()
        self.vl53_thread = threading.Thread(target=self._vl53_loop, daemon=True)
        self.vl53_thread.start()

        self.get_logger().info(
            f"Unified RoboArm started: joints={self.joint_names} channels={self.channels} "
            f"PCA9685=0x{self.pca_address:02X} mux=0x{self.mux_address:02X} servo_mux_ch={self.servo_mux_chan} "
            f"pwm={self.pwm_freq:.1f}Hz enabled={self.enabled}"
        )

    def _now_s(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def angle_to_count(self, angle_deg: float) -> int:
        span_us = self.pulse_us_max - self.pulse_us_min
        frac = max(0.0, min(1.0, angle_deg / 180.0))
        pulse_us = self.pulse_us_min + frac * span_us
        counts = int(round((pulse_us / 1e6) * self.pwm_freq * 4096.0))
        return max(0, min(4095, counts))

    def apply_joint(self, idx: int, angle_deg: float):
        ch = self.channel_by_idx[idx]
        counts = self.angle_to_count(angle_deg)
        with self.lock:
            if self.use_mux:
                self._select_mux(self.servo_mux_chan)
            self.pca.set_pwm(ch, 0, counts)

    def _move_to_neutral(self, idx: int):
        neutral = self.neutral_deg_by_idx[idx]
        self.apply_joint(idx, neutral)
        self.current_deg[idx] = neutral

    def _turn_off(self, idx: int):
        ch = self.channel_by_idx[idx]
        with self.lock:
            if self.use_mux:
                self._select_mux(self.servo_mux_chan)
            self.pca.set_off(ch)

    def _apply_behavior_all(self, behavior: str):
        for idx in range(self.num_joints):
            if behavior == "hold":
                self.apply_joint(idx, self.current_deg[idx])
            elif behavior == "neutral":
                self._move_to_neutral(idx)
            elif behavior == "off":
                self._turn_off(idx)

    def _apply_all(self, deg_list):
        for idx, deg in enumerate(deg_list):
            self.apply_joint(idx, deg)

    def _on_command(self, msg: JointState):
        if not msg.name or not msg.position:
            return

        if not self.enabled:
            self.get_logger().warn("Auto enable outputs, received /command JointState")
            self.enabled = True

        for name, pos in zip(msg.name, msg.position):
            if name not in self.name_to_idx:
                continue
            idx = self.name_to_idx[name]

            angle = float(pos)
            if abs(angle) < 6.3:
                angle = math.degrees(angle)

            target = max(self.limits_min_by_idx[idx], min(self.limits_max_by_idx[idx], angle))

            try:
                self.apply_joint(idx, target)
                self.current_deg[idx] = target
                self.last_cmd_time[idx] = self._now_s()
            except Exception as e:
                self.get_logger().error(f"I2C error command for {name}: {e}")

    def handle_enable(self, _req, resp):
        try:
            self._apply_all(self.current_deg)
            self.enabled = True
            resp.success = True
            resp.message = "Outputs enabled"
            return resp
        except Exception as e:
            resp.success = False
            resp.message = str(e)
            return resp

    def handle_disable(self, _req, resp):
        try:
            self._apply_behavior_all("off")
            self.enabled = False
            resp.success = True
            resp.message = "Outputs disabled (off)"
            return resp
        except Exception as e:
            resp.success = False
            resp.message = str(e)
            return resp

    def handle_home(self, _req, resp):
        try:
            for idx in range(self.num_joints):
                self._move_to_neutral(idx)
            self.enabled = True
            resp.success = True
            resp.message = "All joints neutral"
            return resp
        except Exception as e:
            resp.success = False
            resp.message = str(e)
            return resp

    def _publish_joint_state(self):
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = list(self.joint_names)
        js.position = [math.radians(d) for d in self.current_deg]
        self.pub_joint.publish(js)

    def _watchdog_tick(self):
        if self.command_timeout_sec <= 0:
            return
        now = self._now_s()
        for idx in range(self.num_joints):
            if (now - self.last_cmd_time[idx]) > self.command_timeout_sec:
                if self.timeout_behavior == "hold":
                    pass
                elif self.timeout_behavior == "neutral":
                    self._move_to_neutral(idx)
                elif self.timeout_behavior == "off":
                    self._turn_off(idx)
                self.last_cmd_time[idx] = now

    def _range_msg(self, frame, min_r, max_r, dist_m):
        msg = Range()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame
        msg.radiation_type = Range.INFRARED
        msg.field_of_view = 0.52
        msg.min_range = float(min_r)
        msg.max_range = float(max_r)
        msg.range = float(dist_m) if dist_m is not None else float("nan")
        return msg

    def _read_short(self):
        if not self.sensor_short:
            return None
        try:
            mm = self.sensor_short.range
            if mm is None or mm <= 0:
                return None
            return mm / 1000.0
        except Exception as e:
            self.get_logger().warn(f"VL53L0X read error: {e}")
            return None

    def _read_long(self):
        if not self.sensor_long:
            return None
        try:
            if hasattr(self.sensor_long, "data_ready") and (not self.sensor_long.data_ready):
                return None
            mm = self.sensor_long.distance
            if mm is None or mm <= 0:
                return None
            return mm / 100.0
        except Exception as e:
            self.get_logger().warn(f"VL53L1X read error: {e}")
            return None

    def _vl53_loop(self):
        period = 1.0 / max(self.vl53_rate_hz, 1.0)
        while rclpy.ok() and not self._vl53_stop.is_set():
            with self.lock:
                if self.use_mux:
                    self._select_mux(self.ch_short)
                d_short = self._read_short()

                if self.use_mux:
                    self._select_mux(self.ch_long)
                d_long = self._read_long()

            self.pub_short.publish(self._range_msg(self.frame_short, 0.03, self.vl53_max_short, d_short))
            self.pub_long.publish(self._range_msg(self.frame_long, 0.03, self.vl53_max_long, d_long))
            time.sleep(period)

    def destroy_node(self):
        try:
            self._vl53_stop.set()
            if self.vl53_thread.is_alive():
                self.vl53_thread.join(timeout=0.5)
        except Exception:
            pass

        self.get_logger().warn(f"Shutdown: apply behavior {self.shutdown_behavior} for all servo")
        try:
            self._apply_behavior_all(self.shutdown_behavior)
            time.sleep(0.01)
            with self.lock:
                if self.use_mux:
                    self._select_mux(self.servo_mux_chan)
                self.pca.set_all_off()
            self.bus_smbus.close()
        except Exception as e:
            self.get_logger().error(f"Shutdown error: {e}")
        return super().destroy_node()


def main():
    rclpy.init()
    node = UnifiedRoboArmNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()