#!/usr/bin/env python3
import math
import numpy as np
import rospy
import cv2

from sensor_msgs.msg import CompressedImage
from duckietown_msgs.msg import WheelEncoderStamped, Twist2DStamped


# =========================
# CONFIG
# =========================
ROBOT_NAME = "bear"

CMD_VEL_TOPIC = f"/{ROBOT_NAME}/wheels_driver_node/wheels_cmd"
CAMERA_TOPIC = f"/{ROBOT_NAME}/camera_node/image/compressed"
VIZ_PUB_TOPIC = f"/{ROBOT_NAME}/localization_viz/compressed"

LEFT_TICK_TOPIC = f"/{ROBOT_NAME}/left_wheel_encoder_node/tick"
RIGHT_TICK_TOPIC = f"/{ROBOT_NAME}/right_wheel_encoder_node/tick"

# Fiziksel sabitler
MARKER_SIZE_METERS = 0.065
R = 0.0318
N = 135
L = 0.1
METRE_PER_TICK = (2.0 * math.pi * R) / N

# Kamera kalibrasyonu

K = np.array([
    [270.4563591302591,   0.0,               314.1813567017415],
    [0.0,                 269.2951665378049, 218.88618596346137],
    [0.0,                 0.0,               1.0]
], dtype=np.float32)

D = np.array([
    -0.19162991260105328,
     0.026384790215657535,
     0.005682129590129115,
     0.0006647376545041703,
     0.0
], dtype=np.float32)

# Marker haritası
# Her marker için dünya koordinatı ve yönü

MARKER_MAP = {
    11:  {"x": 1.0, "y": 0.0, "yaw": math.pi},
    10: {"x": 2.0, "y": 1.0, "yaw": math.pi},
}

# Kamera ile robot merkezi arasındaki offset

CAMERA_TO_ROBOT_X = 0.0
CAMERA_TO_ROBOT_Y = 0.0
CAMERA_TO_ROBOT_YAW = 0.0

# Harita çizimi
MAP_W = 520
MAP_H = 520
MAP_SCALE = 150.0
MAP_ORIGIN = (100, 420)


# =========================
# HELPER FUNCTIONS
# =========================
def wrap_angle(a):
    return math.atan2(math.sin(a), math.cos(a))


def world_to_map_px(x, y):
    px = int(MAP_ORIGIN[0] + x * MAP_SCALE)
    py = int(MAP_ORIGIN[1] - y * MAP_SCALE)
    return px, py


def blend_angle(a, b, alpha=0.7):
    diff = wrap_angle(a - b)
    return wrap_angle(b + alpha * diff)


def rotation_2d(yaw):
    c = math.cos(yaw)
    s = math.sin(yaw)
    return np.array([
        [c, -s],
        [s,  c]
    ], dtype=np.float32)


def pose_to_T(x, y, yaw):
    c = math.cos(yaw)
    s = math.sin(yaw)
    T = np.eye(3, dtype=np.float32)
    T[0, 0] = c
    T[0, 1] = -s
    T[1, 0] = s
    T[1, 1] = c
    T[0, 2] = x
    T[1, 2] = y
    return T


def T_to_pose(T):
    x = float(T[0, 2])
    y = float(T[1, 2])
    yaw = math.atan2(T[1, 0], T[0, 0])
    return x, y, yaw


def camera_pose_from_marker(marker_world, rvec, tvec):
    """
    solvePnP çıktısı:
        X_cam = R_cm * X_marker + t_cm

    Buradan kameranın marker frame'indeki pozu:
        R_mc = R_cm^T
        t_mc = -R_cm^T * t_cm
    """

    R_cm, _ = cv2.Rodrigues(rvec)
    t_cm = tvec.reshape(3, 1)

    R_mc = R_cm.T
    t_mc = -R_cm.T @ t_cm

    # Marker düzlemi üstünde 2D yaklaşım
    cam_x_m = float(t_mc[0, 0])
    cam_z_m = float(t_mc[2, 0])

    marker_x = marker_world["x"]
    marker_y = marker_world["y"]
    marker_yaw = marker_world["yaw"]

    # marker local (x,z) -> world (x,y)
    #p_local = np.array([cam_x_m, cam_z_m], dtype=np.float32)
    #
    p_local = np.array([cam_z_m, cam_x_m], dtype=np.float32)
    #p_local = np.array([cam_z_m, -cam_x_m], dtype=np.float32)
    R_wm_2d = rotation_2d(marker_yaw)
    p_world = R_wm_2d @ p_local + np.array([marker_x, marker_y], dtype=np.float32)

    # Kamera yönünü yaklaşık hesapla
    # Kameranın forward eksenini marker frame'inde bul
    # R_mc: camera -> marker
    forward_cam_in_marker = R_mc[:, 2]
    fx = float(forward_cam_in_marker[0])
    fz = float(forward_cam_in_marker[2])
    cam_yaw_local = math.atan2(fz, fx)
    #cam_yaw_world = wrap_angle(marker_yaw + cam_yaw_local)
    cam_yaw_world = wrap_angle(marker_yaw + cam_yaw_local - math.pi / 2)

    return float(p_world[0]), float(p_world[1]), cam_yaw_world


def camera_to_robot_pose(cam_x, cam_y, cam_yaw):
    T_wc = pose_to_T(cam_x, cam_y, cam_yaw)
    T_cr = pose_to_T(CAMERA_TO_ROBOT_X, CAMERA_TO_ROBOT_Y, CAMERA_TO_ROBOT_YAW)
    T_wr = T_wc @ T_cr
    return T_to_pose(T_wr)


# =========================
# NODE
# =========================
class Assignment2Node:
    def __init__(self):
        rospy.init_node("assignment2_localization_node")

        # Pose state
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

        self.vx = 0.0
        self.vy = 0.0
        self.yaw_rate = 0.0

        self.left_tick_prev = None
        self.right_tick_prev = None
        self.d_left = 0.0
        self.d_right = 0.0
        self.robot_dir = 1.0

        self.last_frame = None
        self.source = "FALLBACK (Odom)"
        self.last_pose_update_time = rospy.Time.now().to_sec()

        # ArUco / AprilTag dictionary
        # Eğer setup gerçekten ArUco ise bunu değiştir:
        # cv2.aruco.DICT_4X4_50
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)

        try:
            self.aruco_params = cv2.aruco.DetectorParameters()
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            self.new_api = True
        except AttributeError:
            self.aruco_params = cv2.aruco.DetectorParameters_create()
            self.new_api = False

        self.viz_pub = rospy.Publisher(VIZ_PUB_TOPIC, CompressedImage, queue_size=1)

        rospy.Subscriber(CAMERA_TOPIC, CompressedImage, self.image_cb, queue_size=1, buff_size=2**24)
        rospy.Subscriber(LEFT_TICK_TOPIC, WheelEncoderStamped, self.left_cb)
        rospy.Subscriber(RIGHT_TICK_TOPIC, WheelEncoderStamped, self.right_cb)
        rospy.Subscriber(CMD_VEL_TOPIC, Twist2DStamped, self.cmd_cb)

        rospy.loginfo("Assignment 2 localization node started.")

    # -------------------------
    # Callbacks
    # -------------------------
    def cmd_cb(self, msg):
        self.robot_dir = 1.0 if msg.v >= 0 else -1.0

    def left_cb(self, msg):
        if self.left_tick_prev is not None:
            dticks = msg.data - self.left_tick_prev
            self.d_left += dticks * METRE_PER_TICK * self.robot_dir
        self.left_tick_prev = msg.data

    def right_cb(self, msg):
        if self.right_tick_prev is not None:
            dticks = msg.data - self.right_tick_prev
            self.d_right += dticks * METRE_PER_TICK * self.robot_dir
        self.right_tick_prev = msg.data

    def image_cb(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        self.last_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # -------------------------
    # Odometry fallback
    # -------------------------
    def odom_predict(self):
        dist = (self.d_right + self.d_left) / 2.0
        d_yaw = (self.d_right - self.d_left) / L

        self.x += dist * math.cos(self.yaw)
        self.y += dist * math.sin(self.yaw)
        self.yaw = wrap_angle(self.yaw + d_yaw)

        self.d_left = 0.0
        self.d_right = 0.0
        self.source = "FALLBACK (Odom)"

    def update_velocity_estimate(self, new_x, new_y, new_yaw, now):
        dt = now - self.last_pose_update_time
        if dt <= 1e-6:
            return

        self.vx = (new_x - self.x) / dt
        self.vy = (new_y - self.y) / dt
        self.yaw_rate = wrap_angle(new_yaw - self.yaw) / dt

    # -------------------------
    # Marker detection
    # -------------------------
    def detect_markers(self, gray):
        if self.new_api:
            corners, ids, _ = self.detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
        return corners, ids

    def estimate_pose_from_detections(self, frame, corners, ids):
        """
        Marker'lardan robotun world pose'unu tahmin eder.
        Birden fazla marker varsa ortalama alınır.
        """
        if ids is None or len(ids) == 0:
            return None

        obj_pts = np.array([
            [-MARKER_SIZE_METERS / 2,  MARKER_SIZE_METERS / 2, 0],
            [ MARKER_SIZE_METERS / 2,  MARKER_SIZE_METERS / 2, 0],
            [ MARKER_SIZE_METERS / 2, -MARKER_SIZE_METERS / 2, 0],
            [-MARKER_SIZE_METERS / 2, -MARKER_SIZE_METERS / 2, 0]
        ], dtype=np.float32)

        candidate_poses = []

        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        for i, m_id in enumerate(ids.flatten()):
            m_id = int(m_id)

            image_pts = corners[i].reshape((4, 2)).astype(np.float32)

            success, rvec, tvec = cv2.solvePnP(
                obj_pts,
                image_pts,
                K,
                D,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )

            if not success:
                continue

            cv2.drawFrameAxes(frame, K, D, rvec, tvec, MARKER_SIZE_METERS * 0.5)

            c = image_pts.mean(axis=0).astype(int)
            txt = f"ID:{m_id} tz:{tvec[2][0]:.2f}m"
            cv2.putText(frame, txt, (c[0] - 40, c[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

            rospy.loginfo_throttle(
                0.5,
                f"Detected ID {m_id} | tvec={tvec.flatten()} | rvec={rvec.flatten()}"
            )

            if m_id not in MARKER_MAP:
                continue

            cam_x, cam_y, cam_yaw = camera_pose_from_marker(MARKER_MAP[m_id], rvec, tvec)
            rob_x, rob_y, rob_yaw = camera_to_robot_pose(cam_x, cam_y, cam_yaw)

            candidate_poses.append((rob_x, rob_y, rob_yaw, m_id))

        if not candidate_poses:
            return None

        xs = [p[0] for p in candidate_poses]
        ys = [p[1] for p in candidate_poses]

        cos_vals = [math.cos(p[2]) for p in candidate_poses]
        sin_vals = [math.sin(p[2]) for p in candidate_poses]

        mean_x = float(np.mean(xs))
        mean_y = float(np.mean(ys))
        mean_yaw = math.atan2(np.mean(sin_vals), np.mean(cos_vals))

        used_ids = [str(p[3]) for p in candidate_poses]
        return mean_x, mean_y, mean_yaw, used_ids

    # -------------------------
    # Visualization
    # -------------------------
    def draw_map(self):
        canvas = np.ones((MAP_H, MAP_W, 3), dtype=np.uint8) * 255

        cv2.putText(canvas, "Top-Down Map", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        # markers
        for m_id, data in MARKER_MAP.items():
            px, py = world_to_map_px(data["x"], data["y"])
            cv2.circle(canvas, (px, py), 6, (255, 0, 0), -1)
            cv2.putText(canvas, f"ID:{m_id}", (px + 8, py - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1)

            arrow_len = 26
            ex = int(px + arrow_len * math.cos(data["yaw"]))
            ey = int(py - arrow_len * math.sin(data["yaw"]))
            cv2.arrowedLine(canvas, (px, py), (ex, ey), (255, 0, 0), 2, tipLength=0.25)

        # robot
        rx, ry = world_to_map_px(self.x, self.y)
        color = (0, 180, 0) if "ARUCO" in self.source else (0, 0, 220)

        cv2.circle(canvas, (rx, ry), 8, color, -1)
        ex = int(rx + 30 * math.cos(self.yaw))
        ey = int(ry - 30 * math.sin(self.yaw))
        cv2.arrowedLine(canvas, (rx, ry), (ex, ey), color, 3, tipLength=0.25)

        cv2.putText(canvas, f"State: {self.source}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.putText(canvas, f"X:{self.x:.2f} Y:{self.y:.2f}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        cv2.putText(canvas, f"Yaw:{math.degrees(self.yaw):.1f} deg", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        return canvas

    def publish_visualization(self, frame):
        map_img = self.draw_map()

        h, w = frame.shape[:2]
        new_w = int((MAP_H / h) * w)
        frame_resized = cv2.resize(frame, (new_w, MAP_H))
        combined = np.hstack([frame_resized, map_img])

        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        msg.data = np.array(cv2.imencode(".jpg", combined)[1]).tobytes()
        self.viz_pub.publish(msg)

    # -------------------------
    # Main loop
    # -------------------------
    def run(self):
        rate = rospy.Rate(10)

        while not rospy.is_shutdown():
            now = rospy.Time.now().to_sec()

            # 1) fallback her loop'ta çalışsın
            self.odom_predict()

            if self.last_frame is not None:
                frame = cv2.undistort(self.last_frame.copy(), K, D)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                corners, ids = self.detect_markers(gray)
                aruco_pose = self.estimate_pose_from_detections(frame, corners, ids)

                if aruco_pose is not None:
                    mx, my, myaw, used_ids = aruco_pose

                    # Velocity estimate update
                    self.update_velocity_estimate(mx, my, myaw, now)

                    # fallback'ten aruco'ya geçişte yumuşatma
                    alpha = 0.7
                    self.x = alpha * mx + (1.0 - alpha) * self.x
                    self.y = alpha * my + (1.0 - alpha) * self.y
                    self.yaw = blend_angle(myaw, self.yaw, alpha=alpha)

                    self.source = f"ARUCO_FIX (IDs:{','.join(used_ids)})"
                    self.last_pose_update_time = now
                else:
                    self.last_pose_update_time = now

                self.publish_visualization(frame)

            rospy.loginfo_throttle(
                0.5,
                f"[{self.source}] X:{self.x:.2f} Y:{self.y:.2f} Yaw:{math.degrees(self.yaw):.1f}"
            )


            rate.sleep()


if __name__ == "__main__":
    try:
        Assignment2Node().run()
    except rospy.ROSInterruptException:
        pass