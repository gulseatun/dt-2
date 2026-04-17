#!/usr/bin/env python3
import math
import numpy as np
import rospy
import cv2
from sensor_msgs.msg import CompressedImage
from duckietown_msgs.msg import WheelEncoderStamped, Twist2DStamped

# =========================
# CONFIG (Wolf Ayarları)
# =========================
ROBOT_NAME = "bear"
CMD_VEL_TOPIC = f"/{ROBOT_NAME}/wheels_driver_node/wheels_cmd" 
CAMERA_TOPIC = f"/{ROBOT_NAME}/camera_node/image/compressed"
VIZ_PUB_TOPIC = f"/{ROBOT_NAME}/localization_viz/compressed"

# Fiziksel Sabitler (Madde 1: correct physical marker side length s)
MARKER_SIZE_METERS = 0.065
R, N, L = 0.0318, 135, 0.1
METRE_PER_TICK = (2 * math.pi * R) / N

# Kamera Kalibrasyon (Madde 1: calibrated K and d)
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


# Marker Haritası (Kendi pistine göre güncellemelisin)
MARKER_MAP = {
    11: {"x": 1.0, "y": 0.0, "yaw": 0.0},
    32: {"x": 2.0, "y": 1.0, "yaw": math.pi / 2}
}

# Harita Çizim Ayarları (Madde 3: Top-down map)
MAP_W, MAP_H = 480, 480
MAP_SCALE = 150.0  # 1 metre = 150 piksel
MAP_ORIGIN = (40, 480)

class Assignment2Node:
    def __init__(self):
        rospy.init_node("assignment2_localization_node")
        
        # Robot Pozu ve Odometri Değişkenleri
        self.x, self.y, self.yaw = 0.0, 0.0, 0.0
        self.left_tick_prev, self.right_tick_prev = None, None
        self.d_left, self.d_right = 0.0, 0.0
        self.robot_dir = 1.0
        self.last_frame = None
        self.source = "FALLBACK" 

        # ArUco Setup (OpenCV Sürüm Toleransı)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        try:
            self.aruco_params = cv2.aruco.DetectorParameters()
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            self.new_api = True
        except AttributeError:
            self.aruco_params = cv2.aruco.DetectorParameters_create()
            self.new_api = False

        # Publisher (Madde 3: Görselleştirme Yayını)
        self.viz_pub = rospy.Publisher(VIZ_PUB_TOPIC, CompressedImage, queue_size=1)

        # Subscribers
        rospy.Subscriber(CAMERA_TOPIC, CompressedImage, self.image_cb, queue_size=1, buff_size=2**24)
        rospy.Subscriber(f"/{ROBOT_NAME}/left_wheel_encoder_node/tick", WheelEncoderStamped, self.left_cb)
        rospy.Subscriber(f"/{ROBOT_NAME}/right_wheel_encoder_node/tick", WheelEncoderStamped, self.right_cb)
        rospy.Subscriber(CMD_VEL_TOPIC, Twist2DStamped, self.cmd_cb)

        rospy.loginfo("Assignment 2 Node Başlatıldı. rqt_image_view ile yayını izleyebilirsin.")

    def cmd_cb(self, msg): self.robot_dir = 1.0 if msg.v >= 0 else -1.0
    
    def left_cb(self, msg):
        if self.left_tick_prev is not None: self.d_left += (msg.data - self.left_tick_prev) * METRE_PER_TICK * self.robot_dir
        self.left_tick_prev = msg.data
        
    def right_cb(self, msg):
        if self.right_tick_prev is not None: self.d_right += (msg.data - self.right_tick_prev) * METRE_PER_TICK * self.robot_dir
        self.right_tick_prev = msg.data
        
    def image_cb(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        self.last_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    def draw_map(self):
        # Boş beyaz harita
        canvas = np.ones((MAP_H, MAP_W, 3), dtype=np.uint8) * 255
        cv2.putText(canvas, "Top-Down Map", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Markerları Çiz
        for m_id, data in MARKER_MAP.items():
            px, py = int(MAP_ORIGIN[0] + data["x"] * MAP_SCALE), int(MAP_ORIGIN[1] - data["y"] * MAP_SCALE)
            cv2.circle(canvas, (px, py), 5, (255, 0, 0), -1)
            cv2.putText(canvas, f"ID:{m_id}", (px+10, py-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Robotu Çiz
        rx, ry = int(MAP_ORIGIN[0] + self.x * MAP_SCALE), int(MAP_ORIGIN[1] - self.y * MAP_SCALE)
        
        # Madde 3: Duruma göre renk değiştirme (ArUco: Yeşil, Odom: Kırmızı)
        color = (0, 200, 0) if "ARUCO" in self.source else (0, 0, 200) 
        
        cv2.circle(canvas, (rx, ry), 8, color, -1)
        endx = int(rx + 25 * math.cos(self.yaw))
        endy = int(ry - 25 * math.sin(self.yaw))
        cv2.arrowedLine(canvas, (rx, ry), (endx, endy), color, 3, tipLength=0.3)

        # Bilgi Yazıları
        cv2.putText(canvas, f"State: {self.source}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(canvas, f"X:{self.x:.2f} Y:{self.y:.2f} Yaw:{math.degrees(self.yaw):.0f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
        
        return canvas

    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            # --- Madde 2: Fallback Localization (Wheel Odometry) ---
            dist = (self.d_right + self.d_left) / 2.0
            d_phi = (self.d_right - self.d_left) / L
            self.x += dist * math.cos(self.yaw)
            self.y += dist * math.sin(self.yaw)
            self.yaw = math.atan2(math.sin(self.yaw + d_phi), math.cos(self.yaw + d_phi))
            self.d_left, self.d_right = 0.0, 0.0
            self.source = "FALLBACK (Odom)"

            if self.last_frame is not None:
                # Undistort Image
                frame = cv2.undistort(self.last_frame.copy(), K, D)
                
                # EKLENECEK SATIR: Görüntüyü Gri Tonlamaya (Grayscale) Çevir
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # DEĞİŞECEK SATIR: 'frame' yerine 'gray' veriyoruz
                if self.new_api: 
                    corners, ids, _ = self.detector.detectMarkers(gray)
                else: 
                    corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

                if ids is not None:
                    # Sınırları çiz (Ödev isteri 1.4)
                    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                    
                    for i, m_id in enumerate(ids.flatten()):
                        # KRİTİK DÜZELTME: Gelen Numpy değerini standart int formatına çevir
                        m_id_int = int(m_id)
                        
                        # Pose Estimation
                        obj_pts = np.array([[-MARKER_SIZE_METERS/2, MARKER_SIZE_METERS/2, 0],
                                            [MARKER_SIZE_METERS/2, MARKER_SIZE_METERS/2, 0],
                                            [MARKER_SIZE_METERS/2, -MARKER_SIZE_METERS/2, 0],
                                            [-MARKER_SIZE_METERS/2, -MARKER_SIZE_METERS/2, 0]], dtype=np.float32)
                        
                        success, rvec, tvec = cv2.solvePnP(obj_pts, corners[i], K, D)
                        if success:
                            # Eksenleri çiz (Ödev isteri 1.4)
                            cv2.drawFrameAxes(frame, K, D, rvec, tvec, 0.05)
                            
                            # MADDE 1 EKSİĞİ GİDERİLDİ: "Log the estimated pose for each detected tag ID"
                            rospy.loginfo_throttle(0.5, f"[Madde 1] ID: {m_id_int} Pose -> tvec: {tvec.flatten()}, rvec: {rvec.flatten()}")
                            
                            # Eğer görülen ID haritamızda varsa konumu güncelle
                            if m_id_int in MARKER_MAP:
                                target = MARKER_MAP[m_id_int]
                                self.x = (0.7 * self.x) + (0.3 * target["x"])
                                self.y = (0.7 * self.y) + (0.3 * target["y"])
                                self.source = f"ARUCO_FIX (ID:{m_id_int})"

                # --- Madde 3: Live Visualization Birleştirme ve Yayınlama ---
                h, w = frame.shape[:2]
                new_w = int((MAP_H / h) * w)
                frame_resized = cv2.resize(frame, (new_w, MAP_H))
                map_img = self.draw_map()
                
                # Yan yana birleştir
                combined_img = np.hstack([frame_resized, map_img])

                # Publisher için CompressedImage formatına çevir
                msg = CompressedImage()
                msg.header.stamp = rospy.Time.now()
                msg.format = "jpeg"
                msg.data = np.array(cv2.imencode('.jpg', combined_img)[1]).tobytes()
                self.viz_pub.publish(msg)
            
            # Ana döngü logu
            rospy.loginfo_throttle(0.5, f"[{self.source}] X: {self.x:.2f} m, Y: {self.y:.2f} m, Yaw: {math.degrees(self.yaw):.0f}°, Dir: {'FWD' if self.robot_dir > 0 else 'BWD'}")
            rate.sleep()

if __name__ == "__main__":
    try:
        Assignment2Node().run()
    except rospy.ROSInterruptException:
        pass