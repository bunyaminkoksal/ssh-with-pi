#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import cv2.aruco as aruco
import numpy as np
import socket
import struct
import json
import threading
import time
import math
import os
import glob

from pymavlink import mavutil


# =========================================================
# AYARLAR
# =========================================================

# ---------- Kamera ----------
CAMERA_CANDIDATES = [
    "/dev/video0",
    "/dev/v4l/by-id/usb-046d_0825_AF0F2F20-video-index0",
    0,
    1,
]
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS_LIMIT = 20
JPEG_QUALITY = 70

# ---------- SSH reverse tunnel ile PC video ----------
VIDEO_TARGET_HOST = "127.0.0.1"
VIDEO_TARGET_PORT = 9999

# ---------- GCS ----------
GCS_IP = "192.168.1.10"   # BUNU KENDI BILGISAYAR IP'N YAP
TELEM_PORT = 10000

CMD_HOST = "0.0.0.0"
CMD_PORT = 10001

# ---------- PX4 MAVLink ----------
MAVLINK_BAUD = 921600
HEARTBEAT_TIMEOUT = 10.0

# ---------- PX4 davranışı ----------
AUTO_REQUEST_OFFBOARD_ON_START = True
AUTO_ARM_ON_START = False          # ilk testte False bırak
AUTO_DISARM_ON_FINAL = False       # ilk testte False bırak
OFFBOARD_WARMUP_SEC = 1.3
SETPOINT_RATE_HZ = 20.0

# ---------- ArUco ----------
ARUCO_DICT = aruco.DICT_4X4_50
TARGET_MARKER_ID = 0
MARKER_SIZE_M = 0.20

# ---------- Kamera kalibrasyonu ----------
CALIB_NPZ_PATH = "camera_calibration_charuco.npz"
APPROX_FX = 640.0
APPROX_FY = 640.0
APPROX_CX = FRAME_WIDTH / 2.0
APPROX_CY = FRAME_HEIGHT / 2.0

# ---------- PID ----------
KP = 0.0025
KI = 0.00004
KD = 0.0010

MAX_VEL_X = 0.35
MAX_VEL_Y = 0.35
DESCENT_SPEED = 0.15
MAX_VEL_Z = 0.25

# Gerekirse ilk testlerden sonra değiştir
REVERSE_X = False
REVERSE_Y = False

# ---------- Landing eşikleri ----------
ALIGN_THRESHOLD_PX = 80
DESCEND_START_ALT = 1.20
FINAL_ALT_THRESHOLD = 0.35
FINAL_PIXEL_THRESHOLD = 100
MARKER_LOST_TIMEOUT = 0.5

# ---------- Güvenlik ----------
SEND_ZERO_ON_IDLE = True

# ---------- Takeoff ----------
TAKEOFF_ALT_M = 1.0


# =========================================================
# YARDIMCI
# =========================================================

def clamp(val, vmin, vmax):
    return max(vmin, min(vmax, val))


# =========================================================
# ANA SINIF
# =========================================================

class PerfectLanderPX4:
    def __init__(self):
        self.running = True

        self.auto_enabled = False
        self.manual_override = False
        self.is_armed = False
        self.offboard_requested = False

        self.state = "IDLE"
        self.last_event = "boot"

        # kamera
        self.cap = None
        self.latest_frame = None
        self.latest_jpeg = None
        self.frame_lock = threading.Lock()

        # video
        self.video_sock = None
        self.video_connected = False

        # telemetri
        self.telem_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # komut
        self.cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.cmd_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.cmd_sock.bind((CMD_HOST, CMD_PORT))
        self.cmd_sock.settimeout(0.2)

        # mavlink
        self.master = None
        self.mav_port = None
        self.current_yaw = 0.0
        self.relative_alt_m = None
        self.flight_mode = "UNKNOWN"

        # aruco
        self.marker_detected = False
        self.err_x = 0.0
        self.err_y = 0.0
        self.aruco_altitude = -1.0
        self.last_marker_time = 0.0

        # PID
        self.integral_x = 0.0
        self.integral_y = 0.0
        self.prev_err_x = 0.0
        self.prev_err_y = 0.0
        self.last_control_time = time.time()

        # ortak velocity komutu
        self.cmd_vx = 0.0
        self.cmd_vy = 0.0
        self.cmd_vz = 0.0
        self.cmd_lock = threading.Lock()

        # FPS
        self.fps = 0.0
        self._fps_last_t = time.time()
        self._fps_counter = 0

        # kalibrasyon
        self.camera_matrix = None
        self.dist_coeffs = None
        self.reprojection_error = None
        self.load_camera_calibration()

        # marker 3D köşe noktaları
        ms = MARKER_SIZE_M / 2.0
        self.marker_points = np.array([
            [-ms,  ms, 0.0],
            [ ms,  ms, 0.0],
            [ ms, -ms, 0.0],
            [-ms, -ms, 0.0]
        ], dtype=np.float32)

        # aruco detector
        if hasattr(aruco, "getPredefinedDictionary"):
            self.aruco_dict = aruco.getPredefinedDictionary(ARUCO_DICT)
            self.detector = aruco.ArucoDetector(
                self.aruco_dict,
                aruco.DetectorParameters()
            )
            self.new_api = True
        else:
            self.aruco_dict = aruco.Dictionary_get(ARUCO_DICT)
            self.aruco_params = aruco.DetectorParameters_create()
            self.new_api = False

    # -----------------------------------------------------
    # log
    # -----------------------------------------------------
    def log(self, msg):
        print(f"[INFO] {msg}")
        self.last_event = msg

    # -----------------------------------------------------
    # kalibrasyon
    # -----------------------------------------------------
    def set_approx_camera_calibration(self):
        self.camera_matrix = np.array([
            [APPROX_FX, 0.0, APPROX_CX],
            [0.0, APPROX_FY, APPROX_CY],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        self.dist_coeffs = np.zeros((5, 1), dtype=np.float32)

    def load_camera_calibration(self):
        if os.path.exists(CALIB_NPZ_PATH):
            try:
                data = np.load(CALIB_NPZ_PATH)
                self.camera_matrix = data["camera_matrix"].astype(np.float32)
                self.dist_coeffs = data["dist_coeffs"].astype(np.float32)

                if "reprojection_error" in data:
                    self.reprojection_error = float(data["reprojection_error"])

                self.log("kamera kalibrasyonu yuklendi")
                if self.reprojection_error is not None:
                    self.log(f"reprojection error = {self.reprojection_error:.4f}")

                print("camera_matrix=\n", self.camera_matrix)
                print("dist_coeffs=\n", self.dist_coeffs)

            except Exception as e:
                self.log(f"kalibrasyon yukleme hatasi: {e}")
                self.log("approx camera matrix kullaniliyor")
                self.set_approx_camera_calibration()
        else:
            self.log("kalibrasyon dosyasi bulunamadi, approx matrix kullaniliyor")
            self.set_approx_camera_calibration()

    # -----------------------------------------------------
    # kamera
    # -----------------------------------------------------
    def open_camera(self):
        self.cap = None

        for cam_src in CAMERA_CANDIDATES:
            try:
                print(f"[INFO] Kamera deneniyor: src={cam_src}")
                cap = cv2.VideoCapture(cam_src)

                cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
                cap.set(cv2.CAP_PROP_FPS, FPS_LIMIT)

                time.sleep(0.5)

                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        self.cap = cap
                        self.log(f"kamera hazir: src={cam_src}, shape={frame.shape}")
                        return

                cap.release()

            except Exception as e:
                print(f"[WARN] Kamera deneme hatasi src={cam_src}: {e}")

        raise RuntimeError("USB kamera acilamadi.")

    # -----------------------------------------------------
    # PX4 / MAVLink
    # -----------------------------------------------------
    def connect_mavlink(self):
        candidates = []
        candidates += sorted(glob.glob("/dev/serial/by-id/*"))
        candidates += sorted(glob.glob("/dev/ttyUSB*"))
        candidates += sorted(glob.glob("/dev/ttyACM*"))

        if not candidates:
            raise RuntimeError("MAVLink seri portu bulunamadi.")

        last_err = None

        for port in candidates:
            try:
                self.log(f"px4 baglaniyor: {port} @ {MAVLINK_BAUD}")
                master = mavutil.mavlink_connection(port, baud=MAVLINK_BAUD)
                hb = master.wait_heartbeat(timeout=HEARTBEAT_TIMEOUT)
                if hb is not None:
                    self.master = master
                    self.mav_port = port
                    self.flight_mode = self.master.flightmode
                    self.log(f"px4 heartbeat alindi: {port}")
                    return
                last_err = RuntimeError("heartbeat gelmedi")
            except Exception as e:
                self.log(f"port denenemedi: {port} -> {e}")
                last_err = e

        raise RuntimeError(f"heartbeat gelmedi. Son hata: {last_err}")

    def mavlink_reader_loop(self):
        while self.running:
            try:
                msg = self.master.recv_match(blocking=False)
                if msg is None:
                    time.sleep(0.01)
                    continue

                mtype = msg.get_type()

                if mtype == "ATTITUDE":
                    self.current_yaw = float(msg.yaw)

                elif mtype == "GLOBAL_POSITION_INT":
                    if hasattr(msg, "relative_alt"):
                        self.relative_alt_m = float(msg.relative_alt) / 1000.0

                elif mtype == "HEARTBEAT":
                    try:
                        self.flight_mode = self.master.flightmode
                        # ARM durumunu heartbeat'ten güncelle
                        if msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED:
                            self.is_armed = True
                        else:
                            self.is_armed = False
                    except Exception:
                        pass

            except Exception as e:
                self.log(f"mavlink read hata: {e}")
                time.sleep(0.05)

    def arm_vehicle(self):
        try:
            self.log("arm komutu gonderiliyor")
            self.master.mav.command_long_send(
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0,
                1, 0, 0, 0, 0, 0, 0
            )
            self.is_armed = True
        except Exception as e:
            self.log(f"arm hata: {e}")

    def disarm_vehicle(self, force=False):
        try:
            self.log(f"disarm komutu (force={force})")
            force_flag = 21196 if force else 0
            self.master.mav.command_long_send(
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0,
                0, force_flag, 0, 0, 0, 0, 0
            )
            self.is_armed = False
        except Exception as e:
            self.log(f"disarm hata: {e}")

    def takeoff(self, altitude_m=TAKEOFF_ALT_M):
        """
        Tam otomatik takeoff sekansı:
        1. Offboard warmup (setpoint stream)
        2. OFFBOARD moda geçiş
        3. ARM
        4. MAV_CMD_NAV_TAKEOFF gönder
        """
        try:
            self.state = "TAKEOFF"
            self.log(f"takeoff sekans basliyor: hedef {altitude_m:.1f}m")

            # 1) Offboard warmup — PX4 offboard geçmeden önce
            #    setpoint akışı istiyor
            self.warmup_offboard_stream()

            # 2) OFFBOARD moda geç
            self.request_offboard_mode()
            time.sleep(0.3)

            # 3) ARM
            if not self.is_armed:
                self.arm_vehicle()
                # ARM'ın kabul edilmesi için kısa bekleme
                time.sleep(1.0)

            # 4) Takeoff komutu gönder
            self.log(f"MAV_CMD_NAV_TAKEOFF gonderiliyor: {altitude_m:.1f}m")
            self.master.mav.command_long_send(
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                0,            # confirmation
                0,            # param1: pitch (ignored by PX4)
                0,            # param2: empty
                0,            # param3: empty
                0,            # param4: yaw (NaN = current)
                0,            # param5: lat (ignored, local)
                0,            # param6: lon (ignored, local)
                float(altitude_m)  # param7: altitude
            )

            self.log(f"takeoff baslatildi -> {altitude_m:.1f}m")
        except Exception as e:
            self.log(f"takeoff hata: {e}")

    def land(self):
        """
        PX4 AUTO.LAND moduna geçirir.
        MAV_CMD_NAV_LAND komutu gönderir.
        """
        try:
            self.log("land komutu gonderiliyor")

            self.master.mav.command_long_send(
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_CMD_NAV_LAND,
                0,      # confirmation
                0,      # param1: abort alt (0 = use default)
                0,      # param2: land mode (0 = normal)
                0,      # param3: empty
                0,      # param4: yaw (NaN = keep current)
                0,      # param5: lat (ignored, local)
                0,      # param6: lon (ignored, local)
                0       # param7: alt (ignored)
            )

            self.state = "LANDING"
            self.auto_enabled = False
            self.log("land modu aktif")
        except Exception as e:
            self.log(f"land hata: {e}")

    def force_disarm(self):
        """
        Zorla disarm — motorları anında kapatır.
        DIKKAT: Havadayken tehlikelidir!
        """
        self.log("!!! FORCE DISARM !!!")
        for i in range(5):
            try:
                self.master.mav.command_long_send(
                    self.master.target_system,
                    self.master.target_component,
                    mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                    0,
                    0,      # disarm
                    21196,  # force flag
                    0, 0, 0, 0, 0
                )
            except Exception as e:
                self.log(f"force disarm send hata: {e}")
            time.sleep(0.02)

        self.is_armed = False
        self.auto_enabled = False
        self.state = "DISARMED"
        self.set_commanded_velocity(0.0, 0.0, 0.0)
        self.log("force disarm tamamlandi")

    def request_offboard_mode(self):
        try:
            self.log("OFFBOARD isteniyor")
            self.master.set_mode("OFFBOARD")
            self.offboard_requested = True
        except Exception as e:
            self.log(f"OFFBOARD hata: {e}")

    def request_posctl_mode(self):
        try:
            self.log("POSCTL isteniyor")
            self.master.set_mode("POSCTL")
        except Exception as e:
            self.log(f"POSCTL hata: {e}")

    def send_body_velocity(self, vx, vy, vz, yaw_rate=0.0):
        """
        PX4 BODY_NED
        vx = ileri(+)
        vy = sağ(+)
        vz = aşağı(+)
        """
        try:
            type_mask = (
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE
            )

            self.master.mav.set_position_target_local_ned_send(
                0,
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_FRAME_BODY_NED,
                type_mask,
                0, 0, 0,
                float(vx), float(vy), float(vz),
                0, 0, 0,
                0, float(yaw_rate)
            )
        except Exception as e:
            self.log(f"velocity send hata: {e}")

    def set_commanded_velocity(self, vx, vy, vz):
        with self.cmd_lock:
            self.cmd_vx = float(vx)
            self.cmd_vy = float(vy)
            self.cmd_vz = float(vz)

    def get_commanded_velocity(self):
        with self.cmd_lock:
            return self.cmd_vx, self.cmd_vy, self.cmd_vz

    def warmup_offboard_stream(self, seconds=OFFBOARD_WARMUP_SEC):
        self.log(f"offboard warmup basladi ({seconds:.1f}s)")
        t_end = time.time() + seconds
        while time.time() < t_end and self.running:
            self.send_body_velocity(0.0, 0.0, 0.0)
            time.sleep(1.0 / SETPOINT_RATE_HZ)
        self.log("offboard warmup bitti")

    def setpoint_sender_loop(self):
        period = 1.0 / SETPOINT_RATE_HZ

        while self.running:
            try:
                vx, vy, vz = self.get_commanded_velocity()

                if self.auto_enabled and not self.manual_override:
                    self.send_body_velocity(vx, vy, vz)
                elif SEND_ZERO_ON_IDLE:
                    self.send_body_velocity(0.0, 0.0, 0.0)

                time.sleep(period)

            except Exception as e:
                self.log(f"setpoint loop hata: {e}")
                time.sleep(0.05)

    # -----------------------------------------------------
    # video
    # -----------------------------------------------------
    def connect_video_target(self):
        while self.running:
            try:
                self.log(f"video hedefe baglaniliyor: {VIDEO_TARGET_HOST}:{VIDEO_TARGET_PORT}")
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect((VIDEO_TARGET_HOST, VIDEO_TARGET_PORT))
                self.video_sock = sock
                self.video_connected = True
                self.log("video baglantisi kuruldu")
                return
            except Exception as e:
                self.video_connected = False
                self.log(f"video baglanti hatasi: {e}")
                time.sleep(2)

    def video_sender_loop(self):
        self.connect_video_target()

        while self.running:
            try:
                if self.video_sock is None:
                    self.connect_video_target()

                with self.frame_lock:
                    if self.latest_jpeg is None:
                        time.sleep(0.02)
                        continue
                    data = self.latest_jpeg

                packet = struct.pack("Q", len(data)) + data
                self.video_sock.sendall(packet)
                time.sleep(1.0 / FPS_LIMIT)

            except Exception as e:
                self.video_connected = False
                self.log(f"video gonderim hatasi: {e}")
                try:
                    self.video_sock.close()
                except Exception:
                    pass
                self.video_sock = None
                time.sleep(1.0)

    # -----------------------------------------------------
    # telemetri
    # -----------------------------------------------------
    def send_telem(self):
        msg = {
            "state": self.state,
            "flight_mode": self.flight_mode,
            "marker_detected": self.marker_detected,
            "err_x": int(self.err_x),
            "err_y": int(self.err_y),
            "altitude": float(self.aruco_altitude) if self.aruco_altitude is not None else -1.0,
            "relative_alt": -1.0 if self.relative_alt_m is None else float(self.relative_alt_m),
            "yaw_deg": float(math.degrees(self.current_yaw)),
            "manual_override": self.manual_override,
            "auto_enabled": self.auto_enabled,
            "is_armed": self.is_armed,
            "offboard_requested": self.offboard_requested,
            "fps": float(self.fps),
            "last_event": self.last_event
        }

        try:
            data = json.dumps(msg).encode("utf-8")
            self.telem_sock.sendto(data, (GCS_IP, TELEM_PORT))
        except Exception as e:
            print("[WARN] telemetri gonderilemedi:", e)

    def telemetry_loop(self):
        while self.running:
            self.send_telem()
            time.sleep(0.1)

    # -----------------------------------------------------
    # komut
    # -----------------------------------------------------
    def reset_controller(self):
        self.integral_x = 0.0
        self.integral_y = 0.0
        self.prev_err_x = 0.0
        self.prev_err_y = 0.0
        self.err_x = 0.0
        self.err_y = 0.0
        self.marker_detected = False
        self.aruco_altitude = -1.0
        self.last_marker_time = 0.0
        self.set_commanded_velocity(0.0, 0.0, 0.0)

    def command_listener_loop(self):
        while self.running:
            try:
                data, addr = self.cmd_sock.recvfrom(1024)
                cmd = data.decode("utf-8").strip().upper()

                if cmd == "START":
                    self.auto_enabled = True
                    self.manual_override = False
                    self.state = "SEARCH"
                    self.log(f"START geldi from {addr}")

                    if AUTO_REQUEST_OFFBOARD_ON_START:
                        self.warmup_offboard_stream()
                        self.request_offboard_mode()

                    if AUTO_ARM_ON_START:
                        self.arm_vehicle()

                elif cmd == "HOLD":
                    self.auto_enabled = False
                    self.state = "HOLD"
                    self.log(f"HOLD geldi from {addr}")
                    self.set_commanded_velocity(0.0, 0.0, 0.0)

                elif cmd == "ABORT":
                    self.auto_enabled = False
                    self.state = "ABORT"
                    self.log(f"ABORT geldi from {addr}")
                    self.set_commanded_velocity(0.0, 0.0, 0.0)

                elif cmd == "RESET":
                    self.auto_enabled = False
                    self.manual_override = False
                    self.offboard_requested = False
                    self.state = "IDLE"
                    self.reset_controller()
                    self.log(f"RESET geldi from {addr}")

                elif cmd == "MANUAL_ON":
                    self.manual_override = True
                    self.auto_enabled = False
                    self.state = "MANUAL"
                    self.log(f"MANUAL_ON geldi from {addr}")
                    self.set_commanded_velocity(0.0, 0.0, 0.0)

                elif cmd == "MANUAL_OFF":
                    self.manual_override = False
                    self.state = "IDLE"
                    self.log(f"MANUAL_OFF geldi from {addr}")

                elif cmd == "ARM":
                    self.arm_vehicle()

                elif cmd == "DISARM":
                    self.disarm_vehicle(force=False)

                elif cmd == "TAKEOFF":
                    self.log(f"TAKEOFF geldi from {addr}")
                    self.takeoff(TAKEOFF_ALT_M)

                elif cmd == "LAND":
                    self.log(f"LAND geldi from {addr}")
                    self.land()

                elif cmd == "FORCE_DISARM":
                    self.log(f"FORCE_DISARM geldi from {addr}")
                    self.force_disarm()

                elif cmd == "OFFBOARD":
                    self.warmup_offboard_stream()
                    self.request_offboard_mode()

                elif cmd == "POSCTL":
                    self.request_posctl_mode()

                else:
                    self.log(f"bilinmeyen komut: {cmd}")

            except socket.timeout:
                pass
            except Exception as e:
                self.log(f"komut loop hata: {e}")
                time.sleep(0.1)

    # -----------------------------------------------------
    # aruco
    # -----------------------------------------------------
    def detect_marker(self, gray):
        if self.new_api:
            corners, ids, _ = self.detector.detectMarkers(gray)
        else:
            corners, ids, _ = aruco.detectMarkers(
                gray,
                self.aruco_dict,
                parameters=self.aruco_params
            )

        if ids is not None:
            print("[DEBUG] Gorulen ID'ler:", ids.flatten().tolist())

        if ids is None:
            return None, corners, ids

        ids_flat = ids.flatten()
        for i, mid in enumerate(ids_flat):
            if int(mid) == TARGET_MARKER_ID:
                return corners[i][0], corners, ids

        return None, corners, ids

    # -----------------------------------------------------
    # kontrol
    # -----------------------------------------------------
    def process_frame_and_control(self, frame):
        now = time.time()
        dt = now - self.last_control_time
        self.last_control_time = now
        if dt <= 0:
            dt = 0.02
        if dt > 0.1:
            dt = 0.1

        out = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        target_corners, all_corners, all_ids = self.detect_marker(gray)

        h, w = frame.shape[:2]
        img_cx = w // 2
        img_cy = h // 2

        vx_cmd = 0.0
        vy_cmd = 0.0
        vz_cmd = 0.0

        self.marker_detected = False
        self.err_x = 0.0
        self.err_y = 0.0
        self.aruco_altitude = -1.0

        if target_corners is not None:
            self.marker_detected = True
            self.last_marker_time = now

            tx = int(np.mean(target_corners[:, 0]))
            ty = int(np.mean(target_corners[:, 1]))

            err_x = img_cx - tx
            err_y = img_cy - ty

            if REVERSE_X:
                err_x *= -1
            if REVERSE_Y:
                err_y *= -1

            self.err_x = err_x
            self.err_y = err_y

            dist_px = math.sqrt(err_x ** 2 + err_y ** 2)

            ok, rvec, tvec = cv2.solvePnP(
                self.marker_points,
                target_corners,
                self.camera_matrix,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )

            if ok:
                self.aruco_altitude = float(tvec[2][0])

            if all_ids is not None:
                aruco.drawDetectedMarkers(out, all_corners, all_ids)

            cv2.circle(out, (img_cx, img_cy), 5, (0, 255, 0), -1)
            cv2.circle(out, (tx, ty), 5, (0, 0, 255), -1)
            cv2.line(out, (img_cx, img_cy), (tx, ty), (255, 0, 0), 2)

            cv2.putText(out, f"mode={self.flight_mode}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(out, f"err_x={err_x}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(out, f"err_y={err_y}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(out, f"aruco_z={self.aruco_altitude:.2f} m", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            if self.reprojection_error is not None:
                cv2.putText(out, f"reproj={self.reprojection_error:.3f}", (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

            if not self.auto_enabled or self.manual_override:
                self.state = "MANUAL" if self.manual_override else "IDLE"
                return out, 0.0, 0.0, 0.0

            self.integral_x += err_x * dt
            self.integral_y += err_y * dt
            self.integral_x = clamp(self.integral_x, -300, 300)
            self.integral_y = clamp(self.integral_y, -300, 300)

            der_x = (err_x - self.prev_err_x) / dt
            der_y = (err_y - self.prev_err_y) / dt
            self.prev_err_x = err_x
            self.prev_err_y = err_y

            pid_x = KP * err_x + KI * self.integral_x + KD * der_x
            pid_y = KP * err_y + KI * self.integral_y + KD * der_y

            vx_cmd = clamp(pid_y, -MAX_VEL_X, MAX_VEL_X)
            vy_cmd = clamp(pid_x, -MAX_VEL_Y, MAX_VEL_Y)

            aligned = dist_px < ALIGN_THRESHOLD_PX
            low_enough = (self.aruco_altitude > 0) and (self.aruco_altitude < DESCEND_START_ALT)

            if aligned:
                self.state = "ALIGN"
            else:
                self.state = "TRACK"

            if aligned and low_enough:
                self.state = "DESCEND"
                vz_cmd = clamp(DESCENT_SPEED, 0.0, MAX_VEL_Z)

            if (
                self.aruco_altitude > 0
                and self.aruco_altitude < FINAL_ALT_THRESHOLD
                and dist_px < FINAL_PIXEL_THRESHOLD
            ):
                self.state = "FINAL"
                self.log("final esik saglandi")
                vx_cmd, vy_cmd, vz_cmd = 0.0, 0.0, 0.0

                if AUTO_DISARM_ON_FINAL:
                    self.disarm_vehicle(force=False)
                    self.auto_enabled = False
                    self.state = "DONE"

            cv2.putText(out, f"STATE={self.state}", (10, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(out, f"vx={vx_cmd:.2f} vy={vy_cmd:.2f} vz={vz_cmd:.2f}", (10, 210),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            return out, vx_cmd, vy_cmd, vz_cmd

        else:
            if self.auto_enabled and not self.manual_override:
                self.state = "SEARCH"
            else:
                self.state = "MANUAL" if self.manual_override else "IDLE"

            self.integral_x = 0.0
            self.integral_y = 0.0

            cv2.putText(out, "MARKER YOK", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.putText(out, f"mode={self.flight_mode}", (10, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            return out, 0.0, 0.0, 0.0

    # -----------------------------------------------------
    # ana loop
    # -----------------------------------------------------
    def main_loop(self):
        self.log("ana loop basladi")

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.log("kameradan frame okunamadi")
                time.sleep(0.02)
                continue

            processed, vx, vy, vz = self.process_frame_and_control(frame)

            if self.auto_enabled and not self.manual_override:
                if (time.time() - self.last_marker_time) > MARKER_LOST_TIMEOUT:
                    vx, vy, vz = 0.0, 0.0, 0.0
                    self.state = "SEARCH"

            self.set_commanded_velocity(vx, vy, vz)

            self._fps_counter += 1
            if time.time() - self._fps_last_t >= 1.0:
                self.fps = self._fps_counter / (time.time() - self._fps_last_t)
                self._fps_counter = 0
                self._fps_last_t = time.time()

            ok, enc = cv2.imencode(
                ".jpg",
                processed,
                [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
            )
            if ok:
                with self.frame_lock:
                    self.latest_frame = processed
                    self.latest_jpeg = enc.tobytes()

            time.sleep(1.0 / FPS_LIMIT)

    # -----------------------------------------------------
    # başlat
    # -----------------------------------------------------
    def run(self):
        self.open_camera()
        self.connect_mavlink()

        threading.Thread(target=self.mavlink_reader_loop, daemon=True).start()
        threading.Thread(target=self.setpoint_sender_loop, daemon=True).start()
        threading.Thread(target=self.video_sender_loop, daemon=True).start()
        threading.Thread(target=self.telemetry_loop, daemon=True).start()
        threading.Thread(target=self.command_listener_loop, daemon=True).start()

        try:
            self.main_loop()
        except KeyboardInterrupt:
            self.log("ctrl+c alindi")
        finally:
            self.running = False
            try:
                self.set_commanded_velocity(0.0, 0.0, 0.0)
                time.sleep(0.2)
            except Exception:
                pass
            try:
                self.cap.release()
            except Exception:
                pass
            try:
                self.cmd_sock.close()
            except Exception:
                pass
            try:
                self.telem_sock.close()
            except Exception:
                pass
            try:
                if self.video_sock:
                    self.video_sock.close()
            except Exception:
                pass


if __name__ == "__main__":
    PerfectLanderPX4().run()
