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

# ==========================================================
# PRECISION LANDING PARAMETRELERI
# ==========================================================

# ---------- Multi-Marker Layout ----------
# ID=0: merkez büyük marker (coarse), ID=1..4: köşe küçük markerlar (fine)
CORNER_IDS = [1, 2, 3, 4]
CENTER_MARKER_SIZE = MARKER_SIZE_M   # ID=0 boyutu (metre)
CORNER_MARKER_SIZE = 0.05            # ID=1..4 boyutu (metre)
CORNER_DX = 0.25                     # Merkezden köşe X mesafesi (metre)
CORNER_DY = 0.25                     # Merkezden köşe Y mesafesi (metre)
CORNER_BLEND_WEIGHT = 0.35           # Blend modunda köşe ağırlığı

# ---------- Fine / Coarse mode ----------
FINE_ENTER_ALT = 1.50   # Bu altında fine mode aktif
FINE_EXIT_ALT  = 2.00   # Bu üstünde coarse'a dön (histerezis)

# ---------- Lock mode ----------
LOCK_ALT_M       = 3.00
LOCK_ENTER_PX    = 120
LOCK_EXIT_PX     = 160
LOCK_YAW_ENTER   = 0.25   # rad
LOCK_YAW_EXIT    = 0.35   # rad
LOCK_VZ_DOWN     = 0.25   # m/s
LOCK_KP_MULT     = 2.0
LOCK_KD_MULT     = 2.0
LOCK_DISABLE_INTEGRAL = True

# ---------- Velocity taper ----------
TAPER_BASE_FAR   = 250.0
TAPER_BASE_NEAR  = 180.0
TAPER_MIN_FAR    = 0.20
TAPER_MIN_NEAR   = 0.12

# ---------- Quintic trajectory ----------
TRAJ_BASE_DURATION = 8.0   # saniye

# ---------- Yaw PID ----------
KP_YAW = 1.0

# ---------- Precision kill ----------
PRECISION_KILL_ALT  = 0.40   # m
PRECISION_KILL_DIST = 120    # px
BLIND_KILL_ALT      = 0.50   # m: marker kaybolursa bu altında kill


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
        self.cmd_yaw_rate = 0.0
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

        # köşe marker 3D noktaları
        cs = CORNER_MARKER_SIZE / 2.0
        self.marker_points_corner = np.array([
            [-cs,  cs, 0.0],
            [ cs,  cs, 0.0],
            [ cs, -cs, 0.0],
            [-cs, -cs, 0.0]
        ], dtype=np.float32)

        # Köşe markerdan platform merkezine offset vektörü (marker frame)
        dx, dy = CORNER_DX, CORNER_DY
        self.corner_offsets = {
            1: np.array([+dx, -dy, 0.0], dtype=np.float32),  # sol-üst
            2: np.array([-dx, -dy, 0.0], dtype=np.float32),  # sağ-üst
            3: np.array([+dx, +dy, 0.0], dtype=np.float32),  # sol-alt
            4: np.array([-dx, +dy, 0.0], dtype=np.float32),  # sağ-alt
        }

        # ---- Precision Landing durumu ----
        self.precision_land_active = False
        self.precision_state = "IDLE"   # SEARCH/COARSE/FINE/BLEND/LOCK/TRAJ/KILL
        self.fine_mode = False
        self.corner_count = 0
        self.precision_blind = True
        self.was_landing = False

        # Quintic trajectory
        self.traj_active = False
        self.traj_start_time = 0.0
        self.traj_start_alt = 0.0
        self.traj_duration = TRAJ_BASE_DURATION

        # Center-last tracking
        self.last_seen_center_time = 0.0
        self.last_seen_center_z = 999.0
        self.precision_last_alt = 999.0
        self.max_vel_precision = 0.80   # m/s precision landing max hiz

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
        Tam otomatik takeoff sekansı (ayrı thread'de çalışır):
        1. Offboard warmup (setpoint stream)
        2. OFFBOARD moda geçiş
        3. ARM (bekleme döngüsü ile)
        4. OFFBOARD velocity ile yükselme
        """
        # Ayrı thread'de çalıştır — komut dinleyiciyi bloklamasın
        t = threading.Thread(target=self._takeoff_sequence, args=(altitude_m,), daemon=True)
        t.start()

    def _takeoff_sequence(self, altitude_m):
        """Takeoff sekansının asıl implementasyonu."""
        try:
            self.state = "TAKEOFF"
            self.log(f"takeoff sekans basliyor: hedef {altitude_m:.1f}m")

            # 1) Offboard warmup — PX4 offboard moduna geçmeden önce
            #    minimum ~1 saniye setpoint akışı istiyor
            self.warmup_offboard_stream()

            # 2) OFFBOARD moda geç
            self.request_offboard_mode()
            time.sleep(0.5)

            # 3) ARM — onay döngüsü ile
            if not self.is_armed:
                self.arm_vehicle()
                self.log("ARM bekleniyor...")
                t_start = time.time()
                while not self.is_armed and (time.time() - t_start) < 4.0:
                    # Setpoint akışını devam ettir yoksa PX4
                    # offboard'dan çıkar
                    self.send_body_velocity(0.0, 0.0, 0.0)
                    time.sleep(0.1)

                if not self.is_armed:
                    self.log("TAKEOFF BASARISIZ: ARM kabul edilmedi!")
                    self.state = "IDLE"
                    return

            self.log("ARM onaylandi!")

            # 4) OFFBOARD velocity ile yükselme
            #    NED frame: vz negatif = yukarı
            climb_speed = 0.5   # m/s yukarı
            climb_time = altitude_m / climb_speed

            self.log(f"yukselme basliyor: {climb_speed:.1f} m/s, sure: {climb_time:.1f}s")

            # auto_enabled'ı aç — setpoint_sender_loop
            # cmd velocity'yi göndersin
            self.auto_enabled = True
            self.manual_override = False
            self.set_commanded_velocity(0.0, 0.0, -climb_speed)

            time.sleep(climb_time)

            # 5) Hover — yerinde dur
            self.set_commanded_velocity(0.0, 0.0, 0.0)
            self.state = "HOVER"
            self.log(f"takeoff tamamlandi: {altitude_m:.1f}m")

        except Exception as e:
            self.log(f"takeoff hata: {e}")
            self.state = "IDLE"

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

    def set_commanded_velocity(self, vx, vy, vz, yaw_rate=0.0):
        with self.cmd_lock:
            self.cmd_vx = float(vx)
            self.cmd_vy = float(vy)
            self.cmd_vz = float(vz)
            self.cmd_yaw_rate = float(yaw_rate)

    def get_commanded_velocity(self):
        with self.cmd_lock:
            return self.cmd_vx, self.cmd_vy, self.cmd_vz, self.cmd_yaw_rate

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
                vx, vy, vz, yr = self.get_commanded_velocity()

                if self.auto_enabled and not self.manual_override:
                    self.send_body_velocity(vx, vy, vz, yr)
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
            "last_event": self.last_event,
            # Precision Landing telemetrisi
            "precision_land_active": self.precision_land_active,
            "precision_state": self.precision_state,
            "fine_mode": self.fine_mode,
            "corner_count": self.corner_count,
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

        # Precision landing reset
        self.precision_land_active = False
        self.precision_state = "IDLE"
        self.fine_mode = False
        self.corner_count = 0
        self.precision_blind = True
        self.was_landing = False
        self.traj_active = False
        self.traj_start_time = 0.0
        self.traj_start_alt = 0.0
        self.traj_duration = TRAJ_BASE_DURATION
        self.last_seen_center_time = 0.0
        self.last_seen_center_z = 999.0
        self.precision_last_alt = 999.0

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

                elif cmd == "PRECISION_LAND":
                    self.log(f"PRECISION_LAND geldi from {addr}")
                    self.precision_land_active = True
                    self.precision_state = "SEARCH"
                    self.precision_blind = True
                    self.fine_mode = False
                    self.traj_active = False
                    self.was_landing = False
                    self.auto_enabled = True
                    self.manual_override = False
                    self.state = "PRECISION_LAND"
                    # OFFBOARD'a geç (zaten orada değilse)
                    if self.flight_mode != "OFFBOARD":
                        self.warmup_offboard_stream()
                        self.request_offboard_mode()

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
    # PRECISION LANDING: yardımcı metodlar
    # -----------------------------------------------------
    def calculate_quintic_trajectory(self, current_time):
        """5. derece polinom ile yumuşak iniş profili."""
        t = current_time - self.traj_start_time
        if t >= self.traj_duration:
            return 0.0, 0.0

        tau = t / self.traj_duration
        s_tau = (10 * tau**3) - (15 * tau**4) + (6 * tau**5)
        v_scale = (30 * tau**2) - (60 * tau**3) + (30 * tau**4)

        total_drop = self.traj_start_alt
        desired_z = self.traj_start_alt - (total_drop * s_tau)
        desired_vz = (total_drop * v_scale) / self.traj_duration
        return desired_z, desired_vz

    def calculate_runway_yaw(self, rvec):
        """Marker oryantasyonundan yaw hatası hesapla."""
        R_mat, _ = cv2.Rodrigues(rvec)
        green_x = R_mat[0][1]
        green_y = R_mat[1][1]
        angle_green = math.atan2(green_y, green_x)

        target = -math.pi / 2
        err = angle_green - target
        while err > math.pi:
            err -= 2 * math.pi
        while err < -math.pi:
            err += 2 * math.pi
        return err

    def _center_pixel_from_tvec(self, tvec):
        """tvec'ten piksel konumuna projeksiyon (pinhole model)."""
        fx = float(self.camera_matrix[0, 0])
        fy = float(self.camera_matrix[1, 1])
        cx = float(self.camera_matrix[0, 2])
        cy = float(self.camera_matrix[1, 2])

        z = float(tvec[2])
        if z <= 1e-6:
            return None, None
        u = int(fx * (float(tvec[0]) / z) + cx)
        v = int(fy * (float(tvec[1]) / z) + cy)
        return u, v

    def detect_all_markers(self, gray):
        """Tüm ArUco markerları tespit et, her biri için PnP çöz."""
        if self.new_api:
            corners, ids, _ = self.detector.detectMarkers(gray)
        else:
            corners, ids, _ = aruco.detectMarkers(
                gray, self.aruco_dict, parameters=self.aruco_params
            )

        det = {}
        if ids is None or len(ids) == 0:
            return det, corners, ids

        ids_flat = ids.flatten()
        for i, mid in enumerate(ids_flat):
            mid = int(mid)
            if mid == TARGET_MARKER_ID:
                pts = self.marker_points
            elif mid in CORNER_IDS:
                pts = self.marker_points_corner
            else:
                continue

            c = corners[i][0]
            tx = int(np.mean(c[:, 0]))
            ty = int(np.mean(c[:, 1]))

            ok, rvec, tvec = cv2.solvePnP(
                pts, c, self.camera_matrix, self.dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
            if not ok:
                continue

            det[mid] = {
                "rvec": rvec,
                "tvec": tvec.reshape(3),
                "tx": tx,
                "ty": ty,
                "c": c,
            }

        return det, corners, ids

    # -----------------------------------------------------
    # PRECISION LANDING: ana kontrol döngüsü
    # -----------------------------------------------------
    def precision_land_control(self, frame, dt):
        """
        Gelişmiş hassas iniş kontrolcüsü.
        Return: (out_frame, vx, vy, vz, yaw_rate)
        """
        now = time.time()
        out = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        h, w = frame.shape[:2]
        img_cx = w // 2
        img_cy = h // 2

        # Tüm markerları tespit et
        det, all_corners, all_ids = self.detect_all_markers(gray)

        has0 = (TARGET_MARKER_ID in det)
        z0 = float(det[TARGET_MARKER_ID]["tvec"][2]) if has0 else self.last_seen_center_z

        # ---- Fine / Coarse mode switching ----
        if not self.fine_mode:
            if has0 and (z0 < FINE_ENTER_ALT) and any(cid in det for cid in CORNER_IDS):
                self.fine_mode = True
                self.log("FINE MODE: kose markerlar aktif")
        else:
            if has0 and (z0 > FINE_EXIT_ALT) and (not self.traj_active):
                self.fine_mode = False
                self.log("COARSE MODE: uzaklasildi")

        # ---- Köşelerden platform merkezi kestir ----
        t_center_from_corners = None
        rvec_center_ref = None
        self.corner_count = 0

        if self.fine_mode:
            centers = []
            for cid in CORNER_IDS:
                if cid not in det:
                    continue
                rvec_i = det[cid]["rvec"]
                tvec_i = det[cid]["tvec"]
                R_i, _ = cv2.Rodrigues(rvec_i)
                offset_i = self.corner_offsets[cid]
                t_center_i = (R_i @ offset_i) + tvec_i
                centers.append(t_center_i)
                self.corner_count += 1
                if rvec_center_ref is None:
                    rvec_center_ref = rvec_i

            if self.corner_count >= 1:
                t_center_from_corners = np.mean(np.stack(centers, axis=0), axis=0)

        # ---- Aktif ölçüm seçimi (blend / fallback) ----
        active_tvec = None
        active_rvec = None
        active_tx = None
        active_ty = None
        active_source = "NONE"

        if not self.fine_mode:
            if has0:
                active_tvec = det[TARGET_MARKER_ID]["tvec"]
                active_tx = det[TARGET_MARKER_ID]["tx"]
                active_ty = det[TARGET_MARKER_ID]["ty"]
                active_rvec = det[TARGET_MARKER_ID]["rvec"]
                active_source = "ID0"
        else:
            if has0 and (t_center_from_corners is not None):
                w_blend = CORNER_BLEND_WEIGHT
                active_tvec = (1.0 - w_blend) * det[TARGET_MARKER_ID]["tvec"] + w_blend * t_center_from_corners
                active_tx, active_ty = self._center_pixel_from_tvec(active_tvec)
                active_rvec = det[TARGET_MARKER_ID]["rvec"]
                active_source = f"BLEND(0+{self.corner_count})"
            elif has0:
                active_tvec = det[TARGET_MARKER_ID]["tvec"]
                active_tx = det[TARGET_MARKER_ID]["tx"]
                active_ty = det[TARGET_MARKER_ID]["ty"]
                active_rvec = det[TARGET_MARKER_ID]["rvec"]
                active_source = "ID0_ONLY"
            elif t_center_from_corners is not None:
                active_tvec = t_center_from_corners
                active_tx, active_ty = self._center_pixel_from_tvec(active_tvec)
                active_rvec = rvec_center_ref
                active_source = f"CORNERS({self.corner_count})"

        # Projeksiyon hatası kontrolü
        if active_tvec is not None and (active_tx is None or active_ty is None):
            active_tvec = None

        vx_local = 0.0
        vy_local = 0.0
        vz = 0.0
        yaw_rate_cmd = 0.0

        if active_tvec is not None:
            self.marker_detected = True
            self.last_marker_time = now

            self.aruco_altitude = float(active_tvec[2])
            self.precision_last_alt = self.aruco_altitude
            if has0:
                self.last_seen_center_time = now
                self.last_seen_center_z = float(det[TARGET_MARKER_ID]["tvec"][2])

            tx = int(active_tx)
            ty = int(active_ty)

            err_forward = img_cx - tx
            err_right = img_cy - ty

            if REVERSE_X:
                err_forward *= -1
            if REVERSE_Y:
                err_right *= -1

            self.err_x = err_forward
            self.err_y = err_right
            dist_pixel = math.sqrt(err_forward ** 2 + err_right ** 2)

            # Yaw hatası
            yaw_err = self.calculate_runway_yaw(active_rvec)
            yaw_rate_cmd = float(np.clip(yaw_err * KP_YAW, -1.0, 1.0))

            # Lock mode aktif mi?
            lock_active = (self.aruco_altitude < LOCK_ALT_M)

            # PID kazançları (lock modunda artırılmış)
            kp_eff = KP
            ki_eff = KI
            kd_eff = KD
            if lock_active:
                kp_eff *= LOCK_KP_MULT
                kd_eff *= LOCK_KD_MULT
                if LOCK_DISABLE_INTEGRAL:
                    ki_eff = 0.0

            # İlk yakalama
            if self.precision_blind:
                self.log(f"PRECISION: hedef yakalandi, source={active_source}")
                self.precision_blind = False
                self.prev_err_x = err_forward
                self.prev_err_y = err_right
                self.integral_x = 0.0
                self.integral_y = 0.0

                if self.was_landing:
                    self.log(f"PRECISION: inise devam, alt={self.aruco_altitude:.2f}m")
                    self.traj_active = True
                    self.traj_start_time = now
                    self.traj_start_alt = self.aruco_altitude
                    scale = float(np.clip(self.aruco_altitude / 2.0, 0.4, 1.0))
                    self.traj_duration = TRAJ_BASE_DURATION * scale

            # Precision state güncelle
            mode_txt = "FINE" if self.fine_mode else "COARSE"
            self.precision_state = f"{mode_txt}:{active_source}"

            # ---- GÖRSEL KILL ----
            if self.aruco_altitude < PRECISION_KILL_ALT and dist_pixel < PRECISION_KILL_DIST:
                self.log("PRECISION: gorsel temas -> FORCE DISARM")
                self.precision_state = "KILL"
                self.state = "PRECISION_DONE"
                self.force_disarm()
                return out, 0.0, 0.0, 0.0, 0.0

            # ---- PID ----
            p_x = err_forward * kp_eff
            p_y = err_right * kp_eff

            if dist_pixel < 200:
                self.integral_x += err_forward * dt
                self.integral_y += err_right * dt
            else:
                self.integral_x = 0.0
                self.integral_y = 0.0

            self.integral_x = clamp(self.integral_x, -200, 200)
            self.integral_y = clamp(self.integral_y, -200, 200)

            d_x = (err_forward - self.prev_err_x) / dt
            d_y = (err_right - self.prev_err_y) / dt
            self.prev_err_x = err_forward
            self.prev_err_y = err_right

            pid_x = p_x + (self.integral_x * ki_eff) + (d_x * kd_eff)
            pid_y = p_y + (self.integral_y * ki_eff) + (d_y * kd_eff)

            vx_body = float(np.clip(pid_x, -self.max_vel_precision, self.max_vel_precision))
            vy_body = float(np.clip(pid_y, -self.max_vel_precision, self.max_vel_precision))

            # Velocity taper
            base = TAPER_BASE_FAR
            min_scale = TAPER_MIN_FAR
            if self.aruco_altitude < LOCK_ALT_M:
                base = TAPER_BASE_NEAR
                min_scale = TAPER_MIN_NEAR

            taper_scale = float(np.clip(dist_pixel / base, min_scale, 1.0))
            vx_body *= taper_scale
            vy_body *= taper_scale

            # Body -> Local (yaw rotasyonu)
            cos_y = math.cos(self.current_yaw)
            sin_y = math.sin(self.current_yaw)
            vx_local = vx_body * cos_y - vy_body * sin_y
            vy_local = vx_body * sin_y + vy_body * cos_y

            # ---- Yörünge başlatma ----
            start_cond = ((dist_pixel < 100) and (abs(yaw_err) < 0.3)) or (self.aruco_altitude < 1.0)
            if not self.traj_active and start_cond:
                self.traj_active = True
                self.was_landing = True
                self.traj_start_time = now
                self.traj_start_alt = self.aruco_altitude
                self.traj_duration = TRAJ_BASE_DURATION
                self.log("PRECISION: inis rotasi basladi")

            # ---- Z kontrolü ----
            if self.traj_active:
                desired_z, feed_forward_vz = self.calculate_quintic_trajectory(now)
                z_error = self.aruco_altitude - desired_z
                vz = float(np.clip(feed_forward_vz + (z_error * 1.5), -0.5, 0.8))
            else:
                vz = 0.0

            # ---- Lock mode gating ----
            if lock_active and self.traj_active:
                lock_good = (dist_pixel < LOCK_ENTER_PX) and (abs(yaw_err) < LOCK_YAW_ENTER)
                lock_bad = (dist_pixel > LOCK_EXIT_PX) or (abs(yaw_err) > LOCK_YAW_EXIT)

                if lock_bad:
                    vz = 0.0
                    self.precision_state += "|HOLD_Z"
                elif lock_good:
                    vz = float(np.clip(vz, -0.2, LOCK_VZ_DOWN))
                    self.precision_state += "|LOCK_OK"

            # ---- Görsel overlay ----
            if all_ids is not None:
                aruco.drawDetectedMarkers(out, all_corners, all_ids)

            if has0:
                cv2.drawFrameAxes(out, self.camera_matrix, self.dist_coeffs,
                                  det[TARGET_MARKER_ID]["rvec"],
                                  det[TARGET_MARKER_ID]["tvec"].reshape(3, 1), 0.15)

            cv2.line(out, (img_cx, img_cy), (tx, ty), (255, 0, 0), 2)
            cv2.circle(out, (img_cx, img_cy), 5, (0, 255, 0), -1)
            cv2.circle(out, (tx, ty), 5, (0, 0, 255), -1)

            cv2.putText(out, f"PREC: {self.precision_state}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(out, f"Alt: {self.aruco_altitude:.2f}m  corners:{self.corner_count}", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            landing_txt = "LANDING" if self.traj_active else "ALIGNING"
            cv2.putText(out, landing_txt, (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if self.traj_active else (0, 255, 255), 2)
            cv2.putText(out, f"vx={vx_local:.2f} vy={vy_local:.2f} vz={vz:.2f}", (10, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        else:
            # Marker kayboldu
            self.marker_detected = False
            self.err_x = 0.0
            self.err_y = 0.0

            if not self.precision_blind:
                self.log(f"PRECISION: marker kayboldu! Son alt={self.precision_last_alt:.2f}m")
                self.precision_blind = True

                # Kör nokta inişi kontrolü
                if self.was_landing and self.precision_last_alt < BLIND_KILL_ALT:
                    self.log("PRECISION: kor nokta inisi -> FORCE DISARM")
                    self.precision_state = "BLIND_KILL"
                    self.state = "PRECISION_DONE"
                    self.force_disarm()
                    return out, 0.0, 0.0, 0.0, 0.0

                if self.traj_active:
                    self.traj_active = False
                    self.log("PRECISION: yorunge duraklatildi")

            self.integral_x = 0.0
            self.integral_y = 0.0
            self.precision_state = "SEARCH"

            cv2.putText(out, "PRECISION: MARKER ARANIYOR...", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            vx_local = 0.0
            vy_local = 0.0
            vz = 0.0
            yaw_rate_cmd = 0.0

        return out, vx_local, vy_local, vz, yaw_rate_cmd

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

        # ---- Precision mode delegasyonu ----
        if self.precision_land_active:
            return self.precision_land_control(frame, dt)

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
                return out, 0.0, 0.0, 0.0, 0.0

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

            return out, vx_cmd, vy_cmd, vz_cmd, 0.0

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

            return out, 0.0, 0.0, 0.0, 0.0

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

            processed, vx, vy, vz, yr = self.process_frame_and_control(frame)

            if self.auto_enabled and not self.manual_override:
                if not self.precision_land_active:
                    # Precision modda marker kaybi zaten içerde yönetiliyor
                    if (time.time() - self.last_marker_time) > MARKER_LOST_TIMEOUT:
                        vx, vy, vz, yr = 0.0, 0.0, 0.0, 0.0
                        if not self.precision_land_active:
                            self.state = "SEARCH"

            self.set_commanded_velocity(vx, vy, vz, yr)

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
