#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import cv2
import socket
import struct
import json
import threading
import time
from collections import deque

import numpy as np

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QPalette, QFontDatabase
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QTextEdit,
    QVBoxLayout, QHBoxLayout, QGridLayout, QFrame, QSizePolicy,
    QGraphicsDropShadowEffect
)

# =========================
# AYARLAR
# =========================
VIDEO_HOST = "0.0.0.0"
VIDEO_PORT = 9999

TELEM_HOST = "0.0.0.0"
TELEM_PORT = 10000

RASPI_IP = "10.180.61.244"   # Kendi Pi IP adresinle değiştir
CMD_PORT = 10001


# =========================
# STILLER
# =========================
DARK_BG       = "#0a0e14"
PANEL_BG      = "#111820"
CARD_BG       = "#161d27"
BORDER_COLOR  = "#1e2a38"
ACCENT_CYAN   = "#00e5ff"
ACCENT_ORANGE = "#ff6d00"
ACCENT_GREEN  = "#00e676"
ACCENT_RED    = "#ff1744"
ACCENT_PURPLE = "#b388ff"
TEXT_PRIMARY   = "#e0e6ed"
TEXT_SECONDARY = "#7a8a9e"
TEXT_DIM       = "#3d4f63"

GLOBAL_STYLESHEET = f"""
    QWidget {{
        background-color: {DARK_BG};
        color: {TEXT_PRIMARY};
        font-family: 'Segoe UI', 'Inter', 'Roboto', sans-serif;
        font-size: 13px;
    }}
    QLabel {{
        background: transparent;
        border: none;
    }}
    QTextEdit {{
        background-color: {CARD_BG};
        color: {ACCENT_CYAN};
        border: 1px solid {BORDER_COLOR};
        border-radius: 6px;
        padding: 8px;
        font-family: 'Consolas', 'JetBrains Mono', 'Fira Code', monospace;
        font-size: 11px;
        selection-background-color: #1a3a5c;
    }}
    QPushButton {{
        border: 1px solid {BORDER_COLOR};
        border-radius: 6px;
        padding: 10px 16px;
        font-weight: bold;
        font-size: 13px;
        color: {TEXT_PRIMARY};
        background-color: {CARD_BG};
    }}
    QPushButton:hover {{
        border-color: {ACCENT_CYAN};
        background-color: #1a2636;
    }}
    QPushButton:pressed {{
        background-color: #0d1822;
    }}
"""


def make_card_frame():
    """Palantir-style card container."""
    frame = QFrame()
    frame.setStyleSheet(f"""
        QFrame {{
            background-color: {PANEL_BG};
            border: 1px solid {BORDER_COLOR};
            border-radius: 8px;
        }}
    """)
    return frame


def make_section_title(text):
    """Küçük, soluk section başlığı."""
    lbl = QLabel(text)
    lbl.setStyleSheet(f"""
        QLabel {{
            color: {TEXT_SECONDARY};
            font-size: 10px;
            font-weight: bold;
            letter-spacing: 2px;
            padding: 2px 4px;
            text-transform: uppercase;
        }}
    """)
    return lbl


def make_data_label(initial="-"):
    """Monospace data değeri label."""
    lbl = QLabel(initial)
    lbl.setStyleSheet(f"""
        QLabel {{
            color: {ACCENT_CYAN};
            font-family: 'Consolas', 'JetBrains Mono', 'Fira Code', monospace;
            font-size: 13px;
            font-weight: bold;
        }}
    """)
    return lbl


def make_key_label(text):
    """Soluk anahtar label."""
    lbl = QLabel(text)
    lbl.setStyleSheet(f"""
        QLabel {{
            color: {TEXT_SECONDARY};
            font-size: 12px;
        }}
    """)
    return lbl


def make_command_button(text, color, text_color="white", border_color=None):
    """Styled command button."""
    bc = border_color or color
    btn = QPushButton(text)
    btn.setStyleSheet(f"""
        QPushButton {{
            background-color: {color};
            color: {text_color};
            font-weight: bold;
            font-size: 13px;
            border: 1px solid {bc};
            border-radius: 6px;
            padding: 10px 8px;
        }}
        QPushButton:hover {{
            background-color: {bc};
            border-color: #ffffff33;
        }}
        QPushButton:pressed {{
            background-color: {color};
        }}
    """)
    btn.setCursor(Qt.PointingHandCursor)
    return btn


class GCSWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DRONE GCS — Precision Landing System")
        self.resize(1400, 850)
        self.setMinimumSize(1100, 650)

        self.running = True
        self.latest_frame = None

        self.latest_telem = {
            "state": "-",
            "flight_mode": "UNKNOWN",
            "marker_detected": False,
            "err_x": 0,
            "err_y": 0,
            "altitude": -1.0,
            "relative_alt": -1.0,
            "yaw_deg": 0.0,
            "manual_override": False,
            "auto_enabled": False,
            "is_armed": False,
            "offboard_requested": False,
            "fps": 0.0,
            "last_event": "waiting...",
            "precision_land_active": False,
            "precision_state": "IDLE",
            "fine_mode": False,
            "corner_count": 0,
        }

        self.log_history = deque(maxlen=300)
        self.cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.init_ui()
        self.start_receivers()

        self.ui_timer = QTimer()
        self.ui_timer.timeout.connect(self.refresh_ui)
        self.ui_timer.start(30)

    # =============================================
    # UI KURULUMU
    # =============================================
    def init_ui(self):
        self.setStyleSheet(GLOBAL_STYLESHEET)
        main_layout = QHBoxLayout(self)
        main_layout.setSpacing(12)
        main_layout.setContentsMargins(12, 12, 12, 12)

        # === SOL — VIDEO FEED ===
        left_layout = QVBoxLayout()
        left_layout.setSpacing(8)

        # Üst bar: başlık + durum
        top_bar = QHBoxLayout()
        title_lbl = QLabel("◆  DRONE GCS")
        title_lbl.setStyleSheet(f"""
            QLabel {{
                color: {ACCENT_CYAN};
                font-size: 16px;
                font-weight: bold;
                letter-spacing: 3px;
            }}
        """)
        self.lbl_conn_status = QLabel("● BAĞLANTI BEKLENİYOR")
        self.lbl_conn_status.setStyleSheet(f"""
            QLabel {{
                color: {ACCENT_ORANGE};
                font-size: 11px;
                font-weight: bold;
                letter-spacing: 1px;
            }}
        """)
        self.lbl_conn_status.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        top_bar.addWidget(title_lbl)
        top_bar.addStretch()
        top_bar.addWidget(self.lbl_conn_status)
        left_layout.addLayout(top_bar)

        # Video frame
        video_card = make_card_frame()
        video_inner = QVBoxLayout(video_card)
        video_inner.setContentsMargins(4, 4, 4, 4)

        self.video_label = QLabel("VIDEO BAĞLANTISI BEKLENİYOR...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(780, 560)
        self.video_label.setStyleSheet(f"""
            QLabel {{
                background-color: #080c12;
                color: {TEXT_DIM};
                font-size: 14px;
                font-weight: bold;
                letter-spacing: 2px;
                border: 1px solid {BORDER_COLOR};
                border-radius: 6px;
            }}
        """)
        video_inner.addWidget(self.video_label)

        # Video alt bar: FPS + State
        video_status_bar = QHBoxLayout()
        self.lbl_fps_bar = QLabel("FPS: --")
        self.lbl_fps_bar.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px; font-family: monospace;")
        self.lbl_state_bar = QLabel("STATE: IDLE")
        self.lbl_state_bar.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px; font-family: monospace;")
        self.lbl_mode_bar = QLabel("MODE: UNKNOWN")
        self.lbl_mode_bar.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px; font-family: monospace;")
        video_status_bar.addWidget(self.lbl_fps_bar)
        video_status_bar.addStretch()
        video_status_bar.addWidget(self.lbl_state_bar)
        video_status_bar.addStretch()
        video_status_bar.addWidget(self.lbl_mode_bar)
        video_inner.addLayout(video_status_bar)

        left_layout.addWidget(video_card)

        # === SAĞ — PANEL ===
        right_layout = QVBoxLayout()
        right_layout.setSpacing(10)

        # ---- DURUM KARTI ----
        status_card = make_card_frame()
        status_inner = QVBoxLayout(status_card)
        status_inner.setContentsMargins(14, 10, 14, 10)
        status_inner.setSpacing(4)

        status_inner.addWidget(make_section_title("SİSTEM DURUMU"))

        # ARM durumu — büyük gösterge
        self.lbl_armed = QLabel("DISARMED")
        self.lbl_armed.setAlignment(Qt.AlignCenter)
        self.lbl_armed.setStyleSheet(f"""
            QLabel {{
                color: {ACCENT_GREEN};
                font-size: 18px;
                font-weight: bold;
                letter-spacing: 3px;
                padding: 6px;
                border: 1px solid {BORDER_COLOR};
                border-radius: 6px;
                background-color: #0d1a12;
            }}
        """)
        status_inner.addWidget(self.lbl_armed)

        # Telemetri grid
        telem_grid = QGridLayout()
        telem_grid.setSpacing(3)
        telem_grid.setContentsMargins(0, 6, 0, 0)

        self.lbl_flight_mode = make_data_label("UNKNOWN")
        self.lbl_state = make_data_label("-")
        self.lbl_alt = make_data_label("--.-- m")
        self.lbl_rel_alt = make_data_label("--.-- m")
        self.lbl_yaw = make_data_label("--.- °")
        self.lbl_marker = make_data_label("--")
        self.lbl_errx = make_data_label("0")
        self.lbl_erry = make_data_label("0")

        telem_rows = [
            ("FLIGHT MODE", self.lbl_flight_mode),
            ("STATE",       self.lbl_state),
            ("ARUCO ALT",   self.lbl_alt),
            ("REL ALT",     self.lbl_rel_alt),
            ("YAW",         self.lbl_yaw),
            ("MARKER",      self.lbl_marker),
            ("ERR X",       self.lbl_errx),
            ("ERR Y",       self.lbl_erry),
        ]

        for i, (key, val_lbl) in enumerate(telem_rows):
            telem_grid.addWidget(make_key_label(key), i, 0)
            telem_grid.addWidget(val_lbl, i, 1)

        status_inner.addLayout(telem_grid)
        right_layout.addWidget(status_card)

        # ---- PRECISION LANDING KARTI ----
        prec_card = make_card_frame()
        prec_inner = QVBoxLayout(prec_card)
        prec_inner.setContentsMargins(14, 10, 14, 10)
        prec_inner.setSpacing(4)

        prec_inner.addWidget(make_section_title("PRECISION LANDING"))

        self.lbl_precision_state = make_data_label("OFF")
        self.lbl_fine_mode = make_data_label("--")
        self.lbl_corners = make_data_label("0")

        prec_grid = QGridLayout()
        prec_grid.setSpacing(3)
        prec_rows = [
            ("STATUS",    self.lbl_precision_state),
            ("FINE MODE", self.lbl_fine_mode),
            ("CORNERS",   self.lbl_corners),
        ]
        for i, (key, val_lbl) in enumerate(prec_rows):
            prec_grid.addWidget(make_key_label(key), i, 0)
            prec_grid.addWidget(val_lbl, i, 1)

        prec_inner.addLayout(prec_grid)
        right_layout.addWidget(prec_card)

        # ---- KOMUTLAR KARTI ----
        cmd_card = make_card_frame()
        cmd_inner = QVBoxLayout(cmd_card)
        cmd_inner.setContentsMargins(14, 10, 14, 10)
        cmd_inner.setSpacing(8)

        cmd_inner.addWidget(make_section_title("KOMUTLAR"))

        # Row 1: ARM / DISARM
        row1 = QHBoxLayout()
        self.btn_arm = make_command_button("ARM", "#0d3320", text_color=ACCENT_GREEN, border_color="#1b5e20")
        self.btn_disarm = make_command_button("DISARM", "#3d0a0a", text_color=ACCENT_RED, border_color="#7f1d1d")
        self.btn_arm.clicked.connect(lambda: self.send_command("ARM"))
        self.btn_disarm.clicked.connect(lambda: self.send_command("DISARM"))
        row1.addWidget(self.btn_arm)
        row1.addWidget(self.btn_disarm)
        cmd_inner.addLayout(row1)

        # Row 2: TAKEOFF / LAND
        row2 = QHBoxLayout()
        self.btn_takeoff = make_command_button("▲ TAKEOFF", "#0a1e3d", text_color="#42a5f5", border_color="#0d47a1")
        self.btn_land = make_command_button("▼ LAND", "#3d1a00", text_color="#ff9100", border_color="#e65100")
        self.btn_takeoff.clicked.connect(lambda: self.send_command("TAKEOFF"))
        self.btn_land.clicked.connect(lambda: self.send_command("LAND"))
        row2.addWidget(self.btn_takeoff)
        row2.addWidget(self.btn_land)
        cmd_inner.addLayout(row2)

        # Row 3: OFFBOARD / POSCTL
        row3 = QHBoxLayout()
        self.btn_offboard = make_command_button("OFFBOARD", "#1a0a33", text_color="#b388ff", border_color="#4a148c")
        self.btn_posctl = make_command_button("POSCTL", "#002620", text_color="#4db6ac", border_color="#004d40")
        self.btn_offboard.clicked.connect(lambda: self.send_command("OFFBOARD"))
        self.btn_posctl.clicked.connect(lambda: self.send_command("POSCTL"))
        row3.addWidget(self.btn_offboard)
        row3.addWidget(self.btn_posctl)
        cmd_inner.addLayout(row3)

        # PRECISION LAND — full width, accent
        self.btn_precision_land = QPushButton("◎  PRECISION LAND")
        self.btn_precision_land.setCursor(Qt.PointingHandCursor)
        self.btn_precision_land.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #1a0a33, stop:1 #2d1b69);
                color: {ACCENT_PURPLE};
                font-weight: bold;
                font-size: 14px;
                letter-spacing: 2px;
                border: 1px solid #7c4dff;
                border-radius: 6px;
                padding: 12px;
            }}
            QPushButton:hover {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #2d1b69, stop:1 #4527a0);
                border-color: {ACCENT_PURPLE};
            }}
            QPushButton:pressed {{
                background-color: #1a0a33;
            }}
        """)
        self.btn_precision_land.clicked.connect(lambda: self.send_command("PRECISION_LAND"))
        cmd_inner.addWidget(self.btn_precision_land)

        # RESET — subtle
        self.btn_reset = make_command_button("↺ RESET", "#1a1a22", text_color=TEXT_SECONDARY, border_color=BORDER_COLOR)
        self.btn_reset.clicked.connect(lambda: self.send_command("RESET"))
        cmd_inner.addWidget(self.btn_reset)

        # FORCE DISARM — danger
        self.btn_force_disarm = QPushButton("⚠  FORCE DISARM")
        self.btn_force_disarm.setCursor(Qt.PointingHandCursor)
        self.btn_force_disarm.setStyleSheet(f"""
            QPushButton {{
                background-color: #1a0505;
                color: {ACCENT_RED};
                font-weight: bold;
                font-size: 14px;
                letter-spacing: 2px;
                border: 2px solid #b71c1c;
                border-radius: 6px;
                padding: 12px;
            }}
            QPushButton:hover {{
                background-color: #330a0a;
                border-color: {ACCENT_RED};
            }}
            QPushButton:pressed {{
                background-color: #1a0505;
            }}
        """)
        self.btn_force_disarm.clicked.connect(lambda: self.send_command("FORCE_DISARM"))
        cmd_inner.addWidget(self.btn_force_disarm)

        right_layout.addWidget(cmd_card)

        # ---- EVENT LOG ----
        log_card = make_card_frame()
        log_inner = QVBoxLayout(log_card)
        log_inner.setContentsMargins(14, 10, 14, 10)
        log_inner.setSpacing(4)

        log_inner.addWidget(make_section_title("EVENT LOG"))

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(160)
        log_inner.addWidget(self.log_text)
        right_layout.addWidget(log_card)

        right_layout.addStretch()

        # Layout oranları
        main_layout.addLayout(left_layout, 3)
        main_layout.addLayout(right_layout, 1)

    # =============================================
    # LOG & KOMUT
    # =============================================
    def add_log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        if len(self.log_history) == 0 or self.log_history[-1] != line:
            self.log_history.append(line)

    def send_command(self, cmd: str):
        try:
            self.cmd_sock.sendto(cmd.encode("utf-8"), (RASPI_IP, CMD_PORT))
            self.add_log(f"CMD → {cmd}")
        except Exception as e:
            self.add_log(f"CMD ERROR: {e}")

    # =============================================
    # ALICILAR  (bağlantı mantığı değişmedi)
    # =============================================
    def start_receivers(self):
        threading.Thread(target=self.video_receiver, daemon=True).start()
        threading.Thread(target=self.telem_receiver, daemon=True).start()

    def video_receiver(self):
        while self.running:
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

            conn = None
            try:
                server.bind((VIDEO_HOST, VIDEO_PORT))
                server.listen(1)
                self.add_log(f"Video server listening on {VIDEO_HOST}:{VIDEO_PORT}")

                conn, addr = server.accept()
                self.add_log(f"Video connected: {addr}")

                data = b""
                payload_size = struct.calcsize("Q")

                while self.running:
                    while len(data) < payload_size:
                        packet = conn.recv(4096)
                        if not packet:
                            raise ConnectionError("Video connection closed")
                        data += packet

                    packed_msg_size = data[:payload_size]
                    data = data[payload_size:]
                    msg_size = struct.unpack("Q", packed_msg_size)[0]

                    while len(data) < msg_size:
                        packet = conn.recv(4096)
                        if not packet:
                            raise ConnectionError("Frame stream interrupted")
                        data += packet

                    frame_data = data[:msg_size]
                    data = data[msg_size:]

                    buffer = np.frombuffer(frame_data, dtype=np.uint8)
                    frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

                    if frame is not None:
                        self.latest_frame = frame

            except Exception as e:
                self.add_log(f"Video error: {e}")
                time.sleep(1.0)

            finally:
                try:
                    if conn:
                        conn.close()
                except:
                    pass
                try:
                    server.close()
                except:
                    pass

    def telem_receiver(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((TELEM_HOST, TELEM_PORT))
        sock.settimeout(1.0)

        while self.running:
            try:
                data, _ = sock.recvfrom(4096)
                msg = json.loads(data.decode("utf-8"))
                self.latest_telem = msg

                if msg.get("last_event"):
                    self.add_log(msg["last_event"])

            except socket.timeout:
                pass
            except Exception as e:
                self.add_log(f"Telemetry error: {e}")
                time.sleep(0.1)

        sock.close()

    # =============================================
    # UI GÜNCELLEME
    # =============================================
    def refresh_ui(self):
        t = self.latest_telem

        # State + Flight mode
        state = str(t.get("state", "-"))
        flight_mode = str(t.get("flight_mode", "UNKNOWN"))
        self.lbl_state.setText(state)
        self.lbl_flight_mode.setText(flight_mode)

        # Status bar altları
        fps_val = t.get("fps", 0.0)
        self.lbl_fps_bar.setText(f"FPS: {fps_val:.1f}")
        self.lbl_state_bar.setText(f"STATE: {state}")
        self.lbl_mode_bar.setText(f"MODE: {flight_mode}")

        # Bağlantı göstergesi
        if fps_val > 0:
            self.lbl_conn_status.setText("● BAĞLI")
            self.lbl_conn_status.setStyleSheet(f"color: {ACCENT_GREEN}; font-size: 11px; font-weight: bold; letter-spacing: 1px;")
        else:
            self.lbl_conn_status.setText("● BAĞLANTI BEKLENİYOR")
            self.lbl_conn_status.setStyleSheet(f"color: {ACCENT_ORANGE}; font-size: 11px; font-weight: bold; letter-spacing: 1px;")

        # ARM durumu
        armed = t.get("is_armed", False)
        if armed:
            self.lbl_armed.setText("■  ARMED")
            self.lbl_armed.setStyleSheet(f"""
                QLabel {{
                    color: {ACCENT_RED};
                    font-size: 18px;
                    font-weight: bold;
                    letter-spacing: 3px;
                    padding: 6px;
                    border: 1px solid #7f1d1d;
                    border-radius: 6px;
                    background-color: #1a0505;
                }}
            """)
        else:
            self.lbl_armed.setText("■  DISARMED")
            self.lbl_armed.setStyleSheet(f"""
                QLabel {{
                    color: {ACCENT_GREEN};
                    font-size: 18px;
                    font-weight: bold;
                    letter-spacing: 3px;
                    padding: 6px;
                    border: 1px solid {BORDER_COLOR};
                    border-radius: 6px;
                    background-color: #0d1a12;
                }}
            """)

        # Marker
        marker_det = t.get("marker_detected", False)
        if marker_det:
            self.lbl_marker.setText("● DETECTED")
            self.lbl_marker.setStyleSheet(f"color: {ACCENT_GREEN}; font-family: monospace; font-size: 13px; font-weight: bold;")
        else:
            self.lbl_marker.setText("○ NONE")
            self.lbl_marker.setStyleSheet(f"color: {TEXT_DIM}; font-family: monospace; font-size: 13px; font-weight: bold;")

        self.lbl_errx.setText(str(t.get("err_x", 0)))
        self.lbl_erry.setText(str(t.get("err_y", 0)))
        self.lbl_alt.setText(f"{t.get('altitude', -1.0):.2f} m")
        self.lbl_rel_alt.setText(f"{t.get('relative_alt', -1.0):.2f} m")
        self.lbl_yaw.setText(f"{t.get('yaw_deg', 0.0):.1f}°")

        # Precision landing
        prec_active = t.get("precision_land_active", False)
        prec_state = t.get("precision_state", "IDLE")
        if prec_active:
            self.lbl_precision_state.setText(f"● {prec_state}")
            self.lbl_precision_state.setStyleSheet(f"color: {ACCENT_PURPLE}; font-family: monospace; font-size: 13px; font-weight: bold;")
        else:
            self.lbl_precision_state.setText("○ OFF")
            self.lbl_precision_state.setStyleSheet(f"color: {TEXT_DIM}; font-family: monospace; font-size: 13px; font-weight: bold;")

        fine = t.get("fine_mode", False)
        if fine:
            self.lbl_fine_mode.setText("● FINE")
            self.lbl_fine_mode.setStyleSheet(f"color: {ACCENT_CYAN}; font-family: monospace; font-size: 13px; font-weight: bold;")
        else:
            self.lbl_fine_mode.setText("○ COARSE")
            self.lbl_fine_mode.setStyleSheet(f"color: {TEXT_SECONDARY}; font-family: monospace; font-size: 13px; font-weight: bold;")

        self.lbl_corners.setText(str(t.get("corner_count", 0)))

        # Event log
        self.log_text.setPlainText("\n".join(self.log_history))
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )

        # Video frame
        if self.latest_frame is not None:
            frame = cv2.cvtColor(self.latest_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w

            qt_img = QImage(
                frame.data, w, h, bytes_per_line, QImage.Format_RGB888
            )
            pix = QPixmap.fromImage(qt_img)

            self.video_label.setPixmap(
                pix.scaled(
                    self.video_label.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
            )

    def closeEvent(self, event):
        self.running = False
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = GCSWindow()
    win.show()
    sys.exit(app.exec_())
