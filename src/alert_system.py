# src/alert_system.py
# ═══════════════════════════════════════════════════════════
# ALERT AND NOTIFICATION SYSTEM
# Handles all alert mechanisms when threats are detected:
# 1. Audible alarm (plays sound through speakers)
# 2. Email notification (sends to security email)
# 3. Event logging (records all events)
#
# INPUT:  Alert event dicts from tailgate/piggyback detection
# OUTPUT: Sound plays + Email sent + Event logged
# ═══════════════════════════════════════════════════════════

import time
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from src.config import (
    ALARM_SOUND_PATH,
    ALARM_COOLDOWN,
    EMAIL_ENABLED,
    EMAIL_SENDER,
    EMAIL_PASSWORD,
    EMAIL_RECEIVER,
    SMTP_HOST,
    SMTP_PORT
)


class AlertSystem:
    """
    Handles all alert mechanisms for SecureGate.

    Usage:
        alerts = AlertSystem()
        alerts.trigger_tailgate_alert(event_details)
        alerts.trigger_piggyback_alert(event_details)
    """

    def __init__(self):
        """Initialize the alert system and load alarm sound."""

        # Track when last alarm was played
        # Prevents alarm from replaying every single frame
        self.last_alarm_time = 0

        # Complete log of all alerts this session
        self.alert_log = []

        # Try to initialize pygame for alarm sound
        self.sound_available = False
        self._init_sound()

        print(f"✅ AlertSystem initialized")
        print(f"   Sound available: {self.sound_available}")
        print(f"   Email enabled:   {EMAIL_ENABLED}")
        print(f"   Alarm cooldown:  {ALARM_COOLDOWN}s")

    def _init_sound(self):
        """
        Try to initialize pygame mixer for alarm sound.
        If it fails (no audio device), we gracefully skip sound.
        """
        try:
            import pygame
            # Initialize mixer with explicit format settings
            pygame.mixer.pre_init(
                frequency=44100,
                size=-16,        # 16-bit signed
                channels=1,      # mono
                buffer=512
            )
            pygame.mixer.init()
            self.alarm_sound = pygame.mixer.Sound(ALARM_SOUND_PATH)
            self.sound_available = True
            self.pygame = pygame
        except Exception as e:
            # Sound failed to initialize — common on servers/CI environments
            # We just skip sound and continue
            print(f"   ⚠️  Sound not available: {e}")
            self.sound_available = False

    def trigger_tailgate_alert(self, event_details: dict) -> dict:
        """
        Trigger all alerts for a tailgating event.

        Args:
            event_details: Dict containing tailgate event info:
                          - track_id: who tailgated
                          - behind_id: who they followed
                          - time_gap: seconds between crossings
                          - distance: pixels between them
                          - reason: explanation string

        Returns:
            Dict: The complete alert record that was logged
        """
        # Build the alert record
        alert = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "type": "TAILGATING",
            "tailgater_id": event_details.get("track_id"),
            "authorized_id": event_details.get("behind_id"),
            "time_gap_seconds": event_details.get("time_gap"),
            "distance_pixels": event_details.get("distance"),
            "reason": event_details.get("reason", ""),
            "confidence": "HIGH" if event_details.get("time_gap", 99) < 3.0
                         else "MEDIUM"
        }

        # Add to our log
        self.alert_log.append(alert)

        # Play alarm sound
        self._play_alarm()

        # Send email
        if EMAIL_ENABLED:
            self._send_tailgate_email(alert)

        return alert

    def trigger_piggyback_alert(self, event_details: dict) -> dict:
        """
        Trigger all alerts for a piggybacking event.

        Args:
            event_details: Dict from PiggybackDetector

        Returns:
            Dict: The complete alert record
        """
        alert = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "type": "PIGGYBACKING",
            "track_id": event_details.get("track_id"),
            "height_ratio": event_details.get("height_vs_average"),
            "bbox_height": event_details.get("bbox_height"),
            "reason": event_details.get("reason", ""),
            "confidence": "HIGH"
        }

        self.alert_log.append(alert)
        self._play_alarm()

        if EMAIL_ENABLED:
            self._send_piggyback_email(alert)

        return alert

    def _play_alarm(self):
        """
        Play the alarm sound if cooldown period has passed.

        Cooldown prevents alarm from playing every frame.
        Default: can play once every 10 seconds maximum.
        """
        current_time = time.time()
        time_since_last = current_time - self.last_alarm_time

        if time_since_last < ALARM_COOLDOWN:
            # Still in cooldown period — skip this alarm
            return

        # Update last alarm time BEFORE playing
        # (prevents race conditions)
        self.last_alarm_time = current_time

        if self.sound_available:
            try:
                self.alarm_sound.play()
                print("   🔊 ALARM PLAYING!")
            except Exception as e:
                print(f"   🔊 ALARM TRIGGERED (sound error: {e})")
        else:
            # No sound available — just print
            print("   🔊 ALARM TRIGGERED! (no audio device)")

    def _send_tailgate_email(self, alert: dict):
        """
        Send email notification for tailgating alert.

        Args:
            alert: Complete alert record dict
        """
        subject = "🚨 TAILGATING ALERT — SecureGate Security System"

        body = f"""
SECURITY ALERT — TAILGATING DETECTED
═══════════════════════════════════════

Time of Detection: {alert['timestamp']}
Alert Type: TAILGATING
Confidence: {alert['confidence']}

DETAILS:
  Unauthorized Person ID: {alert['tailgater_id']}
  Followed Behind Person ID: {alert['authorized_id']}
  Time Between Crossings: {alert['time_gap_seconds']} seconds
  Physical Distance: {alert['distance_pixels']} pixels

ANALYSIS:
  {alert['reason']}

ACTION REQUIRED:
  Please review the surveillance footage immediately.
  The tailgater entered the restricted area without authorization.

═══════════════════════════════════════
SecureGate AI Security System
Automated Alert — Do Not Reply
        """

        self._send_email(subject, body)

    def _send_piggyback_email(self, alert: dict):
        """
        Send email notification for piggybacking alert.

        Args:
            alert: Complete alert record dict
        """
        subject = "🚨 PIGGYBACKING ALERT — SecureGate Security System"

        body = f"""
SECURITY ALERT — PIGGYBACKING DETECTED
═══════════════════════════════════════

Time of Detection: {alert['timestamp']}
Alert Type: PIGGYBACKING
Confidence: {alert['confidence']}

DETAILS:
  Detected Person ID: {alert['track_id']}
  Bounding Box Height: {alert['bbox_height']} pixels
  Height Ratio vs Normal: {alert['height_ratio']}x

ANALYSIS:
  {alert['reason']}

ACTION REQUIRED:
  A person may be concealed on another person's back.
  Please verify at the access point immediately.

═══════════════════════════════════════
SecureGate AI Security System
Automated Alert — Do Not Reply
        """

        self._send_email(subject, body)

    def _send_email(self, subject: str, body: str):
        """
        Core email sending function using Gmail SMTP.

        Args:
            subject: Email subject line
            body: Email body text
        """
        # Validate credentials exist
        if not EMAIL_SENDER or not EMAIL_PASSWORD or not EMAIL_RECEIVER:
            print("   📧 Email skipped (credentials not configured in .env)")
            return

        try:
            # Build the email message
            message = MIMEMultipart("alternative")
            message["Subject"] = subject
            message["From"] = EMAIL_SENDER
            message["To"] = EMAIL_RECEIVER

            # Attach the body text
            message.attach(MIMEText(body, "plain"))

            # Create secure SSL connection to Gmail
            # ssl.create_default_context() creates secure connection settings
            context = ssl.create_default_context()

            # Connect to Gmail's SMTP server and send
            with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT,
                                   context=context) as server:
                server.login(EMAIL_SENDER, EMAIL_PASSWORD)
                server.sendmail(
                    EMAIL_SENDER,
                    EMAIL_RECEIVER,
                    message.as_string()
                )

            print(f"   📧 Email sent to {EMAIL_RECEIVER}")

        except smtplib.SMTPAuthenticationError:
            print("   📧 Email FAILED: Wrong credentials.")
            print("      Check EMAIL_SENDER and EMAIL_PASSWORD in .env")
            print("      Make sure you're using Gmail App Password, not regular password")
        except smtplib.SMTPException as e:
            print(f"   📧 Email FAILED: {e}")
        except Exception as e:
            print(f"   📧 Email FAILED (unexpected): {e}")

    def get_alert_log(self) -> list:
        """Get all alerts from this session."""
        return self.alert_log

    def get_alert_count(self) -> int:
        """Get total number of alerts."""
        return len(self.alert_log)

    def get_tailgate_count(self) -> int:
        """Get count of tailgating alerts only."""
        return sum(1 for a in self.alert_log if a["type"] == "TAILGATING")

    def get_piggyback_count(self) -> int:
        """Get count of piggybacking alerts only."""
        return sum(1 for a in self.alert_log if a["type"] == "PIGGYBACKING")

    def reset(self):
        """Reset for a new video session."""
        self.alert_log = []
        self.last_alarm_time = 0
        print("🔄 AlertSystem reset")