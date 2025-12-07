"""
Bildirim Sistemi Modülü
E-posta, SMS ve diğer bildirim kanalları ile kullanıcı bilgilendirme
"""

import smtplib
import requests
import json
import time
import threading
try:
    from email.mime.text import MimeText
    from email.mime.multipart import MimeMultipart
    from email.mime.base import MimeBase
    from email import encoders
    EMAIL_AVAILABLE = True
except ImportError:
    EMAIL_AVAILABLE = False
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum
from collections import deque
import sqlite3
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import pickle

@dataclass
class NotificationConfig:
    """Bildirim konfigürasyonu"""
    email_enabled: bool = True
    sms_enabled: bool = False
    telegram_enabled: bool = False
    discord_enabled: bool = False
    slack_enabled: bool = False
    
    # E-posta ayarları
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    email_username: str = ""
    email_password: str = ""
    from_email: str = ""
    to_emails: List[str] = None
    
    # SMS ayarları
    sms_provider: str = "twilio"  # twilio, nexmo, etc.
    sms_api_key: str = ""
    sms_api_secret: str = ""
    sms_from_number: str = ""
    sms_to_numbers: List[str] = None
    
    # Telegram ayarları
    telegram_bot_token: str = ""
    telegram_chat_ids: List[str] = None
    
    # Discord ayarları
    discord_webhook_url: str = ""
    
    # Slack ayarları
    slack_webhook_url: str = ""
    slack_channel: str = "#trading-alerts"

@dataclass
class NotificationMessage:
    """Bildirim mesajı"""
    id: str
    title: str
    message: str
    priority: str  # 'low', 'medium', 'high', 'critical'
    category: str  # 'trade', 'alert', 'error', 'info', 'warning'
    channels: List[str]  # ['email', 'sms', 'telegram', 'discord', 'slack']
    timestamp: datetime
    sent_channels: List[str] = None
    status: str = 'pending'  # 'pending', 'sent', 'failed', 'partial'

@dataclass
class NotificationRule:
    """Bildirim kuralı"""
    id: str
    name: str
    condition: str  # Python expression
    channels: List[str]
    priority: str
    enabled: bool = True
    cooldown_minutes: int = 0  # Minimum time between notifications
    last_sent: datetime = None

class NotificationChannel(Enum):
    """Bildirim kanalları"""
    EMAIL = "email"
    SMS = "sms"
    TELEGRAM = "telegram"
    DISCORD = "discord"
    SLACK = "slack"

class NotificationPriority(Enum):
    """Bildirim öncelikleri"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class NotificationSystem:
    """Bildirim sistemi sınıfı"""
    
    def __init__(self, config: NotificationConfig = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or NotificationConfig()
        
        # Bildirim geçmişi
        self.notification_history = deque(maxlen=1000)
        self.failed_notifications = deque(maxlen=100)
        
        # Bildirim kuralları
        self.notification_rules = {}
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Bildirim thread'i
        self.notification_thread = None
        self.is_running = False
        
        # Veritabanı
        self.db_path = "notifications.db"
        self._initialize_database()
        
        # Varsayılan kuralları yükle
        self._load_default_rules()
        
        self.logger.info("Bildirim sistemi başlatıldı")
    
    def _initialize_database(self):
        """Veritabanını başlat"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Bildirimler tablosu
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS notifications (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    message TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    category TEXT NOT NULL,
                    channels TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    sent_channels TEXT,
                    status TEXT NOT NULL
                )
            ''')
            
            # Bildirim kuralları tablosu
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS notification_rules (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    condition TEXT NOT NULL,
                    channels TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    enabled BOOLEAN NOT NULL,
                    cooldown_minutes INTEGER NOT NULL,
                    last_sent TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Veritabanı başlatma hatası: {e}")
    
    def _load_default_rules(self):
        """Varsayılan bildirim kurallarını yükle"""
        try:
            default_rules = [
                NotificationRule(
                    id="trade_executed",
                    name="İşlem Gerçekleşti",
                    condition="event_type == 'trade_executed'",
                    channels=["email", "telegram"],
                    priority="medium",
                    cooldown_minutes=5
                ),
                NotificationRule(
                    id="stop_loss_triggered",
                    name="Stop Loss Tetiklendi",
                    condition="event_type == 'stop_loss_triggered'",
                    channels=["email", "sms", "telegram"],
                    priority="high",
                    cooldown_minutes=0
                ),
                NotificationRule(
                    id="take_profit_triggered",
                    name="Take Profit Tetiklendi",
                    condition="event_type == 'take_profit_triggered'",
                    channels=["email", "telegram"],
                    priority="medium",
                    cooldown_minutes=0
                ),
                NotificationRule(
                    id="high_drawdown",
                    name="Yüksek Drawdown",
                    condition="drawdown > 0.05",  # %5'ten büyük drawdown
                    channels=["email", "sms", "telegram"],
                    priority="high",
                    cooldown_minutes=30
                ),
                NotificationRule(
                    id="low_balance",
                    name="Düşük Bakiye",
                    condition="balance < 1000",  # 1000'den düşük bakiye
                    channels=["email", "sms"],
                    priority="critical",
                    cooldown_minutes=60
                ),
                NotificationRule(
                    id="system_error",
                    name="Sistem Hatası",
                    condition="event_type == 'system_error'",
                    channels=["email", "telegram", "discord"],
                    priority="high",
                    cooldown_minutes=10
                ),
                NotificationRule(
                    id="strategy_change",
                    name="Strateji Değişti",
                    condition="event_type == 'strategy_change'",
                    channels=["email", "telegram"],
                    priority="medium",
                    cooldown_minutes=15
                ),
                NotificationRule(
                    id="market_alert",
                    name="Piyasa Uyarısı",
                    condition="event_type == 'market_alert'",
                    channels=["telegram"],
                    priority="medium",
                    cooldown_minutes=5
                )
            ]
            
            for rule in default_rules:
                self.notification_rules[rule.id] = rule
            
            self.logger.info(f"{len(default_rules)} varsayılan bildirim kuralı yüklendi")
            
        except Exception as e:
            self.logger.error(f"Varsayılan kurallar yükleme hatası: {e}")
    
    def start(self):
        """Bildirim sistemini başlat"""
        if self.is_running:
            return
        
        self.is_running = True
        self.notification_thread = threading.Thread(target=self._notification_loop, daemon=True)
        self.notification_thread.start()
        
        self.logger.info("Bildirim sistemi başlatıldı")
    
    def stop(self):
        """Bildirim sistemini durdur"""
        self.is_running = False
        if self.notification_thread:
            self.notification_thread.join(timeout=5)
        
        self.logger.info("Bildirim sistemi durduruldu")
    
    def _notification_loop(self):
        """Bildirim döngüsü"""
        while self.is_running:
            try:
                # Bekleyen bildirimleri işle
                self._process_pending_notifications()
                
                # Başarısız bildirimleri yeniden dene
                self._retry_failed_notifications()
                
                # 30 saniye bekle
                time.sleep(30)
                
            except Exception as e:
                self.logger.error(f"Bildirim döngüsü hatası: {e}")
                time.sleep(60)
    
    def send_notification(self, 
                         title: str, 
                         message: str, 
                         priority: str = "medium",
                         category: str = "info",
                         channels: List[str] = None,
                         custom_data: Dict[str, Any] = None) -> str:
        """Bildirim gönder"""
        try:
            # Bildirim ID'si oluştur
            notification_id = f"NOTIF_{int(time.time())}_{len(self.notification_history)}"
            
            # Varsayılan kanalları belirle
            if channels is None:
                channels = self._get_default_channels(priority)
            
            # Bildirim mesajı oluştur
            notification = NotificationMessage(
                id=notification_id,
                title=title,
                message=message,
                priority=priority,
                category=category,
                channels=channels,
                timestamp=datetime.now(),
                sent_channels=[],
                status='pending'
            )
            
            # Bildirimi kaydet
            with self.lock:
                self.notification_history.append(notification)
                self._save_notification_to_db(notification)
            
            # Bildirimi gönder
            self._send_notification_async(notification)
            
            self.logger.info(f"Bildirim gönderildi: {notification_id}")
            return notification_id
            
        except Exception as e:
            self.logger.error(f"Bildirim gönderme hatası: {e}")
            return None
    
    def _get_default_channels(self, priority: str) -> List[str]:
        """Önceliğe göre varsayılan kanalları belirle"""
        try:
            if priority == "critical":
                return ["email", "sms", "telegram"]
            elif priority == "high":
                return ["email", "telegram"]
            elif priority == "medium":
                return ["email", "telegram"]
            else:  # low
                return ["email"]
                
        except Exception as e:
            self.logger.error(f"Varsayılan kanallar belirleme hatası: {e}")
            return ["email"]
    
    def _send_notification_async(self, notification: NotificationMessage):
        """Bildirimi asenkron olarak gönder"""
        try:
            def send_worker():
                try:
                    sent_channels = []
                    failed_channels = []
                    
                    for channel in notification.channels:
                        try:
                            if channel == "email" and self.config.email_enabled:
                                if self._send_email(notification):
                                    sent_channels.append(channel)
                                else:
                                    failed_channels.append(channel)
                            
                            elif channel == "sms" and self.config.sms_enabled:
                                if self._send_sms(notification):
                                    sent_channels.append(channel)
                                else:
                                    failed_channels.append(channel)
                            
                            elif channel == "telegram" and self.config.telegram_enabled:
                                if self._send_telegram(notification):
                                    sent_channels.append(channel)
                                else:
                                    failed_channels.append(channel)
                            
                            elif channel == "discord" and self.config.discord_enabled:
                                if self._send_discord(notification):
                                    sent_channels.append(channel)
                                else:
                                    failed_channels.append(channel)
                            
                            elif channel == "slack" and self.config.slack_enabled:
                                if self._send_slack(notification):
                                    sent_channels.append(channel)
                                else:
                                    failed_channels.append(channel)
                            
                        except Exception as e:
                            self.logger.error(f"Bildirim gönderme hatası ({channel}): {e}")
                            failed_channels.append(channel)
                    
                    # Durumu güncelle
                    with self.lock:
                        notification.sent_channels = sent_channels
                        if len(sent_channels) == len(notification.channels):
                            notification.status = 'sent'
                        elif len(sent_channels) > 0:
                            notification.status = 'partial'
                        else:
                            notification.status = 'failed'
                            self.failed_notifications.append(notification)
                        
                        self._update_notification_in_db(notification)
                    
                except Exception as e:
                    self.logger.error(f"Bildirim worker hatası: {e}")
            
            # Thread başlat
            send_thread = threading.Thread(target=send_worker, daemon=True)
            send_thread.start()
            
        except Exception as e:
            self.logger.error(f"Asenkron bildirim gönderme hatası: {e}")
    
    def _send_email(self, notification: NotificationMessage) -> bool:
        """E-posta gönder"""
        try:
            if not EMAIL_AVAILABLE:
                self.logger.warning("Email modülü mevcut değil")
                return False
                
            if not self.config.email_enabled or not self.config.email_username:
                return False
            
            # E-posta içeriği oluştur
            msg = MimeMultipart()
            msg['From'] = self.config.from_email or self.config.email_username
            msg['To'] = ", ".join(self.config.to_emails or [])
            msg['Subject'] = f"[{notification.priority.upper()}] {notification.title}"
            
            # HTML içerik
            html_content = f"""
            <html>
            <body>
                <h2>{notification.title}</h2>
                <p><strong>Öncelik:</strong> {notification.priority.upper()}</p>
                <p><strong>Kategori:</strong> {notification.category}</p>
                <p><strong>Tarih:</strong> {notification.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
                <hr>
                <p>{notification.message}</p>
                <hr>
                <p><small>Bu bildirim BTCTURK Trading Bot tarafından gönderilmiştir.</small></p>
            </body>
            </html>
            """
            
            msg.attach(MimeText(html_content, 'html'))
            
            # SMTP bağlantısı
            server = smtplib.SMTP(self.config.smtp_server, self.config.smtp_port)
            server.starttls()
            server.login(self.config.email_username, self.config.email_password)
            
            # E-posta gönder
            text = msg.as_string()
            server.sendmail(self.config.from_email, self.config.to_emails, text)
            server.quit()
            
            self.logger.info(f"E-posta gönderildi: {notification.id}")
            return True
            
        except Exception as e:
            self.logger.error(f"E-posta gönderme hatası: {e}")
            return False
    
    def _send_sms(self, notification: NotificationMessage) -> bool:
        """SMS gönder"""
        try:
            if not self.config.sms_enabled or not self.config.sms_api_key:
                return False
            
            if self.config.sms_provider == "twilio":
                return self._send_twilio_sms(notification)
            elif self.config.sms_provider == "nexmo":
                return self._send_nexmo_sms(notification)
            else:
                self.logger.warning(f"Desteklenmeyen SMS provider: {self.config.sms_provider}")
                return False
                
        except Exception as e:
            self.logger.error(f"SMS gönderme hatası: {e}")
            return False
    
    def _send_twilio_sms(self, notification: NotificationMessage) -> bool:
        """Twilio ile SMS gönder"""
        try:
            from twilio.rest import Client
            
            client = Client(self.config.sms_api_key, self.config.sms_api_secret)
            
            message_body = f"{notification.title}\n{notification.message}"
            
            for phone_number in self.config.sms_to_numbers or []:
                message = client.messages.create(
                    body=message_body,
                    from_=self.config.sms_from_number,
                    to=phone_number
                )
                
                self.logger.info(f"Twilio SMS gönderildi: {message.sid}")
            
            return True
            
        except ImportError:
            self.logger.error("Twilio kütüphanesi bulunamadı")
            return False
        except Exception as e:
            self.logger.error(f"Twilio SMS gönderme hatası: {e}")
            return False
    
    def _send_nexmo_sms(self, notification: NotificationMessage) -> bool:
        """Nexmo ile SMS gönder"""
        try:
            import nexmo
            
            client = nexmo.Client(key=self.config.sms_api_key, secret=self.config.sms_api_secret)
            
            message_body = f"{notification.title}\n{notification.message}"
            
            for phone_number in self.config.sms_to_numbers or []:
                response = client.send_message({
                    'from': self.config.sms_from_number,
                    'to': phone_number,
                    'text': message_body
                })
                
                if response['messages'][0]['status'] == '0':
                    self.logger.info(f"Nexmo SMS gönderildi: {response['messages'][0]['message-id']}")
                else:
                    self.logger.error(f"Nexmo SMS hatası: {response['messages'][0]['error-text']}")
                    return False
            
            return True
            
        except ImportError:
            self.logger.error("Nexmo kütüphanesi bulunamadı")
            return False
        except Exception as e:
            self.logger.error(f"Nexmo SMS gönderme hatası: {e}")
            return False
    
    def _send_telegram(self, notification: NotificationMessage) -> bool:
        """Telegram gönder"""
        try:
            if not self.config.telegram_enabled or not self.config.telegram_bot_token:
                return False
            
            message_text = f"*{notification.title}*\n\n"
            message_text += f"📊 *Öncelik:* {notification.priority.upper()}\n"
            message_text += f"📁 *Kategori:* {notification.category}\n"
            message_text += f"🕒 *Tarih:* {notification.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            message_text += f"{notification.message}\n\n"
            message_text += "🤖 _BTCTURK Trading Bot_"
            
            for chat_id in self.config.telegram_chat_ids or []:
                url = f"https://api.telegram.org/bot{self.config.telegram_bot_token}/sendMessage"
                
                data = {
                    'chat_id': chat_id,
                    'text': message_text,
                    'parse_mode': 'Markdown'
                }
                
                response = requests.post(url, data=data, timeout=10)
                
                if response.status_code == 200:
                    self.logger.info(f"Telegram mesajı gönderildi: {chat_id}")
                else:
                    self.logger.error(f"Telegram mesaj hatası: {response.text}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Telegram gönderme hatası: {e}")
            return False
    
    def _send_discord(self, notification: NotificationMessage) -> bool:
        """Discord gönder"""
        try:
            if not self.config.discord_enabled or not self.config.discord_webhook_url:
                return False
            
            # Önceliğe göre renk
            color_map = {
                "low": 0x00ff00,      # Yeşil
                "medium": 0xffff00,   # Sarı
                "high": 0xff8000,     # Turuncu
                "critical": 0xff0000   # Kırmızı
            }
            
            embed = {
                "title": notification.title,
                "description": notification.message,
                "color": color_map.get(notification.priority, 0x00ff00),
                "fields": [
                    {
                        "name": "Öncelik",
                        "value": notification.priority.upper(),
                        "inline": True
                    },
                    {
                        "name": "Kategori",
                        "value": notification.category,
                        "inline": True
                    },
                    {
                        "name": "Tarih",
                        "value": notification.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                        "inline": True
                    }
                ],
                "footer": {
                    "text": "BTCTURK Trading Bot"
                },
                "timestamp": notification.timestamp.isoformat()
            }
            
            payload = {
                "embeds": [embed]
            }
            
            response = requests.post(self.config.discord_webhook_url, json=payload, timeout=10)
            
            if response.status_code == 204:
                self.logger.info(f"Discord mesajı gönderildi")
                return True
            else:
                self.logger.error(f"Discord mesaj hatası: {response.text}")
                return False
                
        except Exception as e:
            self.logger.error(f"Discord gönderme hatası: {e}")
            return False
    
    def _send_slack(self, notification: NotificationMessage) -> bool:
        """Slack gönder"""
        try:
            if not self.config.slack_enabled or not self.config.slack_webhook_url:
                return False
            
            # Önceliğe göre emoji
            emoji_map = {
                "low": "ℹ️",
                "medium": "⚠️",
                "high": "🚨",
                "critical": "🔥"
            }
            
            message_text = f"{emoji_map.get(notification.priority, 'ℹ️')} *{notification.title}*\n\n"
            message_text += f"*Öncelik:* {notification.priority.upper()}\n"
            message_text += f"*Kategori:* {notification.category}\n"
            message_text += f"*Tarih:* {notification.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            message_text += f"{notification.message}\n\n"
            message_text += "_BTCTURK Trading Bot_"
            
            payload = {
                "channel": self.config.slack_channel,
                "text": message_text,
                "username": "Trading Bot",
                "icon_emoji": ":robot_face:"
            }
            
            response = requests.post(self.config.slack_webhook_url, json=payload, timeout=10)
            
            if response.status_code == 200:
                self.logger.info(f"Slack mesajı gönderildi")
                return True
            else:
                self.logger.error(f"Slack mesaj hatası: {response.text}")
                return False
                
        except Exception as e:
            self.logger.error(f"Slack gönderme hatası: {e}")
            return False
    
    def check_notification_rules(self, event_data: Dict[str, Any]):
        """Bildirim kurallarını kontrol et"""
        try:
            for rule_id, rule in self.notification_rules.items():
                if not rule.enabled:
                    continue
                
                # Cooldown kontrolü
                if rule.last_sent:
                    time_since_last = datetime.now() - rule.last_sent
                    if time_since_last.total_seconds() < rule.cooldown_minutes * 60:
                        continue
                
                # Kural koşulunu değerlendir
                try:
                    # Güvenli değerlendirme için sınırlı namespace
                    safe_globals = {
                        '__builtins__': {},
                        'datetime': datetime,
                        'timedelta': timedelta
                    }
                    
                    # Event verilerini namespace'e ekle
                    safe_globals.update(event_data)
                    
                    # Koşulu güvenli şekilde değerlendir
                    try:
                        condition_result = self._safe_parse_condition(rule.condition, safe_globals)
                        if condition_result:
                            # Bildirim gönder
                            title = f"Kural Tetiklendi: {rule.name}"
                            message = f"Kural '{rule.name}' tetiklendi.\n\nEvent verileri:\n{json.dumps(event_data, indent=2, default=str)}"
                            
                            notification_id = self.send_notification(
                                title=title,
                                message=message,
                                priority=rule.priority,
                                category="rule_triggered",
                                channels=rule.channels
                            )
                            
                            if notification_id:
                                # Son gönderim zamanını güncelle
                                rule.last_sent = datetime.now()
                                self._save_rule_to_db(rule)
                                
                                self.logger.info(f"Kural tetiklendi: {rule.name}")
                    
                    except Exception as e:
                        self.logger.error(f"Kural değerlendirme hatası ({rule.name}): {e}")
                
                except Exception as e:
                    self.logger.error(f"Kural işleme hatası ({rule.name}): {e}")
            
        except Exception as e:
            self.logger.error(f"Bildirim kuralları kontrol hatası: {e}")
    
    def _process_pending_notifications(self):
        """Bekleyen bildirimleri işle"""
        try:
            with self.lock:
                pending_notifications = [
                    n for n in self.notification_history 
                    if n.status == 'pending'
                ]
            
            for notification in pending_notifications:
                self._send_notification_async(notification)
                
        except Exception as e:
            self.logger.error(f"Bekleyen bildirimler işleme hatası: {e}")
    
    def _retry_failed_notifications(self):
        """Başarısız bildirimleri yeniden dene"""
        try:
            with self.lock:
                failed_notifications = list(self.failed_notifications)
                self.failed_notifications.clear()
            
            for notification in failed_notifications:
                # Sadece 1 saat içindeki başarısız bildirimleri yeniden dene
                if datetime.now() - notification.timestamp < timedelta(hours=1):
                    notification.status = 'pending'
                    self._send_notification_async(notification)
                
        except Exception as e:
            self.logger.error(f"Başarısız bildirimler yeniden deneme hatası: {e}")
    
    def _save_notification_to_db(self, notification: NotificationMessage):
        """Bildirimi veritabanına kaydet"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO notifications 
                (id, title, message, priority, category, channels, timestamp, sent_channels, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                notification.id,
                notification.title,
                notification.message,
                notification.priority,
                notification.category,
                json.dumps(notification.channels),
                notification.timestamp.isoformat(),
                json.dumps(notification.sent_channels or []),
                notification.status
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Bildirim veritabanı kaydetme hatası: {e}")
    
    def _update_notification_in_db(self, notification: NotificationMessage):
        """Bildirimi veritabanında güncelle"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE notifications 
                SET sent_channels = ?, status = ?
                WHERE id = ?
            ''', (
                json.dumps(notification.sent_channels or []),
                notification.status,
                notification.id
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Bildirim veritabanı güncelleme hatası: {e}")
    
    def _save_rule_to_db(self, rule: NotificationRule):
        """Kuralı veritabanına kaydet"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO notification_rules 
                (id, name, condition, channels, priority, enabled, cooldown_minutes, last_sent)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                rule.id,
                rule.name,
                rule.condition,
                json.dumps(rule.channels),
                rule.priority,
                rule.enabled,
                rule.cooldown_minutes,
                rule.last_sent.isoformat() if rule.last_sent else None
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Kural veritabanı kaydetme hatası: {e}")
    
    def get_notification_history(self, limit: int = 100) -> List[NotificationMessage]:
        """Bildirim geçmişini al"""
        try:
            with self.lock:
                return list(self.notification_history)[-limit:]
        except Exception as e:
            self.logger.error(f"Bildirim geçmişi alma hatası: {e}")
            return []
    
    def get_notification_stats(self) -> Dict[str, Any]:
        """Bildirim istatistiklerini al"""
        try:
            with self.lock:
                total_notifications = len(self.notification_history)
                sent_notifications = len([n for n in self.notification_history if n.status == 'sent'])
                failed_notifications = len([n for n in self.notification_history if n.status == 'failed'])
                partial_notifications = len([n for n in self.notification_history if n.status == 'partial'])
                
                # Kanal bazında istatistikler
                channel_stats = {}
                for notification in self.notification_history:
                    for channel in notification.sent_channels or []:
                        channel_stats[channel] = channel_stats.get(channel, 0) + 1
                
                return {
                    'total_notifications': total_notifications,
                    'sent_notifications': sent_notifications,
                    'failed_notifications': failed_notifications,
                    'partial_notifications': partial_notifications,
                    'success_rate': (sent_notifications + partial_notifications) / total_notifications if total_notifications > 0 else 0,
                    'channel_stats': channel_stats,
                    'active_rules': len([r for r in self.notification_rules.values() if r.enabled])
                }
                
        except Exception as e:
            self.logger.error(f"Bildirim istatistikleri alma hatası: {e}")
            return {}
    
    def add_notification_rule(self, rule: NotificationRule):
        """Bildirim kuralı ekle"""
        try:
            with self.lock:
                self.notification_rules[rule.id] = rule
                self._save_rule_to_db(rule)
            
            self.logger.info(f"Bildirim kuralı eklendi: {rule.name}")
            
        except Exception as e:
            self.logger.error(f"Bildirim kuralı ekleme hatası: {e}")
    
    def remove_notification_rule(self, rule_id: str):
        """Bildirim kuralını kaldır"""
        try:
            with self.lock:
                if rule_id in self.notification_rules:
                    del self.notification_rules[rule_id]
                    
                    # Veritabanından da kaldır
                    conn = sqlite3.connect(self.db_path)
                    cursor = conn.cursor()
                    cursor.execute('DELETE FROM notification_rules WHERE id = ?', (rule_id,))
                    conn.commit()
                    conn.close()
                    
                    self.logger.info(f"Bildirim kuralı kaldırıldı: {rule_id}")
            
        except Exception as e:
            self.logger.error(f"Bildirim kuralı kaldırma hatası: {e}")
    
    def update_config(self, new_config: NotificationConfig):
        """Konfigürasyonu güncelle"""
        try:
            self.config = new_config
            self.logger.info("Bildirim sistemi konfigürasyonu güncellendi")
            
        except Exception as e:
            self.logger.error(f"Konfigürasyon güncelleme hatası: {e}")
    
    def test_notification(self, channel: str = "email") -> bool:
        """Test bildirimi gönder"""
        try:
            test_title = "Test Bildirimi"
            test_message = "Bu bir test bildirimidir. Sistem düzgün çalışıyor."
            
            notification_id = self.send_notification(
                title=test_title,
                message=test_message,
                priority="low",
                category="test",
                channels=[channel]
            )
            
            return notification_id is not None
            
        except Exception as e:
            self.logger.error(f"Test bildirimi hatası: {e}")
            return False
    
    def _safe_parse_condition(self, condition: str, safe_globals: Dict[str, Any]) -> bool:
        """Güvenli koşul değerlendirme (eval kullanmadan)"""
        try:
            # Basit operatör desteği
            operators = ['>', '<', '>=', '<=', '==', '!=', 'and', 'or', 'not']
            
            # Güvenli değişkenler
            allowed_vars = {
                'price', 'volume', 'change', 'timestamp', 'datetime', 'timedelta',
                'len', 'str', 'int', 'float', 'abs', 'min', 'max', 'sum'
            }
            
            # Koşulu tokenize et
            tokens = self._tokenize_condition(condition)
            
            # Basit parser ile değerlendir
            return self._evaluate_tokens(tokens, safe_globals, allowed_vars)
            
        except Exception as e:
            self.logger.error(f"Koşul değerlendirme hatası: {e}")
            return False
    
    def _tokenize_condition(self, condition: str) -> List[str]:
        """Koşulu tokenlara ayır"""
        import re
        # Basit tokenizer
        tokens = re.findall(r'\b\w+\b|[><=!]+|\d+\.?\d*|and|or|not', condition)
        return tokens
    
    def _evaluate_tokens(self, tokens: List[str], safe_globals: Dict[str, Any], allowed_vars: set) -> bool:
        """Tokenları güvenli şekilde değerlendir"""
        try:
            # Basit ifade değerlendirme
            if len(tokens) < 3:
                return False
            
            # İlk 3 token: var operator value
            if len(tokens) >= 3:
                var_name = tokens[0]
                operator = tokens[1]
                value_str = tokens[2]
                
                if var_name not in allowed_vars:
                    return False
                
                if var_name not in safe_globals:
                    return False
                
                var_value = safe_globals[var_name]
                
                try:
                    if '.' in value_str:
                        compare_value = float(value_str)
                    else:
                        compare_value = int(value_str)
                except ValueError:
                    return False
                
                # Operatör değerlendirme
                if operator == '>':
                    return var_value > compare_value
                elif operator == '<':
                    return var_value < compare_value
                elif operator == '>=':
                    return var_value >= compare_value
                elif operator == '<=':
                    return var_value <= compare_value
                elif operator == '==':
                    return var_value == compare_value
                elif operator == '!=':
                    return var_value != compare_value
                else:
                    return False
            
            return False
            
        except Exception as e:
            self.logger.error(f"Token değerlendirme hatası: {e}")
            return False

# Global bildirim sistemi
notification_system = NotificationSystem()
