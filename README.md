# BTCTURK Trading Bot

Kapsamlı kripto para trading botu - BTCTURK API entegrasyonu ile gelişmiş trading stratejileri

## 🚀 Özellikler

### 📊 Trading Stratejileri
- **Scalping**: Hızlı kazanç stratejisi
- **Grid Trading**: Yatay piyasalarda etkili
- **Trend Following**: Trend takip stratejisi
- **Hedge/Correlation**: Risk azaltma stratejileri

### 🤖 AI Destekli Analiz
- **DeepSeek API Entegrasyonu**: Gelişmiş AI analiz
- **Piyasa Rejimi Tanıma**: LSTM/Transformer ile otomatik sınıflandırma
- **Sentiment Analizi**: Haber ve sosyal medya analizi
- **Dinamik Strateji Seçimi**: AI destekli strateji değişimi

### 📈 Teknik Analiz
- **23+ Teknik Gösterge**: RSI, MACD, Bollinger Bands, ATR, vb.
- **Order Book Analizi**: Derinlik analizi
- **Volatilite Analizi**: ATR ve diğer volatilite göstergeleri
- **Volume Analizi**: OBV, Volume Profile

### 🛡️ Risk Yönetimi
- **Pozisyon Büyüklüğü**: Kelly Kriteri ve diğer yöntemler
- **Stop-Loss/Take-Profit**: Otomatik risk kontrolü
- **Trailing Stop**: Dinamik stop seviyeleri
- **Drawdown Kontrolü**: Maksimum düşüş sınırları

### 📊 Backtest & Analiz
- **Monte Carlo Simülasyonu**: Risk analizi
- **Performans Metrikleri**: Sharpe, Sortino, Max Drawdown
- **Excel/PDF Raporları**: Detaylı performans raporları
- **Forward Testing**: Gerçek zamanlı test

### 🔌 Modüler Yapı
- **Eklenti Sistemi**: Özel stratejiler ve göstergeler
- **Çoklu API Desteği**: BTCTURK API key yönetimi
- **Veritabanı Entegrasyonu**: SQLite/PostgreSQL
- **WebSocket Desteği**: Gerçek zamanlı veri

## 🛠️ Kurulum

### Gereksinimler
- Python 3.8+
- BTCTURK API anahtarları
- DeepSeek API anahtarı (opsiyonel)

### Adım 1: Repository'yi klonlayın
```bash
git clone https://github.com/yourusername/btcturk-trading-bot.git
cd btcturk-trading-bot
```

### Adım 2: Sanal ortam oluşturun
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# veya
venv\Scripts\activate  # Windows
```

### Adım 3: Gerekli paketleri yükleyin
```bash
pip install -r requirements.txt
```

### Adım 4: Konfigürasyonu ayarlayın
```bash
# .env dosyası oluşturun
cp .env.example .env

# API anahtarlarınızı .env dosyasına ekleyin
BTCTURK_API_KEY=your_btcturk_api_key
BTCTURK_SECRET_KEY=your_btcturk_secret_key
DEEPSEEK_API_KEY=your_deepseek_api_key
```

## 🚀 Kullanım

### Hızlı Başlangıç
```bash
python main.py
```

### GUI Üzerinden Kurulum
1. **API Yönetimi**: Ayarlar > API Yönetimi menüsünden API anahtarlarınızı ekleyin
2. **Strateji Seçimi**: İstediğiniz trading stratejisini seçin
3. **Risk Ayarları**: Pozisyon büyüklüğü ve risk limitlerini ayarlayın
4. **Bot'u Başlatın**: Ana ekrandan "Başlat" butonuna tıklayın

### Komut Satırından Kullanım
```bash
# Backtest çalıştır
python -m backtest.run_backtest --strategy scalping --start-date 2024-01-01 --end-date 2024-12-31

# Strateji optimizasyonu
python -m optimization.optimize_strategy --strategy grid_trading --method genetic

# Rapor oluştur
python -m data.generate_report --format excel --output reports/
```

## 📁 Proje Yapısı

```
btcturk-trading-bot/
├── api/                    # API entegrasyonları
│   ├── btcturk_api.py     # BTCTURK REST API
│   ├── websocket_client.py # WebSocket bağlantısı
│   └── multi_api_manager.py # Çoklu API yönetimi
├── strategies/             # Trading stratejileri
│   ├── scalping.py        # Scalping stratejisi
│   ├── grid_trading.py    # Grid trading
│   ├── trend_following.py # Trend takip
│   └── hedge_trading.py   # Hedge stratejileri
├── indicators/             # Teknik göstergeler
│   ├── technical_indicators.py
│   └── custom_indicators.py
├── ai/                     # AI modülleri
│   ├── deepseek_api.py    # DeepSeek entegrasyonu
│   ├── market_analyzer.py # Piyasa analizi
│   └── signal_generator.py # AI sinyal üretimi
├── risk_management/        # Risk yönetimi
│   ├── risk_manager.py    # Ana risk yöneticisi
│   ├── position_sizer.py  # Pozisyon büyüklüğü
│   └── portfolio_optimizer.py # Portföy optimizasyonu
├── backtest/              # Backtest modülü
│   ├── backtest_engine.py # Backtest motoru
│   ├── monte_carlo.py     # Monte Carlo simülasyonu
│   └── performance_analyzer.py # Performans analizi
├── data/                  # Veri yönetimi
│   ├── data_manager.py    # Veri yöneticisi
│   └── report_generator.py # Rapor üretici
├── gui/                   # Kullanıcı arayüzü
│   ├── trading_dashboard.py # Ana dashboard
│   ├── api_management_gui.py # API yönetimi
│   └── settings_gui.py    # Ayarlar
├── plugins/               # Eklenti sistemi
│   ├── plugin_manager.py  # Eklenti yöneticisi
│   └── example_strategy_plugin.py # Örnek eklenti
├── utils/                 # Yardımcı araçlar
│   ├── logger.py          # Logging sistemi
│   └── error_handler.py   # Hata yönetimi
├── config/                # Konfigürasyon
│   ├── config.py          # Ana konfigürasyon
│   └── settings.json      # Kullanıcı ayarları
├── logs/                  # Log dosyaları
├── data/                  # Veri dosyaları
├── reports/               # Raporlar
└── main.py               # Ana uygulama
```

## ⚙️ Konfigürasyon

### API Anahtarları
```python
# .env dosyası
BTCTURK_API_KEY=your_btcturk_api_key
BTCTURK_SECRET_KEY=your_btcturk_secret_key
DEEPSEEK_API_KEY=your_deepseek_api_key
```

### Strateji Parametreleri
```python
# config/settings.json
{
    "strategy_settings": {
        "scalping": {
            "profit_target": 0.005,
            "stop_loss": 0.002,
            "position_size": 0.1
        },
        "grid_trading": {
            "grid_size": 0.01,
            "grid_count": 10,
            "price_range": 0.05
        }
    }
}
```

### Risk Yönetimi
```python
{
    "risk_management": {
        "max_daily_loss_percent": 0.05,
        "position_size_percent": 0.01,
        "max_open_positions": 5,
        "stop_loss_percent": 0.02
    }
}
```

## 📊 Performans Metrikleri

- **Sharpe Ratio**: Risk ayarlı getiri
- **Sortino Ratio**: Aşağı yönlü risk ayarlı getiri
- **Maximum Drawdown**: Maksimum düşüş
- **Profit Factor**: Kazanç/kayıp oranı
- **Win Rate**: Kazanma oranı
- **VaR/CVaR**: Risk değeri metrikleri

## 🔧 Geliştirme

### Yeni Strateji Ekleme
```python
# strategies/my_strategy.py
from strategies.base_strategy import BaseStrategy

class MyStrategy(BaseStrategy):
    def generate_signals(self, market_data):
        # Strateji mantığınızı buraya yazın
        pass
```

### Yeni Eklenti Oluşturma
```python
# plugins/my_plugin.py
from plugins.plugin_manager import BasePlugin

class MyPlugin(BasePlugin):
    def _on_initialize(self):
        # Eklenti başlatma kodları
        pass
```

### Test Çalıştırma
```bash
# Tüm testleri çalıştır
pytest

# Belirli modülü test et
pytest tests/test_strategies.py

# Coverage raporu
pytest --cov=src tests/
```

## 📈 Raporlar

### Excel Raporu
```python
from data.report_generator import report_generator

# Excel raporu oluştur
report_generator.generate_excel_report(
    backtest_results=results,
    performance_analysis=analysis,
    filename="trading_report.xlsx"
)
```

### PDF Raporu
```python
# PDF raporu oluştur
report_generator.generate_pdf_report(
    backtest_results=results,
    performance_analysis=analysis,
    filename="trading_report.pdf"
)
```

## 🚨 Uyarılar

⚠️ **ÖNEMLİ UYARILAR**:
- Bu yazılım eğitim amaçlıdır
- Gerçek para ile trading yapmadan önce kapsamlı testler yapın
- Tüm riskleri değerlendirin
- API anahtarlarınızı güvenli tutun
- Küçük miktarlarla başlayın

## 🤝 Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Commit yapın (`git commit -m 'Add amazing feature'`)
4. Push yapın (`git push origin feature/amazing-feature`)
5. Pull Request oluşturun

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için `LICENSE` dosyasına bakın.

## 📞 Destek

- **GitHub Issues**: Hata bildirimi ve özellik istekleri
- **Discord**: Topluluk desteği
- **Email**: support@btcturk-bot.com

## 🙏 Teşekkürler

- BTCTURK API ekibine
- DeepSeek AI ekibine
- Açık kaynak topluluğuna
- Tüm katkıda bulunanlara

---

**Not**: Bu proje sürekli geliştirilmektedir. Güncellemeler için GitHub'ı takip edin.