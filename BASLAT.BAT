@echo off
setlocal ENABLEDELAYEDEXPANSION

REM ==============================
REM BTCTURK Trading Bot Baslatma
REM ==============================

REM Proje klasorune gec (betik zaten proje kokunde calistirilmali)
REM .env dosyasi main.py icinde python-dotenv ile yuklenir

REM Sanal ortami etkinlestir (varsa)
if exist ".venv\Scripts\activate.bat" (
  call ".venv\Scripts\activate.bat"
) else if exist "venv\Scripts\activate.bat" (
  call "venv\Scripts\activate.bat"
)

REM Python kontrolu
python --version >nul 2>&1
if errorlevel 1 (
  echo [HATA] Python bulunamadi. Lutfen Python'u kurun ve PATH'e ekleyin.
  pause
  exit /b 1
)

REM UTF-8 cikti icin
set PYTHONUTF8=1

REM Uygulamayi baslat
python main.py
set EXITCODE=%ERRORLEVEL%

if not "%EXITCODE%"=="0" (
  echo [HATA] Uygulama hata kodu ile kapandi: %EXITCODE%
  pause
)

endlocal
