@echo off
title LLoCO + llama.cpp Server

REM folder with llama-server.exe
set LLAMA_CPP_DIR=%~dp0llama_cpp

REM folder with the wanted model
set MODEL_DIR=%~dp0models

REM file name of the wanted model
set MODEL_NAME=Qwen2.5-7B-Instruct-Q8_0.gguf

REM server port
set PORT=8080

echo ------------------------------------------
echo start of llama-server.exe
echo Model : %MODEL_DIR%\%MODEL_NAME%
echo Port  : %PORT%
echo ------------------------------------------

REM =======================
REM launch server
REM =======================

start "llama-server" cmd /c %LLAMA_CPP_DIR%\llama-server.exe -m %MODEL_DIR%\%MODEL_NAME% --port %PORT% --ctx-size 32768 --chat-template %~dp0chat_template_qwen.txt

echo wait for the server to start...
timeout /t 50 >nul

REM =======================
REM run LLoCO
REM =======================

echo ------------------------------------------
echo run LLoCO with the local server
echo ------------------------------------------

set LLAMA_SERVER_URL=http://localhost:%PORT%

python main.py -t 7200 batch --dataset IndustryOR --id 12 --problem-timeout 7200

REM =======================
REM close server propely
REM =======================

echo.
echo stop server llama.cpp...
@REM taskkill /FI "WINDOWTITLE eq llama-server*" /F >nul 2>&1

echo ------------------------------------------
echo end
echo ------------------------------------------

pause