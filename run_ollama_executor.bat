@echo off
echo C60.ai Ollama Executor
echo ======================
echo.

REM Check if help is requested
if "%1"=="--help" goto :help
if "%1"=="-h" goto :help

echo Starting Ollama Executor...
python ollama_executer.py %*
echo.
echo Execution completed. Check logs directory for details.
pause
goto :eof

:help
echo Usage: run_ollama_executor.bat [options]
echo.
echo Options:
echo   --model MODEL       Ollama model to use (default: codellama)
echo   --phase NUM         Process only a specific phase number
echo   --resume            Resume from the last completed phase
echo   --list-progress     List progress and exit
echo   --reset-progress    Reset progress tracking
echo.
echo Examples:
echo   run_ollama_executor.bat --model llama3
echo   run_ollama_executor.bat --phase 1
echo   run_ollama_executor.bat --resume
echo.
pause
