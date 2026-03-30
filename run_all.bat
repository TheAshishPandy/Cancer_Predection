@echo off
echo ========================================
echo CANCER PREDICTION PROJECT
echo ========================================
echo.
echo Select an option:
echo 1. Check data
echo 2. Train simple model (fast, on sample)
echo 3. Train on full 9GB data (may take time)
echo 4. Make predictions
echo 5. Exit
echo.
set /p choice="Enter choice (1-5): "

if "%choice%"=="1" python check_data.py
if "%choice%"=="2" python simple_model.py
if "%choice%"=="3" python large_data_model.py
if "%choice%"=="4" python predict.py
if "%choice%"=="5" exit

pause
