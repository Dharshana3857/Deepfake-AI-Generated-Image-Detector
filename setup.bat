@echo off
echo === Deepfake Detector: setup starting ===

python -m venv venv
call venv\Scripts\activate.bat

pip install --upgrade pip
pip install -r requirements.txt

echo === Generating offline demo dataset ===
python generate_sample_data.py --count 150

echo.
echo === Setup complete! ===
echo Virtual env created at .\venv (activate with venv\Scripts\activate.bat)
echo.
echo Next steps:
echo   1. Train:      python src\train.py --epochs 10
echo   2. Evaluate:   python src\evaluate.py
echo   3. Predict:    python src\predict.py --image data\raw\fake\fake_0000.jpg --gradcam
echo   4. Demo UI:    streamlit run app\app.py
echo.
echo To use a REAL dataset instead of the demo one, see download_dataset.py / README.md
