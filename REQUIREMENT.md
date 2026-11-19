
# RUN IT 

-cd D:\TamperVision-Lite-Full.\.venv\Scripts\Activate.ps1.\run_backend.bat

-cd D:\TamperVision-Lite-Full python -m http.server 5500 --directory frontend

# TamperVision-Lite Setup & Run Guide

## 1. Prerequisites

- Python 3.10 or 3.11 recommended
- Git installed
- A browser like Chrome or Edge
- Internet access for installing Python packages

## 2. Create virtual environment

From the project root:

Windows PowerShell:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Windows CMD:
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install dependencies

Install backend requirements:

```bash
pip install -r backend/requirements.txt
```

If you want to install manually, use these packages:

```bash
pip install fastapi>=0.115,<1 uvicorn[standard]>=0.30,<1 python-multipart>=0.0.9,<1 numpy>=1.26,<2.1 opencv-python-headless>=4.10,<5 matplotlib>=3.9,<4 tensorflow>=2.18,<2.21
```

## 4. Start backend

From the project root:

Windows:
```powershell
run_backend.bat
```

Or manually:
```powershell
cd backend
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

macOS / Linux:
```bash
cd backend
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

Backend URL:
```text
http://127.0.0.1:8000
```

Health check:
```text
http://127.0.0.1:8000/health
```

## 5. Start frontend

Open a new terminal and run:

From the project root:

```bash
python -m http.server 5500 --directory frontend
```

Then open in the browser:

```text
http://127.0.0.1:5500
```

Important:
- Do not open the HTML file directly with file:// in some browsers because the API request may fail.
- Use the local HTTP server as shown above.

## 6. Project structure

```text
TamperVision-Lite-Full/
├── backend/
│   ├── main.py
│   ├── model.py
│   ├── requirements.txt
│   └── model/
├── frontend/
│   ├── index.html
│   ├── script.js
│   └── style.css
├── run_backend.bat
├── run_backend.sh
├── README.md
├── README_FRONTEND.md
├── REQUIREMENT.md
└── data/
```

## 7. Common issues

### TensorFlow import errors
Use Python 3.10 or 3.11 and a clean virtual environment.

### Backend not starting
Check that the port 8000 is free and that all dependencies are installed.

### Failed to fetch in browser
Make sure:
- backend is running
- frontend is served over HTTP
- browser uses http://127.0.0.1:5500

## 8. Quick start summary

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt
run_backend.bat
python -m http.server 5500 --directory frontend
```

Then open:
```text
http://127.0.0.1:5500
```
