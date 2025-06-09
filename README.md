# fraud-detection-gnn

        fraud-detection-gnn/
        ├── backend/                         # Backend service (FastAPI + GNN)
        │   ├── app/
        │   │   ├── __init__.py
        │   │   ├── api.py
        │   │   ├── anomaly.py
        │   │   ├── config.py                # Paths and hyperparameters
        │   │   ├── data_preprocess.py       # Preprocessing for 2008 & 2010
        │   │   ├── graph_builder.py         # Build PyG graph
        │   │   ├── inference.py
        │   │   ├── main.py                  # FastAPI server
        │   │   ├── model.py                 # GNN definition
        │   │   ├── train.py                 # Training loop
        │   │   └── utils.py                 # Helper functions
        │   ├── requirements.txt             # Backend dependencies
        │   └── starter.sh                   # Activate env & run server
        │
        ├── frontend/                        # React.js
        │   ├── package.json
        │   ├── package-lock.json
        │   ├── .gitignore
        │   ├── eslint.config.js
        │   ├── vite.config.js               # Vite setup
        │   ├── tailwind.config.js           # tailwind setup
        │   ├── public/
        │   │   ├── image.png
        │   │   └── vite.svg
        │   └── src/
        │       ├── App.jsx
        │       ├── index.css
        │       ├── main.jsx
        │       └── components/
        │           ├── ClaimForm.jsx
        │           └── RiskResult.jsx
        │
        ├── data/
        │   ├── raw/
        │   │   ├── DE1_0_2008_Beneficiary_Summary_File_Sample_2.csv
        │   │   ├── DE1_0_2010_Beneficiary_Summary_File_Sample_2.csv
        │   │   └── DE1_0_2008_to_2010_Inpatient_Claims_Sample_2.csv
        │   └── processed/
        │       ├── beneficiary_2008.csv
        │       ├── beneficiary_2010.csv
        │       └── claims_processed.csv
        │
        ├── models/                          # Saved model checkpoints
        ├── .gitignore
        ├── LICENSE
        └── README.md                        # Project overview and run instructions

## Running the Full Stack

### 1. Backend (FastAPI + GNN)

Navigate to the backend directory:
```bash
cd backend
```

Create and activate a Python virtual environment:
```bash
# Create virtual environment (do this once)
python -m venv .venv

# Activate virtual environment
# Linux/macOS/Git Bash:
source .venv/bin/activate
# Windows Command Prompt:
# .\venv\Scripts\Activate
# Windows PowerShell:
# .\venv\Scripts\Activate.ps1
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Run the data preprocessing script (if needed, usually run once):
```bash
python app/data_preprocess.py
```

Run the graph building script (if needed, usually run once after preprocessing):
```bash
python app/graph_builder.py
```

Start the backend server using the starter script:
```bash
# This script creates and activates the environment, installs dependencies (if needed), and runs the server (This script assumes a Unix-like environment (Linux, macOS, Git Bash on Windows))
sh starter.sh
# Or run uvicorn directly after activating the environment:
# cd app
# uvicorn main:app --reload
```
The backend server will be running at `http://localhost:8000`.

### 2. Frontend (React + Vite)

Navigate to the frontend directory:
```bash
cd frontend
```

Install Node.js dependencies:
```bash
npm install
```

Install tailwindcss for vite:
```bash
npm install tailwindcss @tailwindcss/vite
```

Start the frontend development server:
```bash
npm run dev
```
The frontend application will be accessible at `http://localhost:5173`. The dashboard will fetch data from the backend API.

---
*Make sure both the backend and frontend servers are running simultaneously to use the application.*
