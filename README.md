# fraud-detection-gnn

        fraud-detection-gnn/
        ├── backend/                         # Backend service (FastAPI + GNN)
        │   ├── app/
        │   │   ├── __init__.py
        │   │   ├── config.py                # Paths and hyperparameters
        │   │   ├── main.py                  # FastAPI server
        │   │   ├── data_preprocess.py       # Preprocessing for 2008 & 2010
        │   │   ├── graph_builder.py         # Build PyG graph
        │   │   ├── model.py                 # GNN definition
        │   │   ├── train.py                 # Training loop
        │   │   └── utils.py                 # Helper functions
        │   ├── requirements.txt             # Backend dependencies
        │   └── starter.sh                   # Activate env & run server
        │
        ├── frontend/                        # React.js dashboard
        │   ├── package.json
        │   ├── vite.config.js               # Vite setup
        │   ├── public/
        │   │   └── index.html
        │   └── src/
        │       ├── App.jsx
        │       ├── main.jsx
        │       └── components/
        │           ├── Dashboard.jsx
        │           └── ClaimTable.jsx
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
        ├── notebooks/                       # EDA notebooks
        ├── reports/                         # Figures and logs
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
# .\venv\Scripts\activate.bat
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
# This script activates the environment, installs dependencies (if needed), and runs the server
sh starter.sh
# Or run uvicorn directly after activating the environment:
# uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
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

Start the frontend development server:
```bash
npm run dev
```
The frontend application will be accessible at `http://localhost:3000` (or another port if 3000 is busy). The dashboard will fetch data from the backend API.

---
*Make sure both the backend and frontend servers are running simultaneously to use the application.*