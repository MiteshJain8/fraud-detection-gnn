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