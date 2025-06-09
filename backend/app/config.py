import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DIR = os.path.join(BASE_DIR, '..', 'data', 'raw')
PROCESSED_DIR = os.path.join(BASE_DIR, '..', 'data', 'processed')
MODEL_DIR = os.path.join(BASE_DIR, '..', 'models')

BENEF_2008_CSV = os.path.join(RAW_DIR, 'DE1_0_2008_Beneficiary_Summary_File_Sample_2.csv')
BENEF_2010_CSV = os.path.join(RAW_DIR, 'DE1_0_2010_Beneficiary_Summary_File_Sample_2.csv')
CLAIMS_CSV    = os.path.join(RAW_DIR, 'DE1_0_2008_to_2010_Inpatient_Claims_Sample_2.csv')

BENEF_2008_PROC = os.path.join(PROCESSED_DIR, 'beneficiary_2008.csv')
BENEF_2010_PROC = os.path.join(PROCESSED_DIR, 'beneficiary_2010.csv')
CLAIMS_PROC     = os.path.join(PROCESSED_DIR, 'claims_processed.csv')
GRAPH_PT        = os.path.join(PROCESSED_DIR, 'claim_graph_data.pt')
NODE_MAPPING_PATH   = os.path.join(PROCESSED_DIR, 'node_mapping.json')
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, 'gae_model.pt')
PLOT_PATH = os.path.join(MODEL_DIR, 'training_curve.png')

EMBEDDING_DIM = 64  # Dimension of the node embeddings
EPOCHS = 200
LEARNING_RATE = 0.01
PATIENCE = 10  # Early stopping patience
