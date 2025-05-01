import pandas as pd
from config import BENEF_2008_CSV, BENEF_2010_CSV, CLAIMS_CSV, BENEF_2008_PROC, BENEF_2010_PROC, CLAIMS_PROC
import os

def preprocess_beneficiary(file_in, file_out):
    # Ensure the processed directory exists
    os.makedirs(os.path.dirname(file_out), exist_ok=True)
    df = pd.read_csv(file_in)
    cols = ['DESYNPUF_ID', 'BENE_SEX_IDENT_CD','BENE_RACE_CD','BENE_AGE_CAT_CD','BENE_STATE_CD']
    chronic = [c for c in df.columns if c.startswith('SP_')]
    df = df[cols + chronic].drop_duplicates('DESYNPUF_ID')
    df.to_csv(file_out, index=False)
    print(f"Saved {file_out}")


def preprocess_claims():
    # Ensure the processed directory exists
    os.makedirs(os.path.dirname(CLAIMS_PROC), exist_ok=True)
    df = pd.read_csv(CLAIMS_CSV)
    df['CLM_FROM_DT'] = pd.to_datetime(df['CLM_FROM_DT'], format='%Y%m%d')
    df['CLM_THRU_DT'] = pd.to_datetime(df['CLM_THRU_DT'], format='%Y%m%d')
    cols = ['CLM_ID','DESYNPUF_ID','PRVDR_NUM','AT_PHYSN_NPI','CLM_FROM_DT','CLM_THRU_DT','CLM_PMT_AMT']
    codes = [c for c in df.columns if c.startswith('ICD9_DGNS_CD') or c.startswith('HCPCS_CD')]
    df = df[cols + codes]
    df.to_csv(CLAIMS_PROC, index=False)
    print(f"Saved {CLAIMS_PROC}")


if __name__ == '__main__':
    preprocess_beneficiary(BENEF_2008_CSV, BENEF_2008_PROC)
    preprocess_beneficiary(BENEF_2010_CSV, BENEF_2010_PROC)
    preprocess_claims()