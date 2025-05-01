import pandas as pd
import os
from config import (
    BENEF_2008_CSV,
    BENEF_2010_CSV,
    CLAIMS_CSV,
    BENEF_2008_PROC,
    BENEF_2010_PROC,
    CLAIMS_PROC,
)

def preprocess_beneficiary(file_in: str, file_out: str):
    """
    Load a beneficiary summary CSV (2008 or 2010), select demographic,
    coverage, and chronic-condition columns, dedupe, and save.
    """
    os.makedirs(os.path.dirname(file_out), exist_ok=True)
    df = pd.read_csv(file_in, dtype=str)
    # columns available in both year files
    demographic_cols = [
        "DESYNPUF_ID",
        "BENE_BIRTH_DT",
        "BENE_DEATH_DT",
        "BENE_SEX_IDENT_CD",
        "BENE_RACE_CD",
        "BENE_ESRD_IND",
        "SP_STATE_CODE",
        "BENE_COUNTY_CD",
        "BENE_HI_CVRAGE_TOT_MONS",
        "BENE_SMI_CVRAGE_TOT_MONS",
        "BENE_HMO_CVRAGE_TOT_MONS",
        "PLAN_CVRG_MOS_NUM",
    ]
    # chronic condition flags
    chronic_flags = [
        "SP_ALZHDMTA",
        "SP_CHF",
        "SP_CHRNKIDN",
        "SP_CNCR",
        "SP_COPD",
        "SP_DEPRESSN",
        "SP_DIABETES",
        "SP_ISCHMCHT",
        "SP_OSTEOPRS",
        "SP_RA_OA",
        "SP_STRKETIA",
    ]
    # reimbursement and payment fields
    pay_fields = [
        "MEDREIMB_IP",
        "BENRES_IP",
        "PPPYMT_IP",
        "MEDREIMB_OP",
        "BENRES_OP",
        "PPPYMT_OP",
        "MEDREIMB_CAR",
        "BENRES_CAR",
        "PPPYMT_CAR",
    ]
    # select only columns present
    keep = [c for c in demographic_cols + chronic_flags + pay_fields if c in df.columns]
    df_out = df[keep].drop_duplicates(subset="DESYNPUF_ID")
    df_out.to_csv(file_out, index=False)
    print(f"Saved beneficiary summary to {file_out} (columns: {len(keep)})")


def preprocess_claims():
    """
    Load inpatient claims, parse dates, select key fields + diagnosis/procedure codes, save.
    """
    os.makedirs(os.path.dirname(CLAIMS_PROC), exist_ok=True)
    df = pd.read_csv(CLAIMS_CSV, dtype=str)
    # parse dates
    df["CLM_FROM_DT"] = pd.to_datetime(df["CLM_FROM_DT"], format="%Y%m%d")
    df["CLM_THRU_DT"] = pd.to_datetime(df["CLM_THRU_DT"], format="%Y%m%d")
    df["CLM_ADMSN_DT"] = pd.to_datetime(df["CLM_ADMSN_DT"], format="%Y%m%d")
    df["NCH_BENE_DSCHRG_DT"] = pd.to_datetime(df["NCH_BENE_DSCHRG_DT"], format="%Y%m%d")
    # core columns
    core = [
        "DESYNPUF_ID",
        "CLM_ID",
        "SEGMENT",
        "CLM_FROM_DT",
        "CLM_THRU_DT",
        "PRVDR_NUM",
        "AT_PHYSN_NPI",
        "OP_PHYSN_NPI",
        "OT_PHYSN_NPI",
        "CLM_ADMSN_DT",
        "ADMTNG_ICD9_DGNS_CD",
        "CLM_PASS_THRU_PER_DIEM_AMT",
        "CLM_PMT_AMT",
        "NCH_PRMRY_PYR_CLM_PD_AMT",
        "NCH_BENE_IP_DDCTBL_AMT",
        "NCH_BENE_PTA_COINSRNC_LBLTY_AM",
        "NCH_BENE_BLOOD_DDCTBL_LBLTY_AM",
        "CLM_UTLZTN_DAY_CNT",
        "NCH_BENE_DSCHRG_DT",
        "CLM_DRG_CD",
    ]
    # dynamic ICD9 and HCPCS codes
    icd9 = [c for c in df.columns if c.startswith("ICD9_DGNS_CD_")]
    prcd = [c for c in df.columns if c.startswith("ICD9_PRCDR_CD_")]
    hcpcs = [c for c in df.columns if c.startswith("HCPCS_CD_")]
    keep = [c for c in core + icd9 + prcd + hcpcs if c in df.columns]
    df_out = df[keep]
    df_out.to_csv(CLAIMS_PROC, index=False)
    print(f"Saved processed claims to {CLAIMS_PROC} (fields: {len(keep)})")


if __name__ == '__main__':
    preprocess_beneficiary(BENEF_2008_CSV, BENEF_2008_PROC)
    preprocess_beneficiary(BENEF_2010_CSV, BENEF_2010_PROC)
    preprocess_claims()