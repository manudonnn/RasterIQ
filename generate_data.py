import pandas as pd
import numpy as np
import random
import os

def generate_roster_processing_details(num_rows=200):
    np.random.seed(42)
    random.seed(42)
    
    orgs = [
        "Norton Hospitals", "Cedars-Sinai Medical Care Foundation", "MercyOne Medical Group",
        "Kaiser Permanente", "Mayo Clinic", "Cleveland Clinic", "Johns Hopkins Hospital",
        "Massachusetts General", "UCSF Medical Center", "Northwestern Memorial"
    ]
    states = ["KS", "CA", "NY"]
    lobs = ["Medicaid FFS", "Medicare HMO", "Commercial PPO/EPO", "Medicaid Managed Care"]
    stages = ["PRE_PROCESSING", "MAPPING_APPROVAL", "ISF_GEN", "DART_GEN", "DART_REVIEW", "DART_UI_VALIDATION", "SPS_LOAD", "RESOLVED", "STOPPED"]
    failure_reasons = ["Complete Validation Failure", "Missing NPI", "Invalid Taxonomy", "Address Mismatch", "Network Gap", "Duplicate Record", "N/A"]
    healths = ["Green", "Yellow", "Red"]

    data = {
        "ID": range(1, num_rows + 1),
        "RO_ID": [f"RO_{1000 + i}" for i in range(num_rows)],
        "RA_FILE_DETAILS_ID": [f"F_{2000 + i}" for i in range(num_rows)],
        "RA_ROSTER_DETAILS_ID": [f"R_{3000 + i}" for i in range(num_rows)],
        "RA_PLM_RO_PROF_DATA_ID": [f"P_{4000 + i}" for i in range(num_rows)],
        "SRC_SYS": [random.choice(["AvailityPDM", "Demographic", "ProviderGroup", "Salesforce"]) for _ in range(num_rows)],
        "ORG_NM": [random.choice(orgs) for _ in range(num_rows)],
        "CNT_STATE": [random.choice(states) for _ in range(num_rows)],
        "LOB": [random.choice(lobs) for _ in range(num_rows)],
        "RUN_NO": np.random.randint(1, 5, size=num_rows),
        "FILE_STATUS_CD": np.random.randint(1, 100, size=num_rows),
        "LATEST_STAGE_NM": [random.choice(stages) for _ in range(num_rows)],
        
        # Durations
        "PRE_PROCESSING_DURATION": np.random.randint(1, 25, size=num_rows),
        "MAPPING_APROVAL_DURATION": np.random.randint(1, 15, size=num_rows),
        "ISF_GEN_DURATION": np.random.randint(5, 30, size=num_rows),
        "DART_GEN_DURATION": np.random.randint(10, 80, size=num_rows),
        "DART_REVIEW_DURATION": np.random.randint(0, 40, size=num_rows),
        "DART_UI_VALIDATION_DURATION": np.random.randint(0, 20, size=num_rows),
        "SPS_LOAD_DURATION": np.random.randint(1, 15, size=num_rows),
        
        # Average Durations
        "AVG_PRE_PROCESSING_DURATION": np.full(num_rows, 10),
        "AVG_MAPPING_APROVAL_DURATION": np.full(num_rows, 5),
        "AVG_ISF_GEN_DURATION": np.full(num_rows, 15),
        "AVG_DART_GEN_DURATION": np.full(num_rows, 20),
        "AVG_DART_REVIEW_DURATION": np.full(num_rows, 25),
        "AVG_DART_UI_VALIDATION_DURATION": np.full(num_rows, 12),
        "AVG_SPS_LOAD_DURATION": np.full(num_rows, 10),
        
        # Health
        "PRE_PROCESSING_HEALTH": [random.choice(healths) for _ in range(num_rows)],
        "MAPPING_APROVAL_HEALTH": [random.choice(healths) for _ in range(num_rows)],
        "ISF_GEN_HEALTH": [random.choice(healths) for _ in range(num_rows)],
        "DART_GEN_HEALTH": [random.choice(healths) for _ in range(num_rows)],
        "DART_REVIEW_HEALTH": [random.choice(healths) for _ in range(num_rows)],
        "DART_UI_VALIDATION_HEALTH": [random.choice(healths) for _ in range(num_rows)],
        "SPS_LOAD_HEALTH": [random.choice(healths) for _ in range(num_rows)],
        
        "IS_STUCK": np.random.choice([0, 1], size=num_rows, p=[0.8, 0.2]),
        "IS_FAILED": np.random.choice([0, 1], size=num_rows, p=[0.7, 0.3]),
    }
    
    df = pd.DataFrame(data)
    
    # Logic fixes: 
    # If failed, must have failure status and stage STOPPED/RESOLVED
    df.loc[df['IS_FAILED'] == 1, 'FAILURE_STATUS'] = np.random.choice(failure_reasons[:-1], size=sum(df['IS_FAILED'] == 1))
    df.loc[df['IS_FAILED'] == 0, 'FAILURE_STATUS'] = None
    
    # Introduce anomalies for charts
    # 1. Norton Hospitals in KS has high failure rate
    norton_idx = (df['ORG_NM'] == 'Norton Hospitals') & (df['CNT_STATE'] == 'KS')
    df.loc[norton_idx, 'IS_FAILED'] = np.random.choice([0, 1], size=sum(norton_idx), p=[0.2, 0.8])
    df.loc[norton_idx & (df['IS_FAILED'] == 1), 'FAILURE_STATUS'] = 'Complete Validation Failure'
    df.loc[norton_idx & (df['IS_FAILED'] == 1), 'LATEST_STAGE_NM'] = 'STOPPED'
    
    # 2. DART_GEN has very long durations
    dart_gen_idx = df['LATEST_STAGE_NM'] == 'DART_GEN'
    df.loc[dart_gen_idx, 'DART_GEN_DURATION'] = np.random.randint(60, 120, size=sum(dart_gen_idx))
    
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/roster_processing_details.csv', index=False)
    print(f"Generated {num_rows} rows for data/roster_processing_details.csv")

def generate_aggregated_metrics():
    months = ["Jan-26", "Feb-26", "Mar-26", "Apr-26", "May-26", "Jun-26", "Jul-26", "Aug-26"]
    markets = [("KS", "CLIENT_A"), ("CA", "CLIENT_B"), ("NY", "CLIENT_C")]
    
    rows = []
    
    # Generate trends
    for month in months:
        for market, client in markets:
            # Base logic
            base_overall = random.randint(700, 1000)
            
            # Trend logic: KS declines, NY increases, CA stays flat
            if market == 'KS':
                fail_cnt = int(base_overall * random.uniform(0.15, 0.30)) 
            elif market == 'NY':
                fail_cnt = int(base_overall * random.uniform(0.02, 0.08))
            else:
                fail_cnt = int(base_overall * random.uniform(0.10, 0.20))
                
            scs_cnt = base_overall - fail_cnt
            scs_pct = round((scs_cnt / base_overall) * 100, 1)
            
            first_iter_scs = int(scs_cnt * random.uniform(0.6, 0.9))
            next_iter_scs = scs_cnt - first_iter_scs
            
            first_iter_fail = fail_cnt + random.randint(10, 50)
            next_iter_fail = fail_cnt
            
            rows.append({
                "MONTH": month,
                "MARKET": market,
                "CLIENT_ID": client,
                "FIRST_ITER_SCS_CNT": first_iter_scs,
                "FIRST_ITER_FAIL_CNT": first_iter_fail,
                "NEXT_ITER_SCS_CNT": next_iter_scs,
                "NEXT_ITER_FAIL_CNT": next_iter_fail,
                "OVERALL_SCS_CNT": scs_cnt,
                "OVERALL_FAIL_CNT": fail_cnt,
                "SCS_PERCENT": scs_pct,
                "IS_ACTIVE": 1
            })
            
    df = pd.DataFrame(rows)
    df.to_csv('data/aggregated_operational_metrics.csv', index=False)
    print(f"Generated {len(df)} rows for data/aggregated_operational_metrics.csv")

if __name__ == "__main__":
    print("Generating comprehensive synthetic data...")
    generate_roster_processing_details(1000)
    generate_aggregated_metrics()
