import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
from pydantic import BaseModel
from sklearn.linear_model import LinearRegression

app = FastAPI()

# ── 1. MIDDLEWARE SETUP ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permits any frontend port or host signature
    allow_credentials=True,
    allow_methods=["*"],  # Allows standard GET, POST, and OPTIONS pre-flights
    allow_headers=["*"],
)

# ── 2. DATA INGESTION ENGINE ──
DB_PATH = "./data/your_alpha_data.csv"
try:
    df_full = pd.read_csv(DB_PATH)
    # Split dataset 75/25 for In-Sample (is) vs Out-of-Sample (oos) environments
    train_idx = int(len(df_full) * 0.75)
    datasets = {
        "is": df_full.iloc[:train_idx].copy(),
        "oos": df_full.iloc[train_idx:].copy()
    }
    print(f"Successfully initialized database. IS Rows: {len(datasets['is'])}, OOS Rows: {len(datasets['oos'])}")
except Exception as e:
    print(f"Warning: Could not load {DB_PATH} ({e}). Initializing empty dataset structures.")
    datasets = {"is": pd.DataFrame(), "oos": pd.DataFrame()}

# ── 3. DATA TRANSFER SCHEMAS ──
class UpstreamFilter(BaseModel):
    alphaId: str
    thresholds: List[float]
    selectedBuckets: List[int]

class BucketRequest(BaseModel):
    dataset: str             # "is" or "oos"
    alphaId: str             # Active metric being bucketed
    thresholds: List[float]  # Custom boundary thresholds set by user sliders
    upstreamFilters: List[UpstreamFilter]  # Parent panel segmentations
class RegressionRequest(BaseModel):
    dataset: str         # "is" or "oos"
    features: List[str]  # e.g., ["obi_pressure", "trade_flow_imb"]
    target: str          # "r60", "r300", or "r1800"

# ── 4. LIVE API ENDPOINTS (MUST BE DECLARED FIRST) ──

@app.get("/api/data/meta")
def get_meta():
    """Returns absolute row counts to initial workspace view indicators."""
    return {
        "isRows": len(datasets["is"]),
        "oosRows": len(datasets["oos"])
    }

@app.post("/api/buckets")
def calculate_buckets(req: BucketRequest):
    """
    Executes deep statistical data-bucketing entirely on the server.
    Applies sequential cascade filtering across parent nodes before dividing cuts.
    """
    # Fetch target split
    df = datasets.get(req.dataset, pd.DataFrame()).copy()
    total_rows = len(df)
    
    if df.empty:
        return {"buckets": [], "filteredRows": 0, "totalRows": 0}

    # Apply parent/upstream filters sequentially using digital array indices
    for f in req.upstreamFilters:
        if f.alphaId in df.columns and len(f.thresholds) > 0:
            sorted_thresh = sorted(f.thresholds)
            b_indices = np.searchsorted(sorted_thresh, df[f.alphaId])
            df = df[np.isin(b_indices, f.selectedBuckets)]

    filtered_rows = len(df)
    if filtered_rows == 0:
        return {"buckets": [], "filteredRows": 0, "totalRows": total_rows}

    # Bucket active alpha dimension using current workspace edge cut arrays
    sorted_cuts = sorted(req.thresholds)
    df['bucket_idx'] = np.searchsorted(sorted_cuts, df[req.alphaId])
    
      # Process return averages bucket by bucket inside memory matrix paths
    bucket_stats = []
    num_expected_buckets = len(sorted_cuts) + 1
    
    for b_id in range(num_expected_buckets):
        b_df = df[df['bucket_idx'] == b_id]
        count = len(b_df)
        
        # Build bound string tags dynamically
        if b_id == 0:
            lbl = f"[-inf, {sorted_cuts[0]:.2f}]" if sorted_cuts else "All"
        elif b_id == len(sorted_cuts):
            lbl = f"[{sorted_cuts[-1]:.2f}, +inf]"
        else:
            lbl = f"[{sorted_cuts[b_id-1]:.2f}, {sorted_cuts[b_id]:.2f}]"

        # ── EXACT FRONTEND KEY PAIRS CONFIGURATION ──
        bucket_stats.append({
            "bucketIndex": b_id,
            "label": lbl,
            "n": count,  # Maps directly to b.n
            "r60": float(b_df['r60'].mean()) if count > 0 and not np.isnan(b_df['r60'].mean()) else 0.0,  # Maps to b.r60
            "r300": float(b_df['r300'].mean()) if count > 0 and not np.isnan(b_df['r300'].mean()) else 0.0,
            "r1800": float(b_df['r1800'].mean()) if count > 0 and not np.isnan(b_df['r1800'].mean()) else 0.0,
        })

    return {
        "buckets": bucket_stats,
        "filteredRows": filtered_rows,
        "totalRows": total_rows
    }

@app.post("/api/regression")
def run_alpha_regression(req: RegressionRequest):
    """
    Runs an Ordinary Least Squares (OLS) Linear Regression natively on the server.
    Saves a dynamic combined tracking metric column inside memory data matrices.
    """
    # Grab the current split environment arrays
    df_is = datasets.get("is")
    df_oos = datasets.get("oos")
    
    if df_is is None or df_is.empty or not req.features:
        return {"error": "Invalid training conditions or empty feature selection matrix provided."}
        
    try:
        # Validate that selected feature string tokens exist in database schemas
        valid_features = [f for f in req.features if f in df_is.columns]
        if not valid_features:
            return {"error": "None of the selected alpha tracking features were found in the database."}

        # 1. Extract feature data slices and targets from the In-Sample training slice
        X_train = df_is[valid_features].fillna(0)
        y_train = df_is[req.target].fillna(0)
        
        # 2. Train the linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # 3. Calculate model fit score (R-squared)
        r_squared = float(model.score(X_train, y_train))
        
        # 4. Generate the optimized combined trading signal for both data splits
        # This creates the new temporary column dynamically inside server memory
        for split_key in ["is", "oos"]:
            target_df = datasets[split_key]
            if not target_df.empty:
                X_slice = target_df[valid_features].fillna(0)
                # Compute dot-product weights and overwrite custom tracking label fields
                datasets[split_key]["custom_regression_signal"] = model.predict(X_slice)

        # 5. Format coefficients into a clean dictionary map for the front-end display
        feature_weights = {}
        for feature_name, coeff_val in zip(valid_features, model.coef_):
            feature_weights[feature_name] = float(coeff_val)

        return {
            "status": "success",
            "rSquared": r_squared,
            "intercept": float(model.intercept_),
            "coefficients": feature_weights,
            "message": "Custom tracking signal 'custom_regression_signal' calculated across data matrix maps."
        }
        
    except Exception as e:
        return {"error": f"Mathematical engine calculation exception occurred: {str(e)}"}

 
# ── 5. STATIC WORKSPACE CLIENT ROUTING (MUST BE LAST) ──
if os.path.exists("./dist"):
    print("Static build directory located. Mounting frontend web assets...")
    app.mount("/assets", StaticFiles(directory="./dist/assets"), name="assets")

    @app.get("/{catchall:path}")
    def serve_frontend(catchall: str):
        """Serves UI core shell code for any browser paths routing over standard Port 80."""
        return FileResponse("./dist/index.html")
else:
    print("Notice: './dist' folder not found. Server running in API-only mode.")
