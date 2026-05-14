import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

app = FastAPI()

# ── 1. MIDDLEWARE SETUP ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 2. DATA INGESTION ENGINE ──
DB_PATH = "./data/your_alpha_data.csv"
try:
    df_full = pd.read_csv(DB_PATH)
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
    dataset: str
    alphaId: str
    thresholds: List[float]
    upstreamFilters: List[UpstreamFilter]

# Updated to mirror your frontend state schemas
class RegressionRequest(BaseModel):
    dataset: str
    features: List[str]
    target: str
    name: str = Field(default="custom_alpha")  # Maps directly to regressionName frontend state

# ── 4. LIVE API ENDPOINTS ──

@app.get("/api/data/meta")
def get_meta():
    return {
        "isRows": len(datasets["is"]),
        "oosRows": len(datasets["oos"])
    }

@app.post("/api/regression")
def run_alpha_regression(req: RegressionRequest):
    """
    Runs OLS Regression and stamps the resulting combined array predictions 
    onto a brand new, uniquely named alpha column on the server.
    """
    df_is = datasets.get("is")
    if df_is is None or df_is.empty or not req.features:
        return {"error": "Invalid training conditions or empty feature selection matrix provided."}
        
    try:
        # Sanitize and validate chosen input dimensions
        valid_features = [f for f in req.features if f in df_is.columns]
        if not valid_features:
            return {"error": "None of the selected alpha tracking features were found in the database."}

        # Train model using In-Sample configuration matrices
        X_train = df_is[valid_features].fillna(0)
        y_train = df_is[req.target].fillna(0)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        r_squared = float(model.score(X_train, y_train))
        
        # Ensure the name is clean and won't break dictionary lookups
        sanitized_name = "".join([c for c in req.name if c.isalnum() or c == "_"])
        if not sanitized_name:
            sanitized_name = "custom_alpha"

        # Calculate and store the predictions in the dynamic custom column name
        for split_key in ["is", "oos"]:
            target_df = datasets[split_key]
            if not target_df.empty:
                X_slice = target_df[valid_features].fillna(0)
                # This directly implements 'row[name] = val' across the pandas dataframe block
                datasets[split_key][sanitized_name] = model.predict(X_slice)

        feature_weights = {feat: float(w) for feat, w in zip(valid_features, model.coef_)}

        return {
            "status": "success",
            "rSquared": r_squared,
            "intercept": float(model.intercept_),
            "coefficients": feature_weights,
            "signalName": sanitized_name,
            "message": f"Custom alpha metric '{sanitized_name}' successfully built and cached in memory pools."
        }
        
    except Exception as e:
        return {"error": f"Mathematical engine calculation exception occurred: {str(e)}"}

@app.post("/api/buckets")
def calculate_buckets(req: BucketRequest):
    df = datasets.get(req.dataset, pd.DataFrame()).copy()
    total_rows = len(df)
    
    if df.empty:
        return {"buckets": [], "filteredRows": 0, "totalRows": 0}

    # Apply parent / cascade metric filtering panels
    for f in req.upstreamFilters:
        if f.alphaId in df.columns and len(f.thresholds) > 0:
            sorted_thresh = sorted(f.thresholds)
            b_indices = np.searchsorted(sorted_thresh, df[f.alphaId])
            df = df[np.isin(b_indices, f.selectedBuckets)]

    filtered_rows = len(df)
    
    # Check if target column exists (including newly dynamically generated regression variables)
    if req.alphaId not in df.columns:
        return {
            "error": f"Alpha ID '{req.alphaId}' not found. Ensure you ran the regression model first.",
            "buckets": [], "filteredRows": 0, "totalRows": total_rows
        }

    if filtered_rows == 0:
        return {"buckets": [], "filteredRows": 0, "totalRows": total_rows}

    # Slice out active dataset using current threshold slider configuration boundaries
    sorted_cuts = sorted(req.thresholds)
    df['bucket_idx'] = np.searchsorted(sorted_cuts, df[req.alphaId])
    
    bucket_stats = []
    num_expected_buckets = len(sorted_cuts) + 1
    
    for b_id in range(num_expected_buckets):
        b_df = df[df['bucket_idx'] == b_id]
        count = len(b_df)
        
        if b_id == 0:
            lbl = f"[-inf, {sorted_cuts[0]:.2f}]" if sorted_cuts else "All"
        elif b_id == len(sorted_cuts):
            lbl = f"[{sorted_cuts[-1]:.2f}, +inf]"
        else:
            lbl = f"[{sorted_cuts[b_id-1]:.2f}, {sorted_cuts[b_id]:.2f}]"

        # Explicit key variables mapping safely to frontend HFTGame.tsx line 396 expectations
        bucket_stats.append({
            "bucketIndex": b_id,
            "label": lbl,
            "n": count,
            "r60": float(b_df['r60'].mean()) if count > 0 and not np.isnan(b_df['r60'].mean()) else 0.0,
            "r300": float(b_df['r300'].mean()) if count > 0 and not np.isnan(b_df['r300'].mean()) else 0.0,
            "r1800": float(b_df['r1800'].mean()) if count > 0 and not np.isnan(b_df['r1800'].mean()) else 0.0,
        })

    return {
        "buckets": bucket_stats,
        "filteredRows": filtered_rows,
        "totalRows": total_rows
    }

# ── 5. STATIC WORKSPACE CLIENT ROUTING (MUST BE LAST) ──
if os.path.exists("./dist"):
    print("Static build directory located. Mounting frontend web assets...")
    app.mount("/assets", StaticFiles(directory="./dist/assets"), name="assets")

    @app.get("/{catchall:path}")
    def serve_frontend(catchall: str):
        return FileResponse("./dist/index.html")
else:
    print("Notice: './dist' folder not found. Server running in API-only mode.")

