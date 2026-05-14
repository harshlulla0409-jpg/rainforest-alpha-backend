import os
import requests
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

# ── 1. CONFIGURE SECURITY CORS HANDSHAKES ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 2. NEW PRODUCTION SCHEMA DATA INGESTION ENGINE ──
DB_PATH = "./data/your_alpha_data.csv"
try:
    # Read the data, dropping unlabelled index padding rows cleanly
    df_raw = pd.read_csv(DB_PATH)
    if '""' in df_raw.columns:
        df_raw = df_raw.drop(columns=['""'])
    
    # Sort data linearly by historical time rows if present
    if "Date" in df_raw.columns and "time" in df_raw.columns:
        df_raw = df_raw.sort_values(by=["Date", "time"]).reset_index(drop=True)

    # 75/25 chronological partition split (In-Sample Training vs Out-of-Sample Validation)
    train_idx = int(len(df_raw) * 0.75)
    datasets = {
        "is": df_raw.iloc[:train_idx].copy(),
        "oos": df_raw.iloc[train_idx:].copy()
    }
    print(f"HFT Pipeline Active. Loaded IS Rows: {len(datasets['is'])}, OOS Rows: {len(datasets['oos'])}")
except Exception as e:
    print(f"Database Error: Failed to ingest production csv mapping ({e}). Running on empty fallbacks.")
    datasets = {"is": pd.DataFrame(), "oos": pd.DataFrame()}

# ── 3. DATA PROTOCOL SCHEMAS ──
class AuthRequest(BaseModel):
    code: str

class UpstreamFilter(BaseModel):
    alphaId: str
    thresholds: List[float]
    selectedBuckets: List[int]

class BucketRequest(BaseModel):
    dataset: str
    side: int                # Accepts: 1 (Buys), -1 (Sells), or 0 (All Sides)
    alphaId: str
    thresholds: List[float]
    upstreamFilters: List[UpstreamFilter]

class RegressionRequest(BaseModel):
    dataset: str
    features: List[str]
    target: str                  # Can accept any column key: r5, r10, r60, r1800, etc.
    name: str = Field(default="custom_alpha")

# ── 4. PRODUCTION ENDPOINT ROUTING MANAGEMENT ──

@app.post("/api/auth/github")
def github_authentication_handshake(req: AuthRequest):
    """
    Exchanges GitHub OAuth codes for access token hashes.
    Stores and returns unique user identifier credentials.
    """
    # Grab your secure OAuth keys from your Railway environment configuration variables panel
    client_id = os.getenv("GITHUB_CLIENT_ID")
    client_secret = os.getenv("GITHUB_CLIENT_SECRET")
    
    try:
        # Step A: Exchange the code for a GitHub Access Token
        token_res = requests.post(
            "https://github.com/login/oauth/access_token",
            headers={"Accept": "application/json"},
            data={"client_id": client_id, "client_secret": client_secret, "code": req.code}
        ).json()
        
        access_token = token_res.get("access_token")
        if not access_token:
            return {"status": "error", "message": "Failed to exchange security authorization code token."}
            
        # Step B: Fetch the user's public profile data using the access token
        user_profile = requests.get(
            "https://api.github.com/user",
            headers={"Authorization": f"token {access_token}"}
        ).json()
        
        github_id = str(user_profile.get("id"))
        username = user_profile.get("login")
        avatar_url = user_profile.get("avatar_url")
        
        # Step C: Save or update the user inside your Supabase database instance
        try:
            conn = get_db_connection()
            if conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO app_users (github_id, username, avatar_url)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (github_id) DO UPDATE SET
                        username = EXCLUDED.username,
                        avatar_url = EXCLUDED.avatar_url;
                        """,
                        (github_id, username, avatar_url)
                    )
                    conn.commit()
                    conn.close()
        except NameError:
            print("WARNING: get_db_connection() is undefined. User not saved to DB.")
                
        return {
            "status": "success",
            "user": {
                "id": github_id,
                "username": username,
                "avatarUrl": avatar_url
            }
        }
    except Exception as e:
        return {"status": "error", "message": f"OAuth handshake failure: {str(e)}"}

@app.get("/api/data/meta")
def get_meta():
    """Returns exact absolute data matrix rows available on the cloud container."""
    return {
        "isRows": len(datasets["is"]),
        "oosRows": len(datasets["oos"])
    }

@app.post("/api/regression")
def run_alpha_regression(req: RegressionRequest):
    """
    Fits an OLS Linear Regression on selected microstructural data features.
    Stamps model prediction outputs onto a unique custom signal name parameter.
    """
    df_is = datasets.get("is")
    if df_is is None or df_is.empty or not req.features:
        return {"error": "Insufficient data arrays or empty feature selection parameters provided."}
        
    try:
        # Verify columns exist in current dataframe schema layout
        valid_features = [f for f in req.features if f in df_is.columns]
        if not valid_features:
            return {"error": "None of the chosen tracking alphas were discovered inside the database."}

        if req.target not in df_is.columns:
            return {"error": f"Target horizon matrix '{req.target}' is missing from the dataset schemas."}

        # Handle numeric casts safely
        X_train = df_is[valid_features].fillna(0).apply(pd.to_numeric, errors='coerce').fillna(0)
        y_train = pd.to_numeric(df_is[req.target], errors='coerce').fillna(0) * 100
        
        # Fit SciKit-Learn Regression Engine
        model = LinearRegression()
        model.fit(X_train, y_train)
        r_squared = float(model.score(X_train, y_train))
        
        # Sanitize target signature token label
        sanitized_name = "".join([c for c in req.name if c.isalnum() or c == "_"])
        if not sanitized_name:
            sanitized_name = "custom_alpha"

        # Apply computed beta weights matrix to save the new dynamic tracking row
        for split_key in ["is", "oos"]:
            if not datasets[split_key].empty:
                X_slice = datasets[split_key][valid_features].fillna(0).apply(pd.to_numeric, errors='coerce').fillna(0)
                datasets[split_key][sanitized_name] = model.predict(X_slice)

        feature_weights = {feat: float(w) for feat, w in zip(valid_features, model.coef_)}

        return {
            "status": "success",
            "rSquared": r_squared,
            "intercept": float(model.intercept_),
            "coefficients": feature_weights,
            "signalName": sanitized_name,
            "message": f"Successfully cached OLS target metric under key string '{sanitized_name}'."
        }
        
    except Exception as e:
        return {"error": f"Regression Engine failure validation block: {str(e)}"}

@app.post("/api/buckets")
def calculate_buckets(req: BucketRequest):
    """
    Slices HFT rows into custom bucket brackets.
    Applies execution direction side filtering before executing cascade filters.
    """
    df = datasets.get(req.dataset, pd.DataFrame()).copy()
    total_rows = len(df)
    
    if df.empty:
        return {"buckets": [], "filteredRows": 0, "totalRows": 0}

    # ── NEW: DIRECTIONAL SIDE FILTERING BLOCK ──
    # If the user selects Buy (1) or Sell (-1), mask rows to isolate that specific sub-market execution flow
    if req.side in [1, -1] and "side" in df.columns:
        df = df[df["side"] == req.side]

    # Apply parent / cascade metric filtering panels
    for f in req.upstreamFilters:
        if f.alphaId in df.columns and len(f.thresholds) > 0:
            sorted_thresh = sorted(f.thresholds)
            b_indices = np.searchsorted(sorted_thresh, pd.to_numeric(df[f.alphaId], errors='coerce').fillna(0))
            df = df[np.isin(b_indices, f.selectedBuckets)]

    filtered_rows = len(df)
    
    # Check if target column exists (including dynamic OLS variable columns)
    if req.alphaId not in df.columns:
        return {
            "error": f"Metric key '{req.alphaId}' not initialized. If custom, execute regression first.",
            "buckets": [], "filteredRows": 0, "totalRows": total_rows
        }

    if filtered_rows == 0:
        return {"buckets": [], "filteredRows": 0, "totalRows": total_rows}

    # Compute quantile boundary cuts
    sorted_cuts = sorted(req.thresholds)
    df['bucket_idx'] = np.searchsorted(sorted_cuts, pd.to_numeric(df[req.alphaId], errors='coerce').fillna(0))
    
    bucket_stats = []
    num_expected_buckets = len(sorted_cuts) + 1
    
    # Process return metrics across available data columns
    for b_id in range(num_expected_buckets):
        b_df = df[df['bucket_idx'] == b_id]
        count = len(b_df)
        
        # ── THE FIX: EXTRACT INDEX 0 FROM THE CUTS LIST FOR THE LOWER BOUND ──
        if b_id == 0:
            lbl = f"[-inf, {sorted_cuts[0]:.2f}]" if sorted_cuts else "All"
        elif b_id == len(sorted_cuts):
            lbl = f"[{sorted_cuts[-1]:.2f}, +inf]"
        else:
            lbl = f"[{sorted_cuts[b_id-1]:.2f}, {sorted_cuts[b_id]:.2f}]"

        bucket_stats.append({
            "bucketIndex": b_id,
            "label": lbl,
            "n": count,
            "r60": float(b_df['r60'].mean() * 100) if count > 0 and not np.isnan(b_df['r60'].mean()) else 0.0,
            "r300": float(b_df['r300'].mean() * 100) if count > 0 and not np.isnan(b_df['r300'].mean()) else 0.0,
            "r1800": float(b_df['r1800'].mean() * 100) if count > 0 and not np.isnan(b_df['r1800'].mean()) else 0.0,
        })

    return {
        "buckets": bucket_stats,
        "filteredRows": filtered_rows,
        "totalRows": total_rows
    }

# ── 5. STATIC WORKSPACE CLIENT ASSET MOUNT (MUST BE LAST) ──
if os.path.exists("./dist"):
    print("Production build distribution located. Launching combined web engine port...")
    app.mount("/assets", StaticFiles(directory="./dist/assets"), name="assets")

    @app.get("/{catchall:path}")
    def serve_frontend(catchall: str):
        return FileResponse("./dist/index.html")
