import os
import json
import requests
import psycopg2
from psycopg2.extras import Json
from fastapi import FastAPI, Request, Response  # <-- FIXED: Explicitly imports Request and Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://rainforest-trading.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
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

# ── 3. ENV DATABASE CONNECTION HELPER ──
DATABASE_URL = os.getenv("DATABASE_URL")
def get_db_connection():
    if not DATABASE_URL:
        return None
    return psycopg2.connect(DATABASE_URL)

# ── 4. DATA PROTOCOL SCHEMAS ──
class AuthRequest(BaseModel):
    code: str

class UpstreamFilter(BaseModel):
    alphaId: str
    thresholds: List[float]
    selectedBuckets: List[int]

class BucketRequest(BaseModel):
    dataset: str
    side: int # Accepts: 1 (Buys), -1 (Sells), or 0 (All Sides)
    alphaId: str
    thresholds: List[float]
    upstreamFilters: List[UpstreamFilter]

class RegressionRequest(BaseModel):
    userId: str                       # Appended automatically by authenticated frontend
    features: List[str]
    target: str                       # Can accept any column key: r5, r10, r60, r1800, etc.
    name: str = Field(default="custom_alpha")

class SavedAlphaLevel(BaseModel):
    alphaId: str
    thresholds: List[float]       # The full array of slider cuts (e.g., [-2.0, 0.0, 5.0, 10.0])
    selectedBuckets: List[int]    # The specific bucket indices the user clicked

class SaveStrategyRequest(BaseModel):
    userId: str
    signalName: str
    targetHorizon: str
    features: List[str]
    isRSquared: float
    oosRSquared: float
    intercept: float
    oosScore: float
    coefficients: Dict[str, float]
    oosBucketData: List[Dict[str, Any]]
    activeWorkspaceLevels: List[SavedAlphaLevel]

# ── 5. PRODUCTION ENDPOINT ROUTING MANAGEMENT ──

@app.post("/api/auth/github")
def github_authentication_handshake(req: AuthRequest):
    client_id = os.getenv("GITHUB_CLIENT_ID")
    client_secret = os.getenv("GITHUB_CLIENT_SECRET")
    
    try:
        # THE FIX: GitHub's API strictly requires payload mapping to be sent over form parameters
        payload_data = {
            "client_id": client_id,
            "client_secret": client_secret,
            "code": req.code,
            "redirect_uri": "https://rainforest-trading.com" # Explicitly matches your custom domain callback
        }
        
        token_res = requests.post(
            "https://github.com/login/oauth/access_token",
            headers={"Accept": "application/json"},
            data=payload_data # Form urlencoded parameter injection
        ).json()
        
        access_token = token_res.get("access_token")
        if not access_token:
            # Prints out GitHub's explicit reason to your Railway log terminal window
            print(f"GitHub Rejection Reason Payload: {token_res}")
            # Include GitHub's exact error message so the frontend logs why it failed
            return {"status": "error", "message": f"Failed to exchange security authorization code token. Reason: {token_res.get('error_description', token_res)}"}

        # Fetch profile
        user_profile = requests.get(
            "https://api.github.com/user",
            headers={"Authorization": f"token {access_token}", "Accept": "application/json"}
        ).json()
        
        github_id = str(user_profile.get("id"))
        username = user_profile.get("login")
        avatar_url = user_profile.get("avatar_url")

        # Sync profile to Supabase app_users table
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
        except Exception as db_err:
            print(f"Database sync warning: {db_err}")

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
    Fits an OLS Linear Regression on In-Sample (IS) data and evaluates on Out-of-Sample (OOS).
    Returns ALL calculation metrics to the frontend WITHOUT saving to the database.
    """
    df_is = datasets.get("is")
    df_oos = datasets.get("oos")
    
    if df_is is None or df_is.empty or df_oos is None or df_oos.empty or not req.features:
        return {"error": "Insufficient data arrays or empty feature selection parameters provided."}
        
    try:
        valid_features = [f for f in req.features if f in df_is.columns]
        if not valid_features:
            return {"error": "None of the chosen tracking alphas were discovered inside the database."}
        if req.target not in df_is.columns:
            return {"error": f"Target horizon matrix '{req.target}' is missing from the dataset schemas."}

        X_train = df_is[valid_features].fillna(0).apply(pd.to_numeric, errors='coerce').fillna(0)
        y_train = pd.to_numeric(df_is[req.target], errors='coerce').fillna(0)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        is_r2 = float(model.score(X_train, y_train)) * 100

        X_test = df_oos[valid_features].fillna(0).apply(pd.to_numeric, errors='coerce').fillna(0)
        y_test = pd.to_numeric(df_oos[req.target], errors='coerce').fillna(0)
        oos_r2 = float(model.score(X_test, y_test)) * 100

        sanitized_name = "".join([c for c in req.name if c.isalnum() or c == "_"])
        if not sanitized_name:
            sanitized_name = "custom_alpha"

        for split_key in ["is", "oos"]:
            if not datasets[split_key].empty:
                X_slice = datasets[split_key][valid_features].fillna(0).apply(pd.to_numeric, errors='coerce').fillna(0)
                datasets[split_key][sanitized_name] = model.predict(X_slice)

        df_oos_eval = datasets["oos"].copy()
        df_oos_eval['bucket_idx'] = pd.qcut(df_oos_eval[sanitized_name], q=10, labels=False, duplicates='drop')
        
        oos_total_rows = len(df_oos_eval)
        oos_bucket_stats = []
        has_sufficient_coverage = True

        for b_id in sorted(df_oos_eval['bucket_idx'].unique()):
            b_df = df_oos_eval[df_oos_eval['bucket_idx'] == b_id]
            count = len(b_df)
            coverage_pct = (count / oos_total_rows) * 100
            
            if coverage_pct <= 1.0:
                has_sufficient_coverage = False

            oos_bucket_stats.append({
                "bucketIndex": int(b_id),
                "count": count,
                "coveragePct": round(coverage_pct, 2),
                "r60_bps": float(b_df['r60'].mean() * 100) if not np.isnan(b_df['r60'].mean()) else 0.0,
                "r300_bps": float(b_df['r300'].mean() * 100) if not np.isnan(b_df['r300'].mean()) else 0.0,
                "r1800_bps": float(b_df['r1800'].mean() * 100) if not np.isnan(b_df['r1800'].mean()) else 0.0,
            })

        feature_weights = {feat: float(w) for feat, w in zip(valid_features, model.coef_)}

        # Return everything to frontend state so user can evaluate and push manually
        return {
            "status": "success",
            "signalName": sanitized_name,
            "targetHorizon": req.target,
            "features": valid_features,
            "isRSquared": is_r2,
            "oosRSquared": oos_r2,
            "intercept": float(model.intercept_),
            "coefficients": feature_weights,
            "hasSufficientCoverage": has_sufficient_coverage,
            "oosBucketData": oos_bucket_stats
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

    # Process return metrics across available data columns (scaled to bps by multiplying by 100)
    for b_id in range(num_expected_buckets):
        b_df = df[df['bucket_idx'] == b_id]
        count = len(b_df)
        
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

@app.post("/api/strategies/save")
def save_custom_strategy(req: SaveStrategyRequest):
    """
    Explicitly saves the calculated model parameters alongside the exact 
    alpha slider cuts and bucket definitions active in the workspace panels.
    """
    try:
        conn = get_db_connection()
        if not conn:
            return {"status": "error", "message": "Cloud database connection credentials unavailable."}
            
        # Convert Pydantic schemas to JSON strings for Postgres injection
        coefficients_json = Json(req.coefficients)
        bucket_data_json = Json(req.oosBucketData)
        
        # Serialize the frontend level configurations array cleanly
        levels_payload = [level.model_dump() for level in req.activeWorkspaceLevels]
        levels_json = Json(levels_payload)

        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO alpha_strategies 
                (signal_name, created_by, features, target_horizon, is_r_squared, oos_r_squared, intercept, oos_score, coefficients, oos_bucket_data, active_workspace_levels)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (signal_name) DO UPDATE SET
                is_r_squared = EXCLUDED.is_r_squared,
                oos_r_squared = EXCLUDED.oos_r_squared,
		        oos_score = EXCLUDED.oos_score,
                coefficients = EXCLUDED.coefficients,
                oos_bucket_data = EXCLUDED.oos_bucket_data,
                active_workspace_levels = EXCLUDED.active_workspace_levels;
                """,
                (
                    req.signalName, 
                    req.userId, 
                    req.features, 
                    req.targetHorizon, 
                    req.isRSquared, 
                    req.oosRSquared, 
                    req.intercept,
		            req.oosScore, 
                    coefficients_json, 
                    bucket_data_json,
                    levels_json
                )
            )
            conn.commit()
            return {"status": "success", "message": f"Strategy '{req.signalName}' and its exact configuration levels logged to the cloud vault."}
    except Exception as e:
        if 'conn' in locals() and conn:
            conn.rollback()
        return {"status": "error", "message": f"Database write transaction failure: {str(e)}"}
    finally:
        if 'conn' in locals() and conn:
            conn.close()

@app.get("/api/strategies")
def get_user_strategies(userId: str):
    """
    Retrieves all archived alpha trading strategies belonging to a specific authenticated user account.
    """
    try:
        conn = get_db_connection()
        if not conn:
            return {"status": "error", "message": "Cloud database credentials unavailable."}
            
        with conn.cursor() as cur:
            # Query columns, transforming JSONB arrays cleanly into dictionary results
            cur.execute(
                """
                SELECT id, signal_name, features, target_horizon, is_r_squared, oos_r_squared, intercept, coefficients, oos_bucket_data, active_workspace_levels, created_at, oos_score 
                FROM alpha_strategies 
                WHERE created_by = %s
                ORDER BY oos_r_squared DESC;
                """,
                (userId,)
            )
            rows = cur.fetchall()
            
        strategies_list = []
        for r in rows:
            strategies_list.append({
                "id": r[0],
                "signalName": r[1],
                "features": r[2],
                "targetHorizon": r[3],
                "isRSquared": r[4],
                "oosRSquared": r[5],
                "intercept": r[6],
                "coefficients": r[7], # Automatically parsed as Dict via psycopg2 JSONB
                "oosBucketData": r[8], # Automatically parsed as List via psycopg2 JSONB
                "activeWorkspaceLevels": r[9], # Contains the exact slider cut thresholds
                "createdAt": r[10].isoformat() if r[10] else None,
		        "oosScore": r[11] if len(r) > 11 and r[11] is not None else 0.0 # safely map the new column
            })
            
        return {"status": "success", "strategies": strategies_list}
    except Exception as e:
        return {"status": "error", "message": f"Database read failure: {str(e)}"}
    finally:
        if 'conn' in locals() and conn:
            conn.close()

# ── 6. STATIC WORKSPACE CLIENT ASSET MOUNT (MUST BE LAST) ──
if os.path.exists("./dist"):
    print("Production build distribution located. Launching combined web engine port...")
    app.mount("/assets", StaticFiles(directory="./dist/assets"), name="assets")

    @app.get("/{catchall:path}")
    def serve_frontend(catchall: str):
        return FileResponse("./dist/index.html")
else:
    print("Notice: './dist' folder not found. Server running in API-only mode.")
