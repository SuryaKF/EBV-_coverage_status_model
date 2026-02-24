"""
Coverage Classification API
============================
FastAPI deployment for the coverage classification model.

Run with: uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import os

# ============================================================================
# Pydantic Models (Request/Response Schemas)
# ============================================================================

class PredictionRequest(BaseModel):
    """Single prediction request schema."""
    payer_name: str = Field(..., description="Insurance payer name", example="CenCal")
    state_name: str = Field(..., description="State name", example="California")
    acronym: str = Field(..., description="Coverage acronym", example="PA")
    expansion: str = Field(..., description="Acronym expansion", example="Prior Authorization")
    explanation: str = Field(..., description="Coverage explanation", example="Requires approval before service")

    class Config:
        json_schema_extra = {
            "example": {
                "payer_name": "CenCal",
                "state_name": "California",
                "acronym": "PA",
                "expansion": "Prior Authorization",
                "explanation": "Requires prior authorization approval before the medication can be dispensed"
            }
        }


class BatchPredictionRequest(BaseModel):
    """Batch prediction request schema."""
    payer_name: str = Field(..., description="Insurance payer name (same for all)")
    state_name: str = Field(..., description="State name (same for all)")
    items: List[dict] = Field(..., description="List of acronym items to predict")

    class Config:
        json_schema_extra = {
            "example": {
                "payer_name": "CenCal",
                "state_name": "California",
                "items": [
                    {"acronym": "PA", "expansion": "Prior Authorization", "explanation": "Requires approval"},
                    {"acronym": "ST", "expansion": "Step Therapy", "explanation": "Must try other drugs first"}
                ]
            }
        }


class PredictionResponse(BaseModel):
    """Prediction response schema."""
    coverage_status: str
    confidence: float
    all_probabilities: Dict[str, float]
    input_summary: Dict[str, str]
    timestamp: str


class BatchPredictionResponse(BaseModel):
    """Batch prediction response schema."""
    predictions: List[dict]
    total_processed: int
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str
    model_loaded: bool
    model_type: str
    classes: List[str]
    timestamp: str


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="Coverage Classification API",
    description="Predict insurance coverage status based on payer, state, and acronym information",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global model storage
model_data = None


def get_model_path():
    """Get the model file path from environment or default."""
    # Check environment variable first (for Docker)
    env_path = os.environ.get("MODEL_PATH")
    if env_path and os.path.exists(env_path):
        return env_path
    
    # Fallback to relative path (works in Docker and locally)
    relative_path = os.path.join(os.path.dirname(__file__), "models", "coverage_model.pkl")
    if os.path.exists(relative_path):
        return relative_path
    
    # Final fallback to absolute Windows path (local development)
    return r"c:\Users\VH0000812\Desktop\Coverage\models\coverage_model.pkl"


def load_model():
    """Load the model on startup."""
    global model_data
    model_path = get_model_path()
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    
    model_data = joblib.load(model_path)
    return model_data


def predict_single(payer_name: str, state_name: str, acronym: str, 
                   expansion: str, explanation: str) -> dict:
    """Make a single prediction."""
    if model_data is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Prepare features
    combined_text = f"{expansion} {explanation}".lower().strip()
    acronym_clean = acronym.strip().upper()
    
    # Transform text features
    text_features = model_data['tfidf'].transform([combined_text]).toarray()
    
    # Transform categorical features
    cat_df = pd.DataFrame({
        'PAYER NAME': [payer_name],
        'STATE NAME': [state_name],
        'ACRONYM_clean': [acronym_clean]
    })
    cat_features = model_data['onehot'].transform(cat_df)
    
    # Combine features
    X = np.hstack([text_features, cat_features])
    
    # Predict
    prediction_encoded = model_data['model'].predict(X)
    prediction = model_data['label_encoder'].inverse_transform(prediction_encoded)[0]
    
    # Get probabilities
    probabilities = {}
    confidence = 0.0
    try:
        if hasattr(model_data['model'], 'predict_proba'):
            proba = model_data['model'].predict_proba(X)[0]
            classes = model_data['label_encoder'].classes_
            probabilities = {cls: round(float(prob) * 100, 2) for cls, prob in zip(classes, proba)}
            confidence = probabilities.get(prediction, 0.0)
    except Exception:
        # If predict_proba fails, use decision function or default
        confidence = 100.0  # Model made a prediction but can't provide calibrated probability
    
    return {
        "coverage_status": prediction,
        "confidence": confidence,
        "all_probabilities": probabilities,
        "input_summary": {
            "payer": payer_name,
            "state": state_name,
            "acronym": acronym
        },
        "timestamp": datetime.now().isoformat()
    }


# ============================================================================
# API Endpoints
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    try:
        load_model()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Warning: Could not load model: {e}")


@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API info."""
    return {
        "message": "Coverage Classification API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Check API and model health status."""
    if model_data is None:
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            model_type="N/A",
            classes=[],
            timestamp=datetime.now().isoformat()
        )
    
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        model_type=type(model_data['model']).__name__,
        classes=list(model_data['label_encoder'].classes_),
        timestamp=datetime.now().isoformat()
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict(request: PredictionRequest):
    """
    Predict coverage status for a single record.
    
    Returns the predicted coverage class with confidence scores.
    """
    try:
        result = predict_single(
            payer_name=request.payer_name,
            state_name=request.state_name,
            acronym=request.acronym,
            expansion=request.expansion,
            explanation=request.explanation
        )
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Predictions"])
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict coverage status for multiple acronyms.
    
    All items share the same payer and state.
    """
    try:
        predictions = []
        for item in request.items:
            result = predict_single(
                payer_name=request.payer_name,
                state_name=request.state_name,
                acronym=item.get('acronym', ''),
                expansion=item.get('expansion', ''),
                explanation=item.get('explanation', '')
            )
            result['acronym'] = item.get('acronym', '')
            predictions.append(result)
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_processed=len(predictions),
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/classes", tags=["Model Info"])
async def get_classes():
    """Get all possible coverage classes the model can predict."""
    if model_data is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "classes": list(model_data['label_encoder'].classes_),
        "total": len(model_data['label_encoder'].classes_)
    }


@app.get("/model/info", tags=["Model Info"])
async def model_info():
    """Get detailed model information."""
    if model_data is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": type(model_data['model']).__name__,
        "tfidf_features": model_data['tfidf'].max_features,
        "classes": list(model_data['label_encoder'].classes_),
        "metrics": model_data.get('metrics', {}),
        "categorical_features": ['PAYER NAME', 'STATE NAME', 'ACRONYM_clean']
    }


# ============================================================================
# Run with: uvicorn api:app --host 0.0.0.0 --port 8000 --reload
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
