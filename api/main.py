"""
FastAPI application entry point.

Run with: uvicorn api.main:app --reload
"""

from fastapi import FastAPI

app = FastAPI(
    title="AK Visual Intelligence Platform",
    version="0.1.0",
    description="AI-powered visual inspection API",
)


@app.get("/health")
def health_check():
    """Basic health check — confirms the API process is running."""
    return {"status": "ok", "version": "0.1.0"}
