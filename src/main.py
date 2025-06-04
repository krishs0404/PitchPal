#!/usr/bin/env python3
"""
Email Summarizer - Main application entry point
"""
import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import API routers
from api.routers import email_router, recipient_router, analytics_router

# Initialize FastAPI app
app = FastAPI(
    title="Email Summarizer API",
    description="API for generating personalized email drafts",
    version="0.1.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development - restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(email_router.router, prefix="/api/email", tags=["Email"])
app.include_router(recipient_router.router, prefix="/api/recipient", tags=["Recipient"])
app.include_router(analytics_router.router, prefix="/api/analytics", tags=["Analytics"])

@app.get("/")
async def root():
    return {"message": "Welcome to Email Summarizer API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    # Get port from environment variable or use default
    port = int(os.environ.get("PORT", 8000))
    
    # Run the API server
    logger.info(f"Starting Email Summarizer API on port {port}")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
