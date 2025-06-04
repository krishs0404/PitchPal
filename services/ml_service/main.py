"""
ML Service for PitchPal
Provides lead scoring and other ML predictions using PyTorch
"""
import os
from typing import Dict, List, Optional, Any, Union
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import logging
import json
from datetime import datetime
import uvicorn
from dotenv import load_dotenv
import numpy as np

# Load environment variables
load_dotenv()

# Try to import PyTorch (graceful fallback if not installed)
try:
    import torch
    import torch.nn as nn
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    logging.warning("PyTorch not installed. Using mock ML functionality.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="PitchPal ML Service",
    description="ML service for lead scoring and predictions using PyTorch",
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

# Define models
class RecipientInfo(BaseModel):
    name: str
    company: Optional[str] = None
    role: Optional[str] = None
    interests: Optional[List[str]] = Field(default_factory=list)
    linkedin_data: Optional[Dict[str, Any]] = None

class EmailContent(BaseModel):
    subject: str
    body: str
    
class LeadScoringRequest(BaseModel):
    recipient: RecipientInfo
    goal: str
    email_content: EmailContent
    industry: Optional[str] = None
    interaction_history: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    
class LeadScoringResponse(BaseModel):
    lead_id: str
    score: float  # 0.0 to 1.0
    score_components: Dict[str, float]
    recommendations: List[str]
    timestamp: datetime = Field(default_factory=datetime.now)

# Define a simple PyTorch model for lead scoring
class LeadScoringNN(nn.Module):
    """
    A simple neural network for lead scoring.
    In production, this would be more sophisticated.
    """
    def __init__(self, input_size=10):
        super(LeadScoringNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.layers(x)

# Mock PyTorch lead scoring model
class LeadScoringModel:
    """
    A mock or real PyTorch lead scoring model depending on availability.
    """
    def __init__(self):
        logger.info("Initializing lead scoring model")
        self.use_pytorch = PYTORCH_AVAILABLE
        
        if self.use_pytorch:
            # Initialize PyTorch model
            self.model = LeadScoringNN()
            # In production, we would load pre-trained weights:
            # self.model.load_state_dict(torch.load("model_weights.pth"))
            self.model.eval()
            logger.info("Using PyTorch model for predictions")
        else:
            logger.info("Using mock model for predictions")
        
    def _extract_features(self, data: Dict[str, Any]) -> Union[np.ndarray, torch.Tensor]:
        """Extract numerical features from input data"""
        # This is a simplified feature extraction
        # In production, this would be more sophisticated
        
        # Initialize features with zeros
        features = np.zeros(10)
        
        # Extract basic features
        # Feature 1: Has role information
        features[0] = 1.0 if data.get("role") else 0.0
        
        # Feature 2: Has company information
        features[1] = 1.0 if data.get("company") else 0.0
        
        # Feature 3: Role seniority
        role_keywords = {
            "ceo": 1.0, "chief": 0.9, "vp": 0.8, "vice president": 0.8,
            "director": 0.7, "head": 0.6, "senior": 0.5, "lead": 0.4,
            "manager": 0.3, "specialist": 0.2
        }
        if data.get("role"):
            role_lower = data.get("role", "").lower()
            for keyword, value in role_keywords.items():
                if keyword in role_lower:
                    features[2] = value
                    break
        
        # Feature 4: Number of past interactions
        features[3] = min(1.0, len(data.get("interaction_history", [])) / 10.0)
        
        # Feature 5: Has LinkedIn data
        features[4] = 1.0 if data.get("linkedin_data") else 0.0
        
        # Feature 6: LinkedIn connections/followers (normalized)
        if data.get("linkedin_data") and "connections" in data.get("linkedin_data", {}):
            features[5] = min(1.0, data["linkedin_data"]["connections"] / 500.0)
        
        # Feature 7: Company size indicator
        if data.get("company_data") and "employees" in data.get("company_data", {}):
            employees = data["company_data"]["employees"]
            if isinstance(employees, str):
                if "10,000+" in employees:
                    features[6] = 1.0
                elif "1,000-10,000" in employees:
                    features[6] = 0.8
                elif "500-1,000" in employees:
                    features[6] = 0.6
                elif "50-500" in employees:
                    features[6] = 0.4
                elif "10-50" in employees:
                    features[6] = 0.2
                else:
                    features[6] = 0.1
            elif isinstance(employees, int):
                features[6] = min(1.0, employees / 10000.0)
        
        # Feature 8-10: Reserved for expansion
        
        if self.use_pytorch:
            return torch.tensor(features, dtype=torch.float32)
        return features
        
    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a prediction for lead scoring
        
        Args:
            data: Input data for the model
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Extract features from data
            features = self._extract_features(data)
            
            if self.use_pytorch:
                # Run inference with PyTorch model
                with torch.no_grad():
                    score = self.model(features).item()
            else:
                # Mock implementation - generate score based on features
                # Use a weighted sum of features as a simple scoring method
                weights = np.array([0.15, 0.10, 0.25, 0.15, 0.05, 0.10, 0.20, 0, 0, 0])
                base_score = np.dot(features, weights)
                
                # Add some variance based on recipient name for consistent results
                seed = sum(ord(c) for c in data.get("recipient_name", "unknown"))
                np.random.seed(seed)
                variance = np.random.uniform(-0.1, 0.1)
                
                score = min(0.99, max(0.01, base_score + variance))
            
            # Calculate score components for explainability
            score_components = {
                "role_seniority": float(features[2] * 0.25),
                "past_interactions": float(features[3] * 0.15),
                "company_size": float(features[6] * 0.20),
                "linkedin_presence": float(features[4] * 0.05 + features[5] * 0.10),
                "base_factors": float(features[0] * 0.15 + features[1] * 0.10)
            }
            
            # Generate recommendations based on score
            recommendations = []
            if score >= 0.8:
                recommendations.append("High-value lead: Consider personalized follow-up")
                recommendations.append("Reference specific projects from their LinkedIn")
                recommendations.append("Offer a direct meeting with a decision-maker")
            elif score >= 0.6:
                recommendations.append("Promising lead: Use industry-specific examples")
                recommendations.append("Highlight relevant case studies")
                recommendations.append("Suggest a product demo")
            else:
                recommendations.append("Nurture lead: Focus on educational content")
                recommendations.append("Share relevant articles or webinars")
                recommendations.append("Schedule for follow-up in 30 days")
            
            return {
                "score": float(score),
                "score_components": score_components,
                "recommendations": recommendations
            }
            
        except Exception as e:
            logger.error(f"Error in lead scoring prediction: {e}")
            # Return fallback prediction
            return {
                "score": 0.5,
                "score_components": {"error": 1.0},
                "recommendations": ["Error in prediction, use default approach"]
            }

# Initialize lead scoring model
lead_model = LeadScoringModel()

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "service": "ml_service",
        "pytorch_available": PYTORCH_AVAILABLE
    }

# Lead scoring endpoint
@app.post("/api/ml/score-lead", response_model=LeadScoringResponse)
async def score_lead(request: LeadScoringRequest):
    """
    Score a lead based on available data and the cold email content
    """
    try:
        # Extract recipient data from the nested structure
        recipient_data = {
            "recipient_name": request.recipient.name,
            "role": request.recipient.role,
            "company": request.recipient.company,
            "interests": request.recipient.interests,
            "linkedin_data": request.recipient.linkedin_data,
            "company_data": {"industry": request.industry} if request.industry else {},
            "interaction_history": request.interaction_history,
            "email_content": {
                "subject": request.email_content.subject,
                "body": request.email_content.body
            },
            "goal": request.goal
        }
        
        # Get prediction from model
        prediction = lead_model.predict(recipient_data)
        
        # Create response
        response = LeadScoringResponse(
            lead_id=f"cold-lead-{datetime.now().timestamp()}",
            score=prediction["score"],
            score_components=prediction["score_components"],
            recommendations=prediction["recommendations"]
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error scoring lead: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Batch scoring endpoint
@app.post("/api/ml/batch-score", response_model=List[LeadScoringResponse])
async def batch_score(requests: List[LeadScoringRequest]):
    """
    Score multiple leads in a batch
    """
    try:
        responses = []
        
        for request in requests:
            # Convert request to dictionary for model input
            data = request.dict()
            
            # Get prediction from model
            prediction = lead_model.predict(data)
            
            # Create response
            response = LeadScoringResponse(
                lead_id=f"lead-{datetime.now().timestamp()}",
                score=prediction["score"],
                score_components=prediction["score_components"],
                recommendations=prediction["recommendations"]
            )
            
            responses.append(response)
        
        return responses
        
    except Exception as e:
        logger.error(f"Error batch scoring leads: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Get port from environment variable or use default
    port = int(os.environ.get("PORT", 8002))
    host = os.environ.get("HOST", "0.0.0.0")
    
    # Run the API server
    logger.info(f"Starting PitchPal ML Service on {host}:{port}")
    uvicorn.run("main:app", host=host, port=port, reload=True)
