"""
PitchPal API Service
FastAPI backend with LangChain integration
"""
import os
import json
import httpx
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import logging
from datetime import datetime
from dotenv import load_dotenv
import uvicorn

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="PitchPal API",
    description="API for generating personalized cold emails with LangChain and RAG",
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
    email: str
    company: Optional[str] = None
    role: Optional[str] = None
    linkedin_url: Optional[str] = None
    interests: Optional[List[str]] = Field(default_factory=list)

class EmailRequest(BaseModel):
    recipient: RecipientInfo
    goal: str
    additional_context: Optional[str] = None
    tone: str = "professional"
    use_lead_scoring: bool = False

class EmailResponse(BaseModel):
    email_id: str
    subject: str
    body: str
    recipient_email: str
    personalization_data: Dict
    lead_score: Optional[float] = None
    created_at: datetime = Field(default_factory=datetime.now)

class HubspotPushRequest(BaseModel):
    email_id: str
    contact_id: Optional[str] = None
    deal_id: Optional[str] = None

# Mock database for demo purposes
email_drafts_db = {}

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "api_service"}

# Service URLs
VECTOR_SERVICE_URL = os.getenv("VECTOR_SERVICE_URL", "http://vector_service:8001")
ML_SERVICE_URL = os.getenv("ML_SERVICE_URL", "http://ml_service:8002")

# Async HTTP client
async def get_http_client():
    return httpx.AsyncClient(timeout=60.0)

# Generate email endpoint
@app.post("/api/email/generate", response_model=EmailResponse)
async def generate_email(request: EmailRequest, background_tasks: BackgroundTasks):
    """
    Generate a personalized cold email using LangChain and RAG
    """
    try:
        # 1. First, check if LinkedIn profile is available, if so, index it
        if request.recipient.linkedin_url:
            async with httpx.AsyncClient(timeout=30.0) as client:
                try:
                    # Index LinkedIn profile
                    linkedin_response = await client.post(
                        f"{VECTOR_SERVICE_URL}/api/vector/index-linkedin",
                        json={
                            "url": request.recipient.linkedin_url,
                            "name": request.recipient.name,
                            "company": request.recipient.company,
                            "role": request.recipient.role
                        }
                    )
                    linkedin_response.raise_for_status()
                    logger.info(f"Indexed LinkedIn profile: {linkedin_response.json()}")
                except Exception as e:
                    logger.warning(f"Failed to index LinkedIn profile: {str(e)}")
        
        # 2. Call the vector service to generate personalized email
        email_content = None
        personalization_data = {}
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                # Generate email using vector service RAG
                email_gen_response = await client.post(
                    f"{VECTOR_SERVICE_URL}/api/vector/generate-email",
                    json={
                        "recipient_name": request.recipient.name,
                        "recipient_role": request.recipient.role,
                        "recipient_company": request.recipient.company,
                        "recipient_linkedin": request.recipient.linkedin_url,
                        "goal": request.goal,
                        "tone": request.tone,
                        "additional_context": request.additional_context
                    }
                )
                
                if email_gen_response.status_code == 200:
                    email_result = email_gen_response.json()
                    email_content = {
                        "subject": email_result.get("subject", ""),
                        "body": email_result.get("body", "")
                    }
                    personalization_data = email_result.get("personalization_data", {})
                    logger.info("Successfully generated email from vector service")
                else:
                    logger.warning(f"Vector service returned: {email_gen_response.status_code}")
            except Exception as e:
                logger.error(f"Error calling vector service: {str(e)}")
        
        # 3. If vector service fails or returns empty, use fallback method
        if not email_content or not email_content.get("subject") or not email_content.get("body"):
            logger.warning("Using fallback email generation")
            
            # Fallback email generation
            subject = f"Enhancing {request.goal} at {request.recipient.company}"
            body = f"""
Hi {request.recipient.name},

I noticed your work at {request.recipient.company} as {request.recipient.role}.

Our team has been helping companies like yours improve {request.goal}, and I thought you might be interested in learning how we could help {request.recipient.company} as well.

Would you be open to a brief 15-minute call to discuss this further?

Best regards,
[Your Name]
"""
            email_content = {"subject": subject, "body": body}
            
            # Basic personalization data
            if not personalization_data:
                personalization_data = {
                    "recipient": {
                        "name": request.recipient.name,
                        "role": request.recipient.role,
                        "company": request.recipient.company
                    }
                }
        
        # 4. Calculate lead score if requested
        lead_score = None
        if request.use_lead_scoring:
            try:
                # Call the ML service for lead scoring
                async with httpx.AsyncClient(timeout=15.0) as client:
                    ml_response = await client.post(
                        f"{ML_SERVICE_URL}/api/ml/score-lead",
                        json={
                            "recipient": {
                                "name": request.recipient.name,
                                "company": request.recipient.company,
                                "role": request.recipient.role,
                                "interests": request.recipient.interests
                            },
                            "goal": request.goal,
                            "email_content": email_content
                        }
                    )
                    
                    if ml_response.status_code == 200:
                        lead_score = ml_response.json().get("score")
                        logger.info(f"Lead score calculated: {lead_score}")
            except Exception as e:
                logger.warning(f"Failed to get lead score: {str(e)}")
                # Fallback lead score based on role seniority
                role = request.recipient.role.lower() if request.recipient.role else ""
                if any(senior in role for senior in ["ceo", "cto", "cfo", "founder", "director", "vp", "head"]):
                    lead_score = 0.85
                elif any(mid in role for mid in ["manager", "lead", "senior"]):
                    lead_score = 0.65
                else:
                    lead_score = 0.45
        
        # 5. Generate a unique ID for the email
        email_id = f"cold-email-{datetime.now().timestamp()}"
            
        # 6. Create email response
        email_response = EmailResponse(
            email_id=email_id,
            subject=email_content["subject"],
            body=email_content["body"],
            recipient_email=request.recipient.email,
            personalization_data=personalization_data,
            lead_score=lead_score
        )
        
        # 7. Store in database
        email_drafts_db[email_id] = email_response.dict()
        
        return email_response
        
    except Exception as e:
        logger.error(f"Error generating email: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Push to HubSpot endpoint
@app.post("/api/email/push-to-hubspot")
async def push_to_hubspot(request: HubspotPushRequest):
    """
    Push generated email to HubSpot CRM
    """
    try:
        # Check if email exists
        if request.email_id not in email_drafts_db:
            raise HTTPException(status_code=404, detail="Email draft not found")
        
        # In real implementation, call HubSpot API
        # For now, just return success
        return {
            "status": "success",
            "message": f"Email pushed to HubSpot for contact {request.contact_id}",
            "hubspot_id": f"hubspot-{datetime.now().timestamp()}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error pushing to HubSpot: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Get email draft by ID
@app.get("/api/email/{email_id}", response_model=EmailResponse)
async def get_email(email_id: str):
    """
    Get email draft by ID
    """
    try:
        if email_id not in email_drafts_db:
            raise HTTPException(status_code=404, detail="Email draft not found")
            
        return email_drafts_db[email_id]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting email: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Get port from environment variable or use default
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    # Run the API server
    logger.info(f"Starting PitchPal API on {host}:{port}")
    uvicorn.run("main:app", host=host, port=port, reload=True)
