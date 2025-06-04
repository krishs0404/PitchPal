"""
Vector Service for PitchPal
Handles RAG operations using LangChain and Pinecone
"""
import os
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import logging
from datetime import datetime
import json
import uvicorn
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import langchain components
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, WebBaseLoader
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

# Try to import Pinecone
try:
    import pinecone
except ImportError:
    pinecone = None
    logging.warning("Pinecone not installed. Vector operations will be mocked.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="PitchPal Vector Service",
    description="Vector database service for RAG operations",
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
class LinkedInProfile(BaseModel):
    url: str
    name: Optional[str] = None
    company: Optional[str] = None
    role: Optional[str] = None
    content: Optional[str] = None

class WebContent(BaseModel):
    url: str
    content: Optional[str] = None
    content_type: str = "webpage"  # webpage, news, article, etc.

class SearchQuery(BaseModel):
    query: str
    context: Dict[str, Any] = Field(default_factory=dict)
    top_k: int = 5

class VectorSearchResult(BaseModel):
    query: str
    results: List[Dict]
    source: str
    timestamp: datetime = Field(default_factory=datetime.now)

class EmailGenerationContext(BaseModel):
    recipient_name: str
    recipient_role: Optional[str] = None
    recipient_company: Optional[str] = None
    recipient_linkedin: Optional[str] = None
    goal: str
    tone: str = "professional"
    additional_context: Optional[str] = None

class EmailTemplate(BaseModel):
    subject_template: str
    body_template: str
    context: Dict[str, Any] = Field(default_factory=dict)

# Global variables
pinecone_index = None
embeddings = None

# Initialize OpenAI and Pinecone
def init_services():
    global pinecone_index, embeddings
    
    # Initialize OpenAI embeddings with text-embedding-3-small model
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=openai_api_key
        )
        logger.info("OpenAI embeddings (text-embedding-3-small) initialized")
    
    # Initialize Pinecone client if credentials are available
    api_key = os.getenv("PINECONE_API_KEY")
    environment = os.getenv("PINECONE_ENVIRONMENT", "us-west1-gcp")
    
    if not api_key:
        logger.warning("PINECONE_API_KEY not found. Using mock Pinecone.")
        return None
    
    try:
        pinecone.init(api_key=api_key, environment=environment)
        index_name = os.getenv("PINECONE_INDEX", "email-summarizer")
        
        # Check if index exists, create if not
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=index_name,
                dimension=1536,  # OpenAI embedding dimension
                metric="cosine"
            )
            logger.info(f"Created Pinecone index: {index_name}")
        
        return pinecone.Index(index_name)
    except Exception as e:
        logger.error(f"Error initializing Pinecone: {e}")
        return None

# Mock embedding function for testing without OpenAI API
def get_mock_embeddings(text: str) -> List[float]:
    """Generate mock embeddings for testing"""
    import hashlib
    import numpy as np
    
    # Create a deterministic but unique embedding based on text hash
    hash_object = hashlib.md5(text.encode())
    seed = int(hash_object.hexdigest(), 16) % 10000
    np.random.seed(seed)
    
    # Generate a 1536-dimensional vector (same as OpenAI embeddings)
    return list(np.random.uniform(-1, 1, 1536))

# Query Pinecone for relevant content
async def query_vectors(query_text: str, namespace: str, top_k: int = 5) -> List[Dict]:
    if not embeddings or not pinecone_index:
        logger.warning("Embeddings or Pinecone not initialized. Using mock implementation.")
        # Return mock results
        return [
            {
                "text": f"Relevant content for: {query_text}",
                "metadata": {
                    "source": "mock",
                    "url": "https://example.com",
                    "score": 0.95
                }
            }
        ]
    
    try:
        # Get query embedding
        query_embedding = embeddings.embed_query(query_text)
        
        # Query Pinecone
        results = pinecone_index.query(
            vector=query_embedding,
            top_k=top_k,
            namespace=namespace,
            include_metadata=True
        )
        
        # Format results
        formatted_results = []
        for match in results.matches:
            formatted_results.append({
                "text": match.metadata.get("text", ""),
                "metadata": {
                    **match.metadata,
                    "score": match.score
                }
            })
        
        return formatted_results
    except Exception as e:
        logger.error(f"Error querying vectors: {e}")
        return []

# Fetch content from LinkedIn URL
async def fetch_linkedin_content(url: str) -> str:
    # In a production environment, you would:
    # 1. Use a proper LinkedIn scraper or API
    # 2. Handle authentication and rate limiting
    # 3. Extract structured data from the profile
    
    # For demo purposes, we'll return mock content
    mock_content = f"""
    LinkedIn Profile
    ----------------
    URL: {url}
    
    About:
    Experienced professional with expertise in technology and business. 
    Passionate about innovation and digital transformation.
    
    Experience:
    - Senior Product Manager at TechCorp (2020-Present)
    - Product Manager at InnovateCo (2018-2020)
    - Business Analyst at ConsultFirm (2015-2018)
    
    Education:
    - MBA, Business School (2015)
    - BS Computer Science, Tech University (2013)
    
    Skills:
    Product Management, Strategy, AI/ML, Data Analytics, Leadership
    
    Recent Activity:
    - Shared article: "The Future of AI in Business"
    - Commented on: "Digital Transformation Challenges"
    - Liked: "Product Management Best Practices"
    """
    
    return mock_content

# Fetch content from web URL
async def fetch_web_content(url: str) -> str:
    try:
        # In a production environment, you would:
        # 1. Handle different types of content (HTML, PDF, etc.)
        # 2. Use proper parsing and extraction
        # 3. Handle errors and timeouts
        
        # Simple web request for demo
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.text
        else:
            logger.warning(f"Failed to fetch content from {url}: {response.status_code}")
            return ""
    except Exception as e:
        logger.error(f"Error fetching web content from {url}: {e}")
        return ""

# Initialize on startup
@app.on_event("startup")
async def startup_event():
    init_services()

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "service": "vector_service",
        "openai": embeddings is not None,
        "pinecone": pinecone_index is not None
    }

# Index LinkedIn profile data
@app.post("/api/vector/index-linkedin")
async def index_linkedin_profile(profile: LinkedInProfile, background_tasks: BackgroundTasks):
    """
    Index LinkedIn profile data in the vector database
    """
    try:
        # Get profile content
        profile_content = profile.content
        
        # If content not provided, fetch it
        if not profile_content and profile.url:
            profile_content = await fetch_linkedin_content(profile.url)
        
        if not profile_content:
            raise HTTPException(status_code=400, detail="LinkedIn profile content is required")
        
        # Process the text into chunks
        documents = process_text(profile_content)
        
        # Create metadata
        metadata = {
            "source": "linkedin",
            "url": profile.url,
            "name": profile.name,
            "company": profile.company,
            "role": profile.role,
            "content_type": "profile",
            "indexed_at": datetime.now().isoformat()
        }
        
        # Store embeddings in background
        profile_id = f"linkedin-{datetime.now().timestamp()}"
        background_tasks.add_task(
            store_embeddings,
            documents=documents,
            metadata=metadata,
            namespace="linkedin_profiles"
        )
        
        return {
            "status": "success",
            "message": f"LinkedIn profile for {profile.name or 'unknown'} is being indexed",
            "profile_id": profile_id,
            "chunk_count": len(documents)
        }
        
    except Exception as e:
        logger.error(f"Error indexing LinkedIn profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Index web content (articles, company pages, etc.)
@app.post("/api/vector/index-web-content")
async def index_web_content(content: WebContent, background_tasks: BackgroundTasks):
    """
    Index web content in the vector database
    """
    try:
        # Get content
        web_content = content.content
        
        # If content not provided, fetch it
        if not web_content and content.url:
            web_content = await fetch_web_content(content.url)
        
        if not web_content:
            raise HTTPException(status_code=400, detail="Web content is required")
        
        # Process the text into chunks
        documents = process_text(web_content)
        
        # Create metadata
        metadata = {
            "source": "web",
            "url": content.url,
            "content_type": content.content_type,
            "indexed_at": datetime.now().isoformat()
        }
        
        # Store embeddings in background
        content_id = f"web-{datetime.now().timestamp()}"
        background_tasks.add_task(
            store_embeddings,
            documents=documents,
            metadata=metadata,
            namespace="web_content"
        )
        
        return {
            "status": "success",
            "message": f"Web content from {content.url} is being indexed",
            "content_id": content_id,
            "chunk_count": len(documents)
        }
        
    except Exception as e:
        logger.error(f"Error indexing web content: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Search for relevant context
@app.post("/api/vector/search", response_model=VectorSearchResult)
async def search_vectors(query: SearchQuery):
    """
    Search vector database for relevant context
    """
    try:
        # Determine which namespaces to search
        namespaces = ["linkedin_profiles", "web_content"]
        if "namespace" in query.context:
            namespaces = [query.context["namespace"]]
        
        all_results = []
        for namespace in namespaces:
            results = await query_vectors(
                query_text=query.query,
                namespace=namespace,
                top_k=query.top_k
            )
            all_results.extend(results)
        
        # Sort by score
        all_results = sorted(all_results, key=lambda x: x["metadata"].get("score", 0), reverse=True)
        
        # Limit to top_k
        all_results = all_results[:query.top_k]
        
        return VectorSearchResult(
            query=query.query,
            results=all_results,
            source="pinecone"
        )
        
    except Exception as e:
        logger.error(f"Error searching vectors: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Generate email content based on context
@app.post("/api/vector/generate-email")
async def generate_email(context: EmailGenerationContext):
    """
    Generate personalized email content using RAG
    """
    try:
        # 1. First, retrieve relevant context about the recipient
        recipient_query = f"Information about {context.recipient_name} at {context.recipient_company} who is a {context.recipient_role}"
        
        recipient_results = await query_vectors(
            query_text=recipient_query,
            namespace="linkedin_profiles",
            top_k=3
        )
        
        # 2. Retrieve relevant context about the company/industry
        company_query = f"Information about {context.recipient_company} company and industry"
        
        company_results = await query_vectors(
            query_text=company_query,
            namespace="web_content",
            top_k=3
        )
        
        # 3. Combine all context
        all_context = ""
        
        if recipient_results:
            all_context += "\nRECIPIENT INFORMATION:\n"
            for item in recipient_results:
                all_context += f"- {item['text']}\n"
        
        if company_results:
            all_context += "\nCOMPANY INFORMATION:\n"
            for item in company_results:
                all_context += f"- {item['text']}\n"
        
        if context.additional_context:
            all_context += f"\nADDITIONAL CONTEXT:\n{context.additional_context}\n"
        
        # 4. Create email generation prompt
        email_prompt = f"""
        You are an expert cold email writer. Generate a personalized cold email based on the following information.
        
        RECIPIENT:
        Name: {context.recipient_name}
        Role: {context.recipient_role or 'Unknown'}
        Company: {context.recipient_company or 'Unknown'}
        
        GOAL:
        {context.goal}
        
        TONE:
        {context.tone}
        
        CONTEXT INFORMATION:
        {all_context}
        
        Generate a cold email with:
        1. A compelling subject line
        2. Personalized greeting
        3. Introduction that demonstrates research and relevance
        4. Unique value proposition aligned with their needs
        5. Clear call to action
        
        FORMAT:
        Subject: [Your subject line]
        
        [Email body]
        """
        
        # 5. Call LLM (OpenAI) for generation
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            # Mock response for testing
            return {
                "subject": f"Opportunity for {context.recipient_company} to improve {context.goal}",
                "body": f"""
                Hi {context.recipient_name},
                
                I came across your profile and noticed your work at {context.recipient_company} as {context.recipient_role}. 
                Your experience in this field is impressive.
                
                Based on my research, I believe we can help you with {context.goal}.
                
                Would you be available for a quick 15-minute call next week to discuss this further?
                
                Best regards,
                [Your Name]
                """,
                "personalization_data": {
                    "recipient_info": recipient_results,
                    "company_info": company_results
                }
            }
        
        try:
            from langchain.llms import OpenAI
            llm = OpenAI(
                model_name="gpt-3.5-turbo",
                temperature=0.7, 
                max_tokens=1000, 
                api_key=openai_api_key
            )
            response = llm(email_prompt)
            
            # Parse the response
            lines = response.strip().split("\n")
            subject = ""
            body = ""
            
            # Extract subject and body
            for i, line in enumerate(lines):
                if line.startswith("Subject:"):
                    subject = line.replace("Subject:", "").strip()
                    body = "\n".join(lines[i+1:]).strip()
                    break
            
            # If no subject found, use first line as subject and rest as body
            if not subject and lines:
                subject = lines[0]
                body = "\n".join(lines[1:]).strip()
            
            return {
                "subject": subject,
                "body": body,
                "personalization_data": {
                    "recipient_info": recipient_results,
                    "company_info": company_results
                }
            }
        except Exception as e:
            logger.error(f"Error calling OpenAI: {e}")
            raise HTTPException(status_code=500, detail=f"Error generating email: {e}")
        
    except Exception as e:
        logger.error(f"Error generating email: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Get port from environment variable or use default
    port = int(os.environ.get("PORT", 8001))
    host = os.environ.get("HOST", "0.0.0.0")
    
    # Run the API server
    logger.info(f"Starting Vector Service on {host}:{port}")
    uvicorn.run("main:app", host=host, port=port, reload=True)
