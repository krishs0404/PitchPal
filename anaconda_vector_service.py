"""
PitchPal Vector Service - Anaconda-compatible implementation
Uses Flask, OpenAI, and Pinecone for RAG operations
"""
import os
import json
import logging
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import openai
from pinecone import Pinecone

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PitchPal-Vector-Service")

# Load environment variables
env_path = os.path.join(os.path.dirname(__file__), 'services/vector_service/.env')
load_dotenv(env_path)
logger.info(f"Loaded environment from {env_path}")

# Initialize Flask app
app = Flask(__name__)

# Global variables
pc_index = None

def init_services():
    """Initialize OpenAI and Pinecone services"""
    global pc_index
    
    # Initialize OpenAI
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logger.error("Missing OpenAI API key")
        return False
    
    client = openai.OpenAI(api_key=openai_api_key)
    
    # Test OpenAI connection
    try:
        models = client.models.list()
        logger.info(f"Connected to OpenAI API - Available models: {len(models.data)}")
    except Exception as e:
        logger.error(f"Failed to connect to OpenAI: {e}")
        return False
    
    # Initialize Pinecone
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_environment = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
    pinecone_index_name = os.getenv("PINECONE_INDEX", "pitchpal")
    
    if not pinecone_api_key:
        logger.error("Missing Pinecone API key")
        return False
    
    try:
        # Initialize Pinecone client
        pc = Pinecone(api_key=pinecone_api_key)
        
        # List available indexes
        try:
            index_list = pc.list_indexes()
            logger.info(f"Available Pinecone indexes: {[idx.name for idx in index_list]}")
            
            # Check if our index exists
            index_exists = any(idx.name == pinecone_index_name for idx in index_list)
            if not index_exists:
                logger.warning(f"Pinecone index '{pinecone_index_name}' does not exist in the available indexes")
                return False
        except Exception as e:
            logger.error(f"Error listing Pinecone indexes: {e}")
            return False
        
        # Connect to the specific index
        try:
            pc_index = pc.Index(pinecone_index_name)
            stats = pc_index.describe_index_stats()
            logger.info(f"Connected to Pinecone index '{pinecone_index_name}'")
            logger.info(f"Vector count: {stats.total_vector_count}")
            logger.info(f"Dimension: {stats.dimension}")
            return True
        except Exception as e:
            logger.error(f"Error connecting to index: {e}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to initialize Pinecone: {e}")
        return False

def get_embedding(text, client):
    """Get embedding from OpenAI API"""
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        raise

@app.route('/', methods=['GET'])
def home():
    """Root endpoint"""
    return jsonify({
        "service": "PitchPal Vector Service",
        "status": "running", 
        "endpoints": ["/health", "/embed", "/query", "/index", "/generate-email"]
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Check service health"""
    global pc_index
    
    openai_api_key = os.getenv("OPENAI_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    
    services_ok = pc_index is not None and openai_api_key is not None and pinecone_api_key is not None
    
    return jsonify({
        "status": "healthy" if services_ok else "degraded",
        "openai_configured": openai_api_key is not None,
        "pinecone_configured": pinecone_api_key is not None,
        "pinecone_connected": pc_index is not None,
        "environment": os.getenv("PINECONE_ENVIRONMENT"),
        "index": os.getenv("PINECONE_INDEX")
    })

@app.route('/embed', methods=['POST'])
def embed_text():
    """Generate embeddings for input text"""
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        return jsonify({"error": "OpenAI API key not configured"}), 500
    
    client = openai.OpenAI(api_key=openai_api_key)
    
    data = request.json
    if not data or 'text' not in data:
        return jsonify({"error": "Missing 'text' field in request"}), 400
    
    try:
        text = data['text']
        embedding = get_embedding(text, client)
        return jsonify({
            "embedding": embedding,
            "dimension": len(embedding)
        })
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/index', methods=['POST'])
def index_document():
    """Index a document in Pinecone"""
    global pc_index
    
    if pc_index is None:
        return jsonify({"error": "Pinecone not initialized"}), 503
    
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        return jsonify({"error": "OpenAI API key not configured"}), 500
    
    client = openai.OpenAI(api_key=openai_api_key)
    
    data = request.json
    if not data or 'text' not in data or 'metadata' not in data or 'id' not in data:
        return jsonify({"error": "Request must include 'text', 'metadata', and 'id'"}), 400
    
    try:
        # Generate embedding
        embedding = get_embedding(data['text'], client)
        
        # Index in Pinecone
        namespace = data.get('namespace', 'default')
        pc_index.upsert(
            vectors=[{
                'id': data['id'],
                'values': embedding,
                'metadata': data['metadata']
            }],
            namespace=namespace
        )
        
        return jsonify({
            "success": True,
            "id": data['id'],
            "namespace": namespace
        })
    except Exception as e:
        logger.error(f"Error indexing document: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/query', methods=['POST'])
def query_vectors():
    """Query vectors based on embedding similarity"""
    global pc_index
    
    if pc_index is None:
        return jsonify({"error": "Pinecone not initialized"}), 503
    
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        return jsonify({"error": "OpenAI API key not configured"}), 500
    
    client = openai.OpenAI(api_key=openai_api_key)
    
    data = request.json
    if not data or 'query' not in data:
        return jsonify({"error": "Missing 'query' field in request"}), 400
    
    try:
        # Generate query embedding
        query_embedding = get_embedding(data['query'], client)
        
        # Query Pinecone
        namespace = data.get('namespace', 'default')
        top_k = int(data.get('top_k', 5))
        
        results = pc_index.query(
            vector=query_embedding,
            top_k=top_k,
            namespace=namespace,
            include_metadata=True
        )
        
        # Convert to serializable format
        return jsonify({
            "matches": [
                {
                    "id": match.id,
                    "score": match.score,
                    "metadata": match.metadata
                }
                for match in results.matches
            ]
        })
    except Exception as e:
        logger.error(f"Error querying vectors: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/generate-email', methods=['POST'])
def generate_email():
    """Generate a personalized cold email using OpenAI"""
    global pc_index
    
    if pc_index is None:
        return jsonify({"error": "Pinecone not initialized"}), 503
    
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        return jsonify({"error": "OpenAI API key not configured"}), 500
    
    client = openai.OpenAI(api_key=openai_api_key)
    
    data = request.json
    if not data or 'recipient_info' not in data:
        return jsonify({"error": "Missing required fields in request"}), 400
    
    try:
        # Extract data
        recipient_info = data['recipient_info']
        company_info = data.get('company_info', {})
        sender_info = data.get('sender_info', {})
        
        # Generate embedding for recipient
        recipient_embedding = get_embedding(str(recipient_info), client)
        
        # Query similar contexts
        results = pc_index.query(
            vector=recipient_embedding,
            top_k=5,
            include_metadata=True
        )
        
        # Construct prompt with context
        context_items = []
        for match in results.matches:
            if match.metadata:
                context_items.append(str(match.metadata))
        
        context_text = "\n".join(context_items)
        
        # Create the prompt
        prompt = f"""
        You are an expert cold email copywriter. Write a personalized cold email using the following information:
        
        RECIPIENT INFORMATION:
        {recipient_info}
        
        COMPANY INFORMATION:
        {company_info}
        
        SENDER INFORMATION:
        {sender_info}
        
        SIMILAR CONTEXTS FROM DATABASE:
        {context_text}
        
        Create a compelling, personalized email with:
        1. A personalized subject line based on the recipient's background
        2. A hook that demonstrates you've done your research
        3. A clear value proposition relevant to their needs
        4. A specific call to action
        
        Format your response as JSON with keys for "subject" and "body".
        """
        
        # Generate email using OpenAI
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "You are an expert cold email copywriter."},
                     {"role": "user", "content": prompt}],
            temperature=0.7
        )
        
        # Extract content
        content = response.choices[0].message.content
        
        try:
            # Simple extraction of JSON-like content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
                
            # Format into a proper structure
            try:
                email_data = json.loads(content)
            except:
                # Fallback if the LLM doesn't provide proper JSON
                if "subject" in content.lower() and "body" in content.lower():
                    # Try simple parsing
                    lines = content.split("\n")
                    subject = ""
                    body = []
                    in_body = False
                    
                    for line in lines:
                        if "subject" in line.lower() and ":" in line:
                            subject = line.split(":", 1)[1].strip()
                        elif "body" in line.lower() and ":" in line:
                            in_body = True
                        elif in_body:
                            body.append(line)
                    
                    email_data = {
                        "subject": subject,
                        "body": "\n".join(body).strip()
                    }
                else:
                    email_data = {
                        "subject": "Personalized Outreach",
                        "body": content
                    }
            
            return jsonify({
                "email": email_data,
                "personalization_data": {
                    "recipient": recipient_info,
                    "company": company_info,
                    "similar_contexts_count": len(context_items)
                }
            })
        except Exception as parsing_error:
            logger.error(f"Error parsing OpenAI response: {parsing_error}")
            return jsonify({
                "email": {
                    "subject": "Personalized Outreach",
                    "body": content
                },
                "personalization_data": {
                    "recipient": recipient_info,
                    "company": company_info,
                    "similar_contexts_count": len(context_items)
                },
                "parsing_error": str(parsing_error)
            })
            
    except Exception as e:
        logger.error(f"Error generating email: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Initialize services
    if init_services():
        # Start the Flask server
        port = int(os.getenv("PORT", 8001))
        host = os.getenv("HOST", "0.0.0.0")
        logger.info(f"Starting PitchPal Vector Service on {host}:{port}")
        app.run(host=host, port=port, debug=True)
    else:
        logger.error("Failed to initialize services - exiting")
