"""
Prompt Manager - Handles LLM prompting for email generation
"""
import logging
import os
from typing import Dict, List, Optional
import openai

# Create logger
logger = logging.getLogger(__name__)

class PromptManager:
    """
    Manages prompts and interactions with LLM for email generation
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the prompt manager
        
        Args:
            api_key: OpenAI API key (defaults to environment variable)
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if self.api_key:
            openai.api_key = self.api_key
            logger.info("Prompt manager initialized with API key")
        else:
            logger.warning("No API key provided. LLM functionality will be limited.")
    
    async def generate_email_draft(
        self, 
        recipient_data: Dict, 
        user_goal: str,
        additional_context: Optional[str] = None,
        tone: str = "professional"
    ) -> Dict:
        """
        Generate a personalized email draft
        
        Args:
            recipient_data: Data about the recipient from the personalization engine
            user_goal: The user's goal for the email (e.g., request a coffee chat)
            additional_context: Any additional context provided by the user
            tone: The desired tone of the email
            
        Returns:
            Dictionary containing the generated email subject and body
        """
        try:
            # Build the prompt
            prompt = self._build_email_prompt(
                recipient_data=recipient_data,
                user_goal=user_goal,
                additional_context=additional_context,
                tone=tone
            )
            
            # Call the LLM
            response = await self._call_llm(prompt)
            
            # Parse the response
            email_parts = self._parse_email_response(response)
            
            return email_parts
        except Exception as e:
            logger.error(f"Error generating email draft: {e}")
            raise
    
    def _build_email_prompt(
        self, 
        recipient_data: Dict, 
        user_goal: str,
        additional_context: Optional[str] = None,
        tone: str = "professional"
    ) -> str:
        """
        Build a prompt for email generation
        
        Args:
            recipient_data: Data about the recipient from the personalization engine
            user_goal: The user's goal for the email
            additional_context: Any additional context provided by the user
            tone: The desired tone of the email
            
        Returns:
            Formatted prompt string
        """
        # Extract recipient information
        recipient_name = recipient_data.get("name", "the recipient")
        recipient_role = recipient_data.get("role", "")
        recipient_company = recipient_data.get("company", "")
        
        # Extract personalization data
        linkedin_data = recipient_data.get("personalization_data", {}).get("linkedin", {})
        company_data = recipient_data.get("personalization_data", {}).get("company", {})
        news_data = recipient_data.get("personalization_data", {}).get("news", {})
        past_interactions = recipient_data.get("personalization_data", {}).get("past_interactions", {})
        
        # Format the prompt
        prompt = f"""
You are an expert email writer, creating personalized and effective emails. 
Write a compelling, personalized email to {recipient_name}"""
        
        if recipient_role and recipient_company:
            prompt += f", who is a {recipient_role} at {recipient_company}"
        elif recipient_role:
            prompt += f", who is a {recipient_role}"
        elif recipient_company:
            prompt += f", who works at {recipient_company}"
        
        prompt += f".\n\nThe goal of this email is to: {user_goal}\n\n"
        
        # Add personalization context
        prompt += "Here's some relevant information about the recipient:\n\n"
        
        if linkedin_data:
            prompt += "LinkedIn Information:\n"
            if "headline" in linkedin_data:
                prompt += f"- Headline: {linkedin_data['headline']}\n"
            if "experience" in linkedin_data:
                prompt += "- Recent Experience: "
                prompt += ", ".join([f"{exp['title']} at {exp['company']}" for exp in linkedin_data.get("experience", [])[:2]])
                prompt += "\n"
            if "skills" in linkedin_data:
                prompt += f"- Skills: {', '.join(linkedin_data.get('skills', [])[:5])}\n"
            if "recent_activity" in linkedin_data:
                prompt += f"- Recent Activity: {linkedin_data.get('recent_activity')}\n"
        
        if company_data:
            prompt += "\nCompany Information:\n"
            if "description" in company_data:
                prompt += f"- Description: {company_data.get('description')}\n"
            if "recent_news" in company_data:
                prompt += f"- Recent News: {company_data.get('recent_news')}\n"
        
        if news_data:
            prompt += "\nRecent News:\n"
            for article in news_data.get("articles", [])[:2]:
                prompt += f"- {article.get('title')}: {article.get('summary')}\n"
        
        if past_interactions:
            prompt += "\nPast Interactions:\n"
            if "emails" in past_interactions:
                prompt += f"- Previous Emails: {len(past_interactions.get('emails', []))} emails, "
                prompt += f"last on {past_interactions.get('emails', [{}])[0].get('date', 'unknown date')}\n"
            if "notes" in past_interactions:
                prompt += f"- Notes: {past_interactions.get('notes')}\n"
        
        if additional_context:
            prompt += f"\nAdditional Context:\n{additional_context}\n"
        
        prompt += f"\nTone: {tone}\n\n"
        
        prompt += """
Output Format:
Subject: [Your compelling subject line]

[Your personalized email body]

Write both a subject line and email body. Make the email concise, personalized, and effective. 
Include 1-2 specific details from the provided information to show you've done your research.
The email should be friendly but respectful, and have a clear call to action related to the goal.
Do not use generic placeholders or mention that you're using their data to personalize the email.
"""
        
        return prompt
    
    async def _call_llm(self, prompt: str) -> str:
        """
        Call the LLM with the given prompt
        
        Args:
            prompt: The formatted prompt
            
        Returns:
            LLM response text
        """
        try:
            if not self.api_key:
                # Mock response for testing without API key
                logger.warning("Using mock LLM response due to missing API key")
                return self._get_mock_response()
            
            # Call OpenAI API
            response = await openai.ChatCompletion.acreate(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert email writer who creates personalized, effective emails."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=800
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            raise
    
    def _parse_email_response(self, response: str) -> Dict:
        """
        Parse the LLM response into subject and body
        
        Args:
            response: Raw LLM response
            
        Returns:
            Dictionary with subject and body
        """
        try:
            # Split by the first newline after "Subject:" to separate subject and body
            parts = response.split("Subject:", 1)
            
            if len(parts) < 2:
                # If no "Subject:" is found, treat the whole thing as body with a default subject
                return {
                    "subject": "Connecting with you",
                    "body": response.strip()
                }
            
            # Process the subject and body
            remainder = parts[1].strip()
            subject_and_body = remainder.split("\n\n", 1)
            
            if len(subject_and_body) < 2:
                # If there's no clear separation between subject and body
                subject_and_body = remainder.split("\n", 1)
            
            subject = subject_and_body[0].strip()
            body = subject_and_body[1].strip() if len(subject_and_body) > 1 else ""
            
            return {
                "subject": subject,
                "body": body
            }
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            # Return a default structure if parsing fails
            return {
                "subject": "Connecting with you",
                "body": response.strip()
            }
    
    def _get_mock_response(self) -> str:
        """
        Get a mock LLM response for testing
        
        Returns:
            Mock response string
        """
        return """
Subject: Your AI Product Strategy Insights - Coffee Chat Request

Hi John,

I recently read your article on product management in AI startups and was impressed by your insights on balancing innovation with market needs. Your experience leading AI initiatives at Example Corp, especially after their recent $50M funding round, puts you at the forefront of this rapidly evolving field.

As someone deeply interested in AI product strategy, I'd love to hear more about your approach to prioritizing features in early-stage AI products. Would you be open to a 20-minute virtual coffee chat next week to share some of your experiences?

I'm flexible on Tuesday or Thursday afternoon if either works for your schedule.

Looking forward to potentially connecting,
[Your Name]
"""
