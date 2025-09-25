"""Enhanced prompt templates with source citation and safety constraints."""

from typing import List, Dict, Any, Optional
from .enhanced_data_loader import EnhancedDocument


class PromptBuilder:
    """Builder for creating structured prompts with safety constraints."""
    
    def __init__(self):
        self.system_prompts = {
            "recruiter": self._get_recruiter_system_prompt(),
            "hiring_manager": self._get_hiring_manager_system_prompt(),
            "technical": self._get_technical_system_prompt(),
            "general": self._get_general_system_prompt()
        }
    
    def _get_recruiter_system_prompt(self) -> str:
        """System prompt optimized for recruiters."""
        return """You are a helpful assistant that answers questions about a candidate's resume for recruitment purposes.

CRITICAL INSTRUCTIONS:
1. Answer ONLY using information from the provided resume context
2. Always cite your sources using [source:section] format
3. If information is not in the context, say "I don't have that information in the candidate's records"
4. NEVER provide salary information unless explicitly mentioned in the resume
5. NEVER provide contact information beyond what's in the resume
6. Use professional, concise language suitable for recruiters
7. Focus on skills, experience, and qualifications
8. If asked about sensitive information (salary, personal details), politely decline

RESPONSE FORMAT:
- Provide clear, concise answers
- Include relevant experience and skills
- Cite sources for all claims
- Use bullet points for multiple items when appropriate"""
    
    def _get_hiring_manager_system_prompt(self) -> str:
        """System prompt optimized for hiring managers."""
        return """You are an assistant helping hiring managers evaluate a candidate's qualifications.

CRITICAL INSTRUCTIONS:
1. Answer ONLY using information from the provided resume context
2. Always cite your sources using [source:section] format
3. If information is not in the context, say "I don't have that information in the candidate's records"
4. NEVER provide salary information unless explicitly mentioned in the resume
5. Focus on technical skills, project experience, and achievements
6. Provide specific examples when available
7. Highlight relevant experience for the role being considered

RESPONSE FORMAT:
- Provide detailed, professional answers
- Include specific examples and achievements
- Cite sources for all claims
- Structure information logically"""
    
    def _get_technical_system_prompt(self) -> str:
        """System prompt for technical evaluations."""
        return """You are a technical assistant helping evaluate a candidate's technical qualifications.

CRITICAL INSTRUCTIONS:
1. Answer ONLY using information from the provided resume context
2. Always cite your sources using [source:section] format
3. If information is not in the context, say "I don't have that information in the candidate's records"
4. Focus on technical skills, tools, frameworks, and methodologies
5. Provide specific examples of technical work
6. Highlight relevant technical achievements

RESPONSE FORMAT:
- Provide technical details and specifics
- Include tool/framework names and versions when available
- Cite sources for all claims
- Use technical terminology appropriately"""
    
    def _get_general_system_prompt(self) -> str:
        """General system prompt."""
        return """You are a helpful assistant that answers questions about a candidate's resume.

CRITICAL INSTRUCTIONS:
1. Answer ONLY using information from the provided resume context
2. Always cite your sources using [source:section] format
3. If information is not in the context, say "I don't have that information in the candidate's records"
4. NEVER provide salary information unless explicitly mentioned in the resume
5. NEVER provide personal contact information beyond what's in the resume
6. Use professional language
7. Be helpful but maintain privacy boundaries"""
    
    def _format_context(self, documents: List[EnhancedDocument]) -> str:
        """Format retrieved documents into context string with citations."""
        if not documents:
            return "No relevant information found in the candidate's resume."
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get('source', 'unknown')
            title = doc.metadata.get('title', 'content')
            
            # Create citation reference
            citation = f"[source:{source}:{title}]"
            
            context_parts.append(f"{citation}\n{doc.content}")
        
        return "\n\n".join(context_parts)
    
    def _sanitize_query(self, query: str) -> str:
        """Sanitize query to remove potentially sensitive requests."""
        sensitive_keywords = [
            'salary', 'compensation', 'pay', 'wage', 'income',
            'ssn', 'social security', 'tax id',
            'address', 'phone number', 'email address',
            'birth date', 'age', 'personal information'
        ]
        
        query_lower = query.lower()
        for keyword in sensitive_keywords:
            if keyword in query_lower:
                return f"[SANITIZED] Query contains sensitive information request: {keyword}"
        
        return query
    
    def build_prompt(
        self,
        query: str,
        documents: List[EnhancedDocument],
        prompt_type: str = "recruiter",
        chat_history: Optional[List[tuple]] = None
    ) -> str:
        """Build complete prompt with context and safety constraints."""
        
        # Sanitize query
        sanitized_query = self._sanitize_query(query)
        
        # Get system prompt
        system_prompt = self.system_prompts.get(prompt_type, self.system_prompts["general"])
        
        # Format context
        context = self._format_context(documents)
        
        # Build prompt parts
        prompt_parts = [system_prompt]
        
        # Add chat history if provided
        if chat_history:
            prompt_parts.append("\nPREVIOUS CONVERSATION:")
            for user_msg, assistant_msg in chat_history[-3:]:  # Last 3 exchanges
                prompt_parts.append(f"User: {user_msg}")
                prompt_parts.append(f"Assistant: {assistant_msg}")
        
        # Add current context and query
        prompt_parts.extend([
            "\nCANDIDATE'S RESUME INFORMATION:",
            context,
            f"\nCURRENT QUESTION: {sanitized_query}",
            "\nPlease provide a helpful response based only on the information above."
        ])
        
        return "\n".join(prompt_parts)


class ResponseValidator:
    """Validate responses for safety and accuracy."""
    
    def __init__(self):
        self.forbidden_patterns = [
            r'\$\d+',  # Salary patterns
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN patterns
            r'\b\d{3}-\d{3}-\d{4}\b',  # Phone patterns
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email patterns
        ]
        
        self.required_citations = [
            r'\[source:[^\]]+\]',  # Citation pattern
        ]
    
    def validate_response(self, response: str, has_context: bool = True) -> Dict[str, Any]:
        """Validate response for safety and quality."""
        validation_result = {
            "is_safe": True,
            "has_citations": False,
            "warnings": [],
            "suggestions": []
        }
        
        # Check for forbidden patterns
        for pattern in self.forbidden_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                validation_result["is_safe"] = False
                validation_result["warnings"].append(f"Response contains potentially sensitive information: {pattern}")
        
        # Check for citations
        for pattern in self.required_citations:
            if re.search(pattern, response):
                validation_result["has_citations"] = True
                break
        
        if has_context and not validation_result["has_citations"]:
            validation_result["warnings"].append("Response should include source citations")
            validation_result["suggestions"].append("Add [source:filename:section] citations")
        
        # Check for hallucination indicators
        hallucination_indicators = [
            "I don't have that information",
            "not mentioned in the resume",
            "not specified in the records",
            "I don't have that in my records"
        ]
        
        has_disclaimer = any(indicator in response for indicator in hallucination_indicators)
        if not has_disclaimer and not has_context:
            validation_result["warnings"].append("Response may contain hallucinated information")
            validation_result["suggestions"].append("Add disclaimer when information is not available")
        
        return validation_result
    
    def sanitize_response(self, response: str) -> str:
        """Sanitize response by removing sensitive information."""
        sanitized = response
        
        # Remove salary information
        sanitized = re.sub(r'\$\d+(?:,\d{3})*(?:\.\d{2})?', '[SALARY REDACTED]', sanitized)
        
        # Remove phone numbers
        sanitized = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN REDACTED]', sanitized)
        sanitized = re.sub(r'\b\d{3}-\d{3}-\d{4}\b', '[PHONE REDACTED]', sanitized)
        
        # Remove email addresses
        sanitized = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL REDACTED]', sanitized)
        
        return sanitized


# Import re for regex operations
import re


class CitationExtractor:
    """Extract and format citations from responses."""
    
    @staticmethod
    def extract_citations(response: str) -> List[Dict[str, str]]:
        """Extract citations from response text."""
        citation_pattern = r'\[source:([^\]]+)\]'
        citations = []
        
        for match in re.finditer(citation_pattern, response):
            citation_text = match.group(1)
            parts = citation_text.split(':')
            
            citation = {
                "source": parts[0] if len(parts) > 0 else "unknown",
                "section": parts[1] if len(parts) > 1 else "content",
                "full_text": citation_text,
                "position": match.start()
            }
            citations.append(citation)
        
        return citations
    
    @staticmethod
    def format_citations_for_api(citations: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Format citations for API response."""
        formatted = []
        for citation in citations:
            formatted.append({
                "source": citation["source"],
                "section": citation["section"],
                "reference": citation["full_text"]
            })
        return formatted


# Global instances for easy import
prompt_builder = PromptBuilder()
response_validator = ResponseValidator()
citation_extractor = CitationExtractor()
