"""
Prompt templates for RAG Resume Q&A bot.

Builds structured prompts for LLM with context and safety constraints.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PromptConfig:
    """Configuration for prompt generation."""
    
    system_prompt: str
    max_context_length: int = 4000
    include_scores: bool = False
    citation_format: str = "[source:section]"
    safety_constraints: bool = True


class PromptBuilder:
    """Builds structured prompts for LLM generation."""
    
    def __init__(self, config: Optional[PromptConfig] = None):
        self.config = config or self._get_default_config()
    
    def _get_default_config(self) -> PromptConfig:
        """Get default prompt configuration."""
        return PromptConfig(
            system_prompt=self._get_default_system_prompt(),
            max_context_length=4000,
            include_scores=False,
            citation_format="[source:section]",
            safety_constraints=True
        )
    
    def _get_default_system_prompt(self) -> str:
        """Get default system prompt with safety constraints."""
        return """You are Charlie's resume assistant. Answer questions using only the context provided below.

Remember: Charlie is also known as Yutian Yang—treat references to Charlie, Yutian, or Yang as the same person.

CRITICAL INSTRUCTIONS:
1. Answer ONLY using information from the provided context
2. If information is not in the context, say "I don't have that information in Charlie's records"
3. Always cite sources using [source:section] format
4. Keep answers concise (2-6 sentences)
5. Never hallucinate or make up information about jobs, dates, titles, or companies
6. If asked about salary or personal details not in context, politely decline
7. Use professional, helpful tone appropriate for recruiters and hiring managers

RESPONSE FORMAT:
- Provide clear, concise answers
- Include relevant source citations
- Focus on skills, experience, and achievements
- Use bullet points for multiple items when appropriate"""
    
    def build_prompt(
        self, 
        question: str, 
        search_results: List[Any],
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Build complete prompt with question and context.
        
        Args:
            question: User question
            search_results: Retrieved context documents
            chat_history: Optional conversation history
            
        Returns:
            Formatted prompt string
        """
        # Build context from search results
        context = self._build_context(search_results)
        
        # Build prompt parts
        prompt_parts = [self.config.system_prompt]
        
        # Add conversation history if provided
        if chat_history:
            prompt_parts.append("\nPREVIOUS CONVERSATION:")
            for exchange in chat_history[-3:]:  # Last 3 exchanges
                prompt_parts.append(f"User: {exchange.get('user', '')}")
                prompt_parts.append(f"Assistant: {exchange.get('assistant', '')}")
        
        # Add current context and question
        prompt_parts.extend([
            "\nCONTEXT:",
            context,
            f"\nQUESTION: {question}",
            "\nPlease provide a helpful response based only on the information above."
        ])
        
        # Join and truncate if too long
        full_prompt = "\n".join(prompt_parts)
        if len(full_prompt) > self.config.max_context_length:
            logger.warning(f"Prompt too long ({len(full_prompt)} chars), truncating...")
            full_prompt = full_prompt[:self.config.max_context_length] + "..."
        
        return full_prompt
    
    def _build_context(self, search_results: List[Any]) -> str:
        """Build context string from search results."""
        if not search_results:
            return "No relevant information found in Charlie's records."
        
        context_parts = []
        for result in search_results:
            # Create citation
            citation = f"[{result.record.source}:{result.record.section}]"
            
            # Build context part
            context_part = f"{citation}\n{result.record.text}"
            
            # Add score if requested
            if self.config.include_scores:
                context_part += f"\n(Relevance: {result.score:.3f})"
            
            context_parts.append(context_part)
        
        return "\n\n".join(context_parts)
    
    def build_quick_prompt(self, question: str, context_text: str) -> str:
        """
        Build a quick prompt with pre-formatted context.
        
        Args:
            question: User question
            context_text: Pre-formatted context string
            
        Returns:
            Formatted prompt string
        """
        return f"""{self.config.system_prompt}

CONTEXT:
{context_text}

QUESTION: {question}

Please provide a helpful response based only on the information above."""
    
    def extract_citations(self, text: str) -> List[str]:
        """
        Extract citations from text.
        
        Args:
            text: Text to extract citations from
            
        Returns:
            List of citation strings
        """
        import re
        citation_pattern = r'\[([^\]]+)\]'
        return re.findall(citation_pattern, text)
    
    def validate_response(self, response: str, has_context: bool = True) -> Dict[str, Any]:
        """
        Validate response for safety and quality.
        
        Args:
            response: LLM response to validate
            has_context: Whether context was provided
            
        Returns:
            Validation result dictionary
        """
        validation = {
            "is_safe": True,
            "has_citations": False,
            "warnings": [],
            "suggestions": []
        }
        
        # Check for citations
        citations = self.extract_citations(response)
        if citations:
            validation["has_citations"] = True
        
        # Check for safety issues
        unsafe_patterns = [
            r'\$\d+',  # Salary patterns
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN patterns
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email patterns
        ]
        
        for pattern in unsafe_patterns:
            import re
            if re.search(pattern, response, re.IGNORECASE):
                validation["is_safe"] = False
                validation["warnings"].append(f"Response contains potentially sensitive information: {pattern}")
        
        # Check for appropriate disclaimers
        disclaimer_indicators = [
            "I don't have that information",
            "not mentioned in",
            "not specified in",
            "not in the records"
        ]
        
        has_disclaimer = any(indicator in response for indicator in disclaimer_indicators)
        if not has_disclaimer and not has_context:
            validation["warnings"].append("Response may contain hallucinated information")
            validation["suggestions"].append("Add disclaimer when information is not available")
        
        # Check for required citations
        if has_context and not validation["has_citations"]:
            validation["warnings"].append("Response should include source citations")
            validation["suggestions"].append("Add [source:section] citations")
        
        return validation


class PromptTemplates:
    """Collection of prompt templates for different use cases."""
    
    @staticmethod
    def recruiter_prompt() -> str:
        """Prompt optimized for recruiters."""
        return """You are Charlie's resume assistant helping recruiters evaluate his qualifications.

Remember: Charlie is also known as Yutian Yang—treat references to Charlie, Yutian, or Yang as the same person.

CRITICAL INSTRUCTIONS:
1. Answer ONLY using information from the provided context
2. If information is not in the context, say "I don't have that information in Charlie's records"
3. Always cite sources using [source:section] format
4. Focus on skills, experience, and qualifications relevant to the role
5. Keep answers concise and professional (2-6 sentences)
6. Never provide salary information unless explicitly mentioned in context
7. Never provide personal contact information beyond what's in context

RESPONSE FORMAT:
- Provide clear, concise answers suitable for recruiters
- Include relevant experience and skills
- Cite sources for all claims
- Use bullet points for multiple items when appropriate"""
    
    @staticmethod
    def hiring_manager_prompt() -> str:
        """Prompt optimized for hiring managers."""
        return """You are an assistant helping hiring managers evaluate Charlie's qualifications.

Remember: Charlie is also known as Yutian Yang—treat references to Charlie, Yutian, or Yang as the same person.

CRITICAL INSTRUCTIONS:
1. Answer ONLY using information from the provided context
2. If information is not in the context, say "I don't have that information in Charlie's records"
3. Always cite sources using [source:section] format
4. Focus on technical skills, project experience, and achievements
5. Provide specific examples when available
6. Keep answers detailed but concise (3-6 sentences)
7. Highlight relevant experience for the role being considered

RESPONSE FORMAT:
- Provide detailed, professional answers
- Include specific examples and achievements
- Cite sources for all claims
- Structure information logically"""
    
    @staticmethod
    def technical_prompt() -> str:
        """Prompt for technical evaluations."""
        return """You are a technical assistant helping evaluate Charlie's technical qualifications.

Remember: Charlie is also known as Yutian Yang—treat references to Charlie, Yutian, or Yang as the same person.

CRITICAL INSTRUCTIONS:
1. Answer ONLY using information from the provided context
2. If information is not in the context, say "I don't have that information in Charlie's records"
3. Always cite sources using [source:section] format
4. Focus on technical skills, tools, frameworks, and methodologies
5. Provide specific examples of technical work
6. Highlight relevant technical achievements

RESPONSE FORMAT:
- Provide technical details and specifics
- Include tool/framework names and versions when available
- Cite sources for all claims
- Use technical terminology appropriately"""
    
    @staticmethod
    def general_prompt() -> str:
        """General purpose prompt."""
        return """You are a helpful assistant that answers questions about Charlie's resume.

Remember: Charlie is also known as Yutian Yang—treat references to Charlie, Yutian, or Yang as the same person.

CRITICAL INSTRUCTIONS:
1. Answer ONLY using information from the provided context
2. If information is not in the context, say "I don't have that information in Charlie's records"
3. Always cite sources using [source:section] format
4. Keep answers helpful and professional
5. Never provide salary information unless explicitly mentioned in context
6. Never provide personal contact information beyond what's in context

RESPONSE FORMAT:
- Provide clear, helpful answers
- Include relevant information
- Cite sources for all claims
- Be professional and concise"""


def main():
    """CLI test for prompt builder."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test prompt builder")
    parser.add_argument("--template", choices=["recruiter", "hiring_manager", "technical", "general"], 
                       default="recruiter", help="Prompt template to use")
    parser.add_argument("--question", default="What did Charlie work on at PTC Onshape?", 
                       help="Test question")
    parser.add_argument("--context", help="Custom context text")
    
    args = parser.parse_args()
    
    # Get template
    templates = PromptTemplates()
    if args.template == "recruiter":
        system_prompt = templates.recruiter_prompt()
    elif args.template == "hiring_manager":
        system_prompt = templates.hiring_manager_prompt()
    elif args.template == "technical":
        system_prompt = templates.technical_prompt()
    else:
        system_prompt = templates.general_prompt()
    
    # Create prompt builder
    config = PromptConfig(system_prompt=system_prompt)
    builder = PromptBuilder(config)
    
    # Build prompt
    if args.context:
        prompt = builder.build_quick_prompt(args.question, args.context)
    else:
        # Mock search results for testing
        from data_loader import CorpusRecord
        mock_record = CorpusRecord(
            id="test_id",
            source="resume",
            section="Experience",
            date_range="2024",
            skills=["Python", "SQL"],
            text="Charlie worked on anomaly detection systems at PTC Onshape using Python and machine learning."
        )
        from retriever import SearchResult
        mock_results = [SearchResult(record=mock_record, score=0.95)]
        prompt = builder.build_prompt(args.question, mock_results)
    
    print(f"📝 Generated Prompt ({args.template} template):")
    print("=" * 60)
    print(prompt)
    print("=" * 60)
    
    # Test citation extraction
    test_text = "Charlie worked on [resume:Experience] projects at [website:Projects] companies."
    citations = builder.extract_citations(test_text)
    print(f"\n🔍 Extracted citations: {citations}")


if __name__ == "__main__":
    main()
