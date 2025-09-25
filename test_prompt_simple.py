#!/usr/bin/env python3
"""
Simple test for prompt builder without heavy dependencies.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class MockRecord:
    """Mock record for testing."""
    id: str
    source: str
    section: str
    date_range: str
    skills: List[str]
    text: str
    url: str = None


@dataclass
class MockSearchResult:
    """Mock search result for testing."""
    record: MockRecord
    score: float


class SimplePromptBuilder:
    """Simple prompt builder for testing."""
    
    def __init__(self, system_prompt: str):
        self.system_prompt = system_prompt
    
    def build_prompt(self, question: str, search_results: List[MockSearchResult]) -> str:
        """Build complete prompt."""
        # Build context
        context_parts = []
        for result in search_results:
            citation = f"[{result.record.source}:{result.record.section}]"
            context_parts.append(f"{citation}\n{result.record.text}")
        
        context = "\n\n".join(context_parts) if context_parts else "No relevant information found."
        
        # Build prompt
        prompt = f"""{self.system_prompt}

CONTEXT:
{context}

QUESTION: {question}

Please provide a helpful response based only on the information above."""
        
        return prompt
    
    def extract_citations(self, text: str) -> List[str]:
        """Extract citations from text."""
        citation_pattern = r'\[([^\]]+)\]'
        return re.findall(citation_pattern, text)
    
    def validate_response(self, response: str, has_context: bool = True) -> Dict[str, Any]:
        """Validate response for safety."""
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
        
        # Check for unsafe patterns
        unsafe_patterns = [r'\$\d+', r'\b\d{3}-\d{2}-\d{4}\b']
        for pattern in unsafe_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                validation["is_safe"] = False
                validation["warnings"].append(f"Contains sensitive info: {pattern}")
        
        # Check for disclaimers
        disclaimer_indicators = ["I don't have that information", "not mentioned in", "not in the records"]
        has_disclaimer = any(indicator in response for indicator in disclaimer_indicators)
        if not has_disclaimer and not has_context:
            validation["warnings"].append("May contain hallucinated information")
        
        return validation


def load_sample_data():
    """Load sample data for testing."""
    records = []
    
    # Mock data based on your corpus
    mock_data = [
        {
            "id": "resume#exp_onshape_overview",
            "source": "resume",
            "section": "Experience > PTC Onshape",
            "date_range": "06/2025–08/2025",
            "skills": ["Anomaly Detection", "Prophet", "Python", "AWS"],
            "text": "Built and evaluated unsupervised anomaly detection for API telemetry (Prophet + Isolation Forest, Merlion, LSTM-AE). Deployed a Prophet-based pipeline for performance, interpretability, and ease of deployment. Used AWS Bedrock Titan Embeddings and Claude 3.5 Sonnet for clustering, naming, and sentiment analysis of NPS feedback."
        },
        {
            "id": "website#projects_c1",
            "source": "website",
            "section": "Projects",
            "date_range": "",
            "skills": ["Python", "Prophet", "Anomaly Detection", "AWS"],
            "text": "Built a comprehensive anomaly detection pipeline for API telemetry data at PTC Onshape. Implemented multiple algorithms including Prophet for time series forecasting, Isolation Forest for outlier detection, and LSTM-AE for deep learning-based anomaly detection. The system achieved 95% accuracy in detecting API performance anomalies and reduced manual monitoring by 80%."
        }
    ]
    
    for data in mock_data:
        record = MockRecord(
            id=data["id"],
            source=data["source"],
            section=data["section"],
            date_range=data["date_range"],
            skills=data["skills"],
            text=data["text"]
        )
        records.append(record)
    
    return records


def main():
    """Test simple prompt builder."""
    print("🧪 Testing Simple Prompt Builder")
    print("=" * 40)
    
    # Load sample data
    records = load_sample_data()
    print(f"📚 Loaded {len(records)} sample records")
    
    # Create search results
    search_results = [MockSearchResult(record=r, score=0.95) for r in records]
    
    # Test different system prompts
    prompts = {
        "recruiter": """You are Charlie's resume assistant helping recruiters evaluate his qualifications.

CRITICAL INSTRUCTIONS:
1. Answer ONLY using information from the provided context
2. If information is not in the context, say "I don't have that information in Charlie's records"
3. Always cite sources using [source:section] format
4. Focus on skills, experience, and qualifications relevant to the role
5. Keep answers concise and professional (2-6 sentences)
6. Never provide salary information unless explicitly mentioned in context""",
        
        "technical": """You are a technical assistant helping evaluate Charlie's technical qualifications.

CRITICAL INSTRUCTIONS:
1. Answer ONLY using information from the provided context
2. If information is not in the context, say "I don't have that information in Charlie's records"
3. Always cite sources using [source:section] format
4. Focus on technical skills, tools, frameworks, and methodologies
5. Provide specific examples of technical work
6. Highlight relevant technical achievements"""
    }
    
    test_question = "What did Charlie work on at PTC Onshape?"
    
    for prompt_type, system_prompt in prompts.items():
        print(f"\n📝 Testing {prompt_type.upper()} prompt:")
        print("-" * 30)
        
        # Create prompt builder
        builder = SimplePromptBuilder(system_prompt)
        
        # Build prompt
        prompt = builder.build_prompt(test_question, search_results)
        
        print(f"Prompt length: {len(prompt)} characters")
        print(f"System prompt preview: {system_prompt[:100]}...")
        
        # Show context
        context_lines = prompt.split('\n')
        context_start = next(i for i, line in enumerate(context_lines) if line.startswith('CONTEXT:'))
        context_end = next(i for i, line in enumerate(context_lines) if line.startswith('QUESTION:'))
        context = '\n'.join(context_lines[context_start+1:context_end])
        print(f"Context preview: {context[:200]}...")
        
        # Test mock response
        mock_response = ("Based on Charlie's experience at PTC Onshape, he worked on building unsupervised anomaly detection systems for API telemetry [resume:Experience > PTC Onshape]. "
                        "He implemented multiple algorithms including Prophet for time series forecasting, Isolation Forest for outlier detection, and LSTM-AE for deep learning-based anomaly detection [website:Projects]. "
                        "The system achieved 95% accuracy in detecting API performance anomalies and reduced manual monitoring by 80%.")
        
        # Test citation extraction
        citations = builder.extract_citations(mock_response)
        print(f"Citations extracted: {citations}")
        
        # Test validation
        validation = builder.validate_response(mock_response, has_context=True)
        print(f"Validation: {'✅ Safe' if validation['is_safe'] else '⚠️ Issues'}")
        if validation['warnings']:
            print(f"Warnings: {validation['warnings']}")
    
    # Test full prompt generation
    print(f"\n📄 Full Prompt Example:")
    print("-" * 30)
    
    builder = SimplePromptBuilder(prompts["recruiter"])
    full_prompt = builder.build_prompt(test_question, search_results)
    print(full_prompt)
    
    print(f"\n✅ Simple prompt builder test completed successfully!")


if __name__ == "__main__":
    main()
