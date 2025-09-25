#!/usr/bin/env python3
"""
Quick Resume Chatbot Test - Generate sample questions and responses
"""

import requests
import json
from datetime import datetime


def test_sample_questions():
    """Test a few sample questions and format for ChatGPT evaluation."""
    
    sample_questions = [
        "What programming languages is Yutian proficient in?",
        "Tell me about Yutian's experience with machine learning projects.",
        "What internships has Yutian completed and what did he accomplish?",
        "How many years of experience does Yutian have in data science?",
        "What cloud platforms and data tools has Yutian used?"
    ]
    
    results = []
    print("🧪 Testing sample questions...")
    
    for i, question in enumerate(sample_questions, 1):
        print(f"  [{i}/{len(sample_questions)}] {question}")
        
        try:
            response = requests.post(
                "http://127.0.0.1:8000/api/chat",
                json={"message": question},
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            results.append({
                "question": question,
                "answer": data["answer"],
                "sources": data["sources"]
            })
            
        except Exception as e:
            print(f"    ❌ Error: {e}")
            results.append({
                "question": question,
                "answer": f"ERROR: {str(e)}",
                "sources": []
            })
    
    # Format for ChatGPT
    prompt = f"""# Resume Chatbot Quick Test Results

I tested a resume chatbot about a candidate named Yutian Yang. Please evaluate these responses as a hiring manager for data science roles.

**Evaluation Criteria (Score 1-10):**
- Accuracy: Is the information correct?
- Completeness: Is there enough detail?
- Clarity: Is it well-structured?
- Relevance: Does it answer the question?

**Test Results:**

"""
    
    for i, result in enumerate(results, 1):
        prompt += f"""
**Question {i}:** {result['question']}

**Chatbot Answer:** {result['answer']}

**Sources:** {len(result['sources'])} sources used

---
"""
    
    prompt += """
**Your Task:** Please score each response (1-10) and provide overall feedback on the chatbot's performance for hiring decisions.
"""
    
    # Save to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"quick_test_prompt_{timestamp}.txt"
    
    with open(filename, 'w') as f:
        f.write(prompt)
    
    print(f"\n✅ Quick test complete! Prompt saved to: {filename}")
    print("\n📋 Next steps:")
    print("1. Open the file and copy the content")
    print("2. Paste into ChatGPT Plus")
    print("3. Ask for evaluation of the resume chatbot")
    
    return filename


if __name__ == "__main__":
    test_sample_questions()
