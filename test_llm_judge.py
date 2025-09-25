#!/usr/bin/env python3
"""
LLM Judge Testing Framework for Resume Chatbot

This script generates hiring manager questions, collects chatbot responses,
and formats everything for ChatGPT Plus evaluation.
"""

import json
import time
import requests
from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime


class ResumeChatbotTester:
    """Test the resume chatbot and format results for LLM judge evaluation."""
    
    def __init__(self, chatbot_url: str = "http://127.0.0.1:8000"):
        self.chatbot_url = chatbot_url
        self.test_questions = self._generate_test_questions()
        
    def _generate_test_questions(self) -> List[Dict[str, str]]:
        """Generate comprehensive test questions from a hiring manager perspective."""
        
        questions = [
            # Technical Skills Questions
            {
                "category": "Technical Skills",
                "question": "What programming languages is Yutian proficient in, and can you provide specific examples of how he has used them?",
                "expected_keywords": ["SQL", "Python", "R", "examples", "projects", "experience"]
            },
            {
                "category": "Technical Skills", 
                "question": "What machine learning frameworks and libraries has Yutian worked with? Please provide specific details about his experience.",
                "expected_keywords": ["TensorFlow", "Scikit-learn", "Random Forest", "CNN", "specific projects"]
            },
            {
                "category": "Technical Skills",
                "question": "What experience does Yutian have with cloud platforms and data engineering tools?",
                "expected_keywords": ["AWS", "GCP", "BigQuery", "dbt", "data pipelines"]
            },
            
            # Experience & Background
            {
                "category": "Experience",
                "question": "Can you walk me through Yutian's professional experience and internships? What were his key responsibilities and achievements?",
                "expected_keywords": ["Onshape", "Pinecone", "Allschool", "responsibilities", "achievements", "results"]
            },
            {
                "category": "Experience",
                "question": "What specific projects has Yutian worked on, and what were the outcomes or impact of these projects?",
                "expected_keywords": ["anomaly detection", "churn analysis", "dashboard", "impact", "results", "metrics"]
            },
            {
                "category": "Experience",
                "question": "How many years of relevant experience does Yutian have in data science and analytics?",
                "expected_keywords": ["years", "experience", "internships", "academic", "timeline"]
            },
            
            # Education & Research
            {
                "category": "Education",
                "question": "What is Yutian's educational background, and how does it relate to data science roles?",
                "expected_keywords": ["Statistics", "UC Davis", "Master's", "Bachelor's", "relevant coursework"]
            },
            {
                "category": "Research",
                "question": "Tell me about Yutian's research experience and any publications or significant findings.",
                "expected_keywords": ["research assistant", "UC Davis", "procrastination", "behavioral", "Professor Chakraborty"]
            },
            
            # Soft Skills & Communication
            {
                "category": "Soft Skills",
                "question": "What evidence is there that Yutian can effectively communicate technical findings to stakeholders?",
                "expected_keywords": ["stakeholders", "communication", "dashboards", "presentations", "insights"]
            },
            {
                "category": "Leadership",
                "question": "Does Yutian have any leadership or mentoring experience?",
                "expected_keywords": ["mentor", "WiML", "organizer", "meetup", "leadership"]
            },
            
            # Industry Knowledge
            {
                "category": "Industry Knowledge",
                "question": "What industries has Yutian worked in, and what domain knowledge does he bring?",
                "expected_keywords": ["industries", "domains", "CAD", "vector databases", "education", "analytics"]
            },
            
            # Problem Solving
            {
                "category": "Problem Solving",
                "question": "Can you describe a challenging technical problem Yutian solved and how he approached it?",
                "expected_keywords": ["problem", "solution", "approach", "methodology", "challenge"]
            },
            
            # Data Analysis Skills
            {
                "category": "Data Analysis",
                "question": "What types of data analysis has Yutian performed, and what tools does he use for visualization and reporting?",
                "expected_keywords": ["analysis", "Tableau", "Looker", "Power BI", "visualization", "reporting"]
            },
            
            # A/B Testing & Experimentation
            {
                "category": "Experimentation",
                "question": "Does Yutian have experience with A/B testing and statistical experimentation?",
                "expected_keywords": ["A/B testing", "experimentation", "statistical", "segmentation", "hypothesis"]
            },
            
            # Recent Work & Projects
            {
                "category": "Recent Work",
                "question": "What has Yutian been working on most recently, and what are his current interests or focus areas?",
                "expected_keywords": ["recent", "current", "latest", "2024", "2025", "focus", "interests"]
            }
        ]
        
        return questions
    
    def test_chatbot_responses(self) -> List[Dict[str, Any]]:
        """Collect responses from the chatbot for all test questions."""
        
        results = []
        print(f"🧪 Testing {len(self.test_questions)} questions against resume chatbot...")
        
        for i, q in enumerate(self.test_questions, 1):
            print(f"  [{i}/{len(self.test_questions)}] Testing: {q['question'][:60]}...")
            
            try:
                # Make request to chatbot
                response = requests.post(
                    f"{self.chatbot_url}/api/chat",
                    json={"message": q["question"]},
                    timeout=30
                )
                response.raise_for_status()
                
                data = response.json()
                
                result = {
                    "question_id": i,
                    "category": q["category"],
                    "question": q["question"],
                    "expected_keywords": q["expected_keywords"],
                    "chatbot_answer": data["answer"],
                    "sources": data["sources"],
                    "response_time": response.elapsed.total_seconds(),
                    "timestamp": datetime.now().isoformat()
                }
                
                results.append(result)
                
                # Small delay to avoid overwhelming the server
                time.sleep(0.5)
                
            except Exception as e:
                print(f"    ❌ Error: {e}")
                result = {
                    "question_id": i,
                    "category": q["category"],
                    "question": q["question"],
                    "expected_keywords": q["expected_keywords"],
                    "chatbot_answer": f"ERROR: {str(e)}",
                    "sources": [],
                    "response_time": 0,
                    "timestamp": datetime.now().isoformat(),
                    "error": str(e)
                }
                results.append(result)
        
        print(f"✅ Completed testing {len(results)} questions")
        return results
    
    def format_for_chatgpt_evaluation(self, results: List[Dict[str, Any]]) -> str:
        """Format the test results for ChatGPT Plus evaluation."""
        
        evaluation_prompt = """# Resume Chatbot Evaluation - LLM Judge Assessment

You are an experienced hiring manager for data science and analytics roles. I've tested a resume chatbot that answers questions about a candidate named Yutian Yang. Please evaluate each response based on:

## Evaluation Criteria (Score 1-10 for each):
1. **Accuracy**: Is the information factually correct based on the resume?
2. **Completeness**: Does it provide sufficient detail for a hiring decision?
3. **Clarity**: Is the response clear and well-structured?
4. **Relevance**: Does it directly address the hiring manager's question?
5. **Professionalism**: Is the tone appropriate for a professional setting?

## Scoring Scale:
- 9-10: Excellent - Exceeds expectations
- 7-8: Good - Meets expectations well  
- 5-6: Satisfactory - Meets basic expectations
- 3-4: Below expectations - Significant gaps
- 1-2: Poor - Major issues

## Test Results:

"""
        
        for i, result in enumerate(results, 1):
            evaluation_prompt += f"""
### Question {i} - {result['category']}
**Hiring Manager Question:** {result['question']}

**Expected Keywords:** {', '.join(result['expected_keywords'])}

**Chatbot Response:** 
{result['chatbot_answer']}

**Sources Used:** {len(result['sources'])} sources with scores: {[f"{s['source']} ({s['score']:.3f})" for s in result['sources']]}

**Response Time:** {result['response_time']:.2f} seconds

---
"""
        
        evaluation_prompt += """
## Your Task:
Please provide:
1. **Individual Scores**: Score each response (1-10) for Accuracy, Completeness, Clarity, Relevance, and Professionalism
2. **Overall Assessment**: Overall score for each response (1-10)
3. **Key Strengths**: What the chatbot did well
4. **Areas for Improvement**: What could be better
5. **Hiring Recommendation**: Would you hire this candidate based on the chatbot's responses? (Yes/No/Maybe)
6. **Summary**: Overall evaluation of the chatbot's performance

Please format your response clearly with scores and detailed feedback for each question.
"""
        
        return evaluation_prompt
    
    def save_results(self, results: List[Dict[str, Any]], evaluation_prompt: str):
        """Save test results and evaluation prompt to files."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save raw results
        results_file = f"chatbot_test_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"📄 Raw results saved to: {results_file}")
        
        # Save evaluation prompt
        prompt_file = f"chatgpt_evaluation_prompt_{timestamp}.txt"
        with open(prompt_file, 'w') as f:
            f.write(evaluation_prompt)
        print(f"📄 Evaluation prompt saved to: {prompt_file}")
        
        return results_file, prompt_file
    
    def run_full_evaluation(self):
        """Run the complete evaluation process."""
        
        print("🚀 Starting Resume Chatbot LLM Judge Evaluation")
        print("=" * 60)
        
        # Test chatbot responses
        results = self.test_chatbot_responses()
        
        # Format for ChatGPT evaluation
        evaluation_prompt = self.format_for_chatgpt_evaluation(results)
        
        # Save results
        results_file, prompt_file = self.save_results(results, evaluation_prompt)
        
        print("\n" + "=" * 60)
        print("✅ EVALUATION COMPLETE!")
        print("\n📋 Next Steps:")
        print(f"1. Open the file: {prompt_file}")
        print("2. Copy the entire content")
        print("3. Paste it into ChatGPT Plus")
        print("4. Ask ChatGPT to evaluate the resume chatbot performance")
        print("\n💡 The evaluation prompt includes:")
        print(f"   • {len(results)} test questions from hiring manager perspective")
        print("   • Detailed scoring criteria (1-10 scale)")
        print("   • All chatbot responses with sources")
        print("   • Clear instructions for ChatGPT to evaluate")
        
        return results_file, prompt_file


def main():
    """Main function to run the evaluation."""
    
    # Check if chatbot is running
    try:
        response = requests.get("http://127.0.0.1:8000/", timeout=5)
        if response.status_code != 200:
            raise Exception("Chatbot not responding")
    except Exception as e:
        print("❌ Error: Resume chatbot is not running!")
        print("Please start the chatbot first:")
        print("   uvicorn resume_chatbot.webapp:app --reload")
        return
    
    # Run evaluation
    tester = ResumeChatbotTester()
    tester.run_full_evaluation()


if __name__ == "__main__":
    main()
