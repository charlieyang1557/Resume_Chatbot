#!/usr/bin/env python3
"""
Enhanced LLM Judge Testing Framework for Resume Chatbot

This script tests the enhanced resume chatbot with comprehensive questions
and formats results for ChatGPT Plus evaluation.
"""

import json
import time
import requests
from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime


class EnhancedResumeChatbotTester:
    """Test the enhanced resume chatbot and format results for LLM judge evaluation."""
    
    def __init__(self, chatbot_url: str = "http://127.0.0.1:8000"):
        self.chatbot_url = chatbot_url
        self.test_questions = self._generate_comprehensive_questions()
        
    def _generate_comprehensive_questions(self) -> List[Dict[str, str]]:
        """Generate comprehensive test questions for different roles and contexts."""
        
        questions = [
            # Technical Skills - Programming Languages
            {
                "question": "What programming languages is Yutian proficient in?",
                "category": "technical_skills",
                "context": "recruiter",
                "expected_keywords": ["Python", "SQL", "R", "programming"]
            },
            {
                "question": "Which programming language is Yutian's strongest?",
                "category": "technical_skills", 
                "context": "technical",
                "expected_keywords": ["Python", "proficient", "experience"]
            },
            
            # Technical Skills - ML/AI
            {
                "question": "What machine learning frameworks and tools has Yutian used?",
                "category": "technical_skills",
                "context": "technical",
                "expected_keywords": ["machine learning", "ML", "frameworks", "tools"]
            },
            {
                "question": "Tell me about Yutian's experience with data science and analytics.",
                "category": "experience",
                "context": "hiring_manager", 
                "expected_keywords": ["data science", "analytics", "experience", "projects"]
            },
            
            # Work Experience
            {
                "question": "What companies has Yutian worked for and what were his roles?",
                "category": "work_experience",
                "context": "recruiter",
                "expected_keywords": ["companies", "roles", "experience", "work"]
            },
            {
                "question": "Describe Yutian's most recent work experience at Pinecone.",
                "category": "work_experience",
                "context": "hiring_manager",
                "expected_keywords": ["Pinecone", "recent", "work", "experience"]
            },
            {
                "question": "What internships has Yutian completed?",
                "category": "work_experience",
                "context": "recruiter",
                "expected_keywords": ["internships", "completed", "experience"]
            },
            
            # Education
            {
                "question": "What is Yutian's educational background?",
                "category": "education",
                "context": "recruiter",
                "expected_keywords": ["education", "degree", "university", "background"]
            },
            {
                "question": "Where did Yutian study and what degree did he earn?",
                "category": "education",
                "context": "hiring_manager",
                "expected_keywords": ["university", "degree", "study", "earned"]
            },
            
            # Projects and Achievements
            {
                "question": "What notable projects has Yutian worked on?",
                "category": "projects",
                "context": "technical",
                "expected_keywords": ["projects", "notable", "worked", "achievements"]
            },
            {
                "question": "Tell me about Yutian's achievements and accomplishments.",
                "category": "achievements",
                "context": "hiring_manager",
                "expected_keywords": ["achievements", "accomplishments", "success"]
            },
            
            # Skills and Tools
            {
                "question": "What cloud platforms and data tools has Yutian used?",
                "category": "technical_skills",
                "context": "technical",
                "expected_keywords": ["cloud", "platforms", "tools", "data"]
            },
            {
                "question": "What databases and data storage technologies is Yutian familiar with?",
                "category": "technical_skills",
                "context": "technical",
                "expected_keywords": ["databases", "storage", "technologies", "familiar"]
            },
            
            # Experience Level
            {
                "question": "How many years of experience does Yutian have in data science?",
                "category": "experience",
                "context": "recruiter",
                "expected_keywords": ["years", "experience", "data science", "time"]
            },
            {
                "question": "What is Yutian's level of experience with analytics and experimentation?",
                "category": "experience",
                "context": "hiring_manager",
                "expected_keywords": ["level", "experience", "analytics", "experimentation"]
            },
            
            # Specific Technical Areas
            {
                "question": "Has Yutian worked with A/B testing or experimentation platforms?",
                "category": "technical_skills",
                "context": "technical",
                "expected_keywords": ["A/B testing", "experimentation", "platforms"]
            },
            {
                "question": "What experience does Yutian have with data pipelines and ETL processes?",
                "category": "technical_skills",
                "context": "technical",
                "expected_keywords": ["pipelines", "ETL", "processes", "data"]
            },
            
            # Communication and Collaboration
            {
                "question": "How does Yutian communicate his findings to stakeholders?",
                "category": "soft_skills",
                "context": "hiring_manager",
                "expected_keywords": ["communicate", "findings", "stakeholders"]
            },
            {
                "question": "What experience does Yutian have working with cross-functional teams?",
                "category": "soft_skills",
                "context": "hiring_manager",
                "expected_keywords": ["cross-functional", "teams", "collaboration"]
            },
            
            # Problem-Solving
            {
                "question": "Can you give me examples of how Yutian has solved complex data problems?",
                "category": "problem_solving",
                "context": "technical",
                "expected_keywords": ["examples", "solved", "complex", "problems"]
            },
            
            # Future and Growth
            {
                "question": "What areas of data science is Yutian most interested in?",
                "category": "interests",
                "context": "hiring_manager",
                "expected_keywords": ["interested", "areas", "data science", "focus"]
            }
        ]
        
        return questions
    
    def test_chatbot(self) -> List[Dict[str, Any]]:
        """Test the chatbot with all questions and collect results."""
        results = []
        print(f"🧪 Testing Enhanced Resume Chatbot with {len(self.test_questions)} questions...")
        
        for i, question_data in enumerate(self.test_questions, 1):
            question = question_data["question"]
            category = question_data["category"]
            context = question_data["context"]
            expected_keywords = question_data["expected_keywords"]
            
            print(f"  [{i}/{len(self.test_questions)}] {question[:60]}...")
            
            try:
                # Test with different prompt types
                start_time = time.time()
                
                response = requests.post(
                    f"{self.chatbot_url}/api/chat",
                    json={
                        "message": question,
                        "prompt_type": context,
                        "top_k": 5
                    },
                    timeout=30
                )
                
                response_time = time.time() - start_time
                
                if response.status_code == 200:
                    data = response.json()
                    
                    result = {
                        "question": question,
                        "category": category,
                        "context": context,
                        "expected_keywords": expected_keywords,
                        "answer": data["answer"],
                        "sources": data["sources"],
                        "response_time": data["response_time"],
                        "model_used": data["model_used"],
                        "prompt_type": data["prompt_type"],
                        "cache_hit": data["cache_hit"],
                        "validation_result": data["validation_result"],
                        "citations": data["citations"],
                        "test_timestamp": datetime.now().isoformat(),
                        "success": True
                    }
                    
                    # Check if expected keywords are present
                    answer_lower = data["answer"].lower()
                    found_keywords = [kw for kw in expected_keywords if kw.lower() in answer_lower]
                    result["found_keywords"] = found_keywords
                    result["keyword_coverage"] = len(found_keywords) / len(expected_keywords) if expected_keywords else 0
                    
                else:
                    result = {
                        "question": question,
                        "category": category,
                        "context": context,
                        "expected_keywords": expected_keywords,
                        "error": f"HTTP {response.status_code}: {response.text}",
                        "test_timestamp": datetime.now().isoformat(),
                        "success": False
                    }
                
                results.append(result)
                
                # Small delay to avoid overwhelming the server
                time.sleep(0.5)
                
            except Exception as e:
                result = {
                    "question": question,
                    "category": category,
                    "context": context,
                    "expected_keywords": expected_keywords,
                    "error": str(e),
                    "test_timestamp": datetime.now().isoformat(),
                    "success": False
                }
                results.append(result)
                print(f"    ❌ Error: {e}")
        
        return results
    
    def generate_evaluation_prompt(self, results: List[Dict[str, Any]]) -> str:
        """Generate comprehensive evaluation prompt for ChatGPT Plus."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Calculate statistics
        total_questions = len(results)
        successful_responses = len([r for r in results if r.get("success", False)])
        avg_response_time = sum(r.get("response_time", 0) for r in results if r.get("success")) / max(successful_responses, 1)
        cache_hits = len([r for r in results if r.get("cache_hit", False)])
        
        # Group by category
        category_stats = {}
        for result in results:
            category = result["category"]
            if category not in category_stats:
                category_stats[category] = {"total": 0, "successful": 0, "avg_keyword_coverage": 0}
            
            category_stats[category]["total"] += 1
            if result.get("success", False):
                category_stats[category]["successful"] += 1
                if "keyword_coverage" in result:
                    category_stats[category]["avg_keyword_coverage"] += result["keyword_coverage"]
        
        for category in category_stats:
            if category_stats[category]["successful"] > 0:
                category_stats[category]["avg_keyword_coverage"] /= category_stats[category]["successful"]
        
        prompt = f"""# Enhanced Resume Chatbot Evaluation Report

## Test Summary
- **Total Questions**: {total_questions}
- **Successful Responses**: {successful_responses}
- **Success Rate**: {(successful_responses/total_questions*100):.1f}%
- **Average Response Time**: {avg_response_time:.2f} seconds
- **Cache Hit Rate**: {(cache_hits/successful_responses*100):.1f}% (of successful responses)
- **Test Date**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Category Performance
"""
        
        for category, stats in category_stats.items():
            prompt += f"- **{category.replace('_', ' ').title()}**: {stats['successful']}/{stats['total']} successful, {stats['avg_keyword_coverage']:.1%} keyword coverage\n"
        
        prompt += f"""

## Detailed Question Analysis

Please evaluate each question-response pair below and provide:
1. **Answer Quality Score** (1-10): How well does the answer address the question?
2. **Completeness Score** (1-10): How complete is the information provided?
3. **Accuracy Score** (1-10): How accurate is the information based on the resume?
4. **Source Citation Score** (1-10): How well are sources cited and referenced?
5. **Professional Tone Score** (1-10): How appropriate is the language for the context?
6. **Overall Score** (1-10): Overall assessment
7. **Strengths**: What the response did well
8. **Areas for Improvement**: What could be better
9. **Recommendation**: Hire/Maybe/No Hire based on this response

### Scoring Rubric:
- **9-10**: Excellent - Exceeds expectations, highly professional
- **7-8**: Good - Meets expectations, professional and complete
- **5-6**: Satisfactory - Adequate but could be better
- **3-4**: Poor - Below expectations, missing key information
- **1-2**: Unacceptable - Major issues, inaccurate or inappropriate

---

"""
        
        for i, result in enumerate(results, 1):
            if not result.get("success", False):
                prompt += f"""### Question {i}: ❌ FAILED
**Question**: {result['question']}
**Category**: {result['category']}
**Context**: {result['context']}
**Error**: {result.get('error', 'Unknown error')}

---
"""
                continue
            
            prompt += f"""### Question {i}: ✅ SUCCESS
**Question**: {result['question']}
**Category**: {result['category']}
**Context**: {result['context']}
**Expected Keywords**: {', '.join(result['expected_keywords'])}
**Found Keywords**: {', '.join(result.get('found_keywords', []))}
**Keyword Coverage**: {result.get('keyword_coverage', 0):.1%}

**Response**: {result['answer']}

**Technical Details**:
- Response Time: {result['response_time']:.2f}s
- Model Used: {result['model_used']}
- Cache Hit: {'Yes' if result['cache_hit'] else 'No'}
- Sources: {len(result['sources'])} documents
- Citations: {len(result['citations'])} citations

**Sources**:
"""
            
            for source in result['sources']:
                prompt += f"- {source['source']}: {source['title']}\n"
            
            if result['citations']:
                prompt += f"\n**Citations**:\n"
                for citation in result['citations']:
                    prompt += f"- {citation['full_text']}\n"
            
            prompt += f"""
**Validation Result**: {result['validation_result']}

**Evaluation** (Please fill in):
1. Answer Quality Score (1-10): ___
2. Completeness Score (1-10): ___
3. Accuracy Score (1-10): ___
4. Source Citation Score (1-10): ___
5. Professional Tone Score (1-10): ___
6. Overall Score (1-10): ___
7. Strengths: ___
8. Areas for Improvement: ___
9. Recommendation: ___

---
"""
        
        prompt += f"""

## Overall Assessment

Please provide an overall assessment of the Enhanced Resume Chatbot:

### Summary Scores
- **Average Answer Quality**: ___/10
- **Average Completeness**: ___/10  
- **Average Accuracy**: ___/10
- **Average Source Citation**: ___/10
- **Average Professional Tone**: ___/10
- **Overall Performance**: ___/10

### Key Strengths
1. ___
2. ___
3. ___

### Major Areas for Improvement
1. ___
2. ___
3. ___

### Recommendations for Production Use
1. ___
2. ___
3. ___

### Final Verdict
**Overall Recommendation**: ___ (Ready for Production / Needs Improvement / Not Ready)

**Confidence Level**: ___% (How confident are you in this assessment?)

**Additional Notes**: ___

---

*This evaluation was generated by the Enhanced Resume Chatbot Testing Framework*
*Test completed on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""
        
        return prompt
    
    def save_results(self, results: List[Dict[str, Any]], evaluation_prompt: str):
        """Save test results and evaluation prompt."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save raw results
        results_file = Path(f"enhanced_chatbot_test_results_{timestamp}.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Save evaluation prompt
        prompt_file = Path(f"enhanced_chatgpt_evaluation_prompt_{timestamp}.txt")
        with open(prompt_file, 'w', encoding='utf-8') as f:
            f.write(evaluation_prompt)
        
        print(f"📊 Results saved to: {results_file}")
        print(f"📝 Evaluation prompt saved to: {prompt_file}")
        
        return results_file, prompt_file


def main():
    """Main testing function."""
    print("🚀 Enhanced Resume Chatbot Testing Framework")
    print("=" * 60)
    
    # Check if chatbot is running
    try:
        response = requests.get("http://127.0.0.1:8000/health", timeout=5)
        if response.status_code != 200:
            print("❌ Chatbot not responding. Please start the enhanced webapp:")
            print("   uvicorn resume_chatbot.enhanced_webapp:app --reload")
            return
    except:
        print("❌ Cannot connect to chatbot. Please start the enhanced webapp:")
        print("   uvicorn resume_chatbot.enhanced_webapp:app --reload")
        return
    
    print("✅ Chatbot is running and ready for testing")
    
    # Initialize tester
    tester = EnhancedResumeChatbotTester()
    
    # Run tests
    print(f"\n🧪 Starting comprehensive test with {len(tester.test_questions)} questions...")
    results = tester.test_chatbot()
    
    # Generate evaluation prompt
    print("\n📝 Generating evaluation prompt...")
    evaluation_prompt = tester.generate_evaluation_prompt(results)
    
    # Save results
    print("\n💾 Saving results...")
    results_file, prompt_file = tester.save_results(results, evaluation_prompt)
    
    # Print summary
    successful = len([r for r in results if r.get("success", False)])
    total = len(results)
    
    print(f"\n📊 Test Summary:")
    print(f"   Total Questions: {total}")
    print(f"   Successful: {successful}")
    print(f"   Success Rate: {successful/total*100:.1f}%")
    
    if successful > 0:
        avg_time = sum(r.get("response_time", 0) for r in results if r.get("success")) / successful
        cache_hits = len([r for r in results if r.get("cache_hit", False)])
        print(f"   Average Response Time: {avg_time:.2f}s")
        print(f"   Cache Hit Rate: {cache_hits/successful*100:.1f}%")
    
    print(f"\n📋 Next Steps:")
    print(f"   1. Copy the content from: {prompt_file}")
    print(f"   2. Paste it into ChatGPT Plus")
    print(f"   3. Ask ChatGPT to evaluate the chatbot's performance")
    print(f"   4. Review the detailed assessment and recommendations")
    
    print(f"\n🎉 Enhanced testing completed successfully!")


if __name__ == "__main__":
    main()
