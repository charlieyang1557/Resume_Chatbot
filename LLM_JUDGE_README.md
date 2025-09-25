# 🤖 LLM Judge Testing Framework for Resume Chatbot

This framework allows you to use **ChatGPT Plus** as an LLM judge to evaluate your resume chatbot's performance without using OpenAI APIs.

## 🎯 What This Does

1. **Generates hiring manager questions** for data science/analytics roles
2. **Tests your resume chatbot** with all questions
3. **Collects responses** with sources and timing
4. **Formats everything** for ChatGPT Plus evaluation
5. **Provides scoring criteria** for consistent evaluation

## 🚀 Quick Start

### Option 1: Quick Test (5 questions)
```bash
python quick_test.py
```

### Option 2: Comprehensive Test (15 questions)
```bash
python test_llm_judge.py
```

## 📋 Prerequisites

1. **Resume chatbot running**: `uvicorn resume_chatbot.webapp:app --reload`
2. **ChatGPT Plus account** for evaluation
3. **Python 3.10+** with requests library

## 🔧 Setup

1. **Make sure your chatbot is running:**
   ```bash
   cd /Users/yutianyang/Documents/GitHub/Resume_Chatbot
   uvicorn resume_chatbot.webapp:app --reload
   ```

2. **Install dependencies (if needed):**
   ```bash
   pip install requests
   ```

3. **Run the test:**
   ```bash
   python test_llm_judge.py
   ```

## 📊 Test Questions Categories

The comprehensive test includes questions across:

- **Technical Skills** (3 questions)
  - Programming languages
  - ML frameworks
  - Cloud platforms

- **Experience** (3 questions)
  - Professional background
  - Specific projects
  - Years of experience

- **Education & Research** (2 questions)
  - Academic background
  - Research experience

- **Soft Skills** (2 questions)
  - Communication abilities
  - Leadership experience

- **Industry Knowledge** (1 question)
  - Domain expertise

- **Problem Solving** (1 question)
  - Technical challenges

- **Data Analysis** (1 question)
  - Analysis skills

- **Experimentation** (1 question)
  - A/B testing experience

- **Recent Work** (1 question)
  - Current focus areas

## 📈 Evaluation Criteria

ChatGPT will score each response on:

1. **Accuracy** (1-10): Is the information factually correct?
2. **Completeness** (1-10): Is there sufficient detail?
3. **Clarity** (1-10): Is the response well-structured?
4. **Relevance** (1-10): Does it answer the question?
5. **Professionalism** (1-10): Is the tone appropriate?

## 📁 Output Files

The test generates two files:

1. **`chatbot_test_results_TIMESTAMP.json`**: Raw test data
2. **`chatgpt_evaluation_prompt_TIMESTAMP.txt`**: Formatted prompt for ChatGPT

## 🎯 How to Use ChatGPT Plus

1. **Run the test** and get the evaluation prompt file
2. **Open the `.txt` file** and copy all content
3. **Go to ChatGPT Plus** and paste the content
4. **Ask ChatGPT** to evaluate the resume chatbot performance
5. **Get detailed scores** and feedback for each question

## 📝 Example Evaluation Prompt

```
# Resume Chatbot Evaluation - LLM Judge Assessment

You are an experienced hiring manager for data science and analytics roles...

### Question 1 - Technical Skills
**Hiring Manager Question:** What programming languages is Yutian proficient in?

**Expected Keywords:** SQL, Python, R, examples, projects, experience

**Chatbot Response:** 
According to the resume context, Yutian Yang is skilled in SQL, Python, and R...

**Sources Used:** 2 sources with scores: knowledge_graph.json (0.337), resume_example.md (0.119)

---
```

## 🔍 Sample Questions

### Technical Skills
- "What programming languages is Yutian proficient in, and can you provide specific examples?"
- "What machine learning frameworks has Yutian worked with?"
- "What cloud platforms and data engineering tools has Yutian used?"

### Experience
- "Can you walk me through Yutian's professional experience and internships?"
- "What specific projects has Yutian worked on, and what were the outcomes?"
- "How many years of relevant experience does Yutian have?"

### Education & Research
- "What is Yutian's educational background, and how does it relate to data science?"
- "Tell me about Yutian's research experience and any significant findings."

## 🎯 Expected Outcomes

After running this evaluation, you'll get:

1. **Individual question scores** (1-10 for each criterion)
2. **Overall assessment** of chatbot performance
3. **Key strengths** identified
4. **Areas for improvement** highlighted
5. **Hiring recommendation** based on responses
6. **Summary evaluation** of the chatbot

## 🔧 Customization

### Add Your Own Questions

Edit `test_llm_judge.py` and modify the `_generate_test_questions()` method:

```python
def _generate_test_questions(self) -> List[Dict[str, str]]:
    questions = [
        {
            "category": "Your Category",
            "question": "Your custom question here?",
            "expected_keywords": ["keyword1", "keyword2", "keyword3"]
        },
        # ... more questions
    ]
    return questions
```

### Change Evaluation Criteria

Modify the `format_for_chatgpt_evaluation()` method to adjust scoring criteria:

```python
evaluation_prompt = """# Resume Chatbot Evaluation - LLM Judge Assessment

You are an experienced hiring manager for data science and analytics roles. Please evaluate each response based on:

## Your Custom Criteria (Score 1-10):
1. **Custom Metric 1**: Your description
2. **Custom Metric 2**: Your description
...
```

## 🚨 Troubleshooting

### Chatbot Not Running
```
❌ Error: Resume chatbot is not running!
Please start the chatbot first:
   uvicorn resume_chatbot.webapp:app --reload
```

### Connection Issues
- Check if chatbot is running on `http://127.0.0.1:8000`
- Verify no firewall blocking the connection
- Try restarting the chatbot server

### Import Errors
```bash
pip install requests
```

## 📊 Sample Results

After running the test, you'll get files like:

```
chatbot_test_results_20241201_143022.json
chatgpt_evaluation_prompt_20241201_143022.txt
```

The JSON contains detailed results:
```json
{
  "question_id": 1,
  "category": "Technical Skills",
  "question": "What programming languages is Yutian proficient in?",
  "chatbot_answer": "According to the resume context...",
  "sources": [...],
  "response_time": 2.34,
  "timestamp": "2024-12-01T14:30:22"
}
```

## 🎉 Benefits

✅ **Free evaluation** using ChatGPT Plus (no API costs)  
✅ **Comprehensive testing** with hiring manager perspective  
✅ **Detailed scoring** across multiple criteria  
✅ **Source attribution** for transparency  
✅ **Easy to customize** questions and criteria  
✅ **Automated workflow** from testing to evaluation  

## 🔄 Continuous Improvement

Use this framework to:

1. **Test changes** to your chatbot
2. **Compare different LLM backends** (Simple vs Ollama)
3. **Improve prompts** based on evaluation feedback
4. **Add new resume data** and re-test
5. **Track performance** over time

This gives you a systematic way to evaluate and improve your resume chatbot's performance for real hiring scenarios!
