#!/usr/bin/env python3
"""
Easy LLM Judge Evaluation Runner

This script provides a simple menu to run different types of evaluations.
"""

import os
import sys
import subprocess
from pathlib import Path


def check_chatbot_running():
    """Check if the resume chatbot is running."""
    try:
        import requests
        response = requests.get("http://127.0.0.1:8000/", timeout=5)
        return response.status_code == 200
    except:
        return False


def run_quick_test():
    """Run the quick test (5 questions)."""
    print("🚀 Running Quick Test (5 questions)...")
    subprocess.run([sys.executable, "quick_test.py"])


def run_comprehensive_test():
    """Run the comprehensive test (15 questions)."""
    print("🚀 Running Comprehensive Test (15 questions)...")
    subprocess.run([sys.executable, "test_llm_judge.py"])


def show_instructions():
    """Show instructions for using the evaluation."""
    print("""
📋 How to Use ChatGPT Plus as LLM Judge:

1. 📁 Find the generated .txt file (e.g., chatgpt_evaluation_prompt_YYYYMMDD_HHMMSS.txt)
2. 📖 Open the file and copy ALL content
3. 🌐 Go to ChatGPT Plus (chat.openai.com)
4. 📋 Paste the entire content into ChatGPT
5. 💬 Ask ChatGPT to evaluate the resume chatbot performance
6. 📊 Get detailed scores and feedback!

🎯 The evaluation includes:
   • Professional hiring manager questions
   • Detailed scoring criteria (1-10 scale)
   • All chatbot responses with sources
   • Clear evaluation instructions

✅ This gives you a comprehensive assessment of your resume chatbot's performance!
""")


def main():
    """Main menu for the evaluation runner."""
    
    print("🤖 Resume Chatbot LLM Judge Evaluation")
    print("=" * 50)
    
    # Check if chatbot is running
    if not check_chatbot_running():
        print("❌ Resume chatbot is not running!")
        print("\nPlease start the chatbot first:")
        print("   uvicorn resume_chatbot.webapp:app --reload")
        print("\nThen run this script again.")
        return
    
    print("✅ Resume chatbot is running!")
    print()
    
    while True:
        print("Choose an evaluation type:")
        print("1. 🚀 Quick Test (5 questions, ~2 minutes)")
        print("2. 📊 Comprehensive Test (15 questions, ~5 minutes)")
        print("3. 📋 Show instructions for ChatGPT evaluation")
        print("4. 🚪 Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            run_quick_test()
            print("\n✅ Quick test completed! Check the generated .txt file.")
            break
            
        elif choice == "2":
            run_comprehensive_test()
            print("\n✅ Comprehensive test completed! Check the generated .txt file.")
            break
            
        elif choice == "3":
            show_instructions()
            
        elif choice == "4":
            print("👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")
        
        print("\n" + "-" * 50 + "\n")


if __name__ == "__main__":
    main()
