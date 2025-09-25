#!/usr/bin/env python3
"""
Test script for the FastAPI RAG Resume Q&A API.
"""

import requests
import time
import json
from typing import Dict, Any


def test_api_endpoints(base_url: str = "http://127.0.0.1:8000"):
    """Test all API endpoints."""
    
    print("🧪 Testing RAG Resume Q&A API")
    print("=" * 50)
    
    # Test health endpoint
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Health check passed: {health_data}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to API: {e}")
        print("💡 Make sure the API is running: python app.py")
        return False
    
    # Test stats endpoint
    print("\n📊 Testing stats endpoint...")
    try:
        response = requests.get(f"{base_url}/stats", timeout=5)
        if response.status_code == 200:
            stats_data = response.json()
            print(f"✅ Stats retrieved:")
            print(f"   Total records: {stats_data['total_records']}")
            print(f"   Sources: {stats_data['sources']}")
            print(f"   Top skills: {stats_data['top_skills'][:3]}")
        else:
            print(f"❌ Stats endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Stats endpoint error: {e}")
    
    # Test ask endpoint
    test_questions = [
        "What did Charlie work on at PTC Onshape?",
        "What skills does Charlie have?",
        "What is Charlie's educational background?",
        "Tell me about Charlie's experience at Pinecone."
    ]
    
    print(f"\n❓ Testing ask endpoint with {len(test_questions)} questions...")
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. Question: {question}")
        
        try:
            payload = {
                "question": question,
                "top_k": 3,
                "template": "recruiter"
            }
            
            response = requests.post(
                f"{base_url}/ask",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                answer_data = response.json()
                print(f"   ✅ Answer: {answer_data['answer'][:100]}...")
                print(f"   📚 Sources: {len(answer_data['sources'])} found")
                print(f"   ⏱️  Response time: {answer_data['response_time']:.2f}s")
                print(f"   🔒 Validation: {'✅ Safe' if answer_data['validation']['is_safe'] else '⚠️ Issues'}")
                
                # Show sources
                for source in answer_data['sources'][:2]:  # Show first 2 sources
                    print(f"      - {source['source']}:{source['section']} (score: {source['relevance_score']:.3f})")
                
            else:
                print(f"   ❌ Ask endpoint failed: {response.status_code}")
                print(f"   Error: {response.text}")
        
        except Exception as e:
            print(f"   ❌ Ask endpoint error: {e}")
    
    # Test different templates
    print(f"\n📝 Testing different prompt templates...")
    
    templates = ["recruiter", "hiring_manager", "technical", "general"]
    test_question = "What did Charlie work on at PTC Onshape?"
    
    for template in templates:
        print(f"\n   Template: {template}")
        
        try:
            payload = {
                "question": test_question,
                "top_k": 2,
                "template": template
            }
            
            response = requests.post(
                f"{base_url}/ask",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                answer_data = response.json()
                print(f"   ✅ Answer: {answer_data['answer'][:80]}...")
                print(f"   📏 Prompt length: {answer_data['prompt_length']} chars")
            else:
                print(f"   ❌ Template test failed: {response.status_code}")
        
        except Exception as e:
            print(f"   ❌ Template test error: {e}")
    
    print(f"\n✅ API testing completed!")
    return True


def test_api_validation():
    """Test API input validation."""
    
    print(f"\n🔒 Testing API validation...")
    
    base_url = "http://127.0.0.1:8000"
    
    # Test invalid inputs
    invalid_payloads = [
        {"question": "", "top_k": 3},  # Empty question
        {"question": "Test", "top_k": 0},  # Invalid top_k
        {"question": "Test", "top_k": 15},  # top_k too high
        {"question": "x" * 600, "top_k": 3},  # Question too long
    ]
    
    for i, payload in enumerate(invalid_payloads, 1):
        print(f"   {i}. Testing invalid payload: {payload}")
        
        try:
            response = requests.post(
                f"{base_url}/ask",
                json=payload,
                timeout=5
            )
            
            if response.status_code == 422:  # Validation error
                print(f"      ✅ Validation working: {response.status_code}")
            else:
                print(f"      ⚠️  Unexpected response: {response.status_code}")
        
        except Exception as e:
            print(f"      ❌ Validation test error: {e}")


def main():
    """Main test function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test RAG Resume Q&A API")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="API base URL")
    parser.add_argument("--validation", action="store_true", help="Test input validation")
    parser.add_argument("--wait", type=int, default=0, help="Wait N seconds before testing")
    
    args = parser.parse_args()
    
    if args.wait > 0:
        print(f"⏳ Waiting {args.wait} seconds for API to start...")
        time.sleep(args.wait)
    
    # Test main endpoints
    success = test_api_endpoints(args.url)
    
    if success and args.validation:
        test_api_validation()
    
    if success:
        print(f"\n🎉 All tests passed! Your RAG Resume Q&A API is working correctly.")
        print(f"📖 API Documentation: {args.url}/docs")
        print(f"🔍 Try it manually: curl -X POST {args.url}/ask -H 'Content-Type: application/json' -d '{{\"question\": \"What did Charlie work on?\"}}'")
    else:
        print(f"\n❌ Some tests failed. Check the API server and try again.")


if __name__ == "__main__":
    main()
