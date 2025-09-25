#!/usr/bin/env python3
"""
Enhanced Resume Chatbot Setup Script

This script sets up the enhanced resume chatbot with all dependencies
and creates sample data for testing.
"""

import subprocess
import sys
from pathlib import Path
import json


def run_command(command: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def install_dependencies():
    """Install required dependencies."""
    print("📦 Installing dependencies...")
    
    # Install basic requirements
    if not run_command(
        "pip install -r requirements-enhanced.txt",
        "Installing enhanced requirements"
    ):
        print("⚠️  Some dependencies failed to install. Continuing with available packages...")
    
    # Install the package in development mode
    run_command(
        "pip install -e .",
        "Installing resume chatbot package"
    )


def create_sample_data():
    """Create sample data for testing."""
    print("📝 Creating sample data...")
    
    # Create data directories
    data_dir = Path("data")
    resume_dir = data_dir / "resume"
    cache_dir = Path("cache")
    training_dir = Path("training_data")
    
    for dir_path in [data_dir, resume_dir, cache_dir, training_dir]:
        dir_path.mkdir(exist_ok=True)
    
    # Create sample resume if it doesn't exist
    resume_file = resume_dir / "sample_resume.md"
    if not resume_file.exists():
        sample_resume = """# John Doe - Data Scientist

## Contact Information
- Email: john.doe@email.com
- LinkedIn: linkedin.com/in/johndoe
- Location: San Francisco, CA

## Summary
Experienced data scientist with 5+ years of experience in machine learning, statistical analysis, and data engineering. Passionate about turning data into actionable insights and building scalable ML systems.

## Technical Skills
- **Programming Languages**: Python, R, SQL, JavaScript
- **Machine Learning**: Scikit-learn, TensorFlow, PyTorch, XGBoost
- **Data Tools**: Pandas, NumPy, Spark, Airflow, Docker
- **Cloud Platforms**: AWS, GCP, Azure
- **Databases**: PostgreSQL, MongoDB, Redis

## Professional Experience

### Senior Data Scientist - TechCorp (2021-Present)
- Built and deployed ML models that improved user engagement by 25%
- Led a team of 3 data scientists on customer churn prediction project
- Developed real-time recommendation system serving 1M+ users
- Technologies: Python, TensorFlow, AWS, Kubernetes

### Data Scientist - StartupXYZ (2019-2021)
- Created automated reporting dashboards reducing manual work by 80%
- Implemented A/B testing framework for product optimization
- Built ETL pipelines processing 10TB+ of data daily
- Technologies: Python, SQL, Airflow, PostgreSQL

### Data Analyst - FinanceCorp (2018-2019)
- Analyzed customer behavior patterns to identify growth opportunities
- Created predictive models for risk assessment
- Collaborated with business teams to define KPIs and metrics
- Technologies: R, SQL, Tableau, Excel

## Education
- **Master of Science in Data Science** - Stanford University (2018)
- **Bachelor of Science in Computer Science** - UC Berkeley (2016)

## Projects
- **Real-time Fraud Detection**: Built ML system detecting fraudulent transactions in real-time
- **Customer Segmentation**: Developed clustering models to identify customer segments
- **Recommendation Engine**: Created collaborative filtering system for e-commerce platform

## Certifications
- AWS Certified Machine Learning Specialty
- Google Cloud Professional Data Engineer
- Certified Analytics Professional (CAP)
"""
        
        resume_file.write_text(sample_resume)
        print(f"✅ Created sample resume: {resume_file}")
    
    # Create sample FAQ
    faq_file = resume_dir / "faq.json"
    if not faq_file.exists():
        sample_faq = {
            "questions": [
                {
                    "question": "What is John's strongest programming language?",
                    "answer": "Python is John's strongest programming language, as evidenced by his extensive use of it across multiple ML projects and frameworks.",
                    "category": "technical_skills"
                },
                {
                    "question": "How many years of experience does John have?",
                    "answer": "John has 5+ years of experience in data science and analytics, starting from 2018.",
                    "category": "experience"
                },
                {
                    "question": "What cloud platforms has John worked with?",
                    "answer": "John has experience with AWS, GCP (Google Cloud Platform), and Azure cloud platforms.",
                    "category": "technical_skills"
                }
            ]
        }
        
        faq_file.write_text(json.dumps(sample_faq, indent=2))
        print(f"✅ Created sample FAQ: {faq_file}")


def create_config_file():
    """Create configuration file."""
    config_file = Path("config.json")
    if not config_file.exists():
        config = {
            "chatbot": {
                "llm_backend": "ollama",
                "use_faiss": True,
                "use_cache": True,
                "chunk_size": 1000,
                "overlap": 200
            },
            "vector_store": {
                "model_name": "all-MiniLM-L6-v2",
                "index_type": "flat"
            },
            "cache": {
                "max_age_days": 30,
                "cache_path": "cache/resume_chatbot.db"
            },
            "api": {
                "host": "127.0.0.1",
                "port": 8000,
                "reload": True
            }
        }
        
        config_file.write_text(json.dumps(config, indent=2))
        print(f"✅ Created configuration file: {config_file}")


def test_installation():
    """Test the installation."""
    print("🧪 Testing installation...")
    
    try:
        # Test imports
        import fastapi
        import uvicorn
        print("✅ FastAPI and Uvicorn imported successfully")
        
        # Test FAISS import
        try:
            import faiss
            print("✅ FAISS imported successfully")
        except ImportError:
            print("⚠️  FAISS not available - will use fallback retriever")
        
        # Test sentence transformers
        try:
            import sentence_transformers
            print("✅ Sentence Transformers imported successfully")
        except ImportError:
            print("⚠️  Sentence Transformers not available - will use fallback")
        
        # Test resume chatbot import
        from resume_chatbot.enhanced_chatbot import EnhancedResumeChatbot
        print("✅ Enhanced Resume Chatbot imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Installation test failed: {e}")
        return False


def main():
    """Main setup function."""
    print("🚀 Setting up Enhanced Resume Chatbot...")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Install dependencies
    install_dependencies()
    
    # Create sample data
    create_sample_data()
    
    # Create config file
    create_config_file()
    
    # Test installation
    if test_installation():
        print("\n🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Start Ollama: brew services start ollama")
        print("2. Download Llama model: ollama pull llama3.2:3b")
        print("3. Run the enhanced webapp: uvicorn resume_chatbot.enhanced_webapp:app --reload")
        print("4. Open browser to: http://127.0.0.1:8000")
        print("\nFor more information, see the README.md file.")
    else:
        print("\n❌ Setup completed with errors. Please check the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
