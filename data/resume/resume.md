# Yutian (Charlie) Yang

**Data Scientist & ML Engineer**

- Phone: 501-368-9640
- Email: charlieyang990314@gmail.com
- Portfolio: charlieyang1557.github.io/aboutme/
- LinkedIn: linkedin.com/in/yutianyang

## Education

**University of California, Davis** — Master of Science in Statistics
College of Letters and Science | September 2021 – June 2023

**University of California, Davis** — Bachelor of Science in Statistics and Economics
College of Letters and Science | September 2017 – June 2021

- Relevant Coursework: Advanced Statistical Computing, Algorithm Design & Analysis, Econometrics, Optimization of Big Data Analytics, Statistical Machine Learning I, Statistical Methods of Machine Learning, Time Series Analysis, Probability Theory.

## Skills and Tools

- Programming: SQL, Python, R
- Machine Learning: Scikit-learn, TensorFlow, Random Forests, SVM, CNN, Time Series
- Data Engineering: BigQuery, DBT, AWS, GCP, ETL, Schema Design
- BI & Analytics: Tableau, Power BI, Looker, Sigma, Google Analytics

## Work Experience — Onshape by PTC

**Data Science Intern** | Boston, MA | June 2025 – August 2025
Onshape by PTC is a cloud-based CAD platform.

- Built and evaluated multiple unsupervised anomaly detection models (Prophet + Isolation Forest, Merlion, LSTM-AE) for API telemetry monitoring; selected and deployed Prophet-based pipeline for best performance, interpretability, and ease of deployment.
- Applied AWS Bedrock's Titan Embeddings and Claude 3.5 Sonnet for clustering, naming, and sentiment analysis of NPS feedback, extracting key themes to support product insights from unstructured text.
- Conducted keyword analysis on AI Advisor open-field queries to assess reference URL coverage, uncover content gaps, and improve user-facing query resolution.
- Developed Looker dashboards to track anomaly alerts and AI-driven insights, enhancing visibility across teams and automating Slack-based reporting to reduce manual monitoring efforts.

## Work Experience — Pinecone

**Data Science Intern** | New York, NY | June 2024 – August 2024
Pinecone is an AI vector database unicorn.

- Designed and built the Book of Business and Account 360 dashboards using SQL and Sigma, improving sales operations by 15%. Implemented Row-Level Security (RLS) for tailored views, and documented processes in Notion, reducing onboarding time by 30% and ensuring consistent use across teams.
- Developed the "dim_assistants" schema and implemented it in the pipeline using BigQuery and DBT. Created the Pinecone Assistant dashboard using SQL and Sigma, enabling comprehensive tracking of metrics. Facilitated cross-team collaboration, leading to a 25% increase in product insights.
- Conducted churn analysis using Python and Random Forests, identifying 5 key metrics and setting up alerts, reducing churn by 10%. Overcame data limitations and improved data collection, projected to boost accuracy by 20%.

## Work Experience — Allschool

**Data Analyst Intern** | San Mateo, CA | June 2022 – August 2022
Allschool is an EdTech platform.

- Enhanced impression targeting strategies and boosted customer engagement through A/B testing and segmentation analysis of user traffic and revenue across regional and platform data utilizing Google Analytics.
- Evaluated user behavior across multiple advertising channels, leading to a 15% reduction in project budget and an increase in daily active users employing SQL and BI tools.
- Designed a real-time web scraper with Python and Selenium, accelerating the class selection process by 50%.
- Developed a key metrics dashboard for active users, daily traffic, and revenue, improving business visibility and supporting data-driven decision making using Google Looker Studio.

## Work Experience — UC Davis Department of Economics

**Research Assistant** | Davis, CA | July 2020 – September 2020
Data Analysis & Modeling with Professor Anujit Chakraborty.

- Analyzed behavioral trends in procrastination and present-biased behavior using a generalized linear model (GLM) and Logistic Regression.
- Enhanced the reliability of study results by employing Bootstrapping resampling techniques to expand the sample size to approximately 20,000 data points.
- Addressed multicollinearity among predictors and improved prediction accuracy of procrastination behavior variables with regularization using Lasso Regression.
- Facilitated industry application by uncovering procrastination patterns, offering insights for tech companies to develop user-centric products and services, potentially enhancing user satisfaction, retention, and success.

## Projects — Resume RAG Chatbot

[Live Demo](https://resume-chatbot-production.up.railway.app/) | Python, React, FastAPI, FAISS, Anthropic Claude API

- Built a RAG chatbot with hybrid retrieval (BM25 + FAISS), Claude LLM streaming, and query rewriting. Alpha sweep via LLM-as-judge found pure semantic retrieval achieved 5.0/5.0 relevance.
- Deployed full-stack app (React + FastAPI) with source attribution, dark/light mode, rate limiting, and Docker containerized one-command startup.

## Projects — BoardGameOracle: RAG-Based Board Game Rules Q&A System

[Live Demo](https://boardgame-rules-rag-production.up.railway.app/) | [GitHub](https://github.com/charlieyang1557/boardgame-rules-RAG) | Python, FastAPI, React, TypeScript, Tailwind CSS, Pinecone, Claude Sonnet/Haiku, OpenAI Embeddings, LlamaParse, sentence-transformers, rank-bm25, Docker

- Built a retrieval-augmented generation system that answers board game rules questions with citations from official rulebooks. Supports 4 games spanning simple to heavy complexity: Splendor, Catan, Speakeasy (Vital Lacerda), and Food Chain Magnate (Splotter).
- Hybrid retrieval: Pinecone dense vectors + client-side BM25 with Reciprocal Rank Fusion, cross-encoder reranking (ms-marco-MiniLM).
- 3-tier answer system: Tier 1 direct citation, Tier 2 chain-of-retrieval multi-hop (up to 3 hops), Tier 3 honest uncertainty with suggested interpretations.
- Two-step citation verification: string overlap + LLM entailment check. Zero hallucinations on 186-entry golden dataset across all 4 games.
- Config-driven chunking architecture: adding a new complex game is a config change (section patterns + chunking rules), not a code change.
- Query rewriting with game-specific terminology maps bridging colloquial player language to rulebook terms.
- React frontend with expandable source cards, tier badges, feedback collection. Deployed as single-service FastAPI + static files on Railway.
- Key results: Splendor 100% Tier 1 accuracy, Catan 93% Tier 1 accuracy, FCM Tier 2 multi-hop 90% accuracy (target was 65%), canary query ($125 income calculation requiring 3 rule sections + arithmetic) passes perfectly in all runs.

## Projects — Classification of Mushrooms

UC Davis - STA 221 Advanced Statistical Computing (Python)

- Developed Random Forest, Kernel SVM, and CNN models with grid search hyperparameter tuning and transfer learning using pre-trained ResNet50 to classify mushroom images as edible or poisonous.
