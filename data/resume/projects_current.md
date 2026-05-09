# Current and GitHub Projects

## Public GitHub Profile

Yutian's public GitHub profile is https://github.com/charlieyang1557. Public repositories include portfolio, RAG systems, quantitative finance research, browser automation, and university coursework projects.

## BTC Alpha Pipeline

[GitHub](https://github.com/charlieyang1557/btc-alpha-pipeline) | Python, pandas, NumPy, PyArrow, Backtrader, SQLite, CCXT, Pydantic, pytest, Anthropic SDK

A quantitative research pipeline for discovering, testing, and validating BTC trading strategies. It is an active pre-MVP research system focused on auditable data, backtesting, evaluation, and research-discipline infrastructure rather than rushing to live trading.

- Built and validated BTC/USDT hourly OHLCV ingestion from Binance Vision and CCXT with Parquet market data storage, schema checks, and quality checks.
- Implemented Backtrader-based single-run and walk-forward backtesting, experiment registry tracking, baseline strategies, and manual trade-audit workflows.
- Added factor library, strategy DSL, hypothesis hashing, and AI-assisted hypothesis generation scaffolding.
- Current evaluation focus: regime-aware holdout evaluation, multiple-testing controls, Sharpe deflation screens, and clearer evidence gates to separate signal from overfit artifacts.
- Research conventions include UTC timestamps, bar N signal computation with bar N+1 execution, explicit cost modeling, reproducible hashes, and train/validation/holdout/test boundaries.
- Not positioned as production live trading, capital allocation automation, public performance claims, or a polished demo UI.

## Polymarket Automated Trading System

[GitHub](https://github.com/charlieyang1557/polymarket-arb) | Python, Polymarket US API, Kalshi API, SQLite, pytest

An automated trading system for Polymarket US and Kalshi prediction markets. It was built, deployed live, and systematically tested across four strategies from March 10 to April 7, 2026, with a strong emphasis on empirical falsification and risk controls.

- Tested pre-game passive market making, Pinnacle de-vig odds calibration, cross-market correlation, and WebSocket event trading.
- Found that Polymarket US sports markets were priced within about +/-0.8% of Pinnacle, leaving no systematic mispricing in the tested setup.
- Identified adverse selection as the main issue in passive market making, with a roughly 26% round-trip fill rate before targeted mitigations.
- Implemented live market-making components, paper trading, market scanning, cross-market logging, calibration analysis, and shared market-making engine modules.
- Designed a 4-layer risk system: per-order checks, inventory controls, session P&L stops, and system-level market lifecycle protections.
- Project report documents 36 live maker fills across 5 sessions, about 140 commits, and 639 unit tests across 36+ test files.

## Tock Restaurant Reservation Bot

[GitHub](https://github.com/charlieyang1557/tock-reservation-bot) | Python, Playwright, asyncio, Discord webhooks

An automated reservation sniper for restaurants on Tock that monitors for available slots, enters high-speed polling around release windows, books the first available slot, and sends Discord notifications.

- Built sniper mode with roughly 3-second polling during configurable release windows.
- Implemented release auto-detection by scraping restaurant release announcements and updating the sniper schedule.
- Added session pre-warming, cookie persistence, adaptive concurrency, concurrent booking races, dry-run mode, selector verification, and slot release tracking.
- Designed rich Discord notifications for confirmations, slot detections, sniper activations, booking failures, and dry-run would-have-booked events.
- Built a maintainable project structure around config, selectors, browser/login handling, availability checking, checkout booking, monitoring, notification, release detection, and tracker modules.

## OpenClaw Home Server Infrastructure

A self-hosted Mac Mini configured as an always-on production server running multiple automated systems with monitoring and self-recovery.

- Configured triple-layer sleep prevention with pmset, launchd caffeinate, and per-process caffeinate wrappers.
- Runs a Discord monitoring bot powered by GPT-5.4 through the OpenClaw gateway, enabling natural language server health queries and status checks.
- Uses PM2 startup, PM2 save, crontab scheduling with NVM environment sourcing, auto-restart on power failure, Wake on LAN, and SSH remote management.
- Currently manages the Tock reservation bot, Polymarket trading system, and OpenClaw Discord gateway.
- Full deployment verified through reboot testing so services recover without manual intervention.
- Tech: macOS, PM2, launchd, crontab, OpenClaw, Discord API, ChatGPT Plus OAuth, SSH.

## BoardGameOracle

[Live Demo](https://boardgame-rules-rag-production.up.railway.app/) | [GitHub](https://github.com/charlieyang1557/boardgame-rules-RAG) | Python, FastAPI, React, TypeScript, Tailwind CSS, Pinecone, Claude Sonnet/Haiku, OpenAI Embeddings, LlamaParse, sentence-transformers, rank-bm25, Docker

- Built a RAG-based Q&A system that answers board game rules questions with citations, multi-hop reasoning, and honest uncertainty handling.
- Supports Splendor, Catan, Speakeasy, and Food Chain Magnate.
- Uses query rewriting, Pinecone dense retrieval, BM25 sparse retrieval, Reciprocal Rank Fusion, local cross-encoder reranking, tier routing, and citation verification.
- Evaluation results from the public README: Splendor 91% accuracy, Catan 91%, Speakeasy 82%, Food Chain Magnate 79%, and Food Chain Magnate Tier 2 multi-hop 90%.
- Deployed as a single-service Docker container on Railway with FastAPI serving the React build.

## Resume RAG Chatbot

[Live Demo](https://resume-chatbot-production.up.railway.app/) | [GitHub](https://github.com/charlieyang1557/Resume_Chatbot) | Python, React, FastAPI, FAISS, Anthropic Claude API

- Built a RAG chatbot with BM25 and FAISS hybrid retrieval, query rewriting, Claude streaming responses, and source attribution.
- Uses Reciprocal Rank Fusion with an alpha parameter to control sparse versus dense weighting.
- Frontend includes markdown rendering, collapsible source cards, dark/light theme, and typing indicators.
- Deployed as a full-stack app with Docker and Railway.

## University Projects on GitHub

### STA 221 Mushroom Classification

[GitHub](https://github.com/charlieyang1557/STA-221-Mushroom-Classification) | Jupyter Notebook, Python, machine learning, CNN, transfer learning

- Classified mushroom images as edible or poisonous using a Kaggle dataset with over 3,000 images.
- Compared Random Forest, Kernel SVM, CNN, and ResNet50 transfer learning under limited-image constraints.
- Random Forest reached 0.648 test accuracy after hyperparameter optimization; Kernel SVM reached 0.656 test accuracy.
- The project concluded Random Forest was the most promising overall by accuracy, precision, recall, F1, and discriminatory power, while transfer learning needed further improvement.

### STA 160 Wheat Classification

[GitHub](https://github.com/charlieyang1557/STA-160-Wheat-Classification) | Statistical learning, Random Forest, LOOCV, multinomial logistic regression

- Analyzed UCI seed data for three wheat varieties: Kama, Rosa, and Canadian.
- Used visualization, correlation analysis, Leave-One-Out Cross Validation, grid search with Random Forest, and multiclass classification.
- Random Forest grid search reported 0.952381 training accuracy under the selected configuration.
- Compared multinomial logistic classification against One-vs-Rest approaches using area and perimeter as predictors.

### ECN 190 Information Cascades

[GitHub](https://github.com/charlieyang1557/ECN-190-Information-Cascades) | Behavioral economics, experiment design, information cascades

- Investigated information cascades through an urn-prediction experiment where participants had private signals and could observe prior public predictions.
- Designed the study to test whether matching early predictions cause later participants to ignore private information and follow the observed pattern.
- Reported 25 experimental rounds with 11 cascades forming, or 44% of rounds.
- Proposed an extension studying loss aversion by increasing penalties for incorrect guesses.

### STA 141B Covid Dashboard

[GitHub](https://github.com/charlieyang1557/STA-141B-Covid-Dashboard) | R, dashboarding, Covid-19 data visualization

- Built a Covid-19 status tracker for country-level confirmed cases, deaths, and recoveries over selected time periods.
- Included country comparison, global Covid-19 summary, status plots, and tables for selected countries.
- The README notes the original covid19api.com data source is no longer working.

### STA 237A Covid and Vaccination Policy

[GitHub](https://github.com/charlieyang1557/STA-237-Covid-and-vaccination-policy) | Time series, ARIMA, ARIMAX, policy analysis

- Assessed the effect of vaccination policy on Covid spread and whether policy effectiveness varied across countries.
- Modeled vaccination policy as dummies with increasing stringency and Covid growth rate as new cases per million.
- Controlled for testing rate, compared ARIMA and ARIMAX models, and used gain in variance explained as a proxy for policy effectiveness.
- Planned ANOVA comparisons across countries and internal/external validation using historical subsets and held-out realized data.
