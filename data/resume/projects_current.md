# Current Projects

## Polymarket Automated Trading System

A fully automated market-making and trading system for CFTC-regulated prediction markets, originally built for Kalshi and ported to Polymarket US sports markets.

- Built a complete market-making engine in Python with OBI (Orderbook Imbalance) microprice estimation, continuous inventory skew (gamma=0.5c/contract), and dynamic spread adjustment based on volatility.
- Designed a 4-layer risk management system: Layer 1 fat-finger checks, Layer 2 inventory limits, Layer 3 session-level P&L stops (daily loss limit $20, max drawdown 10%), Layer 4 system-level circuit breakers.
- Implemented a platform-agnostic adapter pattern — normalized Polymarket's SDK responses to match the engine's internal format, allowing the same OBI/skew/risk logic to work across Kalshi and Polymarket without engine changes.
- Conducted a rigorous calibration study on Kalshi sports markets analyzing 10,000+ resolved markets, confirming zero tradeable mispricing — demonstrating quantitative validation discipline before committing capital.
- Built a diagnostic scanner framework using Gamma API and CLOB client that identified 47 viable Polymarket markets with average net spread of 3.6 cents and maker rebates.
- Explored three strategy tracks: (1) Multi-option rebalancing market making, (2) Systematic "No" bias using statistical analysis of historical resolution data (found 81.4% win rate in YES 0.60-0.70 range, p=0.008), (3) Leader-follower settlement-triggered cross-market arbitrage.
- Paper trading results on Kalshi: +151.9 cents realized P&L across 71+ fills in 11 sessions, with best single session +49.7 cents.
- Atomic file operations (tmp+os.rename with .processing pattern) for race-condition-free scanner-to-engine handoff.
- Infrastructure: PM2 process management, Discord webhook alerts, SQLite session logging, Ed25519 keypair authentication, daily crontab-triggered scanning with NVM environment loading.
- Tech: Python, polymarket-us SDK, Gamma API, CLOB API, FAISS, PM2, Discord API, SQLite.

## Tock Restaurant Reservation Bot

An automated reservation sniper bot that monitors high-demand restaurant availability on Tock and books tables the moment slots are released.

- Built with Python and Playwright for headless browser automation, handling the full booking flow: login, calendar navigation, slot detection, and checkout with saved payment.
- Implemented a multi-mode architecture: hourly polling for general availability scanning, plus a precision sniper mode that polls every 3 seconds during known release windows (e.g., Friday 8:00 PM PDT for Fuhuihua restaurant).
- Built robust slot detection with 6-selector fallback strategy to handle Tock's varying UI patterns across different restaurants (verified working on both Benu and Fuhuihua).
- Concurrent booking attempts across multiple target dates, stopping after first successful reservation.
- Session cookie persistence for fast re-authentication, avoiding repeated login flows.
- Discord webhook notifications for slot detection alerts and booking confirmations.
- Deployed on Mac Mini via PM2 with automatic restart on failure, running 24/7 in headless mode.
- Release window detection to identify when restaurants publish new availability.
- Tech: Python, Playwright, asyncio, PM2, Discord webhooks.

## OpenClaw Home Server Infrastructure

A self-hosted Mac Mini configured as an always-on production server running multiple automated systems 24/7 with full monitoring and self-recovery capabilities.

- Mac Mini (username: openclaw) configured with triple-layer sleep prevention: pmset system policy (sleep=0, disksleep=0), launchd caffeinate service with KeepAlive, and per-process caffeinate wrappers.
- Runs a Discord monitoring bot ("Lobster") powered by GPT-5.4 via ChatGPT Plus OAuth through the OpenClaw gateway, enabling natural language server health queries and status checks.
- PM2 for process management with pm2 startup + pm2 save for automatic service recovery across reboots.
- Crontab scheduling with proper NVM environment sourcing for Node.js-dependent tasks.
- Auto-restart on power failure (pmset autorestart=1), Wake on LAN (womp=1) for remote recovery.
- Currently manages: Tock reservation bot, Polymarket trading system, and OpenClaw Discord gateway.
- Full deployment verified through reboot testing — all services auto-recover without manual intervention.
- Tech: macOS, PM2, launchd, crontab, OpenClaw, Discord API, ChatGPT Plus OAuth, SSH remote management.
