# Kraken Algo Trading Bot

Open source trading bot written in Python for Kraken Exchange. Implements both REST/Websocket API functionality [python-kraken-api-docs](https://docs.kraken.com/api/) using threading to manage concurrent connections.

## Disclaimer

**Use this trading bot at your own risk.** The author of this code takes no responsibility for any financial loss, damage, or issues that may occur as a result of using this bot. Trading cryptocurrencies carries a high level of risk, and you should only trade with money you can afford to lose. The author does not guarantee that this bot will be profitable or safe, and by using it, you acknowledge that you are fully responsible for your actions and decisions.

## Features
- Automated trading using Kraken's REST API and WebSocket API
- Real-time market data and order management via WebSocket
- Concurrent thread management for multiple connections
- Supports both market orders and limit orders

## Setup

### Environment

1. Install Required dependencies

2. In the root directory create a file `.env` of the form

```bash
API_KEY='<kraken-public-key>'
PRIVATE_KEY='<kraken-private-key>'
```

3. Launch kraken_bot/kraken_bot.py

### Prerequisites

#### Arch

```bash
sudo pacman -S python-dotenv python-websocket-client python-requests
```

##### Windows
- Python 3.x
- Install required dependencies:

```bash
pip install -r requirements.txt
```
