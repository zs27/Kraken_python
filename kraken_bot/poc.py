import websocket
import time
import asyncio
import signal
import json # TODO: limit json max amount as can brick cpu if maliciously large input
import logging
import threading
import os
from api.kraken_websockets import Kraken_Websocket
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    filename="logs/poc.log",
    filemode="w",
    format="%(asctime)s %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p"
)
logger = logging.getLogger(__name__)

def configure():
    load_dotenv()
    # os.getenv('API_KEY')
    # os.getenv('PRIVATE_KEY')

def main():
    configure()

    # ws_ticker = threading.Thread(target=Kraken_Websocket(), args=())


    return 0


if __name__ == "__main__":
    main()
