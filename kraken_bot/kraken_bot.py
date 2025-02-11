import logging
import os
import time
import queue
from api.kraken_rest import Kraken_Rest_Manager
from api.kraken_websockets import Kraken_Websocket_Manager
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    filename= os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', 'kraken_bot.log'),
    filemode="w",
    format="%(asctime)s %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p"
)
logger = logging.getLogger(__name__)

def configure():
    load_dotenv()

def main():

    configure()

    # Launch rest API manager
    rest_manager = Kraken_Rest_Manager(
        public_api_key = os.getenv('API_KEY'),
        private_api_key = os.getenv('PRIVATE_KEY')
    )

    # Check exchange is up
    if (rest_manager.public_query('SystemStatus')['result']['status'] != "online"):
        logger.critical("Exchange down")
        return 0

    # launch websocket manager
    data_queue = queue.Queue()

    websocket_manager = Kraken_Websocket_Manager(
        public_api_key = os.getenv('API_KEY'),
        private_api_key = os.getenv('PRIVATE_KEY'),
        data_queue=data_queue
    )

    try:
        websocket_manager.launch_sockets()

    except Exception as e:
        print(f"Error in Manager: {e}")

    # TODO: handle when exchange goes down and reconnecting protocol
    while True:
        data = websocket_manager.data_queue.get(block=True)
        print(data)

    return 0


if __name__ == "__main__":
    main()
