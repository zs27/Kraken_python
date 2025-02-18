import time
import queue
import logging
from dataclasses import dataclass
from api.kraken_rest import Kraken_Rest_Manager
from api.kraken_websockets import Kraken_Websocket_Manager

logger = logging.getLogger(__name__)

@dataclass
class Reconnect_Config_Struct:
    max_immediate_retries: int = 5
    long_term_delay: int = 5

class Trading_Manager():
    def __init__(self, public_api_key, private_api_key, config):
        self.queue = queue.Queue()
        self.public_api_key = public_api_key
        self.private_api_key = private_api_key
        self.trading_config = config
        self.reconnect_config = Reconnect_Config_Struct()
        self.rest_manager = None
        self.websocket_manager = None

    def launch(self):
        if not self.initialise_connections():
            self.reconnect(self.config)

        # TODO: handle when exchange goes down and reconnecting protocol

        # Trading Loop
        while True:
            # collect data as json
            data = self.websocket_manager.data_queue.get(block=True)

            evaluate_strategy(data)
            print(data)
    
    # TODO figure out rate limits https://support.kraken.com/hc/en-us/articles/206548367-What-are-the-API-rate-limits-
    # recommended behaviour
    # after drop -> atempt ot reconnect instantly a handful of times
    # after long period -> attempt to reconnect once every 5 seconds
    def reconnect(self):
        for x in range(self.reconnect_config.max_immediate_retries):
            if (self.initialise_connections): 
                logger.info("System immediately reconnected to the Exchange")
                return
        
        logger.debug("Failed immediate reconnection")

        while not self.initialise_connections():
            logger.debug("Failed extended downtime connection")
            time.sleep(self.reconnect_config.long_term_delay)
        
        logger.info("System reconnected to the Exchange after extended downtime")
        

    
    def initialise_connections(self) -> bool:
        # Launch rest API manager
        self.rest_manager = Kraken_Rest_Manager(
            public_api_key = self.public_api_key,
            private_api_key = self.private_api_key
        )

        # Check exchange is up
        if (self.rest_manager.public_query('SystemStatus')['result']['status'] != "online"):
            logger.critical("Exchange down")
            return False

        # Launch Websocket manager
        self.websocket_manager = Kraken_Websocket_Manager(
            public_api_key = self.public_api_key,
            private_api_key = self.private_api_key,
        )

        try:
            self.websocket_manager.launch_sockets()

        except Exception as e:
            print(f"Error in trader: {e}")
            return False
        
        return True
    
    def evaluate_strategy(self):
        pass
