import json
import logging
import sys
import websocket
import threading
import queue
# TODO: Update to autobahn + twisted to handle multiple concurrencies with less overhead for scalability as the base package is too slow 

logger = logging.getLogger(__name__)

class Kraken_Base_Client(threading.Thread):
    def __init__(self, url, data_queue):
        threading.Thread.__init__(self)
        self.websocket_conn = None
        self.is_connected = threading.Event()
        self.url = url
        self.queue = data_queue

    def on_open(self, ws):
        logger.info("Websocket connection opened")
        self.websocket_conn = ws
        self.is_connected.set()

    def on_error(self, ws, error):
        logger.critical(f"Websocket Error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        logger.debug(f"Websocket close status code: {close_status_code}")
        logger.debug(f"Websocket close message: {close_msg}")

    def on_message(self, ws, message):
        logger.info(f"Websocket message: {message}")

        data = json.loads(message)

        # TODO: handle other market data later
        if ('channel' in data):
            if (data['channel'] == "ticker"):
                self.queue.put(data['data'])
        else:
            logger.info(f"Websocket JSON response does not contain channel key")

    
    def subscribe(self, subscription_message):
        self.websocket_conn.send(json.dumps(subscription_message))

    def run(self):
        websocket.enableTrace(False)

        try:
            self.websocket_conn = websocket.WebSocketApp(
                url=self.url,
                on_open=self.on_open,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close
            )
            self.websocket_conn.run_forever()
        except Exception as e:
            print(f"WebSocket error: {e}")
    

class Kraken_Websocket_Manager:
    """Manages WebSocket connections in a separate thread"""
    def __init__(self, public_api_key: str = None, private_api_key: str = None, data_queue: queue.Queue = None):
        self.public_api_key = public_api_key
        self.private_api_key = private_api_key
        self.public_url = "wss://ws.kraken.com/v2"
        self.private_url = "wss://ws-auth.kraken.com/v2"
        self.public_websocket = None
        self.private_websocket = None
        # TODO: Implement Thread safe data transfer
        self.data_queue = data_queue
        # TODO: Keep track of subscriptions to remove

    def launch_sockets(self):
        self.public_websocket = Kraken_Base_Client(self.public_url, self.data_queue)
        self.public_websocket.start()
        self.public_websocket.is_connected.wait()

        self.private_websocket = Kraken_Base_Client(self.private_url, self.data_queue)
        self.private_websocket.start()
        self.private_websocket.is_connected.wait()

    def subscribe_public(self, subscription_message: dict):
        self.public_websocket.subscribe(subscription_message)

    def subscribe_private(self):
        self.private_websocket.subscribe(subscription_message)


    '''
        TODO:
            Buy/Sell orders
            Balance of account

        After LSTM:
            Graceful shutdown of threads
            Data management of other indicators simultaneously
            let model manage subscriptions
    '''
    # def get_subscriptions():
        
    # def remove_subscription():

    # def close_socket():


