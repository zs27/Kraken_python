import websocket
import _thread
import time
import asyncio
import signal
import json
# TODO: limit json max amount as can brick cpu if maliciously large input
import logging

logging.basicConfig(filename='logs/debug.log',level=logging.DEBUG)
logger = logging.getLogger(__name__)

EXIT_FLAG = False

# This function handles incoming messages
def on_message(ws, message):
    logger.debug(f"Websocket message: {message}")

    try:
        json_message = json.loads(message)
        print("Message: " + json.dumps(json_message, indent=4) + "\n")
    except Exception as e:
        print("Error " + e + "\n")
        exit(1)


def on_error(ws, error):
    EXIT_FLAG = True

    logger.debug(f"Websocket Error: {error}")
    print("Error: " + error)


def on_close(ws, close_status_code, close_msg):
    logger.debug(f"Close status code: {close_status_code}")
    logger.debug(f"Close message: {close_msg}")
    print("### closed ###")


def on_open(ws):
    logger.debug("Opened Websocket Connection")
    print("Opened connection")

    # Send a subscription message for the ticker
    subscription_message = {
        "method": "subscribe",
        "params": {
            "channel": "ticker",
            "symbol": [
                "SOL/USD"
            ]
        }
    }

    ws.send(json.dumps(subscription_message))

# This function handles graceful exit
def signal_handler(signum, frame):
    global EXIT_FLAG
    logger.debug(f"Signal handler: signum {signum}, frame {frame}")
    print("\nKeyboardInterrupt caught! Performing cleanup...")
    # This could be used to close the websocket or other cleanup actions
    EXIT_FLAG = True

# Define the WebSocket client setup
def run_websocket():
    websocket.enableTrace(False) # Show logs or not

    ws = websocket.WebSocketApp(
        "wss://ws.kraken.com/v2",
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )

    ws.run_forever()

# Async main function to run the bot
async def main():
    # Setup signal handler for graceful exit
    signal.signal(signal.SIGINT, signal_handler)

    # Create a background thread to run the WebSocket
    _thread.start_new_thread(run_websocket, ())

    # Keep the asyncio event loop alive
    while not EXIT_FLAG:
        await asyncio.sleep(1)

    print("Bot stopped")
    exit(1)

if __name__ == "__main__":
    asyncio.run(main())
