### Rest API

rest_manager.public_query('SystemStatus')
rest_manager.public_query('Assets', {'asset': 'XBT,ETH', 'aclass': 'currency'})

### Websocket API

subscription_message = {
    "method": "subscribe",
    "params": {
        "channel": "ticker",
        "symbol": [
            "SOL/USD"
        ]
    }
}
websocket_manager.subscribe_public(subscription_message)