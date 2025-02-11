#TODO
import time
import requests
import json
import logging

logger = logging.getLogger(__name__)

class Kraken_Rest_Manager():
    def __init__(self, public_api_key: str = None, private_api_key: str = None):
        self.public_api_key = public_api_key
        self.private_api_key = private_api_key
        self.public_url = "https://api.kraken.com/0/public/"
        self.private_url = "https://api.kraken.com/0/private/"
    

    def get_nonce(self) -> str:
        return str(int(time.time()*1000))

    '''
    Query Docs: https://docs.kraken.com/api/docs/rest-api/add-order
    '''
    def public_query(self, method: str, params: dict = None):
        # All public queries are 'GET'

        # TODO: handle each methods optional request modes seperately when needed

        if (params is None):
            params = {}

        headers = {
        'Accept': 'application/json'
        }

        # loop through data to add to url
        final_url = self.public_url+method

        response = requests.request("GET", final_url, headers=headers, data={}, params=params)

        if response.status_code != 200:
            logger.critical("Status code not 200, REST API Request failed")
        
        if (response.text is None):
            logger.critical("Request invalid")

        return json.loads(response.text)


        


    def private_query(self):
        # all private queries are post
        # orders are allegedly faster over websocket
        pass
