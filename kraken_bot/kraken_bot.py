import logging
import os
from dataclasses import dataclass
from trader import Trading_Manager
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


@dataclass
class Trading_Config_Struct:
    trading_symbol: dict = 'BTC/USD'

    

def configure():
    load_dotenv()

def main():
    # load keys
    configure()
    public_api_key = os.getenv('API_KEY')
    private_api_key = os.getenv('PRIVATE_KEY')

    # load trading config
    trading_conf = Trading_Config_Struct()

    # Launch trader
    tm = Trading_Manager(public_api_key, private_api_key, trading_conf)
    tm.launch()

    return 0


if __name__ == "__main__":
    main()
