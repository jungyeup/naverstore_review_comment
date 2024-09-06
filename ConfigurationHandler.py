import os
from dotenv import load_dotenv

class ConfigurationHandler:
    @staticmethod
    def load_environment_variables():
        load_dotenv()
        return {
            'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
            'USER_ID': os.getenv('USER_ID'),
            'PASSWORD': os.getenv('PASSWORD'),
            'STOCK_API_URL': os.getenv('STOCK_API_URL'),
        }