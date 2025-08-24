import os

from dotenv import load_dotenv

load_dotenv()

LOG_LEVEL = int(os.getenv("LOG_LEVEL", 10))