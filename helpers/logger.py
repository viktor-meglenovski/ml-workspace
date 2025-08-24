import logging
from configurations.env_variables import LOG_LEVEL

logger = logging.getLogger("ml_workspace_logger")
logger.setLevel(LOG_LEVEL)

formatter = logging.Formatter("%(asctime)s - %(levelname)s : %(message)s")

console_handler = logging.StreamHandler()
console_handler.setLevel(LOG_LEVEL)
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)