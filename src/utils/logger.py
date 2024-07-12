import sys

import yaml
from loguru import logger

with open("utils/configs.yaml") as f:
    logger_configs = yaml.safe_load(f)["handlers"]


def setup_logger():

    logger.remove()
    logger.add(
        sys.stdout,
        **logger_configs["console"],
    )
    logger.add(**logger_configs["file"])
