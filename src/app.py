from pathlib import Path

import yaml
from loguru import logger

from create_db import DBInfo, write_db
from utils.db_data import customer_base_info
from utils.logger import setup_logger

setup_logger()

with open("utils/configs.yaml") as f:
    db_configs = yaml.safe_load(f)["DB"]


def main():

    if not Path(customer_base_info["db_path"]).exists():

        customer_base_info_db = DBInfo(**customer_base_info)
        write_db(customer_base_info_db)

    else:
        logger.debug(f"DB file {customer_base_info['db_path']} already exists")


if __name__ == "__main__":
    main()
