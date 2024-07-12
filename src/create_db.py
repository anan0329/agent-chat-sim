import datetime
import re
import sqlite3
from dataclasses import dataclass, field

from loguru import logger


def adapt_data_iso(val):
    """Adapt datetime.date to ISO 8601 date.

    Args:
        val (datetime.date): date value
    """
    return val.isoformat()


sqlite3.register_adapter(datetime.date, adapt_data_iso)


@dataclass
class DBInfo:
    db_path: str
    table_name: str
    data: tuple[tuple]
    table_data_schema: str
    table_schema_create: str = field(init=False)

    def __post_init__(self):
        self.table_schema_create = f"""CREATE TABLE IF NOT EXISTS {self.table_name} {self.table_data_schema};"""
        col_len = len(self.table_data_schema.split(","))
        col_names = re.sub(
            "( |\n|INTEGER|VARCHAR|TIMESTAMP|NULL|REAL|\(.*\))",
            "",
            self.table_data_schema,
        )
        self.table_data_insert = f"INSERT INTO {self.table_name} {col_names} VALUES({', '.join(['?'] * col_len)})"


def write_db(db_info: DBInfo) -> None:
    conn = sqlite3.connect(
        db_info.db_path,
        detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
    )

    with conn:
        conn.execute(db_info.table_schema_create)
        conn.executemany(db_info.table_data_insert, db_info.data)
        logger.info(f"Write db {db_info.table_name} to {db_info.db_path} successfully.")

    conn.close()


if __name__ == "__main__":

    from utils.db_data import customer_base_info

    customer_base_info_db = DBInfo(**customer_base_info)

    write_db(customer_base_info_db)
