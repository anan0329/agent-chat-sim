import datetime
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

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
            "( |\n|INTEGER|VARCHAR|TIMESTAMP|NULL|REAL|\(.*\)|primary key not null on conflict ignore)",
            "",
            self.table_data_schema,
        )
        self.table_data_insert = f"INSERT INTO {self.table_name} {col_names} VALUES({', '.join(['?'] * col_len)})"


def write_db(db_info: DBInfo) -> None:

    Path(db_info.db_path).parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(
        db_info.db_path,
        detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
    )

    with conn:
        conn.execute(db_info.table_schema_create)

        try:
            conn.executemany(db_info.table_data_insert, db_info.data)
            logger.info(
                f"Write db {db_info.table_name} to {db_info.db_path} successfully."
            )
        except sqlite3.IntegrityError as e:
            logger.warning(
                f"Found {e}, probably caused by duplicate data inserted, please check if the input data has any duplicate primary key"
            )

    conn.close()


if __name__ == "__main__":

    from utils.db_data import (
        coverage_tbl,
        customer_base_info,
        customer_insure_plan,
        precision_marketing_list,
        upcoming_events,
    )

    customer_base_info_db = DBInfo(**customer_base_info)
    customer_insure_plan_db = DBInfo(**customer_insure_plan)
    company_upcoming_events_db = DBInfo(**upcoming_events)
    coverage_db = DBInfo(**coverage_tbl)
    precision_marketing_db = DBInfo(**precision_marketing_list)

    write_db(customer_base_info_db)
    write_db(customer_insure_plan_db)
    write_db(company_upcoming_events_db)
    write_db(coverage_db)
    write_db(precision_marketing_db)
