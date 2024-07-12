import datetime

import yaml

with open("utils/configs.yaml") as f:
    db_configs = yaml.safe_load(f)["DB"]

customer_base_info = {
    "db_path": db_configs["path"],
    "table_name": "CustomerBaseInfo",
    "data": (
        (1, "王東明", "男性", "22", datetime.date(2001, 8, 25)),
        (2, "李西婷", "女性", "35", datetime.date(1989, 5, 15)),
        (3, "吳北恩", "男性", "52", datetime.date(1972, 3, 16)),
        (4, "張男青", "女性", "3", datetime.date(2021, 3, 16)),
    ),
    "table_data_schema": """(
        ID INTEGER,
        姓名 VARCHAR(8),
        性別 VARCHAR(8),
        年齡 INTEGER,
        生日 TIMESTAMP)""",
}
