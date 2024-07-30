import datetime

import yaml

with open("utils/configs.yaml") as f:
    db_configs = yaml.safe_load(f)["DB"]

# customer_base_info = {
#     "db_path": db_configs["sqlite"],
#     "table_name": "CustomerBaseInfo",
#     "data": (
#         (1, "王東明", "男性", "22", datetime.date(2001, 8, 25)),
#         (2, "李西婷", "女性", "35", datetime.date(1989, 5, 15)),
#         (3, "吳北恩", "男性", "52", datetime.date(1972, 3, 16)),
#         (4, "張男青", "女性", "3", datetime.date(2021, 3, 16)),
#     ),
#     "table_data_schema": """(
#         ID INTEGER primary key not null on conflict ignore,
#         姓名 VARCHAR(8),
#         性別 VARCHAR(8),
#         年齡 INTEGER,
#         生日 TIMESTAMP)""",
# }

customer_base_info = {
    "db_path": db_configs["sqlite"],
    "table_name": "CustomerBaseInfo",
    "data": (
        (1, "王東明", "男性", "年輕族群", "22", datetime.date(2001, 8, 25)),
        (2, "李西婷", "女性", "壯年族群", "35", datetime.date(1989, 8, 15)),
        (3, "吳北恩", "男性", "壯年族群", "52", datetime.date(1972, 8, 16)),
        (4, "張男青", "女性", "嬰幼兒", "3", datetime.date(2021, 8, 16)),
        (5, "林東北", "男性", "退休族群", "66", datetime.date(1958, 3, 16)),
        (6, "江西南", "女性", "嬰幼兒", "1", datetime.date(2023, 3, 16)),
    ),
    "table_data_schema": """(
        ID INTEGER primary key not null on conflict ignore,
        姓名 VARCHAR(8),
        性別 VARCHAR(8),
        類型 VARCHAR(8),
        年齡 INTEGER,
        生日 TIMESTAMP)""",
}

customer_insure_plan = {
    "db_path": db_configs["sqlite"],
    "table_name": "CustomerInsurePlan",
    "data": (
        (
            1,
            "王東明",
            "aaa",
            "全球人壽加倍醫靠終身醫療健康保險 (PHB-S)",
            "住院醫療",
            datetime.date(2022, 7, 15),
        ),
        (
            2,
            "王東明",
            "aaa",
            "全球人壽失扶好照終身健康保險(G版) (LDG-Q)",
            "照護及失能扶助",
            datetime.date(2020, 12, 20),
        ),
        (
            3,
            "王東明",
            "aaa",
            "全球人壽醫療費用健康保險附約 (XHR-P)",
            "住院醫療",
            datetime.date(2020, 12, 20),
        ),
        (
            4,
            "張男青",
            "bbb",
            "全球人壽醫療費用健康保險附約 (XHR-P)",
            "住院醫療",
            datetime.date(2010, 12, 20),
        ),
        (
            5,
            "林東北",
            "ccc",
            "全球人壽失扶好照終身健康保險(G版) (LDG-Q)",
            "照護及失能扶助",
            datetime.date(2014, 12, 20),
        ),
        (
            6,
            "江西南",
            "eee",
            "全球人壽加倍醫靠終身醫療健康保險 (PHB-S)",
            "住院醫療",
            datetime.date(2016, 12, 20),
        ),
    ),
    "table_data_schema": """(
        ID INTEGER primary key not null on conflict ignore,
        姓名 VARCHAR(8),
        保單號碼 VARCHAR(8),
        險種 VARCHAR(32),
        保障 VARCHAR(32),
        生效日 TIMESTAMP)""",
}

coverage_tbl = {
    "db_path": db_configs["sqlite"],
    "table_name": "AllCoverages",
    "data": (
        (1, "住院醫療"),
        (2, "照護及失能扶助"),
        (3, "重疾及特定傷病"),
        (4, "癌症醫療"),
        (5, "傷害醫療"),
    ),
    "table_data_schema": """(
        ID INTEGER primary key not null on conflict ignore,
        保障 VARCHAR(16)
    )""",
}

precision_marketing_list = {
    "db_path": db_configs["sqlite"],
    "table_name": "PrecisionMarketingList",
    "data": (
        (1, "王東明", "近期生日"),
        (2, "王東明", "近期保單周年"),
        (3, "王東明", "目前有保障缺口"),
        (4, "王東明", "年輕族群-可嘗試增員"),
        (5, "林東北", "近期生日"),
        (6, "林東北", "目前有保障缺口"),
    ),
    "table_data_schema": """(
        ID INTEGER primary key not null on conflict ignore,
        姓名 VARCHAR(8),
        銷售切入點 VARCHAR(100)
    )""",
}

# TODO: add upcoming events
upcoming_events = {
    "db_path": db_configs["sqlite"],
    "table_name": "CompanyUpcomingEvents",
    "data": (
        (
            1,
            "全球人壽「因為愛 舞所聚 ZUMBA」公益活動",
            "年輕族群",
            "全球人壽始終相信「因為愛 責任在」，因為有「愛」、有「責任」，面對任何挑戰，我們都無所畏懼，和大家一起以實際行動支持公益、用愛關懷社會角落，一起因為愛，舞所聚",
            datetime.date(2024, 10, 5),
        ),
        (
            2,
            "30周年特別講座 北中南全台巡迴",
            "年輕族群",
            "中南舉辦共3場講座，分享如何打造個人品牌，創立屬於自己的理想人生。三與活動者，更有機會抽種Airpods。邀請懷抱夢想的您，來聽聽他們精彩的人生故事！",
            datetime.date(2024, 8, 17),
        ),
    ),
    "table_data_schema": """(
        ID INTEGER primary key not null on conflict ignore,
        活動名稱 VARCHAR(100),
        適合類型 VARCHAR(32),
        活動描述 VARCHAR(200),
        活動時間 TIMESTAMP)""",
}
