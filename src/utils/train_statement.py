birthday_2m_sql = """
    SELECT 
        姓名,
        性別,
        年齡,
        生日
    FROM CustomerBaseInfo
    WHERE (strftime('%j', 生日) - strftime('%j', 'now')) BETWEEN 0 AND 60;
"""

customer_policy_info = """
    WITH LatestPolicy AS (
        SELECT 保單號碼
        FROM CustomerInsurePlan
        WHERE 姓名 = '林東北'
        ORDER BY 生效日 DESC
        LIMIT 1
    )
        
    SELECT
        險種,
        生效日
    FROM CustomerInsurePlan
    WHERE 保單號碼 = (SELECT 保單號碼 FROM LatestPolicy);
"""

customer_converage_missing = """
    SELECT
        ac.保障 AS 保障缺口
    FROM AllCoverages ac
    LEFT JOIN CustomerInsurePlan ci ON ac.保障 = ci.保障 AND ci.姓名 = '王東明'
    WHERE ci.保障 IS NULL;
"""

precision_marking_strategy = """
    SELECT
        銷售切入點
    FROM PrecisionMarketingList
    WHERE 姓名 = '林東北';
"""

upcoming_events = """
    WITH CustomerType AS (
        SELECT 類型
        FROM CustomerBaseInfo
        WHERE 姓名 = '王東明'
        LIMIT 1
    )

    SELECT 
        活動名稱,
        活動時間,
        活動描述
    FROM CompanyUpcomingEvents
    WHERE 適合類型 = (SELECT 類型 FROM CustomerType);
"""


train_statement = {
    "documentation": [
        # "Our company's cusomer basic information is stored in database CustomerBaseInfo, while the insurance products purchased by customers and their effecive dates are stored in database CustomerInsurePlan.",
        "一張保單可能有一個或多個商品，我們稱為險種，其中一個是主約，其餘的是附約",
        # "欲查詢某客戶的保障缺口，需檢視 CustomerInsurePlan 該客戶的保障欄位，再與 AllCoverages 的保障欄位做差異比較",
        # "欲查詢適合某客戶的活動，需檢視 CustomerBaseInfo 該客戶的分群欄位，再依類型篩選 CompanyUpcomingEvents 對應適合類型的活動內容",
        "健康傘的五大類保障分類分別為住院醫療、照護及失能扶助、重疾及特定傷病、癌症醫療、傷害醫療",
        "年輕族群因可能還在求學或剛出社會階段，建議優先補足傷害保險，推薦險種為全球人壽個人傷害住院日額給付附加條款 (XAH)",
        "王東明已經擁有住院醫療、照護及失能扶助兩類保障",
        # "針對精準行銷名單的年輕族群，推薦銷售切入點有：1.近期生日、2.近期保單周年、3.目前有保障缺口、4.年輕族群-可嘗試增員",
    ],
    "question-sql-pairs": [
        {"question": "請幫我找出那些保戶的生日在兩個月內？", "sql": birthday_2m_sql},
        {
            "question": "林東北上一張保單是甚麼時候投保的？他總共投保那些商品？",
            "sql": customer_policy_info,
        },
        {
            "question": "就健康傘的分析角度，王東明有哪些保障缺口？",
            "sql": customer_converage_missing,
        },
        {
            "question": "近期公司有舉辦那些活動適合王東明參加？",
            "sql": upcoming_events,
        },
        {
            "question": "請幫我查詢王東明是否在我的精準行銷名單裡面？假如在，有哪些推薦的銷售切入點？",
            "sql": precision_marking_strategy,
        },
    ],
}
