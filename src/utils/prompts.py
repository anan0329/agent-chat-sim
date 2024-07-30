from langchain.prompts import PromptTemplate

classifier_template = """Given the user question below, classify it as either being about `Customer Info`, `Sale strategy`, 'Protection gap', 'Company activity', 'Precision marketing', or `Other`.

Do not respond with more than one word.

<question>
{question}
</question>

Classification:"""

# classifier_template = """Given the user question below, classify it as either being about `Customer Info`, `Sale strategy`, 'Protection gap', or `Other`.
# Each type is described below, make sure to follow the guidelines.
# `Customer Info` relates to information such as customer name, ID, age, birthday, or the policy they've bought.
# `Sale strategy` relates to giving sale advices that the insurance products the customer needs
# `Protection gap` relates to what kind of insurace protection the customer already covers, and which isn't.
# `Other` refers to anything not belongs to above three types.

# Do not respond with more than one word.

# <question>
# {question}
# </question>

# Classification:"""


# sale_strategy_template = """你是壽險業專業的顧問，你分析客戶需求，並給出合適的建議。
# 如果問題有提到mpos有關，在答案的最後加上以下資訊：已幫您紀錄完成。
# 回答以下問題:
# Chat History: {chat_history}
# Question: {question}
# Answer:"""
sale_strategy_template = """
Chat History: {chat_history}
Question: {question}"""

customer_info_template = """Question: {question}:"""


general_template = """以繁體中文(zh-TW)回答以下問題:
Chat History: {chat_history}
Question: {question}
Answer:"""


classifier_prompt = PromptTemplate(
    input_variables=["question"], template=classifier_template
)

sale_strategy_prompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template=sale_strategy_template,
)

customer_info_prompt = PromptTemplate(
    input_variables=["question"], template=customer_info_template
)

general_prompt = PromptTemplate(
    input_variables=["chat_history", "question"], template=general_template
)
