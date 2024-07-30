# from dotenv import load_dotenv
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from loguru import logger

from utils.helpers import get_config, get_llm_models
from utils.prompts import (
    classifier_prompt,
    customer_info_prompt,
    general_prompt,
    sale_strategy_prompt,
)

# load_dotenv()


# def get_models(general_config, vanna_config):
#     if general_config["model_factory"] == "openai":
#         model_for_general = ChatOpenAI(
#             model=general_config["model"], temperature=general_config["temperature"]
#         )
#     elif general_config["model_factory"] == "ollama":
#         model_for_general = ChatOllama(
#             model=general_config["model"], temperature=general_config["temperature"]
#         )
#     text2sql = Text2SQL(config=vanna_config)
#     vn = text2sql.connect()
#     model_for_sql = VannaLLM(llm=vn)

#     return model_for_general, model_for_sql


class Chain:
    def __init__(self, general_config, vanna_config) -> None:
        self.model_for_general, self.model_for_sql, self.model_for_rag = get_llm_models(
            general_config, vanna_config
        )
        self.chains = {
            "classifier": classifier_prompt | self.model_for_general,
            "sale_strategy": sale_strategy_prompt
            | self.model_for_rag
            | StrOutputParser(),
            "customer_info": customer_info_prompt | self.model_for_sql,
            "general": general_prompt | self.model_for_general | StrOutputParser(),
        }
        self.route_keywords = {
            "sale_strategy": ["sale strategy"],
            "customer_info": [
                "customer info",
                "protection gap",
                "company activity",
                "precision marketing",
            ],
        }

    def route(self, info):
        logger.info(f"route get {info=}")
        content = info["topic"].content.lower()
        if any(
            keyword
            for keyword in self.route_keywords["sale_strategy"]
            if keyword in content
        ):
            logger.info("route to sale strategy")
            return self.chains["sale_strategy"]
        elif any(
            keyword
            for keyword in self.route_keywords["customer_info"]
            if keyword in content
        ):
            logger.info("route to customer Info")
            return self.chains["customer_info"]
        else:
            logger.info("route to general")
            return self.chains["general"]

    def make_chain(self):
        print(self.chains["classifier"])
        return {
            "topic": self.chains["classifier"],
            "question": lambda x: x["question"],
            "chat_history": lambda x: x["chat_history"],
        } | RunnableLambda(self.route)


# def route(info):
#     print(info)
#     if "sale strategy" in info["topic"].content.lower():
#         logger.info("route to sale strategy")
#         return sale_strategy_chain
#     elif (
#         "customer info" in info["topic"].content.lower()
#         or "protection gap" in info["topic"].content.lower()
#         or "company activity" in info["topic"].content.lower()
#         or "precision marketing" in info["topic"].content.lower()
#     ):
#         logger.info("route to customer Info")
#         return customer_info_chain
#     else:
#         logger.info("route to None")
#         return general_chain


# def make_chain(classifier_chain):
#     return {
#         "topic": classifier_chain,
#         "question": lambda x: x["question"],
#         "chat_history": lambda x: x["chat_history"],
#     } | RunnableLambda(route)


# full_chain = make_chain(classifier_chain)

if __name__ == "__main__":

    from utils.helpers import get_config

    CONFIG_PATH = "utils/configs.yaml"

    gn_cfg, vn_cfg, _ = get_config(CONFIG_PATH)

    chain = Chain(gn_cfg, vn_cfg)
    full_chain = chain.make_chain()
    print(
        full_chain.invoke({"question": "哪些保戶的生日在60天內？", "chat_history": ""})
    )
