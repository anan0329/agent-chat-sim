import os

import pandas as pd
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from loguru import logger

from sql_llm import OllamaChromaVanna, OpenaiChromaVanna
from utils.train_statement import train_statement


class Text2SQL:
    def __init__(self, config: dict[str, str]) -> None:
        self.config = config

    def connect(self) -> None:
        print(f"{self.config=}")
        non_vanna_configs = ["sqlite", "model_factory"]
        chroma_path = self.config.get("sqlite")
        model_factory = self.config.get("model_factory")
        print(f"{chroma_path=}")
        print(f"{model_factory=}")

        self.config = {
            k: v for k, v in self.config.items() if k not in non_vanna_configs
        }

        if self.config.get("embedding_function"):
            embedding_model = self.config.get("embedding_function")
            self.config["embedding_function"] = (
                embedding_functions.SentenceTransformerEmbeddingFunction(
                    embedding_model,
                    device="cuda",
                    normalize_embeddings=True,
                )
            )
            logger.info(f"load embedding model {embedding_model}")

        if model_factory == "ollama":
            self.vn = OllamaChromaVanna(config=self.config)
        elif model_factory == "openai":
            load_dotenv()
            # TODO: add more model types config here, e.g. openai
            self.vn = OpenaiChromaVanna(
                config=self.config | {"api_key": os.environ["OPENAI_API_KEY"]}
            )

        self.vn.connect_to_sqlite(chroma_path)
        logger.info(f"Setting Vanna with config {self.config} successfully")

        return self.vn

    def init_train_db(self):
        df_ddl = self.vn.run_sql(
            "SELECT type, sql FROM sqlite_master WHERE sql is not null"
        )

        for ddl in df_ddl["sql"].to_list():
            self.vn.train(ddl=ddl)

    def add_doc(self, docs: list[str]):
        for doc in docs:
            self.vn.train(documentation=doc)

    def add_query(self, question_sql_pairs: list[dict[str, str]]):
        """add question-sql pairs"""
        for pair in question_sql_pairs:
            self.vn.train(**pair)

    def train(self, train_statement):
        """Only do once"""
        self.init_train_db()
        self.add_doc(train_statement["documentation"])
        self.add_query(train_statement["question-sql-pairs"])

        logger.info("Trained sql completely")


if __name__ == "__main__":

    import yaml

    with open("utils/configs.yaml") as f:
        cfg = yaml.safe_load(f)
        vn_cfg = cfg["vanna"] | cfg["DB"]

    text2sql = Text2SQL(config=vn_cfg)
    text2sql.connect()
    text2sql.train(train_statement)
