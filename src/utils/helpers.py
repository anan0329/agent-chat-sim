import whisper
import yaml

# from dotenv import load_dotenv
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from loguru import logger
from openai import OpenAI

from sql_llm import VannaRAGLLM, VannaSQLLLM
from train_sql import Text2SQL


def get_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
        gn_cfg = cfg["general"]
        vn_cfg = cfg["vanna"] | cfg["DB"]
        stt_cfg = cfg["stt"]
        return gn_cfg, vn_cfg, stt_cfg


def get_llm_models(general_config, vanna_config):
    if general_config["model_factory"] == "openai":
        model_for_general = ChatOpenAI(
            model=general_config["model"], temperature=general_config["temperature"]
        )
    elif general_config["model_factory"] == "ollama":
        model_for_general = ChatOllama(
            model=general_config["model"], temperature=general_config["temperature"]
        )
    text2sql = Text2SQL(config=vanna_config)
    vn = text2sql.connect()
    model_for_sql = VannaSQLLLM(llm=vn)
    model_for_rag = VannaRAGLLM(llm=vn)

    return model_for_general, model_for_sql, model_for_rag


def get_stt_models(stt_config):
    if stt_config["model_factory"] == "openai-api":
        client = OpenAI()
    else:
        client = whisper.load_model(stt_config["model"])
    return client


class Speech2Text:
    def __init__(self, stt_config) -> None:
        self.stt_config = stt_config
        self.model_factory = stt_config["model_factory"]
        self.model = stt_config["model"]
        self.is_cloud = self.model_factory == "openai-api"
        self.client = OpenAI() if self.is_cloud else whisper.load_model(self.model)
        logger.info(f"Speech2Text is using config {self.stt_config=}")

    def local(self, audio_file):
        response = self.client.transcribe(audio_file, language="zh")
        logger.info(f"audio file transcribe result: {response['text']}")
        return response["text"]

    def cloud(self, audio_file):
        response = self.client.audio.transcriptions.create(
            model="whisper-1", file=audio_file
        )
        logger.info(f"audio file transcribe result: {response.text}")

        return response.text

    def transcribe(self, audio_file):
        if self.is_cloud:
            return self.cloud(audio_file)
        else:
            return self.local(audio_file)

    # async def acloud(self, audio_file):
    #     response = await self.client.audio.transcriptions.create(
    #         model="whisper-1", file=audio_file
    #     )
    #     return response.text
