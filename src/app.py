from io import BytesIO

import chainlit as cl
import numpy as np
import torch
import whisper
from chainlit.element import ElementBased
from dotenv import load_dotenv
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from loguru import logger

from route import full_chain
from train_sql import Text2SQL
from utils.logger import setup_logger

# load_dotenv()
setup_logger()

store = {}
session_id = "10112"


# def setup_runnable():
#     memory = cl.user_session.get("memory")  # type: ConversationBufferMemory

#     # def get_session_history(session_id):
#     #     if session_id not in store:
#     #         store[session_id] = memory
#     #     return store[session_id]

#     # runnable_with_history = RunnableWithMessageHistory(
#     #     full_chain,
#     #     RunnableLambda(memory.load_memory_variables) | itemgetter("chat_history"),
#     #     input_messages_key="question",
#     #     history_messages_key="chat_history",
#     # )
#     runnable_with_history = (
#         RunnablePassthrough.assign(
#             history=RunnableLambda(memory.load_memory_variables)
#             | itemgetter("chat_history")
#         )
#         | full_chain
#     )

#     cl.user_session.set("runnable", runnable_with_history)


# @cl.password_auth_callback
# def auth():
#     return cl.User(identifier="test")
# stt = whisper.load_model("base")


# @cl.step(type="tool")
# def speech_to_text(audio_file):
#     # async def speech_to_text(audio_file):
#     # stt = cl.user_session.get("stt")
#     # print(stt)
#     response = stt.transcribe(audio_file, language="zh")
#     # response = await client.audio.transcriptions.create(
#     #     model="whisper-1", file=audio_file
#     # )
#     logger.info(f"{response['text'] = }")

#     return response["text"]


@cl.on_chat_start
async def on_chat_start():
    # cl.user_session.set("memory", ConversationBufferMemory(return_messages=True))
    # setup_runnable()
    # app_user = cl.user_session.get("user")
    # await cl.Message(f"Hello {app_user.identifier}").send()

    chat_history = InMemoryChatMessageHistory()

    def get_session_history(session_id):
        if session_id not in store:
            store[session_id] = chat_history
        return store[session_id]

    runnable_with_history = RunnableWithMessageHistory(
        full_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )

    cl.user_session.set("runnable", runnable_with_history)


@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")

    # runnable_with_history.invoke(
    #     {"question": "請幫我找出那些保戶的生日在70天內"},
    #     config={"configurable": {"session_id": "1"}}
    # )

    msg = cl.Message(content="")

    for chunk in await cl.make_async(runnable.stream)(
        {"question": message.content},
        # config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
        config={"configurable": {"session_id": session_id}},
    ):
        await msg.stream_token(chunk)

    await msg.send()
