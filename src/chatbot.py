import getpass
import os
from io import BytesIO
from operator import itemgetter

import chainlit as cl
import numpy as np
import torch
from chainlit.element import ElementBased
from chainlit.types import ThreadDict

# from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.schema.runnable.config import RunnableConfig
from loguru import logger

from route import Chain
from utils.helpers import Speech2Text, get_config
from utils.logger import setup_logger

# load_dotenv()
setup_logger()


# os.environ["OPENAI_API_KEY"] = getpass.getpass()

CONFIG_PATH = "utils/configs.yaml"
gn_cfg, vn_cfg, stt_cfg = get_config(CONFIG_PATH)

stt = Speech2Text(stt_cfg)


@cl.password_auth_callback
def auth(username: str, password: str):
    username_stored = os.environ.get("CHAINTLIT_USERNAME")
    password_stored = os.environ.get("CHAINTLIT_PASSWORD")

    if username_stored is None or password_stored is None:
        raise ValueError(
            "Username or password not set. Please set CHAINTLIT_USERNAME and "
            "CHAINTLIT_PASSWORD environment variables."
        )

    if (username, password) == (username_stored, password_stored):
        return cl.User(
            identifier="admin", metadata={"role": "admin", "provider": "credentials"}
        )
    else:
        return None


@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="早晨例行活動構思",
            message="你可以幫我設計一個能提升全天生產力的個人化早晨例行活動嗎？請先問問我的目前習慣和哪些活動能讓我早上充滿活力。",
            icon="/public/idea.png",
        ),
        cl.Starter(
            label="解釋超導體",
            message="用五歲小孩能理解的方式解釋超導體。",
            icon="/public/e_learning.png",
        ),
        cl.Starter(
            label="用於每日電子郵件報告的 Python 程式",
            message="寫一個Python腳本自動發送每日電子郵件報告，並說明如何設置它",
            icon="/public/computing.png",
        ),
        cl.Starter(
            label="簡訊邀請朋友參加婚禮",
            message="寫一條簡短、隨性的文字訊息，邀請朋友成為下個月婚禮的伴郎/伴娘，並提供退出選項。",
            icon="/public/pencil.png",
        ),
    ]


@cl.step(type="tool")
def speech_to_text(audio_file):
    return stt.transcribe(audio_file)


def setup_runnable():
    chat_history = cl.user_session.get("memory")
    print(type(chat_history))

    chain = Chain(gn_cfg, vn_cfg)
    full_chain = chain.make_chain()

    runnable_with_history = (
        RunnablePassthrough.assign(
            chat_history=RunnableLambda(chat_history.load_memory_variables)
            | itemgetter("chat_history")
        )
        | full_chain
    )

    # cl.user_session.set("memory", chat_history)
    cl.user_session.set("runnable", runnable_with_history)


@cl.on_chat_start
async def on_chat_start():

    # chat_history = InMemoryChatMessageHistory()
    chat_history = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True
    )
    cl.user_session.set("memory", chat_history)
    setup_runnable()


@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.AudioChunk):
    if chunk.isStart:
        buffer = BytesIO()
        # This is required for whisper to recognize the file type
        buffer.name = f"input_audio.{chunk.mimeType.split('/')[1]}"
        # Initialize the session for a new audio stream
        cl.user_session.set("audio_buffer", buffer)
        cl.user_session.set("audio_mime_type", chunk.mimeType)

    # TODO: Use Gladia to transcribe chunks as they arrive would decrease latency
    # see https://docs-v1.gladia.io/reference/live-audio

    # For now, write the chunks to a buffer and transcribe the whole audio at the end
    cl.user_session.get("audio_buffer").write(chunk.data)


@cl.on_audio_end
async def on_audio_end(elements: list[ElementBased]):
    # Get the audio buffer from the session
    audio_buffer: BytesIO = cl.user_session.get("audio_buffer")
    audio_buffer.seek(0)  # Move the file pointer to the beginning
    audio_file = audio_buffer.read()
    audio_mime_type: str = cl.user_session.get("audio_mime_type")
    # print(f"{audio_buffer.name=}")

    input_audio_el = cl.Audio(
        mime=audio_mime_type, content=audio_file, name=audio_buffer.name
    )
    await cl.Message(
        author="You",
        type="user_message",
        content="",
        elements=[input_audio_el, *elements],
    ).send()

    if stt.is_cloud:
        whisper_input = (audio_buffer.name, audio_file, audio_mime_type)
    else:
        webm_array = np.frombuffer(audio_file, dtype=np.uint8)
        whisper_input = torch.from_numpy(webm_array).float()

    transcription = speech_to_text(whisper_input)

    await cl.Message(
        author="You",
        type="user_message",
        content=transcription,
        elements=[input_audio_el, *elements],
    ).send()

    memory = cl.user_session.get("memory")
    runnable = cl.user_session.get("runnable")

    res = cl.Message(content="")

    async for chunk in runnable.astream(
        {"question": transcription},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await res.stream_token(chunk)

    await res.send()

    memory.chat_memory.add_user_message(transcription)
    memory.chat_memory.add_ai_message(res.content)


@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    chat_history = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True
    )
    root_messages = [m for m in thread["steps"] if m["parentId"] == None]
    for message in root_messages:
        if message["type"] == "user_message":
            chat_history.chat_memory.add_user_message(message["output"])
        else:
            chat_history.chat_memory.add_ai_message(message["output"])

    cl.user_session.set("memory", chat_history)

    setup_runnable()


@cl.on_message
async def on_message(message: cl.Message):
    chat_history = cl.user_session.get("memory")
    runnable = cl.user_session.get("runnable")

    msg = cl.Message(content="")

    for chunk in await cl.make_async(runnable.stream)(
        {"question": message.content},
        # config={"configurable": {"session_id": session_id}},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()

    chat_history.chat_memory.add_user_message(message.content)
    chat_history.chat_memory.add_ai_message(msg.content)
