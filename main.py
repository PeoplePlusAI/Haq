import logging
from core.ai import ragindex
from telegram import Update
import asyncio
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes, 
    MessageHandler,
    CommandHandler, 
    filters,
)
import os
import dotenv
import tempfile
import redis

dotenv.load_dotenv("ops/.env")

token = os.getenv('TELEGRAM_BOT_TOKEN')

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
class BotInitializer:
    _instance = None
    run_once = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(BotInitializer, cls).__new__(cls)
            cls.run_once = True
        return cls._instance

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    await context.bot.send_message(chat_id=chat_id, text="Hello I am Yojana Didi, please tell me the problem you're having.")

async def flow(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    chat_id = update.effective_chat.id
    
    response = ragindex(chat_id, text)
    # response = raglang(chat_id, text)
    # response, history = ingest_doc(chat_id, text)
    await context.bot.send_message(chat_id=chat_id, text=response)

async def flow_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    audio_file = await context.bot.get_file(update.message.voice.file_id)

    # Use a temporary file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as temp_audio_file:
        await audio_file.download_to_drive(custom_path=temp_audio_file.name)
        chat_id = update.effective_chat.id
        print(chat_id)
        response, history = audio_chat(chat_id, audio_file=open(temp_audio_file.name, "rb"))
        await context.bot.send_message(chat_id=update.effective_chat.id, text=response)

if __name__ == '__main__':
    application = ApplicationBuilder().token(token).read_timeout(30).write_timeout(30).build()
    start_handler = CommandHandler('start', start)
    response_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), flow)
    audio_handler = MessageHandler(filters.VOICE & (~filters.COMMAND), flow_voice)
    application.add_handler(response_handler)
    application.add_handler(start_handler)
    application.add_handler(audio_handler)
    application.run_polling()
