import logging
# from core.ai import ragindex
from core.ai_1 import ragindex
# from core.ai_2 import ragindex
from telegram import Update
import os
import dotenv
from functools import lru_cache

from telegram import (
    Update, 
    InlineKeyboardButton, 
    InlineKeyboardMarkup
)
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    MessageHandler,
    CommandHandler,
    filters,
    CallbackContext,
    CallbackQueryHandler,
)

from core.ai import (
    # chat, 
    # audio_chat, 
    bhashini_text_chat
    # bhashini_audio_chat,
    # parse_photo_text,
    # process_image
)
from utils.redis_utils import set_redis
from core.ai import (
    # chat, 
    # audio_chat, 
    bhashini_text_chat
    # parse_photo_text,
    # process_image
)

dotenv.load_dotenv("ops/.env")

token = os.getenv('TELEGRAM_BOT_TOKEN')

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
# setting higher logging level for httpx to avoid all GET and POST requests being logged
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

class BotInitializer:
    _instance = None
    run_once = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(BotInitializer, cls).__new__(cls)
            cls.run_once = True
        return cls._instance

@lru_cache(maxsize=128)
def get_language_message(lang):
    messages = {
        "en": "You have chosen English. \nPlease tell your problem",
        "hi": "आपने हिंदी चुनी है. \nकृपया मुझे बताएं कि आपको क्या समस्या आ रही है।",
        "mr": "तुम्ही मराठीची निवड केली आहे. \कृपया मला तुमची समस्या सांगा"
    }
    return messages.get(lang, "Language not supported")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    BotInitializer()  # To initialize only once

    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="Hello I am Yojana Didi, please tell me the problem you're having."
    )
    await relay_handler(update, context)

async def relay_handler(update: Update, context: CallbackContext):
    await language_handler(update, context)

async def language_handler(update: Update, context: CallbackContext):
    # Handle user's language selection
    keyboard = [
        [InlineKeyboardButton("English", callback_data='1')],
        [InlineKeyboardButton("हिंदी", callback_data='2')],
        [InlineKeyboardButton("मराठी", callback_data='3')],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="Choose a Language:",
        reply_markup=reply_markup
    )

async def preferred_language_callback(update: Update, context: CallbackContext):
    callback_query = update.callback_query
    languages = {"1": "en", "2": "hi", "3": "mr"}
    lang = languages.get(callback_query.data, 'en')
    context.user_data['lang'] = lang
    
    text_message = get_language_message(lang)
    
    await callback_query.answer()
    await callback_query.edit_message_text(text=text_message)
    
    set_redis('lang', lang)

async def response_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await query_handler(update, context)

# def check_change_language_query(text):
#     return text.lower() in ["change language", "set language", "language"]


async def query_handler(update: Update, context: CallbackContext):
    lang = context.user_data.get('lang')
    if not lang:
        await language_handler(update, context)
        return

    if update.message and update.message.text:
        text = update.message.text
        print(f"text is {text}")
        await flow(update, context, text)

async def flow(update: Update, context: ContextTypes.DEFAULT_TYPE, text):
    chat_id = update.effective_chat.id
    lang = context.user_data.get('lang', 'en')

    if lang == 'en':
        response_en = ragindex(chat_id, text)
        await context.bot.send_message(chat_id=chat_id, text=response_en)
    else:
        response, response_en = bhashini_text_chat(chat_id, text, lang)
        await context.bot.send_message(chat_id=chat_id, text=response or f"Sorry, I didn't get that. Reverting to English:\n\n{response_en}")

if __name__ == '__main__':
    application = ApplicationBuilder().token(token).read_timeout(30).write_timeout(30).build()
    start_handler = CommandHandler('start', start)
    language_handler_ = CommandHandler('set_language', language_handler)
    chosen_language = CallbackQueryHandler(preferred_language_callback, pattern='[1-3]')
    response_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), response_handler)
    application.add_handler(start_handler)
    application.add_handler(language_handler_)
    application.add_handler(chosen_language)
    application.add_handler(response_handler)
    application.run_polling()