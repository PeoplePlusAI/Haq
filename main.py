import logging
from core.ai import ragindex
from telegram import Update
import logging
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    MessageHandler,
    CommandHandler,
    filters,
)
import os
import dotenv
import os
import logging
import dotenv

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
    chat, 
    audio_chat, 
    bhashini_text_chat, 
    parse_photo_text,
    process_image
)

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
    try:
        preferred_language = callback_query.data
        lang = languages.get(preferred_language)
        context.user_data['lang'] = lang
    except (AttributeError, ValueError):
        lang = 'en'
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Error getting language! Setting default to English."
        )

    text_message = ""
    if lang == "en":
        text_message = "You have chosen English. \nPlease share your details"
    elif lang == "hi":
        text_message = "आपने हिंदी चुनी है. \nकृपया मुझे अपने बारे में बताएं।"
    elif lang == "mr":
        text_message = "तुम्ही मराठीची निवड केली आहे. \कृपया तुमचे तपशील शेअर करा"

    set_redis('lang', lang)

    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=text_message
    )


async def response_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await query_handler(update, context)


def check_change_language_query(text):
    return text.lower() in ["change language", "set language", "language"]


async def query_handler(update: Update, context: CallbackContext):

    lang = context.user_data.get('lang')
    if not lang:
        await language_handler(update, context)
        return

    if update.message.text:
        text = update.message.text
        print(f"text is {text}")
        if check_change_language_query(text):
            await language_handler(update, context)
            return
        await flow(update, context, text)
    else:
        if update.message.voice:
            voice = await context.bot.get_file(update.message.voice.file_id)
            await flow_voice(update, context, voice)


async def flow(update: Update, context: ContextTypes.DEFAULT_TYPE, text):
    response = ""
    chat_id = update.effective_chat.id
    lang = context.user_data.get('lang')

    if lang == 'en':
        response_en, _ = ragindex(chat_id, text)
    else:
        response, response_en, _ = bhashini_text_chat(chat_id, text, lang)
    if response:
        await context.bot.send_message(chat_id=chat_id, text=response)
    await context.bot.send_message(chat_id=chat_id, text=response_en)


if __name__ == '__main__':
    application = ApplicationBuilder().token(token).read_timeout(30).write_timeout(30).build()
    start_handler = CommandHandler('start', start)
    language_handler_ = CommandHandler('set_language', language_handler)
    chosen_language = CallbackQueryHandler(preferred_language_callback, pattern='[1-3]')
    application.add_handler(start_handler)
    application.add_handler(language_handler_)
    application.add_handler(chosen_language)
    response_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), response_handler)
    application.add_handler(start_handler)
    application.add_handler(response_handler)
    application.run_polling()
    
# from utils.redis_utils import set_redis
# import os
# import pytesseract
# from PIL import Image
# from utils.openai_utils import (
#     get_duration_pydub, 
#     get_random_wait_messages
# )
# async def chat_handler(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str):
#     response = ""
#     chat_id = update.effective_chat.id
#     lang = context.user_data.get('lang')
#     wait_message = get_random_wait_messages(
#         not_always=True,
#         lang=lang
#     )
#     if wait_message:
#         await context.bot.send_message(chat_id=chat_id, text=wait_message)
#     if lang == 'en':
#         response_en, history = chat(chat_id, text)
#     else:
#         response, response_en, history = bhashini_text_chat(chat_id,text, lang)
#     if response:
#         await context.bot.send_message(chat_id=chat_id, text=response)
#     await context.bot.send_message(chat_id=chat_id, text=response_en)

# async def talk_handler(update: Update, context: ContextTypes.DEFAULT_TYPE, voice):    
#     lang = context.user_data.get('lang')
#     # getting audio file
#     audio_file = voice
#     # audio_file = await context.bot.get_file(update.message.voice.file_id)

#     if lang == 'en':
#         with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as temp_audio_file:
#             await audio_file.download_to_drive(custom_path=temp_audio_file.name)
#             chat_id = update.effective_chat.id

#             wait_message = get_random_wait_messages(
#                 not_always=True,
#                 lang=lang
#         )
#             if wait_message:
#                 await context.bot.send_message(chat_id=chat_id, text=wait_message)

#             with open(temp_audio_file.name, "rb") as file:
#                 audio_data = file.read()
#                 audio_base64 = base64.b64encode(audio_data).decode('utf-8')

#                 response_audio, assistant_message, history = audio_chat(
#                     chat_id, audio_file=open(temp_audio_file.name, "rb")
#                 )
#                 response_audio.stream_to_file(temp_audio_file.name)
#                 # fix this error "raise JSONDecodeError("Expecting value", s, err.value) from None" here
#                 # duration = get_duration_pydub(temp_audio_file.name)
#                 await context.bot.send_audio(
#                     chat_id=chat_id, 
#                     audio=open(temp_audio_file.name, "rb"), 
#                     #duration=duration, 
#                     filename="response.wav",
#                     performer="Yojana Didi",
#                 )
#                 await context.bot.send_message(
#                     chat_id=chat_id, text=assistant_message
#                 )
#                 file.close()
#     else:
#         with tempfile.NamedTemporaryFile(suffix='.mp3', delete=True) as temp_audio_file: # suffix='.wav'
#             await audio_file.download_to_drive(custom_path=temp_audio_file.name)
#             chat_id = update.effective_chat.id

#             wait_message = get_random_wait_messages(
#                     not_always=True,
#                     lang=lang
#             )
#             if wait_message:
#                 await context.bot.send_message(chat_id=chat_id, text=wait_message)

#             with open(temp_audio_file.name, "rb") as file:
#                 audio_data = file.read()
#                 audio_base64 = base64.b64encode(audio_data).decode('utf-8')
#                 response_audio, response, history = bhashini_audio_chat(
#                     chat_id, 
#                     audio_file=audio_base64, 
#                     lang=lang
#                 )
#                 file_ = open(temp_audio_file.name, "wb")
#                 file_.write(response_audio.content)
#                 file_.close()
#                 with open(temp_audio_file.name, "rb") as file:
#                     duration = get_duration_pydub(temp_audio_file.name)
#                     await context.bot.send_audio(
#                         chat_id=chat_id, 
#                         audio=open(temp_audio_file.name, "rb"), 
#                         duration=duration, 
#                         filename="response.mp3",
#                         performer="Yojana Didi",
#                     )
#                 await context.bot.send_message(
#                     chat_id=chat_id, text=response
#                 )
#                 file_.close()
