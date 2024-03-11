## Installation

1. Create a file `.env` inside `ops` directory and add the following lines to it:
    ```
    OPENAI_API_KEY=<OPENAI_API_KEY>
    TELEGRAM_BOT_TOKEN=<TELEGRAM BOT TOKEN>
    REDIS_HOST=redis (localhost if running locally without docker)
    USERNAME=<digit-username>
    PASSWORD=<digit-password>
    MODEL_NAME=<model-name eg: gpt-3.5-turbo>
    BHASHINI_KEY=<bhashini-key>
    ```
2. For normal running

    Run the following commands:
    ```
    pip install -r requirements.txt
    ```
    Once the installaton is complete, run the following command to start the bot:
    ```
    python main.py
    ```

3. For docker running
    
    Run the following command to start the bot:
    ```
    docker-compose up
    ```
4. Open Telegram and search for `@YourBotName` and start chatting with the bot.


## Usage

1. To start a conversation with the bot, start having a conversation by asking your question from the bot
2. You can ask follow up questions too 
 
