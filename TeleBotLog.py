
import asyncio
import json
import logging
import os
import threading
import time
import telebot
import pytz
from datetime import datetime, timedelta
from queue import Queue
import sys
import bitmex
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import yfinance as yf
import telegram
from telegram import Bot, Update
from telegram.error import TelegramError
from telegram.ext import Application, CommandHandler, ContextTypes


class TelegramBot:
    def __init__(self, token, chat_id):
        """
        Initialize the Telegram bot.

        :param token: Telegram bot token
        :param chat_id: Chat ID to send messages to
        """
        self.token = token
        self.chat_id = chat_id
        self._bot = None

    async def _async_send_message(self, message=None):
        """
        Async method to send a message.

        :param message: Custom message to send (optional)
        """
        try:
            # Lazy initialization of bot
            if not self._bot:
                self._bot = Bot(token=self.token)

            # If no message provided, create a default test message
            if message is None:
                current_time = datetime.now()
                message = f"{current_time.strftime('%Y-%m-%d %H:%M:%S')},100 - INFO - This is a test message"

            # Send message
            await self._bot.send_message(chat_id=self.chat_id, text=message)
            print(f"Message sent: {message}")

        except Exception as e:
            logger.error(f"Error sending message: {e}")

    def send_message(self, message=None):
        """
        Send a message in different Python environments.

        :param message: Custom message to send (optional)
        """
        try:
            # Check for IPython/Jupyter environment
            get_ipython = sys.modules['__main__'].__dict__.get('get_ipython', None)

            if get_ipython is not None:
                # Jupyter environment
                import nest_asyncio
                nest_asyncio.apply()

            # Run the async method
            asyncio.run(self._async_send_message(message))

        except KeyboardInterrupt:
            print("Message sending stopped by user.")
        except Exception as e:
            print(f"An error occurred: {e}")

def configure_logging(BOT_TOKEN, CHAT_ID):
    """
    Configure logging with a custom Telegram handler.

    :return: Configured logger
    """
    # Create a custom logging handler
    class CustomLoggingHandler(logging.Handler):
        def emit(self, record):
            try:
                # Format the log message
                log_message = self.format(record)
                bot = TelegramBot(BOT_TOKEN, CHAT_ID)
                # Send default test message
                bot.send_message(log_message)
                # Send to Telegram using TelegramLogger
                #TelegramLogger(log_message)
            except Exception as e:
                print(f"Error in custom logging handler: {e}")

    # Configure the logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Create the custom handler
    custom_handler = CustomLoggingHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    custom_handler.setFormatter(formatter)

    # Clear existing handlers to prevent duplicate logging
    logger.handlers.clear()

    # Add the custom handler to the logger
    logger.addHandler(custom_handler)

    # Add standard StreamHandler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger
