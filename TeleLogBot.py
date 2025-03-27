import asyncio
import logging
import sys
from io import BytesIO
import pytz
from datetime import datetime
from telegram import Bot
from telegram.error import TelegramError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_sast_time():
    utc_now = datetime.utcnow()
    sast = pytz.timezone('Africa/Johannesburg')
    return utc_now.replace(tzinfo=pytz.utc).astimezone(sast)
from telegram import Bot
from httpx import AsyncClient, Limits

class TelegramBot:
    def __init__(self, token, chat_id):
        self.token = token
        self.chat_id = chat_id
        self._bot = None
        self._loop = asyncio.get_event_loop()
        # Configure HTTP client with larger pool size and timeout
        self._client = AsyncClient(
            limits=Limits(max_connections=100, max_keepalive_connections=20),
            timeout=30.0  # Increase timeout to 30 seconds
        )

    async def _async_send_message(self, message=None):
        try:
            if not self._bot:
                if not self.token or "your_bot_token" in self.token:
                    raise ValueError("Invalid bot token.")
                self._bot = Bot(token=self.token, http_client=self._client)  # Pass custom client
     
                if message is None:
                current_time = get_sast_time()
                message = f"{current_time.strftime('%Y-%m-%d %H:%M:%S')} - INFO - This is a test message"

            await self._bot.send_message(chat_id=self.chat_id, text=message)

        except TelegramError as e:
            logging.getLogger(__name__).error(f"Telegram error sending message: {e}")
        except Exception as e:
            logging.getLogger(__name__).error(f"Error sending message: {e}")

    async def _async_send_photo(self, photo_buffer, caption=None):
        try:
            if not self._bot:
                if not self.token or "your_bot_token" in self.token:
                    raise ValueError("Invalid bot token. Please provide a valid token from @BotFather")
                self._bot = Bot(token=self.token)

            photo_buffer.seek(0)
            await self._bot.send_photo(
                chat_id=self.chat_id,
                photo=photo_buffer,
                caption=caption if caption else f"Chart generated at {get_sast_time().strftime('%Y-%m-%d %H:%M:%S')}"
            )

        except TelegramError as e:
            logging.getLogger(__name__).error(f"Telegram error sending photo: {e}")
        except Exception as e:
            logging.getLogger(__name__).error(f"Error sending photo: {e}")

    def send_message(self, message=None):
        """Synchronous wrapper for sending text messages"""
        logger = logging.getLogger(__name__)
        try:
            if self._loop.is_running():
                # If we're already in an event loop, create a task
                asyncio.ensure_future(self._async_send_message(message))
            else:
                # If no event loop is running, use run
                asyncio.run(self._async_send_message(message))
        except Exception as e:
            logger.error(f"Error in send_message: {e}")

    def send_photo(self, fig=None, caption=None):
        """Send a matplotlib figure or current plot as a photo"""
        logger = logging.getLogger(__name__)
        try:
            import matplotlib.pyplot as plt
            buffer = BytesIO()
            
            if fig is None:
                plt.savefig(buffer, format='png', bbox_inches='tight')
            else:
                fig.savefig(buffer, format='png', bbox_inches='tight')
            
            buffer.seek(0)
            
            if self._loop.is_running():
                asyncio.ensure_future(self._async_send_photo(buffer, caption))
            else:
                asyncio.run(self._async_send_photo(buffer, caption))
            
            buffer.close()
            if fig is not None:
                plt.close(fig)

        except Exception as e:
            logger.error(f"Error in send_photo: {e}")

class CustomLoggingHandler(logging.Handler):
    def __init__(self, bot):
        super().__init__()
        self.bot = bot
        self._emitting = False  # Flag to prevent recursive calls

    def emit(self, record):
        if self._emitting:  # Prevent recursive logging
            return
        try:
            self._emitting = True
            log_message = self.format(record)
            self.bot.send_message(log_message)
        except Exception as e:
            print(f"Error in custom logging handler: {e}", file=sys.stderr)
        finally:
            self._emitting = False

def configure_logging(bot_token, chat_id):
    logger = logging.getLogger(__name__)
    if not logger.handlers:  # Only configure if not already configured
        logger.setLevel(logging.INFO)
        
        bot = TelegramBot(bot_token, chat_id)
        custom_handler = CustomLoggingHandler(bot)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        custom_handler.setFormatter(formatter)
        
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        
        logger.addHandler(custom_handler)
        logger.addHandler(stream_handler)
    
    return logger

# Configure logger at module level
#BOT_TOKEN = "your_bot_token"  # Replace with actual token
#CHAT_ID = "your_chat_id"      # Replace with actual chat ID
#logger = configure_logging(BOT_TOKEN, CHAT_ID)

if __name__ == "__main__":
    bot = TelegramBot(BOT_TOKEN, CHAT_ID)
    
    # Test message
    logger.info("Starting script")
    bot.send_message("Test message")





