import time
import logging
import sys
from BitMEXSMCTrader import  BitMEXLiveTrader
from TeleBotLog import configure_logging

#Telegram creds
TOKEN =  os.getenv("TOKEN") 
CHAT_ID =  os.getenv("CHAT_ID") 
#Bitmex creds
API_KEY =  os.getenv("API_KEY") 
API_SECRET= os.getenv("API_SECRET") 

logger = configure_logging(TOKEN, CHAT_ID)

def timed_job():
    """
    Simulates a long-running task with periodic logging.
    Runs for approximately 2 minutes.
    """
    start_time = time.time()
    max_runtime = 2 * 60  # 2 minutes in seconds
    
    try:
        iteration = 0
        while time.time() - start_time < max_runtime
            #Trader Bot Start
            BitMEXLiveTrader(API_KEY, API_SECRET)
            
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        sys.exit(1)
    
    logger.info("Task completed within 5-minute time limit")

def GitActionLoop():
    """
    Main function to run the script
    """
    logger.info("Starting script")
    timed_job()
    logger.info("Script finished")

if __name__ == "__main__":
    GitActionLoop()
