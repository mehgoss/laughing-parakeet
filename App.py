import time
import logging
import sys
from BitMEXSMCTrader import  BitMEXLiveTrader
from TeleBotLog import configure_logging




#Telegram creds
TOKEN =  os.getenv("API_SECRET") 
CHAT_ID =  os.getenv("API_SECRET") 
#Bitmex creds
API_KEY =  os.getenv("API_SECRET") 
API_SECRET= os.getenv("API_SECRET") 


def long_running_task():
    """
    Simulates a long-running task with periodic logging.
    Runs for approximately 5 minutes.
    """
    start_time = time.time()
    max_runtime = 5 * 60  # 5 minutes in seconds
    
    try:
        iteration = 0
        while time.time() - start_time < max_runtime:
            iteration += 1
            #logger.info(f"Running iteration {iteration}"
            # Set the correct time zone
            
            def get_sast_time():
                utc_now = datetime.utcnow()
                sast = pytz.timezone('Africa/Johannesburg')
                return utc_now.replace(tzinfo=pytz.utc).astimezone(sast)
            
            
            BitMEXLiveTrader(API_KEY, API_SECRET)
            # Simulate some work
            #ime.sleep(10)  # Sleep for 10 seconds between iterations
            
            # Optional: Add some meaningful work here
            # For example, you could:
            # - Check a condition
            # - Perform a calculation
            # - Make an API call
            # - Process some data
    
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        sys.exit(1)
    
    logger.info("Task completed within 5-minute time limit")

def GitActionLoop():
    """
    Main function to run the script
    """
    logger.info("Starting script")
    long_running_task()
    logger.info("Script finished")

if __name__ == "__main__":
    logger = configure_logging(TOKEN, CHAT_ID)
    GitActionLoop()
