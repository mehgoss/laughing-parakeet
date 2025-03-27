import time
import logging
import sys
import os
from datetime import datetime
import signal
import traceback

# Import your trading modules
from BitMEXSMCTrader import BitMEXLiveTrader
from TeleLogBot import configure_logging

# Environment variables for credentials
TOKEN = os.getenv("TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")

# Maximum runtime configuration
MAX_RUNTIME = int(os.getenv("MAX_RUNTIME", 5*60))  # Default 5 minutes
SUBPROCESS_RUNTIME = 5 * 60# 2 minutes per subprocess

class TimeoutException(Exception):
    """Custom exception to handle timeout scenarios."""
    pass

def timeout_handler(signum, frame):
    """
    Signal handler to raise a TimeoutException when time limit is reached.
    """
    raise TimeoutException("Subprocess execution time exceeded")

def configure_main_logger():
    """
    Configure the main logger with error handling.
    """
    try:
        logger, telegram_bot = configure_logging(TOKEN, CHAT_ID)
        return logger
    except Exception as e:
        print(f"Failed to configure logging: {e}")
        sys.exit(1)

def run_trading_subprocess(logger):
    """
    Run a single trading subprocess with strict time monitoring.
    
    Returns:
    - bool: True if subprocess completed successfully, False otherwise
    """
    # Set up signal-based timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(SUBPROCESS_RUNTIME)
    
    start_time = time.time()
    iteration = 0
    
    try:
        trader, api = BitMEXLiveTrader(API_KEY, API_SECRET)
        
        while True:
            # Check elapsed time before each iteration
            elapsed_time = time.time() - start_time
            if elapsed_time >= SUBPROCESS_RUNTIME:
                logger.info(f"Subprocess time limit reached after {elapsed_time:.2f} seconds")
                break
            
            logger.info(f"Subprocess iteration {iteration} starting")
            
            # Execute trading logic with time tracking
            iteration_start = time.time()
            trader.run()  # Adjust based on your actual implementation
            iteration_duration = time.time() - iteration_start
            
            logger.info(f"Iteration {iteration} took {iteration_duration:.2f} seconds")
            
            iteration += 1
            
            # Prevent overwhelming the API and ensure we don't exceed time limit
            remaining_time = SUBPROCESS_RUNTIME - (time.time() - start_time)
            if remaining_time > 5:
                time.sleep(min(5, remaining_time))
            else:
                break
        
        logger.info(f"Subprocess completed after {iteration} iterations")
        return True
    
    except TimeoutException:
        logger.warning("Subprocess timed out")
        return False
    except Exception as e:
        logger.error(f"Subprocess error: {e}")
        logger.error(traceback.format_exc())
        return False
    finally:
        # Cancel the alarm
        signal.alarm(0)

def main():
    """
    Main function to manage the entire trading strategy execution.
    Runs multiple 2-minute subprocesses within the total max runtime.
    """
    logger = configure_main_logger()
    logger.info(f"Starting BitMEX trading strategy backtest")
    logger.info(f"Maximum runtime: {MAX_RUNTIME} seconds")
    
    total_start_time = time.time()
    subprocess_count = 0
    successful_subprocesses = 0
    
    try:
        while time.time() - total_start_time < MAX_RUNTIME:
            # Check if we have time for another full subprocess
            if time.time() + SUBPROCESS_RUNTIME > total_start_time + MAX_RUNTIME:
                logger.info("Not enough time for another full subprocess")
                break
            
            logger.info(f"Starting subprocess {subprocess_count}")
            
            # Run a single subprocess
            subprocess_success = run_trading_subprocess(logger)
            
            subprocess_count += 1
            if subprocess_success:
                successful_subprocesses += 1
            
            # Optional: Add a small buffer between subprocesses
            time.sleep(10)
        
        # Log final summary
        total_runtime = time.time() - total_start_time
        logger.info(f"Backtest completed")
        logger.info(f"Total runtime: {total_runtime:.2f} seconds")
        logger.info(f"Total subprocesses run: {subprocess_count}")
        logger.info(f"Successful subprocesses: {successful_subprocesses}")
    
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
