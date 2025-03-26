import time
import logging
import sys
import os
from datetime import datetime

# Import your trading modules
from BitMEXSMCTrader import BitMEXLiveTrader
from TeleLogBot import configure_logging

# Environment variables for credentials
TOKEN = os.getenv("TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")

# Maximum runtime configuration
MAX_RUNTIME = int(os.getenv("MAX_RUNTIME", 300))  # Default 5 minutes
SUBPROCESS_RUNTIME = 120  # 2 minutes per subprocess

def configure_main_logger():
    """
    Configure the main logger with error handling.
    """
    try:
        logger = configure_logging(TOKEN, CHAT_ID)
        return logger
    except Exception as e:
        print(f"Failed to configure logging: {e}")
        sys.exit(1)

def run_trading_subprocess(logger):
    """
    Run a single trading subprocess for a fixed duration.
    
    Returns:
    - bool: True if subprocess completed successfully, False otherwise
    """
    start_time = time.time()
    iteration = 0
    
    try:
        trader = BitMEXLiveTrader(API_KEY, API_SECRET)
        
        while time.time() - start_time < SUBPROCESS_RUNTIME:
            logger.info(f"Subprocess iteration {iteration} starting")
            
            # Execute trading logic
            trader.trade()  # Adjust based on your actual implementation
            
            iteration += 1
            time.sleep(5)  # Prevent overwhelming the API
        
        logger.info(f"Subprocess completed after {iteration} iterations")
        return True
    
    except Exception as e:
        logger.error(f"Subprocess error: {e}")
        return False

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
        logger.info(f"Backtest completed")
        logger.info(f"Total runtime: {time.time() - total_start_time:.2f} seconds")
        logger.info(f"Total subprocesses run: {subprocess_count}")
        logger.info(f"Successful subprocesses: {successful_subprocesses}")
    
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
