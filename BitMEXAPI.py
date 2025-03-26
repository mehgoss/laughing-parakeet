# -*- coding: utf-8 -*-
import json
import logging
import os
import threading
import time
import pytz
from datetime import datetime, timedelta
import sys
import bitmex
from dotenv import load_dotenv
from TeleLogBot import configure_logging
import pandas as pd

load_dotenv()

# Telegram credentials
TOKEN = os.getenv("TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

logger = configure_logging(TOKEN, CHAT_ID)

# Set the correct time zone
utc_now = datetime.utcnow()
sast = pytz.timezone('Africa/Johannesburg')
sast_now = utc_now.replace(tzinfo=pytz.utc).astimezone(sast)


class BitMEXTestAPI:
    def __init__(self, api_key, api_secret, test=True, symbol='SOL-USD'):
        """
        Initialize BitMEX API client

        :param api_key: BitMEX API key
        :param api_secret: BitMEX API secret
        :param test: Whether to use testnet (default True)
        :param symbol: Trading symbol (default SOL-USD)
        """
        try:
            self.client = bitmex.bitmex(
                test=test,
                api_key=api_key,
                api_secret=api_secret
            )
            self.symbol = symbol  # Symbol parsed here

            # Log initialization
            network_type = 'testnet' if test else 'mainnet'
            logger.info(f"BitMEXAPI initialized for {symbol} on {network_type}")
            print(f"BitMEXTestAPI initialized for {symbol} on {network_type}")

        except Exception as e:
            logger.error(f"Initialization error: {str(e)}")
            print(f"Initialization error: {str(e)}")
            raise

    def get_profile_info(self):
        """
        Retrieve comprehensive account profile information

        :return: Dictionary with user, balance, and position detail
        """
        try:
            # Get user information
            user_info = self.client.User.User_get().result()[0]

            # Get account balance
            margin = self.client.User.User_getMargin().result()[0]

            # Get open positions
            positions = self.client.Position.Position_get(
                filter=json.dumps({"symbol": self.symbol})
            ).result()[0]

            # Fetch current BTC/USD price for conversion
            btc_price_data = self.client.Trade.Trade_getBucketed(
                symbol="XBTUSD",
                binSize="1m",
                count=1,
                reverse=True
            ).result()[0]
            btc_usd_price = btc_price_data[0]['close'] if btc_price_data else 40000  # Fallback price

            # Convert wallet balance from satoshis (1e8) to BTC and USD
            wallet_balance_btc = margin.get('walletBalance') / 100000000
            wallet_balance_usd = wallet_balance_btc * btc_usd_price

            # Format profile information
            profile_info = {
                "user": {
                    "id": user_info.get('id'),
                    "username": user_info.get('username'),
                    "email": user_info.get('email'),
                    "account": user_info.get('account')
                },
                "balance": {
                    "wallet_balance": margin.get('walletBalance'),
                    "margin_balance": margin.get('marginBalance'),
                    "available_margin": margin.get('availableMargin'),
                    "unrealized_pnl": margin.get('unrealisedPnl'),
                    "realized_pnl": margin.get('realisedPnl')
                    "usd" : wallet_balance_usd
                },
                "positions": [{
                    "symbol": pos.get('symbol'),
                    "current_qty": pos.get('currentQty'),
                    "avg_entry_price": pos.get('avgEntryPrice'),
                    "leverage": pos.get('leverage'),
                    "liquidation_price": pos.get('liquidationPrice'),
                    "unrealized_pnl": pos.get('unrealisedPnl'),
                    "realized_pnl": pos.get('realisedPnl')
                } for pos in positions] if positions else []
            }

            # Logging profile details
            logger.info("Profile information retrieved successfully")
            print("Profile information retrieved successfully")

            logger.info(f"Account: {profile_info['user']['username']}")
            print(f"Account: {profile_info['user']['username']}")

            logger.info(f"Wallet Balance: {wallet_balance_btc:.8f} BTC ({wallet_balance_usd:.2f} USD)")
            print(f"Wallet Balance: {wallet_balance_btc:.8f} BTC ({wallet_balance_usd:.2f} USD)")

            logger.info(f"Available Margin: {profile_info['balance']['available_margin'] / 100000000:.8f} BTC")
            print(f"Available Margin: {profile_info['balance']['available_margin'] / 100000000:.8f} BTC")

            if profile_info['positions']:
                for pos in profile_info['positions']:
                    logger.info(f"📈🔵🔴Position: {pos['symbol']} | Qty: {pos['current_qty']} | Entry: {pos['avg_entry_price']}")
                    print(f"Position: {pos['symbol']} | Qty: {pos['current_qty']} | Entry: {pos['avg_entry_price']}")
            else:
                logger.info("No open positions")
                print("No open positions")

            return profile_info

        except Exception as e:
            logger.error(f"Error getting profile information: {str(e)}")
            print(f"Error getting profile information: {str(e)}")
            return None

    def get_candle(self, timeframe='1m', count=100):
        """
        Retrieve candlestick (OHLCV) data for the specified symbol

        :param timeframe: Candle timeframe (default '1m' for 1 minute)
        :param count: Number of candles to retrieve (default 100)
        :return: DataFrame with candle data or None if error occurs
        """
        try:
            # Extended timeframe mapping
            timeframe_map = {
                '1m': '1m',
                '2m': '1m',
                '5m': '5m',
                '10m': '5m',
                '15m': '5m',
                '30m': '5m',
                '1h': '1h',
                '4h': '1h',
                '1d': '1d'
            }

            if timeframe not in timeframe_map:
                raise ValueError(f"Invalid timeframe. Supported: {', '.join(timeframe_map.keys())}")

            base_timeframe = timeframe_map[timeframe]
            multiplier = {
                '2m': 2, '10m': 2, '15m': 3, '30m': 6, '4h': 4
            }.get(timeframe, 1)

            # Adjust count for aggregation
            adjusted_count = count * multiplier if multiplier > 1 else count

            # Retrieve candle data
            candles = self.client.Trade.Trade_getBucketed(
                symbol=self.symbol,
                binSize=base_timeframe,
                count=adjusted_count,
                reverse=True
            ).result()[0]

            # Format initial candle data
            formatted_candles = [{
                'timestamp': candle['timestamp'],
                'open': candle['open'],
                'high': candle['high'],
                'low': candle['low'],
                'close': candle['close'],
                'volume': candle['volume']
            } for candle in candles]

            # Convert to DataFrame
            df = pd.DataFrame(formatted_candles)
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Aggregate if needed
            if multiplier > 1:
                df = df.sort_values('timestamp')
                df = df.resample(f'{timeframe}', on='timestamp').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna().tail(count)

            # Log retrieval success
            logger.info(f"Retrieved {len(df)} {timeframe} candles for {self.symbol}")
            print(f"Retrieved {len(df)} {timeframe} candles for {self.symbol}")

            return df

        except Exception as e:
            logger.error(f"Error retrieving candle data: {str(e)}")
            print(f"Error retrieving candle data: {str(e)}")
            return None

    def open_test_position(self, side="Buy", quantity=100, order_type="Market"):
        """
        Open a test trading position

        :param side: Buy or Sell
        :param quantity: Number of contracts
        :param order_type: Type of order (default Market)
        :return: Order details or None if error
        """
        try:
            logger.info(f"Opening test {side} position for {quantity} contracts")

            # Execute the order
            order = self.client.Order.Order_new(
                symbol=self.symbol,
                side=side,
                orderQty=quantity,
                ordType=order_type
            ).result()[0]

            # Log order details
            logger.info(f"Order placed: {order['ordStatus']} | OrderID: {order['orderID']}")
            print(f"Order placed: {order['ordStatus']} | OrderID: {order['orderID']}")

            logger.info(f"Order details: {side} {quantity} contracts at {order.get('price', 'market price')}")
            print(f"Order details: {side} {quantity} contracts at {order.get('price', 'market price')}")

            # Wait for order to settle
            time.sleep(2)
            self.get_profile_info()

            return order

        except Exception as e:
            logger.error(f"Error opening test position: {str(e)}")
            print(f"Error opening test position: {str(e)}")
            return None

    def _close_position(self, position):
        """
        Close a single position

        :param position: Position dictionary from Position_get
        :return: Order result or None if error
        """
        try:
            symbol = position['symbol']
            current_qty = position['currentQty']

            if current_qty == 0:
                logger.info(f"No open position for {symbol}")
                print(f"No open position for {symbol}")
                return None

            # Determine closing side
            side = "Sell" if current_qty > 0 else "Buy"
            qty = abs(current_qty)

            logger.info(f"Closing position: {symbol} | {current_qty} contracts | Side: {side} | Qty: {qty}")
            print(f"Closing position: {symbol} | {current_qty} contracts | Side: {side} | Qty: {qty}")

            # Place closing order
            order = self.client.Order.Order_new(
                symbol=symbol,
                side=side,
                orderQty=qty,
                ordType="Market"
            ).result()[0]

            logger.info(f"🔴📈⁉️❗Position closed: {order['ordStatus']} | OrderID: {order['orderID']}")
            print(f"Position closed: {order['ordStatus']} | OrderID: {order['orderID']}")

            return order

        except Exception as e:
            logger.error(f"Error closing position {position['symbol']}: {str(e)}")
            print(f"Error closing position {position['symbol']}: {str(e)}")
            return None

    def close_all_positions(self):
        """
        Close all open positions for the current symbol

        :return: True if successful, None if error
        """
        try:
            # Get current positions
            positions = self.client.Position.Position_get(
                filter=json.dumps({"symbol": self.symbol})
            ).result()[0]

            if not positions:
                logger.info("No positions to close")
                print("No positions to close")
                return None

            # Close each position
            for position in positions:
                self._close_position(position)

            # Wait for orders to settle
            time.sleep(2)
            self.get_profile_info()

            return True

        except Exception as e:
            logger.error(f"Error closing positions: {str(e)}")
            print(f"Error closing positions: {str(e)}")
            return None

    def run_test_sequence(self):
        """
        Run a comprehensive test sequence of trading operations

        :return: Final profile information or None if error
        """
        try:
            logger.info("Starting test sequence")
            print("Starting test sequence")

            # Initial profile
            logger.info("=== INITIAL PROFILE ===")
            print("=== INITIAL PROFILE ===")
            self.get_profile_info()

            # Open long position
            logger.info("=== OPENING LONG POSITION(BUY)🔵  ===")
            print("=== OPENING LONG POSITION (BUY)🔵 ===")
            self.open_test_position(side="Buy", quantity=1)

            # Wait and check profile
            wait_time = 1
            logger.info(f"Waiting for {wait_time} seconds...")
            print(f"Waiting for {wait_time} seconds...")
            time.sleep(wait_time)
            self.get_profile_info()

            # Open short position
            logger.info("=== OPENING SHORT POSITION(SELL)🔴  ===")
            print("=== OPENING SHORT POSITION(SELL)🔴 ===")
            self.open_test_position(side="Sell", quantity=1)

            # Wait and check profile
            logger.info(f"Waiting for {wait_time} seconds...")
            time.sleep(wait_time)
            self.get_profile_info()

            # Close all positions
            logger.info("=== CLOSING ALL POSITIONS ===")
            print("=== CLOSING ALL POSITIONS ===")
            self.close_all_positions()

            # Final profile check
            logger.info("=== FINAL PROFILE ===")
            print("=== FINAL PROFILE ===")
            final_profile = self.get_profile_info()

            logger.info("Test sequence completed successfully")
            print("Test sequence completed successfully")
            return final_profile

        except Exception as e:
            logger.error(f"Error in test sequence: {str(e)}")
            print(f"Error in test sequence: {str(e)}")
            return None

# Example usage (optional, for testing):
if __name__ == "__main__":
    api_key = os.getenv("BITMEX_API_KEY")
    api_secret = os.getenv("BITMEX_API_SECRET")
    api = BitMEXTestAPI(api_key, api_secret, test=True, symbol="SOL-USD")
    api.run_test_sequence()
