
# -*- coding: utf-8 -*-
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
            self.symbol = symbol

            # Log initialization
            network_type = 'testnet' if test else 'mainnet'
            logger.info(f"BitMEXTestAPI initialized for {symbol} on {network_type}")
            print(f"BitMEXTestAPI initialized for {symbol} on {network_type}")

        except Exception as e:
            logger.error(f"Initialization error: {str(e)}")
            print(f"Initialization error: {str(e)}")
            raise

    def get_profile_info(self):
        """
        Retrieve comprehensive account profile information

        :return: Dictionary with user, balance, and position details
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

            logger.info(f"Wallet Balance: {profile_info['balance']['wallet_balance'] / 100000000:.8f} XBT")
            print(f"Wallet Balance: {profile_info['balance']['wallet_balance'] / 100000000:.8f} XBT")

            logger.info(f"Available Margin: {profile_info['balance']['available_margin'] / 100000000:.8f} XBT")
            print(f"Available Margin: {profile_info['balance']['available_margin'] / 100000000:.8f} XBT")

            if profile_info['positions']:
                for pos in profile_info['positions']:
                    logger.info(f"Position: {pos['symbol']} | Qty: {pos['current_qty']} | Entry: {pos['avg_entry_price']}")
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
        :return: List of candle dictionaries or None if error occurs
        """
        try:
            # Mapping of timeframe to BitMEX binSize parameter
            timeframe_map = {
                '1m': '1',
                '5m': '5',
                '1h': '1h',
                '1d': '1d'
            }

            # Validate and get the correct bin size
            bin_size = timeframe_map.get(timeframe)
            if not bin_size:
                raise ValueError(f"Invalid timeframe. Supported: {', '.join(timeframe_map.keys())}")

            # Retrieve candle data
            candles = self.client.Trade.Trade_getBucketed(
                symbol=self.symbol,
                binSize=bin_size,
                count=count,
                reverse=True  # Most recent candles first
            ).result()[0]

            # Format candle data
            formatted_candles = [{
                'timestamp': candle['timestamp'],
                'open': candle['open'],
                'high': candle['high'],
                'low': candle['low'],
                'close': candle['close'],
                'volume': candle['volume']
            } for candle in candles]

            # Log retrieval success
            logger.info(f"Retrieved {len(formatted_candles)} {timeframe} candles for {self.symbol}")
            print(f"Retrieved {len(formatted_candles)} {timeframe} candles for {self.symbol}")

            return formatted_candles

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
                symbol = position['symbol']
                current_qty = position['currentQty']

                if current_qty == 0:
                    logger.info(f"No open position for {symbol}")
                    print(f"No open position for {symbol}")
                    continue

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

                logger.info(f"Position closed: {order['ordStatus']} | OrderID: {order['orderID']}")
                print(f"Position closed: {order['ordStatus']} | OrderID: {order['orderID']}")

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
            logger.info("=== OPENING LONG POSITION ===")
            print("=== OPENING LONG POSITION ===")
            self.open_test_position(side="Buy", quantity=1)

            # Wait and check profile
            wait_time = 1
            logger.info(f"Waiting for {wait_time} seconds...")
            print(f"Waiting for {wait_time} seconds...")
            time.sleep(wait_time)
            self.get_profile_info()

            # Open short position
            logger.info("=== OPENING SHORT POSITION ===")
            print("=== OPENING SHORT POSITION ===")
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
