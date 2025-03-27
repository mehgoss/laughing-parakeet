
# -*- coding: utf-8 -*-
import asyncio
import json
import logging
import os
import threading
import time
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
from BitMEXAPI import BitMEXTestAPI
from TeleLogBot import configure_logging

# Load .env file
load_dotenv()

# Telegram creds
TOKEN = os.getenv("TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

logger = configure_logging(TOKEN, CHAT_ID)

def get_sast_time():
    utc_now = datetime.utcnow()
    sast = pytz.timezone('Africa/Johannesburg')
    return utc_now.replace(tzinfo=pytz.utc).astimezone(sast)

class SMC:
    def __init__(self, api_key="", api_secret="", test=True, symbol="SOL-USD",
                 timeframe="15m", risk_per_trade=0.02, lookback_periods=100):
        """
        Initialize the SMC trading class
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.test = test
        self.symbol = symbol
        self.timeframe = timeframe
        self.risk_per_trade = risk_per_trade
        self.lookback_periods = lookback_periods

        # Initialize BitMEX API client
        self.api = BitMEXTestAPI(
            api_key=self.api_key,
            api_secret=self.api_secret,
            test=self.test,
            symbol=str(self.symbol).replace('-USD', 'USD')
        )

        # Trading state
        self.initial_balance = 0
        self.current_balance = 0
        self.df = pd.DataFrame()
        self.in_trade = False
        self.entry_price = 0
        self.stop_loss = 0
        self.take_profit = 0
        self.position_size = 0
        self.trade_type = None
        self.trades = []
        self.equity_curve = []

        logger.info(f"BitMEXLiveTrader initialized for {symbol} on {timeframe} timeframe")

    def get_market_data(self):
        """
        Fetch market data from BitMEX API or fallback to yfinance.
        """
        try:
            logger.info(f"Fetching {self.symbol} market data from BitMEX")
            data = self.api.get_candle(timeframe=self.timeframe)
            df = pd.DataFrame(data)
            logger.info(f"Retrieved {len(df)} candles from BitMEX")
            self.df = df
            self.df.columns = [col.lower() for col in self.df.columns]
            self.df['higher_high'] = False
            self.df['lower_low'] = False
            self.df['bos_up'] = False
            self.df['bos_down'] = False
            self.df['choch_up'] = False
            self.df['choch_down'] = False
            self.df['bullish_fvg'] = False
            self.df['bearish_fvg'] = False
            return df
        except Exception as e:
            logger.warning(f"Failed to get data from BitMEX API: {str(e)}. Falling back to yfinance.")
            crypto_ticker = self.symbol if self.symbol.endswith('USD') else f"{self.symbol}-USD"
            sast_now = get_sast_time()
            end_date = sast_now
            start_date = end_date - timedelta(days=2)
            try:
                data = yf.download(
                    crypto_ticker,
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d'),
                    interval=self.timeframe
                )
                logger.info(f"Retrieved {len(data)} candles from yfinance")
                self.df = data
                if not data.empty:
                    if isinstance(self.df.columns, pd.MultiIndex):
                        self.df.columns = [col[0].lower() if col[1] else col[0].lower() for col in self.df.columns]
                    else:
                        self.df.columns = [col.lower() for col in self.df.columns]
                    required_columns = ['open', 'high', 'low', 'close']
                    if all(col in self.df.columns for col in required_columns):
                        self.df['higher_high'] = False
                        self.df['lower_low'] = False
                        self.df['bos_up'] = False
                        self.df['bos_down'] = False
                        self.df['choch_up'] = False
                        self.df['choch_down'] = False
                        self.df['bullish_fvg'] = False
                        self.df['bearish_fvg'] = False
                return data
            except Exception as e:
                logger.error(f"yfinance fallback failed: {str(e)}")
                return pd.DataFrame()

    def identify_structure(self):
        """Identify market structure including highs, lows, BOS and CHoCH"""
        logger.info("Identifying market structure")
        df = self.df
        window = 5
        for i in range(window, len(df)):
            if df.iloc[i]['high'] > max(df.iloc[i-window:i]['high']):
                df.loc[df.index[i], 'higher_high'] = True
            if df.iloc[i]['low'] < min(df.iloc[i-window:i]['low']):
                df.loc[df.index[i], 'lower_low'] = True

        prev_structure_high = df.iloc[0]['high']
        prev_structure_low = df.iloc[0]['low']
        structure_points_high = []
        structure_points_low = []

        for i in range(1, len(df)):
            current_high = df.iloc[i]['high']
            current_low = df.iloc[i]['low']
            if df.iloc[i]['higher_high']:
                structure_points_high.append((i, current_high))
            if df.iloc[i]['lower_low']:
                structure_points_low.append((i, current_low))

            if len(structure_points_high) >= 2:
                last_high_idx, last_high = structure_points_high[-1]
                prev_high_idx, prev_high = structure_points_high[-2]
                if current_low > prev_high and i > last_high_idx + 1:
                    df.loc[df.index[i], 'bos_up'] = True
                    print(f"Bullish BOS detected at index {i}, price: {current_low}")

            if len(structure_points_low) >= 2:
                last_low_idx, last_low = structure_points_low[-1]
                prev_low_idx, prev_low = structure_points_low[-2]
                if current_high < prev_low and i > last_low_idx + 1:
                    df.loc[df.index[i], 'bos_down'] = True
                    print(f"Bearish BOS detected at index {i}, price: {current_high}")

        for i in range(window+1, len(df)):
            if df.iloc[i-1]['bos_up']:
                recent_lows = df.iloc[i-window:i]['low'].tolist()
                if min(recent_lows[:-1]) < recent_lows[-1]:
                    df.loc[df.index[i], 'choch_up'] = True
                    print(f"Bullish CHoCH detected at index {i}, price: {current_high}")
            if df.iloc[i-1]['bos_down']:
                recent_highs = df.iloc[i-window:i]['high'].tolist()
                if max(recent_highs[:-1]) > recent_highs[-1]:
                    df.loc[df.index[i], 'choch_down'] = True
                    print(f"Bearish CHoCH detected at index {i}")
        return df

    def identify_fvg(self):
        """Identify Fair Value Gaps (FVGs)"""
        logger.info("Identifying Fair Value Gaps")
        df = self.df
        if 'bullish_fvg_low' not in df.columns:
            df['bullish_fvg_low'] = np.nan
            df['bullish_fvg_high'] = np.nan
            df['bullish_fvg_sl_index'] = np.nan
        if 'bearish_fvg_low' not in df.columns:
            df['bearish_fvg_low'] = np.nan
            df['bearish_fvg_high'] = np.nan
            df['bearish_fvg_sl_index'] = np.nan
        if 'bullish_fvg_mitigated' not in df.columns:
            df['bullish_fvg_mitigated'] = False
        if 'bearish_fvg_mitigated' not in df.columns:
            df['bearish_fvg_mitigated'] = False

        for i in range(2, len(df)):
            if df.iloc[i-2]['low'] > df.iloc[i]['high']:
                high_point = df.iloc[i-2]['high']
                low_point = df.iloc[i]['low']
                price_range = high_point - low_point
                fvg_low = df.iloc[i]['high']
                fvg_high = df.iloc[i-2]['low']
                if price_range > 0:
                    relative_pos = (fvg_high - low_point) / price_range
                    if 0 <= relative_pos <= 0.5:
                        df.loc[df.index[i], 'bullish_fvg'] = True
                        df.loc[df.index[i], 'bullish_fvg_low'] = fvg_low
                        df.loc[df.index[i], 'bullish_fvg_high'] = fvg_high
                        df.loc[df.index[i], 'bullish_fvg_sl_index'] = i
                        print(f"Bullish FVG detected at index {i}, range: {fvg_low}-{fvg_high}")

            if df.iloc[i-2]['high'] < df.iloc[i]['low']:
                high_point = df.iloc[i]['high']
                low_point = df.iloc[i-2]['low']
                price_range = high_point - low_point
                fvg_low = df.iloc[i-2]['high']
                fvg_high = df.iloc[i]['low']
                if price_range > 0:
                    relative_pos = (fvg_high - low_point) / price_range
                    if 0.5 <= relative_pos <= 1:
                        df.loc[df.index[i], 'bearish_fvg'] = True
                        df.loc[df.index[i], 'bearish_fvg_low'] = fvg_low
                        df.loc[df.index[i], 'bearish_fvg_high'] = fvg_high
                        df.loc[df.index[i], 'bearish_fvg_sl_index'] = i
                        print(f"Bearish FVG detected at index {i}, range: {fvg_low}-{fvg_high}")
        return df

    def check_fvg_mitigation(self, current_idx):
        """Check if any previously identified FVGs have been mitigated"""
        df = self.df
        for i in range(current_idx):
            if df.iloc[i].get('bullish_fvg', False) and pd.notna(df.iloc[i].get('bullish_fvg_low')):
                fvg_low = df.iloc[i]['bullish_fvg_low']
                fvg_high = df.iloc[i]['bullish_fvg_high']
                for j in range(i+1, current_idx+1):
                    if df.iloc[j]['low'] <= fvg_high and df.iloc[j]['high'] >= fvg_low:
                        df.loc[df.index[i], 'bullish_fvg_mitigated'] = True
                        print(f"Bullish FVG at index {i} has been mitigated at index {j}")
                        break
            if df.iloc[i].get('bearish_fvg', False) and pd.notna(df.iloc[i].get('bearish_fvg_low')):
                fvg_low = df.iloc[i]['bearish_fvg_low']
                fvg_high = df.iloc[i]['bearish_fvg_high']
                for j in range(i+1, current_idx+1):
                    if df.iloc[j]['low'] <= fvg_high and df.iloc[j]['high'] >= fvg_low:
                        df.loc[df.index[i], 'bearish_fvg_mitigated'] = True
                        print(f"Bearish FVG at index {i} has been mitigated at index {j}")
                        break
        return df

    def execute_trades(self):
        """
        Execute trades based on SMC signals for live trading.
        Modified position size by multiplying raw size by 100.
        """
        logger.info("Starting live trade execution")
        df = self.df
        candles_per_hour = 4 if self.timeframe == "15m" else 1 if self.timeframe == "5m" else 1
        min_candles = candles_per_hour * 4
        if len(df) < min_candles:
            logger.warning(f"Not enough data: {len(df)} candles available, need at least {min_candles}")
            return [], self.equity_curve

        current_candle_idx = len(df) - 1
        current_price = df.iloc[current_candle_idx]['close']
        lookback_start_idx = max(0, current_candle_idx - min_candles)
        self.check_fvg_mitigation(current_candle_idx)

        if self.in_trade:
            if self.trade_type == 'long':
                if df.iloc[current_candle_idx]['low'] <= self.stop_loss:
                    self.execute_signal({'price': self.stop_loss, 'action': "exit", 'reason': 'stoploss'})
                    logger.info(f"Long trade stopped out at {self.stop_loss}")
                elif df.iloc[current_candle_idx]['high'] >= self.take_profit:
                    self.execute_signal({'price': self.take_profit, 'action': "exit", 'reason': 'takeprofit'})
                    logger.info(f"Long trade took profit at {self.take_profit}")
            elif self.trade_type == 'short':
                if df.iloc[current_candle_idx]['high'] >= self.stop_loss:
                    self.execute_signal({'price': self.stop_loss, 'action': "exit", 'reason': 'stoploss'})
                    logger.info(f"Short trade stopped out at {self.stop_loss}")
                elif df.iloc[current_candle_idx]['low'] <= self.take_profit:
                    self.execute_signal({'price': self.take_profit, 'action': "exit", 'reason': 'takeprofit'})
                    logger.info(f"Short trade took profit at {self.take_profit}")
        else:
            signal_detected = False
            if df.iloc[current_candle_idx - 1]['bos_up'] and df.iloc[current_candle_idx]['choch_up']:
                for j in range(lookback_start_idx, current_candle_idx):
                    if df.iloc[j].get('bullish_fvg', False) and not df.iloc[j].get('bullish_fvg_mitigated', False):
                        fvg_low = df.iloc[j]['bullish_fvg_low']
                        fvg_high = df.iloc[j]['bullish_fvg_high']
                        sl_idx = int(df.iloc[j]['bullish_fvg_sl_index'])
                        if fvg_low <= current_price <= fvg_high or \
                           (current_price > fvg_high and df.iloc[current_candle_idx]['low'] >= fvg_low):
                            stop_loss = df.iloc[sl_idx]['low']
                            risk = current_price - stop_loss
                            if risk <= 0:
                                continue
                            risk_amount = self.current_balance * self.risk_per_trade
                            raw_position_size = risk_amount / risk
                            self.position_size = int(raw_position_size * 100)  # Multiply by 100 and ensure integer
                            take_profit = current_price + (risk * 2)
                            signal = {
                                'action': 'entry',
                                'side': 'long',
                                'price': current_price,
                                'stop_loss': stop_loss,
                                'take_profit': take_profit
                            }
                            self.execute_signal(signal)
                            logger.info(f"Entered long trade at {current_price}, SL: {stop_loss}, TP: {take_profit}, Qty: {self.position_size}")
                            signal_detected = True
                            break
            elif df.iloc[current_candle_idx - 1]['bos_down'] and df.iloc[current_candle_idx]['choch_down']:
                for j in range(lookback_start_idx, current_candle_idx):
                    if df.iloc[j].get('bearish_fvg', False) and not df.iloc[j].get('bearish_fvg_mitigated', False):
                        fvg_low = df.iloc[j]['bearish_fvg_low']
                        fvg_high = df.iloc[j]['bearish_fvg_high']
                        sl_idx = int(df.iloc[j]['bearish_fvg_sl_index'])
                        if fvg_low <= current_price <= fvg_high or \
                           (current_price < fvg_low and df.iloc[current_candle_idx]['high'] <= fvg_high):
                            stop_loss = df.iloc[sl_idx]['high']
                            risk = stop_loss - current_price
                            if risk <= 0:
                                continue
                            risk_amount = self.current_balance * self.risk_per_trade
                            raw_position_size = risk_amount / risk
                            self.position_size = int(raw_position_size * 100)  # Multiply by 100 and ensure integer
                            take_profit = current_price - (risk * 2)
                            signal = {
                                'action': 'entry',
                                'side': 'short',
                                'price': current_price,
                                'stop_loss': stop_loss,
                                'take_profit': take_profit
                            }
                            self.execute_signal(signal)
                            logger.info(f"Entered short trade at {current_price}, SL: {stop_loss}, TP: {take_profit}, Qty: {self.position_size}")
                            signal_detected = True
                            break
            if not signal_detected:
                logger.info("No trading signal detected, exiting trade check")
                return self.trades, self.equity_curve
        return self.trades, self.equity_curve

    def calculate_performance(self):
        """Calculate and return performance metrics"""
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'total_return_pct': 0,
                'max_drawdown_pct': 0
            }
        winning_trades = [t for t in self.trades if t['pnl'] > 0]
        win_rate = len(winning_trades) / len(self.trades)
        gross_profit = sum([t['pnl'] for t in self.trades if t['pnl'] > 0])
        gross_loss = abs(sum([t['pnl'] for t in self.trades if t['pnl'] <= 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        total_return = (self.current_balance - self.initial_balance) / self.initial_balance if self.initial_balance else 0
        peak = self.initial_balance
        max_drawdown = 0
        for balance in self.equity_curve:
            if balance > peak:
                peak = balance
            drawdown = (peak - balance) / peak
            max_drawdown = max(max_drawdown, drawdown)
        performance = {
            'total_trades': len(self.trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_return_pct': total_return * 100,
            'max_drawdown_pct': max_drawdown * 100
        }
        return performance

    def visualize_results(self, start_idx=0, end_idx=None):
        """Visualize backtesting results with trades and SMC patterns"""
        if end_idx is None:
            end_idx = len(self.df)
        fig, ax = plt.subplots(figsize=(15, 8))
        subset = self.df.iloc[start_idx:end_idx]
        ax.plot(subset.index, subset['close'], label='Close Price', color='black', linewidth=1)
        for i in range(start_idx, min(end_idx, len(self.df))):
            if self.df.iloc[i].get('bullish_fvg', False) and pd.notna(self.df.iloc[i].get('bullish_fvg_low')):
                fvg_low = self.df.iloc[i]['bullish_fvg_low']
                fvg_high = self.df.iloc[i]['bullish_fvg_high']
                mitigated = self.df.iloc[i].get('bullish_fvg_mitigated', False)
                color = 'lightgreen' if not mitigated else 'darkgreen'
                rect = patches.Rectangle((i-0.5, fvg_low), 1, fvg_high-fvg_low, linewidth=1,
                                        edgecolor=color, facecolor=color, alpha=0.3)
                ax.add_patch(rect)
            if self.df.iloc[i].get('bearish_fvg', False) and pd.notna(self.df.iloc[i].get('bearish_fvg_low')):
                fvg_low = self.df.iloc[i]['bearish_fvg_low']
                fvg_high = self.df.iloc[i]['bearish_fvg_high']
                mitigated = self.df.iloc[i].get('bearish_fvg_mitigated', False)
                color = 'lightcoral' if not mitigated else 'darkred'
                rect = patches.Rectangle((i-0.5, fvg_low), 1, fvg_high-fvg_low, linewidth=1,
                                        edgecolor=color, facecolor=color, alpha=0.3)
                ax.add_patch(rect)
        bos_up_idx = subset[subset['bos_up'] == True].index
        bos_down_idx = subset[subset['bos_down'] == True].index
        choch_up_idx = subset[subset['choch_up'] == True].index
        choch_down_idx = subset[subset['choch_down'] == True].index
        ax.scatter(bos_up_idx, subset.loc[bos_up_idx, 'low'], color='green', marker='^', s=100, label='BOS Up')
        ax.scatter(bos_down_idx, subset.loc[bos_down_idx, 'high'], color='red', marker='v', s=100, label='BOS Down')
        ax.scatter(choch_up_idx, subset.loc[choch_up_idx, 'low'], color='blue', marker='^', s=80, label='CHoCH Up')
        ax.scatter(choch_down_idx, subset.loc[choch_down_idx, 'high'], color='purple', marker='v', s=80, label='CHoCH Down')
        for trade in self.trades:
            if start_idx <= trade['entry_index'] < end_idx:
                color = 'green' if trade['type'] == 'long' else 'red'
                marker = '^' if trade['type'] == 'long' else 'v'
                ax.scatter(trade['entry_index'], trade['entry_price'], color=color, marker=marker, s=120, zorder=5)
                if trade['exit_index'] < end_idx:
                    color = 'green' if trade['pnl'] > 0 else 'red'
                    ax.scatter(trade['exit_index'], trade['exit_price'], color=color, marker='o', s=120, zorder=5)
                    ax.plot([trade['entry_index'], trade['exit_index']],
                           [trade['entry_price'], trade['exit_price']],
                           color=color, linewidth=1, linestyle='--')
                    ax.annotate(f"{trade['pnl']:.2f}",
                              (trade['exit_index'], trade['exit_price']),
                              textcoords="offset points",
                              xytext=(0,10),
                              ha='center')
        ax.set_title('SMC Backtest Results')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        fig2, ax2 = plt.subplots(figsize=(15, 5))
        ax2.plot(self.equity_curve, label='Account Balance', color='blue')
        ax2.set_title('Equity Curve')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        plt.tight_layout()
        return fig, fig2

    def execute_signal(self, signal):
        if signal is None:
            logger.info("No trading signal detected")
            return
        if signal['action'] == 'entry':
            self.execute_entry(signal)
        elif signal['action'] == 'exit':
            self.execute_exit(signal)

    def execute_entry(self, signal):
        side = signal['side']
        price = signal['price']
        stop_loss = signal['stop_loss']
        take_profit = signal['take_profit']
        sast_now = get_sast_time()
        try:
            order_side = "Buy" if side == "long" else "Sell"
            order_result = self.api.open_test_position(side=order_side, quantity=self.position_size)
            self.in_trade = True
            self.trade_type = side
            self.entry_price = price
            self.stop_loss = stop_loss
            self.take_profit = take_profit
            trade = {
                'entry_time': sast_now.strftime('%Y-%m-%d %H:%M:%S'),
                'entry_price': price,
                'entry_index': len(self.df) - 1,
                'type': side,
                'position_size': self.position_size,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk_amount': self.risk_per_trade * self.current_balance
            }
            self.trades.append(trade)
            self.equity_curve.append(self.current_balance)
        except Exception as e:
            logger.error(f"Error opening {side} position: {str(e)}")

    def execute_exit(self, signal):
        reason = signal['reason']
        price = signal['price']
        sast_now = get_sast_time()
        try:
            self.api.close_all_positions()
            if self.trade_type == 'long':
                pnl = (price - self.entry_price) * self.position_size
            else:
                pnl = (self.entry_price - price) * self.position_size
            current_trade = self.trades[-1]
            current_trade.update({
                'exit_time': sast_now.strftime('%Y-%m-%d %H:%M:%S'),
                'exit_price': price,
                'exit_index': len(self.df) - 1,
                'exit_reason': reason,
                'pnl': pnl
            })
            self.current_balance += pnl
            self.equity_curve.append(self.current_balance)
            self.in_trade = False
            self.trade_type = None
            self.entry_price = 0
            self.stop_loss = 0
            self.take_profit = 0
            self.position_size = 0
        except Exception as e:
            logger.error(f"Error closing position: {str(e)}")

    def run(self, scan_interval=120):
        """
        Main loop for live trading with the SMC strategy.
        Runs for approximately 3 minutes: 2 scans with a 2-minute sleep between.
        """
        sast_now = get_sast_time()
        logger.info(f"Starting BitMEXLiveTrader at {sast_now.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Starting BitMEXLiveTrader at {sast_now.strftime('%Y-%m-%d %H:%M:%S')}")

        try:
            profile = self.api.get_profile_info()
            self.initial_balance = f"{float(profile['balance']['usd']):.2f}"
            self.current_balance = self.initial_balance
            self.equity_curve = [self.initial_balance]
            logger.info(f"Initial balance set to {self.initial_balance}")
        except Exception as e:
            logger.error(f"Failed to initialize balance: {str(e)}")
            return

        for iteration in range(2):
            sast_now = get_sast_time()
            logger.info(f"Scan {iteration + 1}/2 started at {sast_now.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Scan {iteration + 1}/2 started at {sast_now.strftime('%Y-%m-%d %H:%M:%S')}")

            self.get_market_data()
            if self.df.empty or len(self.df) < 16:
                logger.warning(f"Insufficient data: {len(self.df)} candles retrieved")
                if iteration < 1:
                    time.sleep(scan_interval)
                continue

            self.identify_structure()
            self.identify_fvg()
            trades, equity_curve = self.execute_trades()

            try:
                profile = self.api.get_profile_info()
                api_balance = profile['balance']['usd'] 
                if abs(api_balance - self.current_balance) > 0.01:
                    logger.info(f"Balance updated from API: {self.current_balance} -> {api_balance}")
                    self.current_balance = api_balance
                    self.equity_curve.append(self.current_balance)
            except Exception as e:
                logger.warning(f"Failed to sync balance with API: {str(e)}")

            performance = self.calculate_performance()
            logger.info(f"Performance snapshot: {performance}")

            if self.trades:
                try:
                    lookback_candles = 16 if self.timeframe == "15m" else 48 if self.timeframe == "5m" else 4
                    fig, fig2 = self.visualize_results(start_idx=max(0, len(self.df) - lookback_candles))
                    plt.show()
                    plt.close(fig)
                    plt.close(fig2)
                except Exception as e:
                    logger.warning(f"Visualization failed: {str(e)}")

            if iteration < 1:
                logger.info(f"Waiting {scan_interval} seconds for next scan...")
                print(f"Waiting {scan_interval} seconds for next scan...")
                time.sleep(scan_interval)

        logger.info("Completed 2 scans, stopping BitMEXLiveTrader")
        if self.in_trade:
            try:
                self.api.close_all_positions()
                logger.info("All open positions closed")
            except Exception as e:
                logger.error(f"Failed to close positions on exit: {str(e)}")
        final_performance = self.calculate_performance()
        logger.info(f"Final performance metrics: {final_performance}")
    
    
def BitMEXLiveTrader(API_KEY, API_SECRET):
    """
    Main function to run the BitMEXLiveTrader
    """
    # BitMEX API credentials (use your test API key and secret)
    #API_KEY = os.getenv("API_KEY")  # Your test API key
    #API_SECRET = os.getenv("API_SECRET")  # Your test API secret


    try:
        
        print("Current time in SAST:", get_sast_time().strftime('%Y-%m-%d %H:%M:%S'))
        # Example log calls
        logger.info("BitmexSMCLiveTrader version 0.1 ")
        logger.info(f"Current time in SAST: {get_sast_time().strftime('%Y-%m-%d %H:%M:%S')}")
        # Initialize and run BitMEXLiveTrader
        api= BitMEXTestAPI(
            api_key=API_KEY,
            api_secret=API_SECRET,
            test=True
        )
        trader = SMC(
            api_key=API_KEY,
            api_secret=API_SECRET,
            test=True,  # Use testnet
            symbol="SOLUSD",  # Solana/USD
            timeframe="5m",  # 5m candles
            risk_per_trade=0.02  # 2% risk per trade
            )
         
        #api.run_test_sequence()
        logger.info("Welcome to BitmexSMCLiveTrader version 0.1🥺🥺\n - Bitmax Api Looks good to go👍👍👍\n -Start Trading with Smart Money Concept Strategy🤲")
        # Start trading loop
        trader.run(scan_interval=120)  # Scan every 2 minutes

    except KeyboardInterrupt:
        logger.info("BitMEXLiveTrader stopped by user")
        print("BitMEXLiveTrader stopped by user")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        print(f"Unexpected error: {str(e)}")
# Example usage
if __name__ == "__main__":
    API_KEY = os.getenv("BITMEX_API_KEY")
    API_SECRET = os.getenv("BITMEX_API_SECRET")
    trader = SMC(
        api_key=API_KEY,
        api_secret=API_SECRET,
        test=True,
        symbol="SOL-USD",
        timeframe="5m",
        risk_per_trade=0.02
    )
    trader.run(scan_interval=120)
