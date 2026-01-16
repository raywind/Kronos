#!/usr/bin/env python3
"""
独立的黄金数据获取测试脚本
"""

import pandas as pd
import sys
import yfinance as yf
from datetime import datetime, timedelta
import time

# 导入 yfinance 的异常类
try:
    from yfinance.exceptions import YFRateLimitError, YFException
except ImportError:
    # 兼容旧版本的 yfinance
    YFRateLimitError = Exception
    YFException = Exception


def get_yfinance_data_batch(symbol, start_date, end_date, period="daily", max_retries=3):
    """
    分批获取 yfinance 数据，避免单次请求过大导致的速率限制

    参数:
        symbol: 股票代码，如 "GC=F", "BTC-USD"
        start_date: 开始日期
        end_date: 结束日期
        period: 数据周期
        max_retries: 最大重试次数

    返回:
        合并后的DataFrame
    """
    # 计算时间跨度
    total_days = (end_date - start_date).days

    # 如果时间跨度不大，直接获取
    if total_days <= 365:  # 一年以内直接获取
        return _get_single_batch(symbol, start_date, end_date, period, max_retries)

    # 分批获取，每批最多一年数据
    all_data = []
    current_start = start_date

    while current_start < end_date:
        batch_end = min(current_start + timedelta(days=365), end_date)
        print(f"正在获取 {current_start.strftime('%Y-%m-%d')} 至 {batch_end.strftime('%Y-%m-%d')} 的数据...")

        batch_data = _get_single_batch(symbol, current_start, batch_end, period, max_retries)
        if batch_data is not None and not batch_data.empty:
            all_data.append(batch_data)

        # 移动到下一批，并增加短暂延迟
        current_start = batch_end + timedelta(days=1)
        if current_start < end_date:
            time.sleep(1)  # 批次间暂停1秒

    if not all_data:
        raise ValueError(f"未能获取到 {symbol} 的任何数据")

    # 合并所有批次数据
    combined_data = pd.concat(all_data, ignore_index=False)
    combined_data = combined_data.sort_index()  # 按时间排序
    combined_data = combined_data[~combined_data.index.duplicated(keep='first')]  # 去重

    return combined_data


def _get_single_batch(symbol, start_date, end_date, period, max_retries):
    """获取单批次数据"""
    for attempt in range(1, max_retries + 1):
        try:
            ticker = yf.Ticker(symbol)

            if period == "daily":
                data = ticker.history(start=start_date, end=end_date, interval="1d")
            else:
                minutes = int(period)
                if minutes >= 60:
                    interval = f"{minutes//60}h"
                else:
                    interval = f"{minutes}m"
                data = ticker.history(start=start_date, end=end_date, interval=interval)

            if data is not None and not data.empty:
                return data

        except YFRateLimitError as e:
            wait_time = min(120 * attempt, 600)  # 更长的等待时间，最多10分钟
            print(f"[警告] 批次获取遇到速率限制，等待 {wait_time} 秒后重试...")
            if attempt < max_retries:
                time.sleep(wait_time)
            else:
                raise Exception(f"Yahoo Finance 速率限制: {e}")
        except YFException as e:
            print(f"[警告] 批次获取遇到 yfinance 错误: {e}")
            if attempt < max_retries:
                time.sleep(3.0)
            else:
                raise
        except Exception as e:
            print(f"[警告] 批次获取失败: {e}")
            if attempt < max_retries:
                time.sleep(2.0)
            else:
                raise

    return None


def get_gold_data(period="daily", days=500):
    """
    使用yfinance获取黄金数据

    参数:
        period: 数据周期，"daily" 表示日线
        days: 获取最近多少天的数据

    返回:
        处理后的DataFrame，包含 open, high, low, close, volume, amount, timestamps 列
    """
    try:
        print(f"正在获取黄金 (GC=F) 的数据...")

        # 计算日期范围
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        print(f"日期范围: {start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')}")

        # 使用分批获取机制获取黄金数据
        gold_data = get_yfinance_data_batch("GC=F", start_date, end_date, period, max_retries=5)  # 增加重试次数

        if gold_data is None or gold_data.empty:
            raise ValueError("未能获取到黄金数据，请检查网络连接。")

        # yfinance返回的数据列名是英文的，索引是DatetimeIndex
        # 将索引转换为timestamps列
        gold_data = gold_data.reset_index()
        # yfinance返回的索引列名可能是 'Date' 或其他
        if 'Date' in gold_data.columns:
            gold_data['timestamps'] = gold_data['Date']
        elif 'Datetime' in gold_data.columns:
            gold_data['timestamps'] = gold_data['Datetime']
        else:
            # 如果没有找到日期列，使用索引
            gold_data['timestamps'] = gold_data.index

        # 处理时间戳并转换为UTC时间
        gold_data['timestamps'] = pd.to_datetime(gold_data['timestamps'])
        # yfinance返回的时间戳通常没有时区信息，假设是UTC时间
        if gold_data['timestamps'].dt.tz is None:
            gold_data['timestamps'] = gold_data['timestamps'].dt.tz_localize('UTC')
        else:
            gold_data['timestamps'] = gold_data['timestamps'].dt.tz_convert('UTC')

        # yfinance的列名已经是英文：Open, High, Low, Close, Volume
        # 转换为小写
        column_mapping = {
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }
        gold_data = gold_data.rename(columns=column_mapping)

        # 按时间排序
        gold_data = gold_data.sort_values('timestamps').reset_index(drop=True)

        # 如果数据太多，只取最近的部分
        if len(gold_data) > days:
            gold_data = gold_data.tail(days).reset_index(drop=True)

        # 转换数值列
        numeric_cols = ["open", "high", "low", "close", "volume", "amount"]
        for col in numeric_cols:
            if col in gold_data.columns:
                gold_data[col] = pd.to_numeric(gold_data[col], errors="coerce")

        # 确保有必要的列
        required_cols = ['open', 'high', 'low', 'close']
        for col in required_cols:
            if col not in gold_data.columns:
                raise ValueError(f"数据中缺少必要的列: {col}")

        # 确保有volume和amount列
        if 'volume' not in gold_data.columns:
            gold_data['volume'] = 0.0
        if 'amount' not in gold_data.columns:
            gold_data['amount'] = 0.0

        # 修复缺失的成交额
        if gold_data["amount"].isna().all() or (gold_data["amount"] == 0).all():
            gold_data["amount"] = gold_data["close"] * gold_data["volume"]

        # 填充任何剩余的NaN值
        gold_data = gold_data.ffill().bfill()

        # 选择需要的列
        result_df = gold_data[['timestamps', 'open', 'high', 'low', 'close', 'volume', 'amount']].copy()

        # 确保数据类型正确
        for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
            result_df[col] = pd.to_numeric(result_df[col], errors='coerce')

        print(f"[成功] 成功获取 {len(result_df)} 条数据")
        print(f"数据范围: {result_df['timestamps'].min()} 至 {result_df['timestamps'].max()}")
        return result_df

    except Exception as e:
        print(f"获取黄金数据时发生错误: {e}")
        import traceback
        traceback.print_exc()
        raise


def main():
    """主测试函数"""
    print("=" * 60)
    print("黄金数据获取测试")
    print("=" * 60)

    try:
        # 测试获取30天的数据
        print("测试获取30天的黄金日线数据...")
        df = get_gold_data(period="daily", days=30)

        print(f"\n[成功] 测试成功！获取到 {len(df)} 条记录")
        print("\n数据基本信息:")
        print(f"- 时间范围: {df['timestamps'].min()} 至 {df['timestamps'].max()}")
        print(f"- 平均收盘价: ${df['close'].mean():.2f}")
        print(f"- 最高价: ${df['high'].max():.2f}")
        print(f"- 最低价: ${df['low'].min():.2f}")

        print("\n数据预览 (最近5条):")
        print(df.tail().to_string(index=False))

    except Exception as e:
        print(f"\n[失败] 测试失败: {e}")
        return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
