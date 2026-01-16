import pandas as pd
import matplotlib.pyplot as plt
import sys
import akshare as ak
import yfinance as yf
from datetime import datetime, timedelta
import time
import random
sys.path.append("../")
from model import Kronos, KronosTokenizer, KronosPredictor

# 导入 yfinance 的异常类
try:
    from yfinance.exceptions import YFRateLimitError, YFException
except ImportError:
    # 兼容旧版本的 yfinance
    YFRateLimitError = Exception
    YFException = Exception

# 设置 matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


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
    import time

    # 计算时间跨度
    total_days = (end_date - start_date).days

    # 如果时间跨度不大，直接获取
    if total_days <= 180:  # 半年以内直接获取
        return _get_single_batch(symbol, start_date, end_date, period, max_retries)

    # 分批获取，每批最多半年数据，避免触发速率限制
    all_data = []
    current_start = start_date
    batch_count = 0

    while current_start < end_date:
        batch_count += 1
        batch_end = min(current_start + timedelta(days=180), end_date)  # 每批6个月
        batch_days = (batch_end - current_start).days

        print(f"正在获取批次 {batch_count}: {current_start.strftime('%Y-%m-%d')} 至 {batch_end.strftime('%Y-%m-%d')} ({batch_days}天)...")

        batch_data = _get_single_batch(symbol, current_start, batch_end, period, max_retries)
        if batch_data is not None and not batch_data.empty:
            all_data.append(batch_data)
            print(f"✓ 批次 {batch_count} 获取成功: {len(batch_data)} 条记录")
        else:
            print(f"⚠️ 批次 {batch_count} 获取失败，跳过此批次")

        # 移动到下一批，并增加较长延迟避免速率限制
        current_start = batch_end + timedelta(days=1)
        if current_start < end_date:
            # 批次间暂停5-10秒，避免连续请求
            batch_delay = random.uniform(5, 10)
            print(f"等待 {batch_delay:.1f} 秒后继续下一批次...")
            time.sleep(batch_delay)

    if not all_data:
        raise ValueError(f"未能获取到 {symbol} 的任何数据")

    # 合并所有批次数据
    combined_data = pd.concat(all_data, ignore_index=False)
    combined_data = combined_data.sort_index()  # 按时间排序
    combined_data = combined_data[~combined_data.index.duplicated(keep='first')]  # 去重

    return combined_data


def _get_single_batch(symbol, start_date, end_date, period, max_retries):
    """获取单批次数据，增强版速率限制处理"""
    for attempt in range(1, max_retries + 1):
        try:
            # 在每次尝试前增加基础延迟
            base_delay = 2.0
            time.sleep(base_delay)

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
            # 指数退避 + 随机抖动
            base_wait = 60
            exponential_wait = base_wait * (2 ** (attempt - 1))
            jitter = random.uniform(0.5, 1.5)
            wait_time = min(exponential_wait * jitter, 600)  # 最大10分钟

            print(f"⚠️  Yahoo Finance 速率限制 (尝试 {attempt}/{max_retries})")
            print(f"   等待 {wait_time:.1f} 秒后重试...")
            print("   提示: Yahoo Finance API 限制很严格，建议使用代理或稍后再试")

            if attempt < max_retries:
                time.sleep(wait_time)
            else:
                print("\n🔄 建议解决方案:")
                print("   1. 等待 30-60 分钟后重试")
                print("   2. 使用 VPN 或代理 IP")
                print("   3. 减少数据请求量 (缩短时间范围)")
                print("   4. 考虑使用其他数据源 (如 Alpha Vantage, IEX Cloud)")
                raise Exception(f"Yahoo Finance 速率限制已达最大重试次数: {e}")
        except YFException as e:
            print(f"[警告] yfinance 错误 (尝试 {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                time.sleep(3.0)
            else:
                raise
        except Exception as e:
            print(f"[警告] 网络错误 (尝试 {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                time.sleep(2.0)
            else:
                raise

    return None


def get_crypto_data_alternative(symbol="BTC", days=500, period="daily"):
    """
    备用的加密货币数据获取方法，当 yfinance 受限时使用

    参数:
        symbol: 加密货币符号 ("BTC", "ETH", etc.)
        days: 获取最近多少天的数据
        period: 数据周期

    返回:
        DataFrame 或 None (如果无法获取)
    """
    print("正在尝试备用的数据获取方法...")

    try:
        # 方法1: 尝试使用更小的批次和更长的延迟
        print("方法1: 使用更保守的获取策略...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # 使用更小的批次 (3个月) 和更长的延迟
        all_data = []
        current_start = start_date

        while current_start < end_date:
            batch_end = min(current_start + timedelta(days=90), end_date)

            for attempt in range(1, 6):  # 最多尝试6次
                try:
                    print(f"获取 {current_start.strftime('%Y-%m-%d')} 至 {batch_end.strftime('%Y-%m-%d')} (尝试 {attempt}/6)...")
                    ticker = yf.Ticker(f"{symbol}-USD")
                    data = ticker.history(start=current_start, end=batch_end, interval="1d")

                    if data is not None and not data.empty:
                        all_data.append(data)
                        print(f"✓ 获取成功: {len(data)} 条记录")
                        break

                except Exception as e:
                    print(f"⚠️ 尝试 {attempt} 失败: {e}")
                    if attempt < 6:
                        wait_time = 30 + attempt * 10  # 逐渐增加等待时间
                        print(f"等待 {wait_time} 秒...")
                        time.sleep(wait_time)
                    else:
                        print("此批次获取失败，跳过")

            current_start = batch_end + timedelta(days=1)
            if current_start < end_date:
                time.sleep(random.uniform(10, 15))  # 批次间更长延迟

        if all_data:
            combined = pd.concat(all_data, ignore_index=False)
            combined = combined.sort_index()
            combined = combined[~combined.index.duplicated(keep='first')]
            return combined

    except Exception as e:
        print(f"备用方法1失败: {e}")

    # 方法2: 建议用户使用代理或其他解决方案
    print("\n🔄 建议解决方案:")
    print("   1. 使用 VPN 或代理服务器")
    print("   2. 等待 1-2 小时后重试")
    print("   3. 使用更短的时间范围 (减少 'days' 参数)")
    print("   4. 考虑使用其他数据源:")
    print("      - Alpha Vantage API")
    print("      - CoinGecko API")
    print("      - Binance API")
    print("      - IEX Cloud API")

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
    import time

    try:
        print(f"正在获取黄金 (GC=F) 的数据...")

        # 计算日期范围
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        print(f"日期范围: {start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')}")

        # 使用分批获取机制获取黄金数据
        gold_data = get_yfinance_data_batch("GC=F", start_date, end_date, period, max_retries=3)

        # 如果主要方法失败，尝试备用方法
        if gold_data is None or gold_data.empty:
            print("主要获取方法失败，尝试备用方法...")
            gold_data = get_crypto_data_alternative("GC=F", days, period)

        if gold_data is None or gold_data.empty:
            raise ValueError("所有获取方法都失败了。请尝试：1) 使用代理/VPN 2) 等待一段时间后重试 3) 减少数据量")

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

        print(f"✅ 成功获取 {len(result_df)} 条数据")
        print(f"数据范围: {result_df['timestamps'].min()} 至 {result_df['timestamps'].max()}")
        return result_df

    except Exception as e:
        print(f"获取黄金数据时发生错误: {e}")
        import traceback
        traceback.print_exc()
        raise


def get_bitcoin_data(period="daily", days=500):
    """
    使用yfinance获取比特币数据
    
    参数:
        period: 数据周期，"daily" 表示日线
        days: 获取最近多少天的数据
    
    返回:
        处理后的DataFrame，包含 open, high, low, close, volume, amount, timestamps 列
    """
    import time
    
    try:
        print(f"正在获取比特币 (BTC-USD) 的数据...")
        
        # 计算日期范围
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        print(f"日期范围: {start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')}")
        
        # 使用分批获取机制获取比特币数据
        crypto_data = get_yfinance_data_batch("BTC-USD", start_date, end_date, period, max_retries=3)

        # 如果主要方法失败，尝试备用方法
        if crypto_data is None or crypto_data.empty:
            print("主要获取方法失败，尝试备用方法...")
            crypto_data = get_crypto_data_alternative("BTC", days, period)

        if crypto_data is None or crypto_data.empty:
            raise ValueError("所有获取方法都失败了。请尝试：1) 使用代理/VPN 2) 等待一段时间后重试 3) 减少数据量")
        
        # yfinance返回的数据列名是英文的，索引是DatetimeIndex
        # 将索引转换为timestamps列
        crypto_data = crypto_data.reset_index()
        # yfinance返回的索引列名可能是 'Date' 或其他
        if 'Date' in crypto_data.columns:
            crypto_data['timestamps'] = crypto_data['Date']
        elif 'Datetime' in crypto_data.columns:
            crypto_data['timestamps'] = crypto_data['Datetime']
        else:
            # 如果没有找到日期列，使用索引
            crypto_data['timestamps'] = crypto_data.index
        
        # 处理时间戳并转换为UTC时间
        crypto_data['timestamps'] = pd.to_datetime(crypto_data['timestamps'])
        # yfinance返回的时间戳通常没有时区信息，假设是UTC时间
        if crypto_data['timestamps'].dt.tz is None:
            crypto_data['timestamps'] = crypto_data['timestamps'].dt.tz_localize('UTC')
        else:
            crypto_data['timestamps'] = crypto_data['timestamps'].dt.tz_convert('UTC')
        
        # yfinance的列名已经是英文：Open, High, Low, Close, Volume
        # 转换为小写
        column_mapping = {
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }
        crypto_data = crypto_data.rename(columns=column_mapping)
        
        # 按时间排序
        crypto_data = crypto_data.sort_values('timestamps').reset_index(drop=True)
        
        # 如果数据太多，只取最近的部分
        if len(crypto_data) > days:
            crypto_data = crypto_data.tail(days).reset_index(drop=True)
        
        # 转换数值列
        numeric_cols = ["open", "high", "low", "close", "volume", "amount"]
        for col in numeric_cols:
            if col in crypto_data.columns:
                crypto_data[col] = pd.to_numeric(crypto_data[col], errors="coerce")
        
        # 确保有必要的列
        required_cols = ['open', 'high', 'low', 'close']
        for col in required_cols:
            if col not in crypto_data.columns:
                raise ValueError(f"数据中缺少必要的列: {col}")
        
        # 确保有volume和amount列
        if 'volume' not in crypto_data.columns:
            crypto_data['volume'] = 0.0
        if 'amount' not in crypto_data.columns:
            crypto_data['amount'] = 0.0
        
        # 修复缺失的成交额
        if crypto_data["amount"].isna().all() or (crypto_data["amount"] == 0).all():
            crypto_data["amount"] = crypto_data["close"] * crypto_data["volume"]
        
        # 填充任何剩余的NaN值
        crypto_data = crypto_data.ffill().bfill()
        
        # 选择需要的列
        result_df = crypto_data[['timestamps', 'open', 'high', 'low', 'close', 'volume', 'amount']].copy()
        
        # 确保数据类型正确
        for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
            result_df[col] = pd.to_numeric(result_df[col], errors='coerce')
        
        print(f"✅ 成功获取 {len(result_df)} 条数据")
        print(f"数据范围: {result_df['timestamps'].min()} 至 {result_df['timestamps'].max()}")
        return result_df
        
    except Exception as e:
        print(f"获取比特币数据时发生错误: {e}")
        import traceback
        traceback.print_exc()
        raise


def get_stock_data_from_akshare(symbol, period="daily", days=500):
    """
    从akshare获取股票数据
    
    参数:
        symbol: 股票代码，例如 "000001" 或 "600977"
        period: 数据周期，"daily" 表示日线，"5" 表示5分钟线
        days: 获取最近多少天的数据
    
    返回:
        处理后的DataFrame，包含 open, high, low, close, volume, amount, timestamps 列
    """
    import time
    
    try:
        # 计算日期范围
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')
        
        print(f"正在从akshare获取股票 {symbol} 的数据...")
        print(f"日期范围: {start_date} 至 {end_date}")
        
        # 获取股票历史数据（带重试机制）
        max_retries = 3
        stock_data = None
        
        for attempt in range(1, max_retries + 1):
            try:
                if period == "daily":
                    # 日线数据，不指定日期范围可以获取更多数据
                    stock_data = ak.stock_zh_a_hist(
                        symbol=symbol, 
                        period="daily", 
                        start_date=start_date, 
                        end_date=end_date, 
                        adjust="qfq"  # 前复权
                    )
                else:
                    # 对于分钟级数据，暂时不支持，提示用户使用日线
                    raise ValueError("分钟级数据暂不支持，请使用日线数据（daily）")
                
                if stock_data is not None and not stock_data.empty:
                    break
            except Exception as e:
                print(f"[警告] 尝试 {attempt}/{max_retries} 失败: {e}")
                if attempt < max_retries:
                    time.sleep(1.5)
                else:
                    raise
        
        if stock_data is None or stock_data.empty:
            raise ValueError(f"未能获取到股票代码 {symbol} 的数据，请检查股票代码是否正确。")
        
        # 转换列名：akshare返回的是中文列名，需要转换为英文
        column_mapping = {
            '日期': 'timestamps',
            '开盘': 'open',
            '收盘': 'close',
            '最高': 'high',
            '最低': 'low',
            '成交量': 'volume',
            '成交额': 'amount'
        }
        
        # 重命名列
        stock_data = stock_data.rename(columns=column_mapping)
        
        # 处理时间戳并转换为UTC时间
        if 'timestamps' in stock_data.columns:
            # 转换为datetime，假设原始数据是本地时间（中国时区）
            stock_data['timestamps'] = pd.to_datetime(stock_data['timestamps'])
            # 如果时间戳没有时区信息，假设是北京时间（UTC+8），然后转换为UTC
            if stock_data['timestamps'].dt.tz is None:
                # 假设原始数据是北京时间（UTC+8）
                stock_data['timestamps'] = stock_data['timestamps'].dt.tz_localize('Asia/Shanghai')
            # 转换为UTC时间
            stock_data['timestamps'] = stock_data['timestamps'].dt.tz_convert('UTC')
        else:
            raise ValueError("数据中缺少日期列")
        
        # 按时间排序
        stock_data = stock_data.sort_values('timestamps').reset_index(drop=True)
        
        # 转换数值列（处理可能的逗号分隔符和无效值）
        numeric_cols = ["open", "high", "low", "close", "volume", "amount"]
        for col in numeric_cols:
            if col in stock_data.columns:
                # 转换为字符串，移除逗号，处理无效值
                stock_data[col] = (
                    stock_data[col]
                    .astype(str)
                    .str.replace(",", "", regex=False)
                    .replace({"--": None, "": None, "nan": None})
                )
                stock_data[col] = pd.to_numeric(stock_data[col], errors="coerce")
        
        # 确保有必要的列
        required_cols = ['open', 'high', 'low', 'close']
        for col in required_cols:
            if col not in stock_data.columns:
                raise ValueError(f"数据中缺少必要的列: {col}")
        
        # 修复无效的开盘价
        open_bad = (stock_data["open"] == 0) | (stock_data["open"].isna())
        if open_bad.any():
            print(f"[信息] 修复了 {open_bad.sum()} 个无效的开盘价")
            stock_data.loc[open_bad, "open"] = stock_data["close"].shift(1)
            stock_data["open"].fillna(stock_data["close"], inplace=True)
        
        # 确保有volume和amount列
        if 'volume' not in stock_data.columns:
            stock_data['volume'] = 0.0
        if 'amount' not in stock_data.columns:
            stock_data['amount'] = 0.0
        
        # 修复缺失的成交额
        if stock_data["amount"].isna().all() or (stock_data["amount"] == 0).all():
            stock_data["amount"] = stock_data["close"] * stock_data["volume"]
        
        # 填充任何剩余的NaN值
        stock_data = stock_data.ffill().bfill()
        
        # 选择需要的列
        result_df = stock_data[['timestamps', 'open', 'high', 'low', 'close', 'volume', 'amount']].copy()
        
        # 确保数据类型正确
        for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
            result_df[col] = pd.to_numeric(result_df[col], errors='coerce')
        
        print(f"✅ 成功获取 {len(result_df)} 条数据")
        print(f"数据范围: {result_df['timestamps'].min()} 至 {result_df['timestamps'].max()}")
        return result_df
        
    except Exception as e:
        print(f"获取数据时发生错误: {e}")
        import traceback
        traceback.print_exc()
        raise


def plot_prediction(kline_df, pred_df, lookback):
    # 确保使用时间戳作为索引（如果是UTC时间）
    if 'timestamps' in kline_df.columns:
        kline_df_indexed = kline_df.set_index('timestamps')
    else:
        kline_df_indexed = kline_df.copy()
    
    # 确保pred_df的索引是UTC时间（pred_df应该已经使用y_timestamp作为索引）
    if isinstance(pred_df.index, pd.DatetimeIndex):
        if pred_df.index.tz is None:
            pred_df.index = pred_df.index.tz_localize('UTC')
        elif pred_df.index.tz != pd.Timestamp.now(tz='UTC').tz:
            pred_df.index = pred_df.index.tz_convert('UTC')
    else:
        pred_df.index = pd.to_datetime(pred_df.index).tz_localize('UTC')
    
    # 确保kline_df的索引也是UTC时间
    if isinstance(kline_df_indexed.index, pd.DatetimeIndex):
        if kline_df_indexed.index.tz is None:
            kline_df_indexed.index = kline_df_indexed.index.tz_localize('UTC')
        elif kline_df_indexed.index.tz != pd.Timestamp.now(tz='UTC').tz:
            kline_df_indexed.index = kline_df_indexed.index.tz_convert('UTC')
    else:
        kline_df_indexed.index = pd.to_datetime(kline_df_indexed.index).tz_localize('UTC')
    
    # 分离历史数据和真实值（用于对比的部分）
    # 历史数据：前lookback条
    historical_df = kline_df_indexed.iloc[:lookback]
    # 真实值（如果有）：lookback之后的部分，用于与预测值对比
    if len(kline_df_indexed) > lookback:
        actual_df = kline_df_indexed.iloc[lookback:]
    else:
        actual_df = pd.DataFrame()
    
    # 历史数据的收盘价和成交量
    sr_close_hist = historical_df['close']
    sr_volume_hist = historical_df['volume']
    
    # 预测值（使用pred_df自己的索引，这是未来时间戳）
    sr_pred_close = pred_df['close']
    sr_pred_volume = pred_df['volume']
    
    # 如果有真实值用于对比
    if len(actual_df) > 0:
        sr_close_actual = actual_df['close']
        sr_volume_actual = actual_df['volume']
        
        # 合并历史、真实值和预测值
        close_df = pd.DataFrame({
            '历史值': sr_close_hist,
            '真实值': sr_close_actual,
            '预测值': sr_pred_close
        })
        volume_df = pd.DataFrame({
            '历史值': sr_volume_hist,
            '真实值': sr_volume_actual,
            '预测值': sr_pred_volume
        })
    else:
        # 只有历史值和预测值
        close_df = pd.DataFrame({
            '历史值': sr_close_hist,
            '预测值': sr_pred_close
        })
        volume_df = pd.DataFrame({
            '历史值': sr_volume_hist,
            '预测值': sr_pred_volume
        })

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # 绘制历史数据
    ax1.plot(close_df['历史值'].index, close_df['历史值'], label='历史值', color='gray', linewidth=1.5, alpha=0.7)
    
    # 如果有真实值，绘制真实值
    if '真实值' in close_df.columns:
        ax1.plot(close_df['真实值'].index, close_df['真实值'], label='真实值', color='blue', linewidth=1.5)
    
    # 绘制预测值（使用pred_df自己的索引，这是未来时间戳）
    ax1.plot(pred_df.index, pred_df['close'], label='预测值', color='red', linewidth=1.5, linestyle='--')
    
    ax1.set_ylabel('收盘价', fontsize=14)
    ax1.legend(loc='lower left', fontsize=12)
    ax1.grid(True)
    ax1.set_title('收盘价预测对比', fontsize=16)
    
    # 格式化x轴时间显示，根据数据频率选择格式
    # 合并所有时间索引
    all_times = close_df['历史值'].index.union(pred_df.index)
    if len(all_times) > 0:
        time_span = (all_times.max() - all_times.min()).total_seconds()
        if time_span > 86400 * 30:  # 超过30天，只显示日期
            ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d', tz='UTC'))
        else:  # 少于30天，显示日期和时间
            ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d %H:%M', tz='UTC'))

    # 绘制成交量
    ax2.plot(volume_df['历史值'].index, volume_df['历史值'], label='历史值', color='gray', linewidth=1.5, alpha=0.7)
    
    # 如果有真实值，绘制真实值
    if '真实值' in volume_df.columns:
        ax2.plot(volume_df['真实值'].index, volume_df['真实值'], label='真实值', color='blue', linewidth=1.5)
    
    # 绘制预测值
    ax2.plot(pred_df.index, pred_df['volume'], label='预测值', color='red', linewidth=1.5, linestyle='--')
    
    ax2.set_ylabel('成交量', fontsize=14)
    ax2.set_xlabel('时间', fontsize=14)
    ax2.legend(loc='upper left', fontsize=12)
    ax2.grid(True)
    ax2.set_title('成交量预测对比', fontsize=16)
    
    # 格式化x轴时间显示
    if len(all_times) > 0:
        if time_span > 86400 * 30:  # 超过30天，只显示日期
            ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d', tz='UTC'))
        else:  # 少于30天，显示日期和时间
            ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d %H:%M', tz='UTC'))
    
    # 旋转x轴标签以便更好地显示
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.show()


def get_user_input():
    """获取用户输入的参数"""
    print("=" * 70)
    print("Kronos 金融数据预测系统 (支持akshare+yfinance数据源)")
    print("=" * 70)
    print("数据源说明:")
    print("  akshare: A股、港股、美股、期货等，支持1/5/15/30/60分钟、日线数据")
    print("  yfinance: 美股、加密货币等，支持多种时间频率")
    print()

    # 选择数据源类型
    data_source = input("请选择数据源（1=股票[akshare], 2=比特币[yfinance], 3=黄金[yfinance]，默认1）: ").strip()
    if not data_source:
        data_source = "1"

    if data_source == "2":
        # 比特币数据
        symbol = "BTC"
        print("已选择比特币 (BTC/USDT) 数据")
    elif data_source == "3":
        # 黄金数据
        symbol = "GOLD"
        print("已选择黄金 (GC=F) 数据")
    else:
        # 股票数据
        symbol = input("请输入股票代码（例如：000001, 600977）: ").strip()
        if not symbol:
            symbol = "600977"  # 默认值
            print(f"使用默认股票代码: {symbol}")

    # 获取数据周期 (akshare/yfinance支持的频率)
    period_input = input("请输入数据周期（daily=日线, 1/5/15/30/60=分钟线，默认daily）: ").strip().lower()
    if not period_input:
        period = "daily"
    elif period_input == "daily":
        period = "daily"
    elif period_input in ["1", "5", "15", "30", "60"]:
        period = period_input
        if period_input == "1":
            print("注意：1分钟数据需要大量历史数据，建议lookback至少200")
        elif period_input in ["5", "15"]:
            print("注意：高频数据波动较大，建议适当调整预测参数")
    else:
        print("无效的周期，使用默认周期: daily")
        period = "daily"

    # 获取历史数据长度 (根据akshare/yfinance数据源特性调整)
    while True:
        try:
            if period == "daily":
                default_lookback = 400
                prompt = f"请输入历史数据长度（用于预测，日线建议200-1000，默认{default_lookback}）: "
                max_limit = 2000  # 日线可以支持更多历史数据
            elif period in ["5", "15"]:
                default_lookback = 300
                prompt = f"请输入历史数据长度（用于预测，{period}分钟线建议100-800，默认{default_lookback}）: "
                max_limit = 1000
            elif period in ["30", "60"]:
                default_lookback = 200
                prompt = f"请输入历史数据长度（用于预测，{period}分钟线建议50-600，默认{default_lookback}）: "
                max_limit = 800
            else:
                default_lookback = 400
                prompt = f"请输入历史数据长度（用于预测，默认{default_lookback}）: "
                max_limit = 512

            lookback_input = input(prompt).strip()
            lookback = int(lookback_input) if lookback_input else default_lookback

            if lookback < 50:
                print("历史数据长度不能小于50，请重新输入")
                continue
            if lookback > max_limit:
                print(f"警告：历史数据长度超过{max_limit}，模型会自动截断")
            break
        except ValueError:
            print("请输入有效的数字")
    
    # 获取预测长度 (根据akshare/yfinance数据源特性调整)
    while True:
        try:
            if period == "daily":
                default_pred_len = 120
                prompt = f"请输入预测长度（预测未来多少个交易日，日线建议30-500，默认{default_pred_len}）: "
                max_limit = 500
            elif period in ["5", "15"]:
                default_pred_len = 100
                prompt = f"请输入预测长度（预测未来多少个{period}分钟，{period}分钟线建议20-200，默认{default_pred_len}）: "
                max_limit = 300
            elif period in ["30", "60"]:
                default_pred_len = 80
                prompt = f"请输入预测长度（预测未来多少个{period}分钟，{period}分钟线建议10-150，默认{default_pred_len}）: "
                max_limit = 200
            else:
                default_pred_len = 120
                prompt = f"请输入预测长度（预测未来多少个时间点，默认{default_pred_len}）: "
                max_limit = 500

            pred_len_input = input(prompt).strip()
            pred_len = int(pred_len_input) if pred_len_input else default_pred_len
            if pred_len < 1:
                print("预测长度必须大于0，请重新输入")
                continue
            if pred_len > max_limit:
                print(f"建议预测长度不超过{max_limit}，但仍继续执行")
            break
        except ValueError:
            print("请输入有效的数字")

    # 获取设备
    device = input("请输入设备（cpu/cuda:0，默认cpu）: ").strip().lower()
    if not device:
        device = "cpu"
    if device not in ["cpu", "cuda:0", "cuda:1"]:
        print("使用默认设备: cpu")
        device = "cpu"
    
    # 计算需要获取的数据天数（根据akshare/yfinance数据源特性调整）
    if period == "daily":
        # 日线数据：A股通常交易日约240天/年，留出足够缓冲
        days = max((lookback + pred_len) * 2, 1000)  # 至少2年数据
    elif period in ["5", "15"]:
        # 5/15分钟线：通常一天4-8小时交易时间，转换为天数
        trading_hours_per_day = 4  # 假设4小时交易时间
        data_points_per_day = trading_hours_per_day * 60 / int(period) if period != "daily" else 1
        days = max(int((lookback + pred_len) / data_points_per_day * 3), 60)  # 至少2个月数据
    elif period in ["1", "30", "60"]:
        # 其他分钟线：根据频率计算
        minutes_per_day = 4 * 60  # 假设4小时交易时间
        data_points_per_day = minutes_per_day / int(period) if period != "daily" else 1
        days = max(int((lookback + pred_len) / data_points_per_day * 2), 30)  # 至少1个月数据
    else:
        days = max((lookback + pred_len) * 2, 500)  # 默认值
    
    print()
    print("=" * 70)
    print("参数确认 (基于akshare/yfinance数据源优化):")
    if symbol == "BTC":
        print(f"  数据源: 比特币 (yfinance)")
    elif symbol == "GOLD":
        print(f"  数据源: 黄金 (yfinance)")
    else:
        print(f"  数据源: 股票 (akshare)")
        print(f"  股票代码: {symbol}")
    print(f"  历史数据长度: {lookback} (用于训练模型)")
    print(f"  预测长度: {pred_len} (预测未来{period}周期)")
    print(f"  数据周期: {period} ({'日线' if period == 'daily' else period + '分钟线'})")

    # 显示数据源特性提示
    if symbol in ["BTC", "GOLD"]:
        data_source_name = "yfinance"
    else:
        data_source_name = "akshare"

    if period == "daily":
        print(f"  数据获取: 约{days}个交易日 ({data_source_name}日线数据)")
    else:
        print(f"  数据获取: 约{days}个交易日 ({data_source_name}分钟线数据)")

    print(f"  计算设备: {device}")
    print(f"  模型配置: Kronos-base (适合akshare/yfinance数据)")
    print("=" * 70)
    print()
    
    return symbol, lookback, pred_len, period, device, days


def main():
    """主函数"""
    try:
        # 获取用户输入
        symbol, lookback, pred_len, period, device, days = get_user_input()
        
        # 1. 获取数据
        if symbol == "BTC":
            df = get_bitcoin_data(period=period, days=days)
        elif symbol == "GOLD":
            df = get_gold_data(period=period, days=days)
        else:
            df = get_stock_data_from_akshare(symbol, period=period, days=days)
        
        # 检查数据是否足够
        if len(df) < lookback + pred_len:
            print(f"警告：数据量不足。当前数据量: {len(df)}，需要至少: {lookback + pred_len}")
            print("将使用所有可用数据进行预测")
            lookback = min(lookback, len(df) - pred_len)
            if lookback < 50:
                raise ValueError("数据量太少，无法进行预测")
        
        # 2. 加载模型和分词器
        print("\n正在加载模型...")
        tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
        model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
        print("模型加载完成")
        
        # 3. 实例化预测器
        print("正在初始化预测器...")
        predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)
        print("预测器初始化完成")
        
        # 4. 准备数据
        print("\n正在准备预测数据...")
        # 使用最后lookback条数据作为历史数据（用于预测）
        x_df = df.tail(lookback)[['open', 'high', 'low', 'close', 'volume', 'amount']]
        x_timestamp = df.tail(lookback)['timestamps'].copy()
        
        # 确保x_timestamp是UTC时间
        if x_timestamp.dt.tz is None:
            x_timestamp = x_timestamp.dt.tz_localize('UTC')
        elif x_timestamp.dt.tz != pd.Timestamp.now(tz='UTC').tz:
            x_timestamp = x_timestamp.dt.tz_convert('UTC')
        
        # 生成未来时间戳（基于最后一个时间戳，使用UTC时间）
        last_timestamp = x_timestamp.iloc[-1]
        # 确保时间戳是UTC时间
        if hasattr(last_timestamp, 'tz') and last_timestamp.tz is None:
            last_timestamp = pd.Timestamp(last_timestamp).tz_localize('UTC')
        elif hasattr(last_timestamp, 'tz') and last_timestamp.tz != pd.Timestamp.now(tz='UTC').tz:
            last_timestamp = last_timestamp.tz_convert('UTC')
        
        if period == "daily":
            # 日线数据，每天一个点
            y_timestamp = pd.Series(pd.date_range(
                start=last_timestamp + timedelta(days=1),
                periods=pred_len,
                freq='D',
                tz='UTC'
            ))
        else:
            # 分钟线数据
            minutes = int(period)
            y_timestamp = pd.Series(pd.date_range(
                start=last_timestamp + timedelta(minutes=minutes),
                periods=pred_len,
                freq=f'{minutes}min',
                tz='UTC'
            ))
        
        # 5. 进行预测
        print(f"\n开始预测，预测长度: {pred_len}...")
        print(f"历史数据时间范围: {x_timestamp.iloc[0]} 至 {x_timestamp.iloc[-1]}")
        print(f"预测时间范围: {y_timestamp.iloc[0]} 至 {y_timestamp.iloc[-1]}")
        
        pred_df = predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=pred_len,
            T=1.0,
            top_p=0.9,
            sample_count=1,
            verbose=True
        )
        
        # 确保pred_df使用y_timestamp作为索引
        if not pred_df.index.equals(y_timestamp):
            print(f"警告：pred_df索引与y_timestamp不匹配，正在修正...")
            pred_df.index = y_timestamp
        
        # 6. 显示预测结果
        print("\n预测数据预览:")
        print(pred_df.head(10))
        print(f"\n预测数据统计:")
        print(pred_df.describe())
        print(f"\n预测数据时间范围: {pred_df.index[0]} 至 {pred_df.index[-1]}")
        
        # 7. 可视化
        print("\n正在生成可视化图表...")
        # 使用所有数据用于可视化，包括历史数据和可能的真实值
        kline_df = df.copy()
        plot_prediction(kline_df, pred_df, lookback)
        
        print("\n预测完成！")
        
    except KeyboardInterrupt:
        print("\n\n用户中断操作")
    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()


def test_gold_data():
    """测试黄金数据获取功能"""
    try:
        print("测试黄金数据获取...")
        df = get_gold_data(period="daily", days=30)
        print(f"✅ 黄金数据获取成功，共 {len(df)} 条记录")
        print("数据预览:")
        print(df.head())
        return True
    except Exception as e:
        print(f"❌ 黄金数据获取失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rate_limit_handling():
    """测试改进的速率限制处理"""
    print("🧪 测试改进的 Yahoo Finance 速率限制处理...")
    print("=" * 60)

    try:
        print("正在测试比特币数据获取 (短时间范围)...")
        df = get_bitcoin_data(period="daily", days=30)
        print(f"✅ 比特币数据获取成功，共 {len(df)} 条记录")
    except Exception as e:
        print(f"⚠️ 比特币数据获取失败: {e}")

    print("\n" + "=" * 60)

    try:
        print("正在测试黄金数据获取 (短时间范围)...")
        df = get_gold_data(period="daily", days=30)
        print(f"✅ 黄金数据获取成功，共 {len(df)} 条记录")
    except Exception as e:
        print(f"⚠️ 黄金数据获取失败: {e}")

    print("\n" + "=" * 60)
    print("💡 如果测试失败，建议:")
    print("   1. 使用 VPN 或代理服务器")
    print("   2. 等待 30-60 分钟后重试")
    print("   3. 运行时使用更短的时间范围")
    print("   4. 考虑使用其他数据源 API")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "--test-gold":
            test_gold_data()
        elif sys.argv[1] == "--test-rate-limit":
            test_rate_limit_handling()
        else:
            print("可用测试命令:")
            print("  --test-gold        测试黄金数据获取")
            print("  --test-rate-limit  测试改进的速率限制处理")
            print("  (无参数)           运行主程序")
    else:
        main()
