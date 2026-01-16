# Yahoo Finance 速率限制解决方案

## 问题描述

Yahoo Finance (yfinance) API 存在严格的速率限制，导致频繁出现 "Too Many Requests" 错误。

## 改进的解决方案

### 1. 增强的重试机制
- **指数退避**: 失败后等待时间呈指数增长 (60s, 120s, 240s...)
- **随机抖动**: 在退避时间上添加随机因子，避免多个请求同时重试
- **分批获取**: 将大时间范围的数据分成小批次获取
- **批次间延迟**: 批次间增加 5-10 秒的随机延迟

### 2. 更小的批次大小
- 从原来的 1 年批次减少到 6 个月批次
- 进一步降低单次请求的频率

### 3. 备用获取方法
- 当主要方法失败时，自动尝试更保守的备用策略
- 使用更小的批次 (3 个月) 和更长的延迟 (30-60 秒)

## 使用方法

### 正常使用
```bash
python prediction_example.py
```

### 测试改进的处理
```bash
python prediction_example.py --test-rate-limit
```

## 替代解决方案

如果仍然遇到速率限制，可以考虑以下替代方案：

### 1. 使用代理/VPN
```bash
# 使用代理服务器
export HTTP_PROXY=http://your-proxy:port
export HTTPS_PROXY=http://your-proxy:port
```

### 2. 使用其他数据源 API

#### Alpha Vantage
```python
import requests

def get_alpha_vantage_data(symbol, api_key):
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={api_key}"
    response = requests.get(url)
    return response.json()
```

#### CoinGecko API (免费)
```python
import requests

def get_coingecko_data(coin_id="bitcoin", days=30):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days={days}"
    response = requests.get(url)
    return response.json()
```

#### IEX Cloud
```python
import requests

def get_iex_data(symbol, token):
    url = f"https://cloud.iexapis.com/stable/stock/{symbol}/chart/1y?token={token}"
    response = requests.get(url)
    return response.json()
```

### 3. 本地数据缓存
- 下载历史数据到本地 CSV 文件
- 修改代码从本地文件读取数据

```python
def load_local_data(filepath):
    return pd.read_csv(filepath, parse_dates=['timestamp'])
```

## 最佳实践

1. **避免高峰期使用**: Yahoo Finance 在交易时间限制更严格
2. **使用较短时间范围**: 从 30-90 天开始测试
3. **实现请求缓存**: 避免重复请求相同数据
4. **设置合理的重试策略**: 不要设置过高的重试次数
5. **监控使用情况**: 添加日志记录请求频率

## 配置建议

### 减少默认数据量
```python
# 修改默认参数
default_days = 180  # 从 500 天减少到 180 天
default_lookback = 100  # 从 400 减少到 100
```

### 增加延迟设置
```python
# 在配置文件中设置延迟参数
BATCH_DELAY_MIN = 5.0
BATCH_DELAY_MAX = 10.0
RETRY_BASE_DELAY = 60
MAX_RETRY_DELAY = 600
```

## 故障排除

### 常见错误及解决方案

1. **YFRateLimitError**: 速率限制
   - 解决方案: 等待更长时间，使用代理，或切换到其他数据源

2. **YFException**: yfinance 内部错误
   - 解决方案: 检查网络连接，尝试备用方法

3. **ConnectionError**: 网络连接问题
   - 解决方案: 检查网络设置，使用代理服务器

### 调试模式
启用详细日志来诊断问题：
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 性能优化

1. **并发请求**: 使用异步请求减少总等待时间
2. **数据压缩**: 在传输大量数据时启用压缩
3. **连接复用**: 复用 HTTP 连接减少握手开销
4. **智能缓存**: 实现基于时间的缓存策略
