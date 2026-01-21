from datetime import time
from typing import List

# ============================================================
# 数据路径配置
# ============================================================
DEFAULT_TICK_DATA_PATH = "/local/g04nfs/data/tick/eq/std/"
LEGION_OUTPUT_PATH = "/home/ctsu/tosct/feature_framework"
EXCHANGES = ["SH", "SZ"]

# ============================================================
# 时间戳配置（复用自 ajdata）
# ============================================================
DAILY_TIMESTAMPS = ['15:00:00.000']

M10_TIMESTAMPS = [
    '09:40:00.000', '09:50:00.000', '10:00:00.000', '10:10:00.000',
    '10:20:00.000', '10:30:00.000', '10:40:00.000', '10:50:00.000',
    '11:00:00.000', '11:10:00.000', '11:20:00.000', '11:30:00.000',
    '13:10:00.000', '13:20:00.000', '13:30:00.000', '13:40:00.000',
    '13:50:00.000', '14:00:00.000', '14:10:00.000', '14:20:00.000',
    '14:30:00.000', '14:40:00.000', '14:50:00.000', '15:00:00.000'
]

M5_TIMESTAMPS = [
    '09:35:00.000', '09:40:00.000', '09:45:00.000', '09:50:00.000',
    '09:55:00.000', '10:00:00.000', '10:05:00.000', '10:10:00.000',
    '10:15:00.000', '10:20:00.000', '10:25:00.000', '10:30:00.000',
    '10:35:00.000', '10:40:00.000', '10:45:00.000', '10:50:00.000',
    '10:55:00.000', '11:00:00.000', '11:05:00.000', '11:10:00.000',
    '11:15:00.000', '11:20:00.000', '11:25:00.000', '11:30:00.000',
    '13:05:00.000', '13:10:00.000', '13:15:00.000', '13:20:00.000',
    '13:25:00.000', '13:30:00.000', '13:35:00.000', '13:40:00.000',
    '13:45:00.000', '13:50:00.000', '13:55:00.000', '14:00:00.000',
    '14:05:00.000', '14:10:00.000', '14:15:00.000', '14:20:00.000',
    '14:25:00.000', '14:30:00.000', '14:35:00.000', '14:40:00.000',
    '14:45:00.000', '14:50:00.000', '14:55:00.000', '15:00:00.000'
]

M1_TIMESTAMPS = [f'{h:02d}:{m:02d}:00.000' 
                 for h, m_range in [(9, range(31, 60)), (10, range(60)), 
                                    (11, range(31)), (13, range(1, 60)), 
                                    (14, range(60)), (15, range(1))]
                 for m in m_range]

# ============================================================
# 交易时段配置
# ============================================================
TRADING_SESSIONS = [
    (time(9, 15), time(11, 32)),   # 上午11:32之前的数据都要归到11:30的bar中（保证所有闭盘数据都保存）
    (time(13, 0), time(15, 15)),    # 下午15:15之前的数据都要归到15:00的bar中
]

# ============================================================
# 特殊时段配置
# ============================================================

# 时段边界
OPENING_AUCTION_END = time(9, 30)        # 集合竞价结束
MORNING_END = time(11, 30)                # 上午收盘
AFTERNOON_START = time(13, 0)             # 下午开盘
CLOSING_AUCTION_START = time(15, 0)       # 收盘竞价开始

# 调整目标时间
OPENING_AUCTION_ADJUST_TO = time(9, 30, 1)      # 集合竞价 → 09:30:01
NOON_BREAK_ADJUST_TO = time(11, 29, 59)          # 午休 → 11:29:59
CLOSING_AUCTION_ADJUST_TO = time(14, 59, 59)     # 收盘竞价 → 14:59:59

LUNCH_BREAK_MINUTES = 90  # 午休时长（分钟）

# ============================================================
# 工具函数
# ============================================================
def get_timestamps(freq: str) -> List[str]:
    """获取指定频率的时间戳列表"""
    freq_map = {
        'M1': M1_TIMESTAMPS,
        'M5': M5_TIMESTAMPS,
        'M10': M10_TIMESTAMPS,
        'DAILY': DAILY_TIMESTAMPS,
        'EOD': DAILY_TIMESTAMPS,
    }
    return freq_map.get(freq.upper(), [])


def get_bar_count_per_day(freq: str) -> int:
    """获取每天的 bar 数量"""
    return len(get_timestamps(freq))

# ============================================================
# Surge Factor 时段配置
# ============================================================

def get_trading_time_slice(bar_freq: str, trading_time: str):
    """
    获取指定交易时段的 bar_time 切片
    
    Args:
        bar_freq: bar 频率 "1m"/"5m"/"10m"
        trading_time: 时段标识
    
    Returns:
        list of time objects: 该时段内的有效 bar_time
    """
    from datetime import time
    
    # 获取该频率的所有时间戳
    all_timestamps = get_timestamps(bar_freq.upper())
    all_times = [time.fromisoformat(t.replace(".000", "")) for t in all_timestamps]
    
    # 定义时段边界
    time_ranges = {
        "all_day": (time(9, 31), time(15, 0)),      # 全天
        "morning": (time(9, 31), time(11, 30)),      # 上午
        "afternoon": (time(13, 1), time(15, 0)),     # 下午
        "opening": (time(9, 31), time(10, 0)),       # 开盘半小时
        "closing": (time(14, 31), time(15, 0)),      # 尾盘半小时
        "morning_mid": (time(10, 1), time(11, 30)),  # 上午中段
        "afternoon_mid": (time(13, 1), time(14, 30)),# 下午中段
    }
    
    if trading_time not in time_ranges:
        raise ValueError(f"Unknown trading_time: {trading_time}")
    
    start_time, end_time = time_ranges[trading_time]
    
    # 过滤出该时段的时间
    return [t for t in all_times if start_time <= t <= end_time]


def get_bars_per_trading_time(bar_freq: str, trading_time: str) -> int:
    """获取指定时段的 bar 数量"""
    return len(get_trading_time_slice(bar_freq, trading_time))
