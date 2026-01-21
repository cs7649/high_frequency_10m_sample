#data_loader.py
import os
import polars as pl
import argparse
import datetime
from typing import List, Optional, Union
from pathlib import Path
from functools import wraps

from config import DEFAULT_TICK_DATA_PATH, EXCHANGES, TRADING_SESSIONS


# ============================================================
# 工具函数
# ============================================================

def add_exchange_suffix(lf: pl.LazyFrame, symbol_col: str = "inst_id") -> pl.LazyFrame:
    """
    添加交易所后缀
    
    Args:
        lf: LazyFrame
        symbol_col: 原始股票代码列名（默认 "inst_id"）
    
    Returns:
        添加了 "symbol" 列（带后缀）的 LazyFrame
    """
    # 补齐6位
    lf = lf.with_columns(
        pl.col(symbol_col)
        .cast(pl.Utf8)
        .str.pad_start(6, "0")
        .alias(symbol_col)
    )
    
    # 根据前缀添加后缀，创建新的 symbol 列
    lf = lf.with_columns(
        pl.when(
            pl.col(symbol_col).str.starts_with("60") | 
            pl.col(symbol_col).str.starts_with("68")
        ).then(pl.col(symbol_col) + ".SH")
        .when(
            pl.col(symbol_col).str.starts_with("00") | 
            pl.col(symbol_col).str.starts_with("30")
        ).then(pl.col(symbol_col) + ".SZ")
        .otherwise(pl.col(symbol_col))
        .alias("symbol")
    )
    
    # 过滤掉非 SH/SZ 的代码（ETF、债券等）
    return lf.filter(pl.col("symbol").str.contains(r"\.SH|\.SZ"))


def filter_trading_hours(lf: pl.LazyFrame, time_col: str = "xts") -> pl.LazyFrame:
    """
    过滤交易时段（包含竞价时段）
    
    时间范围：
    - 上午：09:15~11:32（包含集合竞价和午休前数据）
    - 下午：13:00~15:15（包含收盘竞价后数据）
    """
    morning_start, morning_end = TRADING_SESSIONS[0]
    afternoon_start, afternoon_end = TRADING_SESSIONS[1]
    
    return lf.filter(
        ((pl.col(time_col).dt.time() >= morning_start) & 
         (pl.col(time_col).dt.time() <= morning_end)) |
        ((pl.col(time_col).dt.time() >= afternoon_start) & 
         (pl.col(time_col).dt.time() <= afternoon_end))
    )

def adjust_special_time(lf: pl.LazyFrame, time_col: str = "xts") -> pl.LazyFrame:
    """
    调整特殊时段的时间戳，使其归入正确的 bar
    
    处理规则：
    1. 集合竞价（<= 09:30:00）→ 调整到 09:30:01（归入第一个 bar）
    2. 午休（> 11:30:00 且 < 13:00:00）→ 调整到 11:29:59（归入上午最后一个 bar）
    3. 收盘竞价（>= 15:00:00）→ 调整到 14:59:59（归入最后一个 bar）
    
    Args:
        lf: LazyFrame
        time_col: 时间列名，默认 "xts"
    
    Returns:
        调整后的 LazyFrame
    """
    from config import (
        OPENING_AUCTION_END, MORNING_END, AFTERNOON_START, CLOSING_AUCTION_START,
        OPENING_AUCTION_ADJUST_TO, NOON_BREAK_ADJUST_TO, CLOSING_AUCTION_ADJUST_TO
    )
    
    return lf.with_columns(
        pl.when(pl.col(time_col).dt.time() <= OPENING_AUCTION_END)
        # 集合竞价：调整到 09:30:01
        .then(
            pl.datetime(
                pl.col(time_col).dt.year(),
                pl.col(time_col).dt.month(),
                pl.col(time_col).dt.day(),
                pl.lit(OPENING_AUCTION_ADJUST_TO.hour),
                pl.lit(OPENING_AUCTION_ADJUST_TO.minute),
                pl.lit(OPENING_AUCTION_ADJUST_TO.second)
            )
        )
        .when(
            (pl.col(time_col).dt.time() > MORNING_END) & 
            (pl.col(time_col).dt.time() < AFTERNOON_START)
        )
        # 午休：调整到 11:29:59
        .then(
            pl.datetime(
                pl.col(time_col).dt.year(),
                pl.col(time_col).dt.month(),
                pl.col(time_col).dt.day(),
                pl.lit(NOON_BREAK_ADJUST_TO.hour),
                pl.lit(NOON_BREAK_ADJUST_TO.minute),
                pl.lit(NOON_BREAK_ADJUST_TO.second)
            )
        )
        .when(pl.col(time_col).dt.time() >= CLOSING_AUCTION_START)
        # 收盘竞价：调整到 14:59:59
        .then(
            pl.datetime(
                pl.col(time_col).dt.year(),
                pl.col(time_col).dt.month(),
                pl.col(time_col).dt.day(),
                pl.lit(CLOSING_AUCTION_ADJUST_TO.hour),
                pl.lit(CLOSING_AUCTION_ADJUST_TO.minute),
                pl.lit(CLOSING_AUCTION_ADJUST_TO.second)
            )
        )
        .otherwise(pl.col(time_col))
        .alias(time_col)
    )
# ============================================================
# DataLoader 类
# ============================================================

class DataLoader:
    """
    Tick 数据加载器
    
    使用示例：
        loader = DataLoader()
        trade_lf = loader.load_trade(["20220104", "20220105"])
        trade_df = trade_lf.collect()
    
    对应旧框架：
        trade = pl.read_parquet(f"{data_path}trade/{exchange}/{date}")
    """
    
    def __init__(
        self, 
        data_path: str = DEFAULT_TICK_DATA_PATH, 
        exchanges: List[str] = None
    ):
        self.data_path = data_path.rstrip("/") + "/"
        self.exchanges = exchanges if exchanges is not None else EXCHANGES
    
    def _load_single_file(
        self,
        data_type: str,
        exchange: str,
        date: str,
        columns: Optional[List[str]] = None
    ) -> pl.LazyFrame:
        """
        加载单个 Parquet 文件（lazy）
        
        Args:
            data_type: 数据类型 ("trade", "quote", "snap")
            exchange: 交易所 ("SH", "SZ")
            date: 日期 YYYYMMDD
            columns: 需要的列（None 表示全部）
        """
        file_path = f"{self.data_path}{data_type}/{exchange}/{date}"
        lf = pl.scan_parquet(file_path)
        
        # 选择需要的列
        if columns:
            lf = lf.select(columns)
        
        # 添加 date 和 exchange 列
        lf = lf.with_columns([
            pl.lit(date).alias("date"),
            pl.lit(exchange).alias("exchange")
        ])
        
        return lf
    
    def _load_data(
        self,
        date_list: List[str],
        data_type: str,
        columns: Optional[List[str]] = None,
        filter_hours: bool = True,
        add_suffix: bool = True,
    ) -> pl.LazyFrame:
        """
        加载数据的通用方法
        
        Args:
            date_list: 日期列表
            data_type: 数据类型 ("trade", "quote", "snap")
            columns: 需要的列（None 表示全部）
            filter_hours: 是否过滤交易时段
            add_suffix: 是否添加交易所后缀
        
        Returns:
            合并后的 LazyFrame
        """
        # 加载所有文件
        lazy_frames = []
        for date in date_list:
            for exchange in self.exchanges:
                lf = self._load_single_file(data_type, exchange, date, columns)
                lazy_frames.append(lf)
        
        # 合并数据
        combined_lf = pl.concat(lazy_frames, how="diagonal")
        
        # 过滤交易时段
        if filter_hours:
            combined_lf = filter_trading_hours(combined_lf, time_col="xts")
        
        # 调整特殊时段
        combined_lf = adjust_special_time(combined_lf, time_col="xts")
        
        # 添加交易所后缀
        if add_suffix:
            combined_lf = add_exchange_suffix(combined_lf, symbol_col="inst_id")
        
        return combined_lf
    
    def load_trade(
        self,
        date_list: Union[str, List[str]],
        columns: Optional[List[str]] = None,
        filter_hours: bool = True,
        add_suffix: bool = True
    ) -> pl.LazyFrame:
        """
        加载 Trade（逐笔成交）数据
        
        Args:
            date_list: 日期或日期列表，格式 "YYYYMMDD"
            columns: 需要的列，None 表示全部
            filter_hours: 是否过滤交易时段（默认 True）
            add_suffix: 是否添加交易所后缀（默认 True）
        
        Returns:
            pl.LazyFrame: Trade 数据
        """
        if isinstance(date_list, str):
            date_list = [date_list]
        
        return self._load_data(
            date_list=date_list,
            data_type="trade",
            columns=columns,
            filter_hours=filter_hours,
            add_suffix=add_suffix
        )
    
    def load_quote(
        self,
        date_list: Union[str, List[str]],
        columns: Optional[List[str]] = None,
        filter_hours: bool = True,
        add_suffix: bool = True
    ) -> pl.LazyFrame:
        """加载 Quote（逐笔委托）数据"""
        if isinstance(date_list, str):
            date_list = [date_list]
        
        return self._load_data(
            date_list=date_list,
            data_type="quote",
            columns=columns,
            filter_hours=filter_hours,
            add_suffix=add_suffix
        )
    
    def load_snap(
        self,
        date_list: Union[str, List[str]],
        columns: Optional[List[str]] = None,
        filter_hours: bool = True,
        add_suffix: bool = True
    ) -> pl.LazyFrame:
        """加载 Snap（快照）数据"""
        if isinstance(date_list, str):
            date_list = [date_list]
        
        return self._load_data(
            date_list=date_list,
            data_type="snap",
            columns=columns,
            filter_hours=filter_hours,
            add_suffix=add_suffix
        )
