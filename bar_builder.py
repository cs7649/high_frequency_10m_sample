"""
bar_builder.py - Bar 构建器

基于 bar_M10_xts.py 重写，支持：
1. 灵活的 bar 频率（1m/5m/10m）
2. 为所有数据添加 bar_time 标签
3. Snap 数据的基础 OHLCV 计算（需要 diff）
4. Trade/Quote 数据的自定义聚合（不计算基础数据）

注意：特殊时段处理已在 data_loader.py 中完成
"""

import polars as pl
from datetime import time
from typing import List, Literal, Optional

from config import M1_TIMESTAMPS, M5_TIMESTAMPS, M10_TIMESTAMPS


# ============================================================
# 配置
# ============================================================

FREQ_MAP = {
    "1m": "1m",
    "5m": "5m",
    "10m": "10m",
    "1min": "1m",
    "5min": "5m",
    "10min": "10m",
}

# 每种频率的有效 bar_time（用于过滤）
VALID_BAR_TIMES = {
    "1m": [time.fromisoformat(t.replace(".000", "")) for t in M1_TIMESTAMPS],
    "5m": [time.fromisoformat(t.replace(".000", "")) for t in M5_TIMESTAMPS],
    "10m": [time.fromisoformat(t.replace(".000", "")) for t in M10_TIMESTAMPS],
}

# 每种频率的第一个 bar_time（用于处理 09:30 边界）
FIRST_BAR_TIME = {
    "1m": time(9, 31),
    "5m": time(9, 35),
    "10m": time(9, 40),
}

# 下午第一个 bar_time（用于处理 13:00 边界）
AFTERNOON_FIRST_BAR_TIME = {
    "1m": time(13, 1),
    "5m": time(13, 5),
    "10m": time(13, 10),
}


# ============================================================
# BarBuilder 类
# ============================================================

class BarBuilder:
    """
    Bar 构建器
    
    使用左开右闭 (start, end] 的窗口划分：
    - (09:30, 09:35] → bar_time = 09:35
    - (09:35, 09:40] → bar_time = 09:40
    - ...
    - (14:55, 15:00] → bar_time = 15:00
    
    使用示例：
        builder = BarBuilder(freq="5m")
        
        # 1. 只添加 bar_time 标签（适用于所有数据）
        labeled_lf = builder.add_bar_time(trade_lf)
        
        # 2. 自定义聚合（适用于 Trade/Quote）
        bar_df = builder.group_by_bar(trade_lf, agg_exprs=[...])
        
        # 3. 计算基础 OHLCV（适用于 Snap）
        bar_df = builder.group_by_bar_snap(snap_lf)
    """
    
    def __init__(self, freq: str = "5m"):
        """
        初始化 BarBuilder
        
        Args:
            freq: bar 频率，"1m"/"5m"/"10m"
        """
        self.freq = FREQ_MAP.get(freq.lower(), freq)
        self.valid_bar_times = VALID_BAR_TIMES[self.freq]
        self.first_bar_time = FIRST_BAR_TIME[self.freq]
        self.afternoon_first_bar_time = AFTERNOON_FIRST_BAR_TIME[self.freq]
    
    def add_bar_time(
        self,
        lf: pl.LazyFrame,
        time_col: str = "xts"
    ) -> pl.LazyFrame:
        """
        给数据添加 bar_time 列
        
        使用左开右闭 (start, end]：
        - 09:30:01 ~ 09:35:00 → bar_time = 09:35
        - 09:35:01 ~ 09:40:00 → bar_time = 09:40
        
        特殊处理（已在 data_loader 中调整时间）：
        - 09:30:01（原集合竞价）→ 归入第一个 bar
        - 11:29:59（原午休前）→ 归入 11:30 bar
        - 14:59:59（原收盘竞价）→ 归入 15:00 bar
        
        Args:
            lf: LazyFrame
            time_col: 时间列名
        
        Returns:
            添加了 bar_time 列的 LazyFrame
        """
        # 左开右闭的计算逻辑：
        # 如果 xts 刚好在边界上（如 09:35:00），属于该 bar
        # 否则，bar_time = truncate(xts) + freq
        
        bar_time_expr = (
            pl.when(
                # 刚好在边界上（如 09:35:00.000），属于当前 bar
                pl.col(time_col).dt.truncate(self.freq) == pl.col(time_col)
            )
            .then(pl.col(time_col))
            .otherwise(
                # 普通情况：truncate + offset
                pl.col(time_col).dt.truncate(self.freq).dt.offset_by(self.freq)
            )
            .alias("bar_time")
        )
        
        return lf.with_columns(bar_time_expr)
    
    def _filter_valid_bar_times(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        过滤有效的 bar_time
        
        只保留在 VALID_BAR_TIMES 中的时间点
        """
        return df.filter(
            pl.col("bar_time").dt.time().is_in(self.valid_bar_times)
        )
    
    def group_by_bar(
        self,
        lf: pl.LazyFrame,
        agg_exprs: List[pl.Expr],
        time_col: str = "xts",
        by: List[str] = None,
        filter_valid: bool = True
    ) -> pl.DataFrame:
        """
        按 bar 分组并自定义聚合（适用于 Trade/Quote）
        
        不计算基础 OHLCV，由因子自己定义聚合逻辑
        
        Args:
            lf: LazyFrame
            agg_exprs: 聚合表达式列表
            time_col: 时间列名
            by: 分组列，默认 ["symbol", "date"]
            filter_valid: 是否过滤有效 bar_time
        
        Returns:
            聚合后的 DataFrame
        """
        if by is None:
            by = ["symbol", "date"]
        
        # 1. 添加 bar_time 列
        lf = self.add_bar_time(lf, time_col)
        
        # 2. 排序
        lf = lf.sort([*by, time_col])
        
        # 3. 按 bar 聚合
        result = (
            lf
            .group_by([*by, "bar_time"])
            .agg(agg_exprs)
            .sort([*by, "bar_time"])
            .collect()
        )
        
        # 4. 过滤有效 bar_time
        if filter_valid:
            result = self._filter_valid_bar_times(result)
        
        return result
    
    def group_by_bar_trade(
        self,
        lf: pl.LazyFrame,
        time_col: str = "xts",
        price_col: str = "px",
        qty_col: str = "qty",
        amt_col: str = "amt",
        flag_col: str = "flag",
        by: List[str] = None,
        filter_valid: bool = True
    ) -> pl.DataFrame:
        """
        Trade 数据的 bar 聚合，计算基础 OHLCV
        
        与 Snap 的区别：
        - Trade 的 qty/amt 是每笔成交的值，直接 sum 即可，不需要 diff
        - OHLC 直接用 px 的 first/max/min/last
        
        输出字段：
        - open, high, low, close: OHLC
        - vol: 成交量
        - amt: 成交金额
        - vwap: 成交均价
        - ret: 收益率
        - pcls: 前收价
        - trade_count: 成交笔数
        
        Args:
            lf: Trade 数据的 LazyFrame
            time_col: 时间列名
            price_col: 价格列名
            qty_col: 成交量列名
            amt_col: 成交金额列名
            by: 分组列，默认 ["symbol", "date"]
            filter_valid: 是否过滤有效 bar_time
        
        Returns:
            聚合后的 DataFrame
        """
        if by is None:
            by = ["symbol", "date"]

        # 0. 过滤掉撤单记录和无效记录
        # flag=52表示撤单，flag=70表示成交
        # 同时过滤掉 px <= 0 的异常记录
        lf = lf.filter(pl.col(flag_col) != 52)
        
        # 1. 添加 bar_time 列
        lf = self.add_bar_time(lf, time_col)
        
        # 2. 排序
        lf = lf.sort([*by, time_col])
        
        # 3. 按 bar 聚合（Trade 直接聚合，不需要 diff）
        agg_df = (
            lf
            .group_by([*by, "bar_time"])
            .agg([
                # OHLC：直接用 px
                pl.col(price_col).first().alias("open"),
                pl.col(price_col).max().alias("high"),
                pl.col(price_col).min().alias("low"),
                pl.col(price_col).last().alias("close"),
                
                # vol/amt：直接 sum
                pl.col(qty_col).sum().alias("vol"),
                pl.col(amt_col).sum().alias("amt"),
                
                # 成交笔数
                pl.len().alias("trade_count"),
            ])
            .sort([*by, "bar_time"])
            .collect()
        )
        
        # 4. 过滤有效 bar_time
        if filter_valid:
            agg_df = self._filter_valid_bar_times(agg_df)
        
        # 5. 计算 vwap
        agg_df = agg_df.with_columns(
            pl.when(pl.col("vol") <= 0)
            .then(None)
            .otherwise(pl.col("amt") / pl.col("vol"))
            .alias("vwap")
        )
        
        # 6. 计算 pcls（前一个 bar 的 close）
        agg_df = agg_df.with_columns(
            pl.col("close").shift(1).over("symbol").alias("pcls")
        )
        
        # 7. 计算 ret
        agg_df = agg_df.with_columns(
            pl.when((pl.col("pcls").is_null()) | (pl.col("pcls") <= 0))
            .then(None)
            .otherwise(pl.col("close") / pl.col("pcls") - 1)
            .alias("ret")
        )
        
        # 8. 整理列顺序
        agg_df = agg_df.select([
            *by,
            "bar_time",
            "open",
            "high",
            "low",
            "close",
            "vol",
            "amt",
            "vwap",
            "ret",
            "pcls",
            "trade_count",
        ])
        
        return agg_df

    def group_by_bar_snap(
        self,
        lf: pl.LazyFrame,
        time_col: str = "xts",
        by: List[str] = None,
        filter_valid: bool = True
    ) -> pl.DataFrame:
        """
        Snap 数据的 bar 聚合，计算基础 OHLCV
        
        与 bar_M10_xts.py 逻辑一致：
        1. snap 的 qty/turnover 是累积值，需要 diff 得到增量
        2. high/low 需要检测 daily high/low 是否变化
        3. open 使用前一个 bar 的 close
        
        输出字段：
        - open, high, low, close: OHLC
        - vol: 成交量
        - amt: 成交金额
        - vwap: 成交均价
        - ret: 收益率
        - pcls: 前收价
        
        Args:
            lf: Snap 数据的 LazyFrame
            time_col: 时间列名
            by: 分组列，默认 ["symbol", "date"]
            filter_valid: 是否过滤有效 bar_time
        
        Returns:
            聚合后的 DataFrame
        """
        if by is None:
            by = ["symbol", "date"]
        
        # 1. 排序（在 diff 之前必须排序）
        lf = lf.sort([*by, time_col])
        
        # 2. 计算增量（snap 的 qty/turnover 是累积值）
        lf = lf.with_columns([
            pl.col("turnover").diff().over("symbol").alias("turnover_diff"),
            pl.col("qty").diff().over("symbol").alias("qty_diff"),
            # 记录前一个快照的 high/low
            pl.col("high").shift(1).over("symbol").alias("prev_high"),
            pl.col("low").shift(1).over("symbol").alias("prev_low"),
        ])
        
        # 3. 处理第一条记录（diff 为 null）
        lf = lf.with_columns([
            pl.when(pl.col("turnover_diff").is_null())
              .then(pl.col("turnover"))
              .otherwise(pl.col("turnover_diff"))
              .alias("turnover_incr"),
            pl.when(pl.col("qty_diff").is_null())
              .then(pl.col("qty"))
              .otherwise(pl.col("qty_diff"))
              .alias("qty_incr"),
        ])
        
        # 4. 计算 intra-snap 的 high/low
        # 如果 daily high 增加，说明两个快照之间有更高价成交
        lf = lf.with_columns([
            pl.when(pl.col("prev_high").is_null())
              .then(pl.col("high"))
              .when(pl.col("high") > pl.col("prev_high"))
              .then(pl.col("high"))
              .otherwise(pl.col("last"))
              .alias("intra_high"),
            
            pl.when(pl.col("prev_low").is_null())
              .then(pl.col("low"))
              .when(pl.col("low") < pl.col("prev_low"))
              .then(pl.col("low"))
              .otherwise(pl.col("last"))
              .alias("intra_low"),
        ])
        
        # 5. 添加 bar_time 列
        lf = self.add_bar_time(lf, time_col)
        
        # 6. 按 bar 聚合
        agg_df = (
            lf
            .group_by([*by, "bar_time"])
            .agg([
                # amt: 成交金额增量之和
                pl.col("turnover_incr").sum().alias("amt"),
                # vol: 成交量增量之和
                pl.col("qty_incr").sum().alias("vol"),
                # close: 窗口内最后一个 last
                pl.col("last").last().alias("close"),
                # high: 窗口内 intra_high 的最大值
                pl.col("intra_high").max().alias("high"),
                # low: 窗口内 intra_low 的最小值
                pl.col("intra_low").min().alias("low"),
                # first_last: 窗口内第一个 last（用于计算 open）
                pl.col("last").first().alias("first_last"),
                # pcls_orig: 昨收价（用于第一个 bar）
                pl.col("pcls").first().alias("pcls_orig"),
            ])
            .sort([*by, "bar_time"])
            .collect()
        )
        
        # 7. 过滤有效 bar_time
        if filter_valid:
            agg_df = self._filter_valid_bar_times(agg_df)
        
        # 8. 计算 open（前一个 bar 的 close）
        agg_df = agg_df.with_columns(
            pl.col("close").shift(1).over("symbol").alias("prev_close")
        )
        
        agg_df = agg_df.with_columns([
            # open: 前一个 bar 的 close，第一个 bar 用 first_last
            pl.when(pl.col("prev_close").is_null())
              .then(pl.col("first_last"))
              .otherwise(pl.col("prev_close"))
              .alias("open"),
            
            # pcls: 前一个 bar 的 close，第一个 bar 用昨收
            pl.when(pl.col("prev_close").is_null())
              .then(pl.col("pcls_orig"))
              .otherwise(pl.col("prev_close"))
              .alias("pcls"),
        ])
        
        # 9. 计算 vwap 和 ret
        agg_df = agg_df.with_columns([
            # vwap: amt / vol
            pl.when(pl.col("vol") <= 0)
              .then(None)
              .otherwise(pl.col("amt") / pl.col("vol"))
              .alias("vwap"),
            
            # ret: close / pcls - 1
            pl.when(pl.col("pcls") <= 0)
              .then(None)
              .otherwise(pl.col("close") / pl.col("pcls") - 1)
              .alias("ret"),
        ])
        
        # 10. 删除临时列，整理列顺序
        agg_df = agg_df.select([
            *by,
            "bar_time",
            "open",
            "high",
            "low",
            "close",
            "vol",
            "amt",
            "vwap",
            "ret",
            "pcls",
        ])
        
        return agg_df
