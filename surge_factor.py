"""
surge_factor.py - Surge因子计算器（修复版）

修复内容：
1. M10输出：无论用1m/5m/10m挖掘，最终都聚合到M10的24个时间点
2. EOD输出：bar_time统一为15:00:00.000
"""

import polars as pl
import numpy as np
from datetime import time, datetime, timedelta
from typing import List, Literal, Optional, Union, Dict

from dux.cal import bizdays, bizday

from data_loader import DataLoader
from bar_builder import BarBuilder
from config import (
    get_timestamps, 
    get_bar_count_per_day,
    get_trading_time_slice,
    get_bars_per_trading_time,
    M10_TIMESTAMPS,
    DAILY_TIMESTAMPS,
)


# ============================================================
# M10时间映射工具
# ============================================================

def get_m10_bar_time(bar_time: time) -> time:
    """
    将任意bar_time映射到对应的M10 bar_time
    
    规则：使用左开右闭 (start, end]
    - (09:30, 09:40] → 09:40
    - (09:40, 09:50] → 09:50
    - ...
    
    Args:
        bar_time: 原始bar时间（可能是1m/5m/10m）
    
    Returns:
        对应的M10 bar时间
    """
    # M10的时间点列表
    m10_times = [time.fromisoformat(t.replace(".000", "")) for t in M10_TIMESTAMPS]
    
    # 特殊处理：如果是午休前的时间(11:30之后，13:00之前)，归到11:30
    if time(11, 30) < bar_time < time(13, 0):
        return time(11, 30)
    
    # 特殊处理：如果是收盘后(15:00之后)，归到15:00
    if bar_time > time(15, 0):
        return time(15, 0)
    
    # 找到第一个 >= bar_time 的M10时间点
    for m10_time in m10_times:
        if m10_time >= bar_time:
            return m10_time
    
    # 默认返回最后一个（15:00）
    return m10_times[-1]


def build_m10_bar_time_mapping() -> Dict[time, time]:
    """
    构建从1m/5m bar_time到M10 bar_time的映射表
    
    Returns:
        Dict[原始time, M10 time]
    """
    from config import M1_TIMESTAMPS, M5_TIMESTAMPS
    
    mapping = {}
    
    # 处理所有1m时间点
    for t_str in M1_TIMESTAMPS:
        t = time.fromisoformat(t_str.replace(".000", ""))
        mapping[t] = get_m10_bar_time(t)
    
    # 处理所有5m时间点
    for t_str in M5_TIMESTAMPS:
        t = time.fromisoformat(t_str.replace(".000", ""))
        mapping[t] = get_m10_bar_time(t)
    
    # 处理所有10m时间点（映射到自己）
    for t_str in M10_TIMESTAMPS:
        t = time.fromisoformat(t_str.replace(".000", ""))
        mapping[t] = t
    
    return mapping


class SurgeFactor:
    """
    Surge因子计算器（修复版）
    
    修复内容：
    1. M10输出：无论用1m/5m/10m挖掘，最终都聚合到M10的24个时间点
    2. EOD输出：bar_time统一为15:00:00.000
    """
    
    def __init__(
        self,
        # ===== 基础参数 =====
        bar_freq: str = "1m",
        output_freq: str = "EOD",
        
        # ===== Surge识别参数 =====
        threshold: float = 1.0,
        
        # ===== EOD专用参数 =====
        trading_time: str = "all_day",
        factor_type: str = "surge_ret",
        surge_window: int = 5,
        
        # ===== M10专用参数 =====
        m10_method: str = "same_time",
        lookback_days: int = 20,
        lookback_bars: int = 48,
        
        # ===== 聚合统计量 =====
        intraday_stat: str = "mean",
        
        # ===== 其他参数 =====
        price_type: str = None,
        data_path: str = None,
    ):
        # 转换 "1m" -> "M1", "5m" -> "M5", "10m" -> "M10"
        freq_map = {
            "1m": "M1", "1min": "M1", "m1": "M1",
            "5m": "M5", "5min": "M5", "m5": "M5",
            "10m": "M10", "10min": "M10", "m10": "M10",
        }
        self.bar_freq = freq_map.get(bar_freq.lower(), bar_freq.upper())
        self.output_freq = output_freq.upper()
        self.threshold = threshold
        
        # EOD参数
        self.trading_time = trading_time
        self.factor_type = factor_type
        self.surge_window = surge_window
        
        # M10参数
        self.m10_method = m10_method
        self.lookback_days = lookback_days
        self.lookback_bars = lookback_bars
        
        # 聚合统计量
        self.intraday_stat = intraday_stat
        
        # 其他参数
        self.price_type = price_type
        
        # 初始化loader和builder
        self.loader = DataLoader(data_path=data_path) if data_path else DataLoader()
        self.bar_builder = BarBuilder(freq=bar_freq)
        
        # 参数验证
        self._validate_params()
        
        # 存储数据的属性
        self.bar_data = None
        
        # 构建M10映射表（用于聚合）
        self._m10_mapping = build_m10_bar_time_mapping()
    
    def _validate_params(self):
        """参数验证"""
        if self.output_freq not in ["EOD", "M10"]:
            raise ValueError(f"output_freq必须是'EOD'或'M10'，当前: {self.output_freq}")
        
        if self.output_freq == "EOD":
            if self.factor_type not in ["surge_ret", "surge_vol"]:
                raise ValueError(f"factor_type必须是'surge_ret'或'surge_vol'，当前: {self.factor_type}")
        
        if self.output_freq == "M10":
            if self.m10_method not in ["same_time", "rolling"]:
                raise ValueError(f"m10_method必须是'same_time'或'rolling'，当前: {self.m10_method}")
            if self.factor_type != "surge_ret":
                print(f"⚠️  M10模式只支持surge_ret，已自动设置")
                self.factor_type = "surge_ret"
        
        print(f"✓ 参数验证通过")
        print(f"  - 输出模式: {self.output_freq}")
        print(f"  - Bar频率: {self.bar_freq}")
        print(f"  - 因子类型: {self.factor_type}")
        print(f"  - 聚合统计量: {self.intraday_stat}")
        if self.output_freq == "EOD":
            print(f"  - 交易时段: {self.trading_time}")
        else:
            print(f"  - M10方法: {self.m10_method}")
            if self.m10_method == "same_time":
                print(f"  - 回看天数: {self.lookback_days}")
            else:
                print(f"  - 回看Bar数: {self.lookback_bars}")

    # ============================================================
    # 数据加载部分
    # ============================================================
    
    def load_and_build_bars(
        self, 
        bizdays_str: str = None,
        date_list: List[str] = None,
        add_intraday_ret: bool = True
    ) -> pl.DataFrame:
        """加载trade数据并合成bar"""
        if date_list is not None:
            dates = date_list
        elif bizdays_str is not None:
            dates = bizdays(bizdays_str)
        else:
            raise ValueError("必须提供 bizdays_str 或 date_list")
        
        print(f"📊 加载数据: {dates[0]} ~ {dates[-1]}，共 {len(dates)} 天，频率: {self.bar_freq}")
        
        trade_lf = self.loader.load_trade(
            date_list=dates,
            columns=["inst_id", "xts", "px", "qty", "amt", "flag"]
        )
        
        bar_df = self.bar_builder.group_by_bar_trade(
            lf=trade_lf,
            time_col="xts",
            price_col="px",
            qty_col="qty",
            amt_col="amt",
            flag_col="flag",
            filter_valid=True
        )
        
        print(f"✓ Bar数据生成完成: {len(bar_df)} 条记录")
        print(f"  - 股票数: {bar_df['symbol'].n_unique()}")
        print(f"  - 日期范围: {bar_df['date'].min()} ~ {bar_df['date'].max()}")
        
        if add_intraday_ret:
            bar_df = self._add_bar_returns(bar_df)
        
        self.bar_data = bar_df
        return bar_df
    
    def _add_bar_returns(self, bar_df: pl.DataFrame) -> pl.DataFrame:
        """添加bar内收益率"""
        return bar_df.with_columns(
            pl.when(pl.col("open") <= 0)
            .then(None)
            .otherwise((pl.col("close") - pl.col("open")) / pl.col("open"))
            .alias("bar_ret")
        )

    # ============================================================
    # 【核心修复】添加M10 bar_time映射
    # ============================================================
    
    def _add_m10_bar_time(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        添加m10_bar_time列，将任意频率的bar_time映射到M10时间点
        
        用于M10输出时的聚合
        """
        # 提取bar_time的时间部分
        bar_times = df["bar_time"].dt.time().to_list()
        
        # 映射到M10时间
        m10_times = [self._m10_mapping.get(t, t) for t in bar_times]
        
        # 重建完整的datetime（保留日期部分）
        dates = df["bar_time"].dt.date().to_list()
        m10_datetimes = [
            datetime.combine(d, t) for d, t in zip(dates, m10_times)
        ]
        
        return df.with_columns(
            pl.Series("m10_bar_time", m10_datetimes).cast(pl.Datetime)
        )

    # ============================================================
    # Surge识别部分 - EOD模式
    # ============================================================
    
    def _identify_surge_eod(self, bar_df: pl.DataFrame) -> pl.DataFrame:
        """EOD模式的surge识别"""
        print(f"🔍 EOD Surge识别: {self.trading_time}, threshold={self.threshold}")
        
        valid_times = get_trading_time_slice(self.bar_freq, self.trading_time)
        
        print(f"  - Bar频率: {self.bar_freq}")
        print(f"  - 有效时间段数: {len(valid_times)}")
        
        df = bar_df.filter(
            pl.col("bar_time").dt.time().is_in(valid_times)
        )
        
        print(f"  - 筛选后bar数: {len(df)}")
        
        df = df.with_columns([
            pl.col("vol").mean().over(["symbol", "date"]).alias("vol_mean"),
            pl.col("vol").std().over(["symbol", "date"]).alias("vol_std"),
        ])
        
        df = df.with_columns(
            pl.when(pl.col("vol_std").is_null() | (pl.col("vol_std") == 0))
            .then(False)
            .otherwise(
                pl.col("vol") > (pl.col("vol_mean") + self.threshold * pl.col("vol_std"))
            )
            .alias("is_surge")
        )
        
        surge_ratio = df["is_surge"].sum() / len(df)
        print(f"  - Surge占比: {surge_ratio:.2%}")
        
        return df

    # ============================================================
    # Surge识别部分 - M10 same_time模式
    # ============================================================
    
    def _identify_surge_m10_same_time(self, bar_df: pl.DataFrame) -> pl.DataFrame:
        """
        M10模式 - same_time方法
        
        【重要】Surge识别使用原始bar_time（1m/5m/10m），不是M10时间点
        M10聚合在后续的_aggregate_surge_ret中进行
        """
        print(f"🔍 M10 Surge识别 (same_time): H={self.lookback_days}天, threshold={self.threshold}")
        
        # 使用原始bar_time的时间部分进行同时刻比较
        bar_df = bar_df.with_columns(
            pl.col("bar_time").dt.time().alias("bar_time_only")
        )
        
        dates = sorted(bar_df["date"].unique().to_list())
        
        result_list = []
        
        for target_date in dates:
            date_idx = dates.index(target_date)
            lookback_dates = dates[max(0, date_idx - self.lookback_days):date_idx]
            
            if len(lookback_dates) < self.lookback_days:
                print(f"  ⚠️  {target_date}: 历史数据不足({len(lookback_dates)}天 < {self.lookback_days}天)，跳过")
                continue
            
            baseline_df = bar_df.filter(pl.col("date").is_in(lookback_dates))
            
            # 使用原始bar_time_only进行同时刻比较（如09:31比09:31，09:32比09:32）
            baseline_stats = (
                baseline_df
                .group_by(["symbol", "bar_time_only"])
                .agg([
                    pl.col("vol").mean().alias("vol_mean_baseline"),
                    pl.col("vol").std().alias("vol_std_baseline"),
                ])
            )
            
            target_df = bar_df.filter(pl.col("date") == target_date)
            
            target_df = target_df.join(
                baseline_stats,
                on=["symbol", "bar_time_only"],
                how="left"
            )
            
            target_df = target_df.with_columns(
                pl.when(
                    pl.col("vol_std_baseline").is_null() | 
                    (pl.col("vol_std_baseline") == 0)
                )
                .then(False)
                .otherwise(
                    pl.col("vol") > (pl.col("vol_mean_baseline") + self.threshold * pl.col("vol_std_baseline"))
                )
                .alias("is_surge")
            )
            
            result_list.append(target_df)
        
        if not result_list:
            raise ValueError(f"所有日期的历史数据都不足{self.lookback_days}天，无法计算surge")
        
        result_df = pl.concat(result_list)
        result_df = result_df.drop("bar_time_only")
        
        surge_ratio = result_df["is_surge"].sum() / len(result_df)
        print(f"  - 有效日期数: {len(result_list)}/{len(dates)}")
        print(f"  - Surge占比: {surge_ratio:.2%}")
        
        return result_df

    # ============================================================
    # Surge识别部分 - M10 rolling模式
    # ============================================================
    
    def _identify_surge_m10_rolling(self, bar_df: pl.DataFrame) -> pl.DataFrame:
        """
        M10模式 - rolling方法
        
        【重要】Surge识别使用原始bar_time（1m/5m/10m）的顺序
        M10聚合在后续的_aggregate_surge_ret中进行
        """
        print(f"🔍 M10 Surge识别 (rolling): k={self.lookback_bars}根, threshold={self.threshold}")
        
        # 不在这里添加m10_bar_time，保持原始bar顺序进行rolling
        df = bar_df.sort(["symbol", "date", "bar_time"])
        
        df = df.with_columns([
            pl.col("vol")
              .rolling_mean(window_size=self.lookback_bars, min_periods=self.lookback_bars)
              .shift(1)
              .over("symbol")
              .alias("vol_mean_baseline"),
            
            pl.col("vol")
              .rolling_std(window_size=self.lookback_bars, min_periods=self.lookback_bars)
              .shift(1)
              .over("symbol")
              .alias("vol_std_baseline"),
        ])
        
        df = df.with_columns(
            pl.when(
                pl.col("vol_std_baseline").is_null() | 
                (pl.col("vol_std_baseline") == 0)
            )
            .then(False)
            .otherwise(
                pl.col("vol") > (pl.col("vol_mean_baseline") + self.threshold * pl.col("vol_std_baseline"))
            )
            .alias("is_surge")
        )
        
        valid_count = df["is_surge"].is_not_null().sum()
        surge_count = df["is_surge"].sum()
        surge_ratio = surge_count / valid_count if valid_count > 0 else 0
        
        print(f"  - 有效bar数: {valid_count}/{len(df)}")
        print(f"  - Surge占比: {surge_ratio:.2%}")
        
        return df

    # ============================================================
    # 【核心修复】因子聚合部分 - surge_ret
    # ============================================================
    
    def _aggregate_surge_ret(self, surge_df: pl.DataFrame) -> pl.DataFrame:
        """
        聚合surge_ret因子
        
        【修复流程】
        1. 先筛选is_surge=True的bar（使用原始1m/5m/10m bar_time）
        2. 筛选完之后，再映射到M10时间点进行聚合
        3. EOD模式：聚合到每天一个值，bar_time设为15:00:00.000
        """
        print(f"📊 聚合surge_ret因子: intraday_stat={self.intraday_stat}")
        
        # Step 1: 筛选surge时刻（此时还是原始的1m/5m/10m bar）
        surge_moments = surge_df.filter(pl.col("is_surge") == True)
        
        if len(surge_moments) == 0:
            raise ValueError("没有检测到surge时刻，无法计算因子")
        
        print(f"  - Surge时刻数: {len(surge_moments)}")
        
        # Step 2: 根据输出频率选择分组方式
        if self.output_freq == "EOD":
            # EOD: 按(symbol, date)聚合，不需要m10映射
            group_cols = ["symbol", "date"]
            agg_expr = pl.col("bar_ret").__getattribute__(self.intraday_stat)().alias("factor_value")
            
            factor_df = (
                surge_moments
                .group_by(group_cols)
                .agg(agg_expr)
            )
        else:
            # M10: 筛选完surge bar后，再添加m10_bar_time进行聚合
            # 【关键】这里才进行1m/5m到M10的映射
            surge_moments = self._add_m10_bar_time(surge_moments)
            
            print(f"  - 映射到M10时间点后进行聚合")
            
            group_cols = ["symbol", "date", "m10_bar_time"]
            agg_expr = pl.col("bar_ret").__getattribute__(self.intraday_stat)().alias("factor_value")
            
            factor_df = (
                surge_moments
                .group_by(group_cols)
                .agg(agg_expr)
            )
            
            # 重命名为bar_time（保持输出格式一致）
            factor_df = factor_df.rename({"m10_bar_time": "bar_time"})
        
        print(f"  - 聚合后记录数: {len(factor_df)}")
        
        # Step 3: EOD模式添加标准的bar_time (15:00:00.000)
        if self.output_freq == "EOD":
            factor_df = self._add_eod_bar_time(factor_df)
        
        return factor_df
    
    def _add_eod_bar_time(self, factor_df: pl.DataFrame) -> pl.DataFrame:
        """
        为EOD因子添加标准的bar_time列 (15:00:00.000)
        
        Legion保存EOD因子时需要这个时间戳
        """
        # 从date列构建完整的datetime
        # date列可能是int (20220104) 或 str ("20220104")
        
        dates = factor_df["date"].to_list()
        
        # 统一转换为datetime，时间部分为15:00:00
        bar_times = []
        for d in dates:
            if isinstance(d, int):
                d_str = str(d)
            else:
                d_str = d
            dt = datetime.strptime(d_str, "%Y%m%d").replace(hour=15, minute=0, second=0)
            bar_times.append(dt)
        
        return factor_df.with_columns(
            pl.Series("bar_time", bar_times).cast(pl.Datetime)
        )

    # ============================================================
    # 因子聚合部分 - surge_vol (保持原有逻辑，添加EOD bar_time)
    # ============================================================
    
    def _aggregate_surge_vol(self, surge_df: pl.DataFrame) -> pl.DataFrame:
        """聚合surge_vol因子"""
        print(f"📊 聚合surge_vol因子: window={self.surge_window}, intraday_stat={self.intraday_stat}")
        
        surge_df = surge_df.with_columns(
            pl.col("is_surge").alias("is_surge_start")
        )
        
        surge_df = self._mark_surge_periods(surge_df)
        
        if self.price_type is not None:
            data_col = self.price_type
            print(f"  - 使用价格数据: {self.price_type}")
        else:
            data_col = "bar_ret"
            print(f"  - 使用收益率数据: bar_ret")
        
        period_vol_df = self._calculate_period_volatility(surge_df, data_col)
        
        print(f"  - Surge period数: {len(period_vol_df)}")
        
        agg_expr = pl.col("period_vol").__getattribute__(self.intraday_stat)().alias("factor_value")
        
        factor_df = (
            period_vol_df
            .group_by(["symbol", "date"])
            .agg(agg_expr)
        )
        
        print(f"  - 聚合后记录数: {len(factor_df)}")
        
        # 添加标准的bar_time (15:00:00.000)
        factor_df = self._add_eod_bar_time(factor_df)
        
        return factor_df
    
    def _mark_surge_periods(self, surge_df: pl.DataFrame) -> pl.DataFrame:
        """标记surge period"""
        df = surge_df.sort(["symbol", "date", "bar_time"])
        
        df = df.with_columns(
            pl.when(pl.col("is_surge_start"))
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .alias("surge_start_flag")
        )
        
        df = df.with_columns(
            pl.col("surge_start_flag")
            .cum_sum()
            .over(["symbol", "date"])
            .alias("period_id")
        )
        
        df = df.with_columns(
            pl.col("bar_time").rank("ordinal").over(["symbol", "date"]).alias("bar_rank")
        )
        
        df = df.with_columns(
            pl.when(pl.col("is_surge_start"))
            .then(pl.lit(True))
            .otherwise(pl.lit(False))
            .alias("in_surge_period")
        )
        
        return df
    
    def _calculate_period_volatility(
        self, 
        surge_df: pl.DataFrame, 
        data_col: str
    ) -> pl.DataFrame:
        """计算每个surge period的波动率"""
        df = surge_df.sort(["symbol", "date", "bar_time"])
        
        df = df.with_columns(
            pl.col(data_col)
            .rolling_std(window_size=self.surge_window, min_periods=self.surge_window)
            .over(["symbol", "date"])
            .alias("period_vol")
        )
        
        period_vol_df = df.filter(
            pl.col("is_surge_start") & pl.col("period_vol").is_not_null()
        )
        
        return period_vol_df.select(["symbol", "date", "bar_time", "period_vol"])

    # ============================================================
    # 主计算流程
    # ============================================================
    
    def calculate(self, bizdays_str: str) -> pl.DataFrame:
        """计算surge因子的主流程"""
        print(f"\n{'='*60}")    
        print(f"开始计算Surge因子")
        print(f"{'='*60}")
        
        print(f"\n[1/4] 加载数据...")
        bar_df = self.load_and_build_bars(bizdays_str=bizdays_str, add_intraday_ret=True)
        
        print(f"\n[2/4] 识别Surge...")
        surge_df = self._identify_surge(bar_df)
        
        print(f"\n[3/4] 聚合因子...")
        factor_df = self._aggregate_factor(surge_df)
        
        print(f"\n[4/4] 生成因子名称...")
        factor_name = self._generate_factor_name()
        factor_df = factor_df.with_columns(
            pl.lit(factor_name).alias("factor_name")
        )
        
        # 整理输出列顺序
        factor_df = self._format_output(factor_df)
        
        print(f"\n{'='*60}")
        print(f"✓ 因子计算完成: {factor_name}")
        print(f"  - 记录数: {len(factor_df)}")
        print(f"  - 股票数: {factor_df['symbol'].n_unique()}")
        print(f"  - 日期数: {factor_df['date'].n_unique()}")
        if "bar_time" in factor_df.columns:
            print(f"  - 时刻数: {factor_df['bar_time'].n_unique()}")
        print(f"{'='*60}\n")
        
        return factor_df
    
    def _format_output(self, factor_df: pl.DataFrame) -> pl.DataFrame:
        """
        整理输出格式
        
        确保输出包含：symbol, date, bar_time, factor_value, factor_name
        """
        # 确定输出列
        output_cols = ["symbol", "date", "bar_time", "factor_value", "factor_name"]
        
        # 过滤存在的列
        existing_cols = [col for col in output_cols if col in factor_df.columns]
        
        return factor_df.select(existing_cols)
    
    def _identify_surge(self, bar_df: pl.DataFrame) -> pl.DataFrame:
        """识别surge（根据output_freq选择方法）"""
        if self.output_freq == "EOD":
            return self._identify_surge_eod(bar_df)
        elif self.m10_method == "same_time":
            return self._identify_surge_m10_same_time(bar_df)
        else:
            return self._identify_surge_m10_rolling(bar_df)
    
    def _aggregate_factor(self, surge_df: pl.DataFrame) -> pl.DataFrame:
        """聚合因子（根据factor_type选择方法）"""
        if self.factor_type == "surge_ret":
            return self._aggregate_surge_ret(surge_df)
        else:
            return self._aggregate_surge_vol(surge_df)
    
    def _generate_factor_name(self) -> str:
        """生成因子名称"""
        factor_type_str = "ret" if self.factor_type == "surge_ret" else "vol"
        bar_freq_str = self.bar_freq.lower()
        output_freq_str = self.output_freq.lower()
        
        parts = [f"surge_{factor_type_str}", bar_freq_str, output_freq_str]
        
        if self.output_freq == "EOD":
            trading_time_str = self.trading_time.replace("_", "")
            parts.append(trading_time_str)
            
            if self.factor_type == "surge_vol":
                parts.append(f"w{self.surge_window}")
                if self.price_type:
                    parts.append(self.price_type)
        else:
            if self.m10_method == "same_time":
                parts.append("sametime")
                parts.append(f"d{self.lookback_days}")
            else:
                parts.append("rolling")
                parts.append(f"k{self.lookback_bars}")
        
        parts.append(f"t{self.threshold}")
        parts.append(self.intraday_stat)

        factor_name = "_".join(parts)
        
        return factor_name

    def get_lookback_days(self) -> int:
        """获取该因子需要的回看天数"""
        if self.output_freq == "EOD":
            return 0
        elif self.m10_method == "same_time":
            return self.lookback_days
        else:
            bars_per_day = get_bar_count_per_day(self.bar_freq)
            return (self.lookback_bars // bars_per_day) + 1

    def calculate_single_day(
        self, 
        settlement_date: str,
        bar_data: pl.DataFrame = None
    ) -> pl.DataFrame:
        """计算单个结算日的因子"""
        if bar_data is None:
            lookback = self.get_lookback_days()
            start_date = bizday(settlement_date, -lookback) if lookback > 0 else settlement_date
            date_list = bizdays(f"{start_date}-{settlement_date}")
            bar_data = self.load_and_build_bars(date_list=date_list, add_intraday_ret=True)
        
        self.bar_data = bar_data
        
        surge_df = self._identify_surge(bar_data)
        factor_df = self._aggregate_factor(surge_df)
        
        date_col_dtype = factor_df["date"].dtype
        if date_col_dtype == pl.Utf8:
            filter_value = settlement_date
        else:
            filter_value = int(settlement_date)
        
        factor_df = factor_df.filter(pl.col("date") == filter_value)
        
        factor_name = self._generate_factor_name()
        factor_df = factor_df.with_columns(
            pl.lit(factor_name).alias("factor_name")
        )
        
        # 整理输出格式
        factor_df = self._format_output(factor_df)
        
        return factor_df
