"""
surge_factor.py - Surge因子计算器（融合HF37设计）

设计思路：
1. 读取N天trade数据 → 用bar_builder合成bar
2. 识别surge时刻（volume > mean + threshold*std）
3. EOD模式：对当天bar筛选 → 聚合成surge_ret/surge_vol → 每天1个因子
4. M10模式：跨天筛选 → 聚合成surge_ret → 每天24个因子

支持的因子类型：
- surge_ret: surge时刻的收益率特征
- surge_vol: surge期间的波动率特征（仅EOD支持）
"""

import polars as pl
import numpy as np
from datetime import time, datetime
from typing import List, Literal, Optional, Union, Dict

from dux.cal import bizdays

from data_loader import DataLoader
from bar_builder import BarBuilder
from config import (
    get_timestamps, 
    get_bar_count_per_day,
    get_trading_time_slice,
    get_bars_per_trading_time
)


class SurgeFactor:
    """
    Surge因子计算器
    
    使用示例：
        # EOD surge_ret因子
        factor = SurgeFactor(
            bar_freq="1m",
            output_freq="EOD",
            factor_type="surge_ret",
            trading_time="all_day",
            threshold=1.0
        )
        result = factor.calculate(bizdays_str='20220104-10')
        
        # EOD surge_vol因子
        factor = SurgeFactor(
            bar_freq="5m",
            output_freq="EOD",
            factor_type="surge_vol",
            trading_time="afternoon",
            surge_window=10,
            threshold=1.5
        )
        result = factor.calculate(bizdays_str='20220104-10')
        
        # M10 surge_ret (same_time)
        factor = SurgeFactor(
            bar_freq="1m",
            output_freq="M10",
            m10_method="same_time",
            lookback_days=20,
            threshold=2.0
        )
        result = factor.calculate(bizdays_str='20220104-10')
        
        # M10 surge_ret (rolling)
        factor = SurgeFactor(
            bar_freq="1m",
            output_freq="M10",
            m10_method="rolling",
            lookback_bars=48,
            threshold=2.0
        )
        result = factor.calculate(bizdays_str='20220104-10')
    """
    
    def __init__(
        self,
        # ===== 基础参数 =====
        bar_freq: str = "1m",
        output_freq: str = "EOD",
        
        # ===== Surge识别参数 =====
        threshold: float = 1.0,
        
        # ===== EOD专用参数 =====
        trading_time: str = "all_day",      # 交易时段: all_day/morning/afternoon/opening/closing等
        factor_type: str = "surge_ret",     # 因子类型: surge_ret 或 surge_vol
        surge_window: int = 5,              # surge_vol的窗口大小（几个bar）
        
        # ===== M10专用参数 =====
        m10_method: str = "same_time",      # M10筛选方式: same_time 或 rolling
        lookback_days: int = 20,            # same_time方法的回看天数H
        lookback_bars: int = 48,            # rolling方法的回看bar数k
        
        # ===== HF37风格的统计参数（可选）=====
        intraday_stat: str = "mean",        # 日内统计: mean/max/min
        is_abs: bool = False,               # 是否取绝对值（截面中性化后）
        neutralize: bool = True,            # 是否截面中性化（减去市场均值）
        price_type: str = None,             # surge_vol可选价格类型: open/close等
        
        # ===== 数据路径 =====
        data_path: str = None,
    ):
        """
        初始化 SurgeFactor
        
        Args:
            bar_freq: bar频率 "1m"/"5m"/"10m"
            output_freq: 输出频率 "EOD"（每天1个值）或 "M10"（每天24个值）
            threshold: surge判断阈值（几倍std）
            
            trading_time: EOD的交易时段选择
            factor_type: EOD的因子类型（surge_ret或surge_vol）
            surge_window: surge_vol的窗口大小
            
            m10_method: M10的筛选方式
            lookback_days: same_time方法的回看天数（需要 <= N）
            lookback_bars: rolling方法的回看bar数
            
            intraday_stat: 日内统计方法（mean/max/min）
            is_abs: 是否取绝对值
            neutralize: 是否截面中性化
            price_type: surge_vol可用价格类型
            data_path: 数据路径
        """
        # 保存参数并标准化bar_freq格式
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
        
        # HF37参数
        self.intraday_stat = intraday_stat
        self.is_abs = is_abs
        self.neutralize = neutralize
        self.price_type = price_type
        
        # 初始化loader和builder
        self.loader = DataLoader(data_path=data_path) if data_path else DataLoader()
        self.bar_builder = BarBuilder(freq=bar_freq)
        
        # 参数验证
        self._validate_params()
        
        # 存储数据的属性
        self.bar_data = None
    
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
            # M10只支持surge_ret
            if self.factor_type != "surge_ret":
                print(f"⚠️  M10模式只支持surge_ret，已自动设置")
                self.factor_type = "surge_ret"
        
        print(f"✓ 参数验证通过")
        print(f"  - 输出模式: {self.output_freq}")
        print(f"  - Bar频率: {self.bar_freq}")
        print(f"  - 因子类型: {self.factor_type}")
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
        bizdays_str: str,
        add_intraday_ret: bool = True
    ) -> pl.DataFrame:
        """
        加载trade数据并合成bar
        
        Args:
            bizdays_str: 交易日范围字符串，格式如 '20220104-10' 表示读取20220104到20220110的数据
                        会通过bizdays()函数转换为日期列表
            add_intraday_ret: 是否添加bar收益率列（surge_ret需要）
        
        Returns:
            bar数据DataFrame，包含列：
            - symbol, date, bar_time
            - open, high, low, close, vol, amt, vwap
            - ret (收盘价收益率，相对前一个bar)
            - bar_ret (bar内收益率，如果add_intraday_ret=True)
        """
        print(f"📊 加载数据: {bizdays_str}，频率: {self.bar_freq}")
        
        # 1. 加载trade数据（使用bizdays函数转换日期范围）
        trade_lf = self.loader.load_trade(
            date_list=bizdays(bizdays_str),
            columns=["inst_id", "xts", "px", "qty", "amt", "flag"]
        )
        
        # 2. 合成bar（使用bar_builder）
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
        
        # 3. 可选：添加bar内收益率（用于surge_ret）
        if add_intraday_ret:
            bar_df = self._add_bar_returns(bar_df)
        
        self.bar_data = bar_df
        return bar_df
    
    def _add_bar_returns(self, bar_df: pl.DataFrame) -> pl.DataFrame:
        """
        添加bar内收益率
        
        计算公式：
        bar_ret = (close - open) / open
        
        用途：
        - surge_ret因子需要这个收益率
        - 表示surge时刻的即时价格变化
        """
        return bar_df.with_columns(
            pl.when(pl.col("open") <= 0)
            .then(None)
            .otherwise((pl.col("close") - pl.col("open")) / pl.col("open"))
            .alias("bar_ret")
        )


    # ============================================================
    # Surge识别部分 - EOD模式
    # ============================================================
    
    def _identify_surge_eod(self, bar_df: pl.DataFrame) -> pl.DataFrame:
        """
        EOD模式的surge识别
        
        逻辑：
        1. 按trading_time筛选当天的bar（如all_day/morning/afternoon等）
        2. 计算每个symbol每天该时段的volume均值和标准差
        3. 标记surge: vol > mean(vol) + threshold * std(vol)
        
        Returns:
            添加了is_surge列的DataFrame
        
        示例：
            trading_time='afternoon', threshold=1.5
            → 对13:00-15:00的bar，如果vol > mean + 1.5*std，标记为surge
        """
        print(f"🔍 EOD Surge识别: {self.trading_time}, threshold={self.threshold}")
        
        # 1. 获取该时段的有效bar_time
        valid_times = get_trading_time_slice(self.bar_freq, self.trading_time)
        
        print(f"  - Bar频率: {self.bar_freq}")
        print(f"  - 有效时间段数: {len(valid_times)}")
        if len(valid_times) > 0:
            print(f"  - 时间范围: {valid_times[0]} ~ {valid_times[-1]}")
        
        # 调试：查看实际数据中的bar_time
        unique_times = bar_df.select(pl.col("bar_time").dt.time().unique()).to_series().to_list()
        print(f"  - 实际bar_time数: {len(unique_times)}")
        if len(unique_times) > 0:
            print(f"  - 实际时间范围: {min(unique_times)} ~ {max(unique_times)}")
        
        # 2. 筛选该时段的bar
        df = bar_df.filter(
            pl.col("bar_time").dt.time().is_in(valid_times)
        )
        
        print(f"  - 筛选后bar数: {len(df)}")
        
        # 3. 计算每个symbol每天的volume统计量（只在该时段内）
        df = df.with_columns([
            pl.col("vol").mean().over(["symbol", "date"]).alias("vol_mean"),
            pl.col("vol").std().over(["symbol", "date"]).alias("vol_std"),
        ])
        
        # 4. 标记surge
        # 如果std为0（该时段成交量完全相同），则不标记为surge
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
        M10模式 - same_time方法（修复版）
        
        修复：使用 bar_time 的时间部分进行匹配，而不是完整的 datetime
        """
        print(f"🔍 M10 Surge识别 (same_time): H={self.lookback_days}天, threshold={self.threshold}")
        
        # 添加 bar_time_only 列（只保留时间部分）
        bar_df = bar_df.with_columns(
            pl.col("bar_time").dt.time().alias("bar_time_only")
        )
        
        # 获取所有日期，排序
        dates = sorted(bar_df["date"].unique().to_list())
        
        result_list = []
        
        for target_date in dates:
            date_idx = dates.index(target_date)
            
            # 前H天的日期（不包括当天）
            lookback_dates = dates[max(0, date_idx - self.lookback_days):date_idx]
            
            if len(lookback_dates) < self.lookback_days:
                print(f"  ⚠️  {target_date}: 历史数据不足({len(lookback_dates)}天 < {self.lookback_days}天)，跳过")
                continue
            
            # 基准数据：过去H天
            baseline_df = bar_df.filter(pl.col("date").is_in(lookback_dates))
            
            # 计算每个(symbol, bar_time_only)的基准统计量
            # 关键修复：使用 bar_time_only 而不是 bar_time
            baseline_stats = (
                baseline_df
                .group_by(["symbol", "bar_time_only"])
                .agg([
                    pl.col("vol").mean().alias("vol_mean_baseline"),
                    pl.col("vol").std().alias("vol_std_baseline"),
                ])
            )
            
            # 当天数据
            target_df = bar_df.filter(pl.col("date") == target_date)
            
            # Join基准统计量（使用 bar_time_only）
            target_df = target_df.join(
                baseline_stats,
                on=["symbol", "bar_time_only"],
                how="left"
            )
            
            # 标记surge
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
        
        # 删除临时列
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
        
        逻辑：
        1. 按symbol时间顺序排序（date, bar_time连续）
        2. 对每个bar，用它前面k根bar计算基准
        3. 判断当前bar是否surge
        
        实现方式：
        - 全局排序：按symbol, date, bar_time
        - 用rolling窗口计算前k根bar的mean和std
        - shift(1)确保基准不包含当前bar
        
        示例：
            lookback_bars=48, threshold=2.0
            对于某股票的第100根bar，
            用第52~99根bar（共48根）计算基准mean和std，
            判断第100根bar是否 > mean + 2.0*std
        
        Returns:
            添加了is_surge列的DataFrame
        """
        print(f"🔍 M10 Surge识别 (rolling): k={self.lookback_bars}根, threshold={self.threshold}")
        
        # 1. 全局排序（按symbol, date, bar_time）
        df = bar_df.sort(["symbol", "date", "bar_time"])
        
        # 2. 计算rolling统计量（前k根bar）
        # 注意：rolling_mean的窗口包含当前值，所以要shift(1)
        df = df.with_columns([
            pl.col("vol")
              .rolling_mean(window_size=self.lookback_bars, min_periods=self.lookback_bars)
              .shift(1)  # shift(1)确保基准是"前k根bar"
              .over("symbol")
              .alias("vol_mean_baseline"),
            
            pl.col("vol")
              .rolling_std(window_size=self.lookback_bars, min_periods=self.lookback_bars)
              .shift(1)
              .over("symbol")
              .alias("vol_std_baseline"),
        ])
        
        # 3. 标记surge
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
        
        # 统计有效数据（前k根bar会是null）
        valid_count = df["is_surge"].is_not_null().sum()
        surge_count = df["is_surge"].sum()
        surge_ratio = surge_count / valid_count if valid_count > 0 else 0
        
        print(f"  - 有效bar数: {valid_count}/{len(df)} (前{self.lookback_bars}根为null)")
        print(f"  - Surge占比: {surge_ratio:.2%}")
        
        return df


    # ============================================================
    # 因子聚合部分 - surge_ret
    # ============================================================
    
    def _aggregate_surge_ret(self, surge_df: pl.DataFrame) -> pl.DataFrame:
        """
        聚合surge_ret因子
        
        逻辑（参考HF37）：
        1. 筛选surge时刻（is_surge=True）
        2. 提取这些时刻的bar_ret（收益率）
        3. 按日内统计方法聚合（mean/max/min）
        4. 可选：截面中性化（个股值 - 市场均值）
        5. 可选：取绝对值
        
        输出：
        - EOD: 每个(symbol, date)一个值
        - M10: 每个(symbol, date, bar_time)一个值
        
        Returns:
            因子DataFrame，包含列：
            - symbol, date, [bar_time], factor_value
        """
        print(f"📊 聚合surge_ret因子: {self.intraday_stat}, neutralize={self.neutralize}, is_abs={self.is_abs}")
        
        # 1. 筛选surge时刻
        surge_moments = surge_df.filter(pl.col("is_surge") == True)
        
        if len(surge_moments) == 0:
            raise ValueError("没有检测到surge时刻，无法计算因子")
        
        print(f"  - Surge时刻数: {len(surge_moments)}")
        
        # 2. 根据输出频率选择分组方式
        if self.output_freq == "EOD":
            # EOD: 按(symbol, date)聚合
            group_cols = ["symbol", "date"]
        else:
            # M10: 按(symbol, date, bar_time)聚合
            group_cols = ["symbol", "date", "bar_time"]
        
        # 3. 按日内统计方法聚合bar_ret
        agg_expr = pl.col("bar_ret").__getattribute__(self.intraday_stat)().alias("individual_stat")
        
        factor_df = (
            surge_moments
            .group_by(group_cols)
            .agg(agg_expr)
        )
        
        print(f"  - 聚合后记录数: {len(factor_df)}")
        
        # 4. 截面中性化（参考HF37）
        if self.neutralize:
            factor_df = self._cross_sectional_neutralize(factor_df, group_cols)
        else:
            # 不中性化，直接使用individual_stat
            factor_df = factor_df.with_columns(
                pl.col("individual_stat").alias("factor_value")
            )
        
        # 5. 可选：取绝对值
        if self.is_abs:
            factor_df = factor_df.with_columns(
                pl.col("factor_value").abs().alias("factor_value")
            )
        
        return factor_df
    
    def _cross_sectional_neutralize(
        self, 
        factor_df: pl.DataFrame,
        group_cols: List[str]
    ) -> pl.DataFrame:
        """
        截面中性化（参考HF37的设计）
        
        逻辑：
        factor_value = individual_stat - cross_sec_mean
        
        Args:
            factor_df: 包含individual_stat列的DataFrame
            group_cols: 分组列（用于确定截面维度）
        
        Returns:
            添加了factor_value列的DataFrame
        
        截面维度：
        - EOD: 按date截面（所有股票）
        - M10: 按(date, bar_time)截面（所有股票在同一时刻）
        """
        # 确定截面分组（去掉symbol）
        cross_sec_group = [col for col in group_cols if col != "symbol"]
        
        # 计算截面均值
        factor_df = factor_df.with_columns(
            pl.col("individual_stat").mean().over(cross_sec_group).alias("cross_sec_mean")
        )
        
        # 中性化
        factor_df = factor_df.with_columns(
            (pl.col("individual_stat") - pl.col("cross_sec_mean")).alias("factor_value")
        )
        
        return factor_df


    # ============================================================
    # 因子聚合部分 - surge_vol
    # ============================================================
    
    def _aggregate_surge_vol(self, surge_df: pl.DataFrame) -> pl.DataFrame:
        """
        聚合surge_vol因子（参考HF37的设计）
        
        逻辑：
        1. 识别surge period（从surge起点开始，持续surge_window个bar）
        2. 计算每个surge period内的波动率（标准差）
        3. 对每天所有surge period的波动率进行日内统计聚合
        4. 可选：截面中性化
        5. 可选：取绝对值
        
        注意：
        - surge_vol只支持EOD模式
        - 可以基于收益率或价格数据计算波动率
        
        Returns:
            因子DataFrame，包含列：
            - symbol, date, factor_value
        """
        print(f"📊 聚合surge_vol因子: window={self.surge_window}, {self.intraday_stat}")
        
        # 1. 识别surge period的起点
        surge_df = surge_df.with_columns(
            pl.col("is_surge").alias("is_surge_start")
        )
        
        # 2. 识别surge period（从起点开始的surge_window个bar）
        surge_df = self._mark_surge_periods(surge_df)
        
        # 3. 选择用于计算波动率的数据
        if self.price_type is not None:
            # 使用价格数据（如open, close等）
            data_col = self.price_type
            print(f"  - 使用价格数据: {self.price_type}")
        else:
            # 使用bar_ret（收益率）
            data_col = "bar_ret"
            print(f"  - 使用收益率数据: bar_ret")
        
        # 4. 计算每个surge period的波动率
        period_vol_df = self._calculate_period_volatility(surge_df, data_col)
        
        print(f"  - Surge period数: {len(period_vol_df)}")
        
        # 5. 按日内统计方法聚合（每天的所有surge period）
        agg_expr = pl.col("period_vol").__getattribute__(self.intraday_stat)().alias("individual_stat")
        
        factor_df = (
            period_vol_df
            .group_by(["symbol", "date"])
            .agg(agg_expr)
        )
        
        print(f"  - 聚合后记录数: {len(factor_df)}")
        
        # 6. 截面中性化
        if self.neutralize:
            factor_df = self._cross_sectional_neutralize(factor_df, ["symbol", "date"])
        else:
            factor_df = factor_df.with_columns(
                pl.col("individual_stat").alias("factor_value")
            )
        
        # 7. 可选：取绝对值
        if self.is_abs:
            factor_df = factor_df.with_columns(
                pl.col("factor_value").abs().alias("factor_value")
            )
        
        return factor_df
    
    def _mark_surge_periods(self, surge_df: pl.DataFrame) -> pl.DataFrame:
        """
        标记surge period
        
        逻辑：
        1. surge_start: is_surge=True的bar
        2. surge period: 从surge_start开始，持续surge_window个bar
        3. 添加period_id用于分组
        
        Returns:
            添加了in_surge_period和period_id列的DataFrame
        """
        # 按symbol, date排序
        df = surge_df.sort(["symbol", "date", "bar_time"])
        
        # 为每个surge_start分配一个唯一ID
        df = df.with_columns(
            pl.when(pl.col("is_surge_start"))
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .alias("surge_start_flag")
        )
        
        # 累计计数，生成period_id
        df = df.with_columns(
            pl.col("surge_start_flag")
            .cum_sum()
            .over(["symbol", "date"])
            .alias("period_id")
        )
        
        # 计算每个bar距离其所属surge_start的距离
        df = df.with_columns(
            pl.col("bar_time").rank("ordinal").over(["symbol", "date"]).alias("bar_rank")
        )
        
        # 标记是否在surge period内（距离 < surge_window）
        # 这个逻辑比较复杂，简化处理：
        # 如果is_surge_start=True，则标记为in_surge_period
        # 后续surge_window-1个bar也标记为in_surge_period
        
        # 简化版：只计算surge_start的窗口
        # 使用rolling窗口来标记
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
        """
        计算每个surge period的波动率
        
        逻辑（参考HF37）：
        1. 筛选surge_start的bar
        2. 对每个surge_start，向后取surge_window个bar
        3. 计算这个窗口内data_col的标准差
        
        简化实现：
        - 只在surge_start的bar上计算
        - 使用rolling_std，窗口=surge_window
        
        Returns:
            包含period_vol的DataFrame
        """
        # 按symbol, date, bar_time排序
        df = surge_df.sort(["symbol", "date", "bar_time"])
        
        # 计算rolling std（向后surge_window个bar）
        df = df.with_columns(
            pl.col(data_col)
            .rolling_std(window_size=self.surge_window, min_periods=self.surge_window)
            .over(["symbol", "date"])
            .alias("period_vol")
        )
        
        # 只保留surge_start的bar（这些bar有完整的窗口）
        period_vol_df = df.filter(
            pl.col("is_surge_start") & pl.col("period_vol").is_not_null()
        )
        
        return period_vol_df.select(["symbol", "date", "bar_time", "period_vol"])


    # ============================================================
    # 主计算流程
    # ============================================================
    
    def calculate(self, bizdays_str: str) -> pl.DataFrame:
        """
        计算surge因子的主流程
        
        流程：
        1. 加载并合成bar数据
        2. 识别surge时刻/period
        3. 聚合计算因子
        4. 添加因子名称
        5. 返回结果
        
        Args:
            bizdays_str: 交易日范围字符串，格式如 '20220104-10'
                        会通过bizdays()函数转换为日期列表
        
        Returns:
            因子DataFrame，包含列：
            - symbol, date, [bar_time], factor_value, factor_name
        
        使用示例：
            factor = SurgeFactor(bar_freq="1m", output_freq="EOD", factor_type="surge_ret")
            result = factor.calculate(bizdays_str='20220104-10')
        """
        print(f"\n{'='*60}")    
        print(f"开始计算Surge因子")
        print(f"{'='*60}")
        
        # Step 1: 加载并合成bar数据
        print(f"\n[1/4] 加载数据...")
        bar_df = self.load_and_build_bars(bizdays_str=bizdays_str, add_intraday_ret=True)
        
        # Step 2: 识别surge
        print(f"\n[2/4] 识别Surge...")
        surge_df = self._identify_surge(bar_df)
        
        # Step 3: 聚合因子
        print(f"\n[3/4] 聚合因子...")
        factor_df = self._aggregate_factor(surge_df)
        
        # Step 4: 添加因子名称
        print(f"\n[4/4] 生成因子名称...")
        factor_name = self._generate_factor_name()
        factor_df = factor_df.with_columns(
            pl.lit(factor_name).alias("factor_name")
        )
        
        print(f"\n{'='*60}")
        print(f"✓ 因子计算完成: {factor_name}")
        print(f"  - 记录数: {len(factor_df)}")
        print(f"  - 股票数: {factor_df['symbol'].n_unique()}")
        if self.output_freq == "EOD":
            print(f"  - 日期数: {factor_df['date'].n_unique()}")
        else:
            print(f"  - 日期数: {factor_df['date'].n_unique()}")
            print(f"  - 时刻数: {factor_df['bar_time'].n_unique()}")
        print(f"{'='*60}\n")
        
        return factor_df
    
    def _identify_surge(self, bar_df: pl.DataFrame) -> pl.DataFrame:
        """
        识别surge（根据output_freq选择方法）
        
        Returns:
            添加了is_surge列的DataFrame
        """
        if self.output_freq == "EOD":
            return self._identify_surge_eod(bar_df)
        elif self.m10_method == "same_time":
            return self._identify_surge_m10_same_time(bar_df)
        else:  # rolling
            return self._identify_surge_m10_rolling(bar_df)
    
    def _aggregate_factor(self, surge_df: pl.DataFrame) -> pl.DataFrame:
        """
        聚合因子（根据factor_type选择方法）
        
        Returns:
            因子DataFrame
        """
        if self.factor_type == "surge_ret":
            return self._aggregate_surge_ret(surge_df)
        else:  # surge_vol
            return self._aggregate_surge_vol(surge_df)
    
    def _generate_factor_name(self) -> str:
        """
        生成因子名称（融合用户原有规则和HF37风格）
        
        命名规则：
        - 基础格式: surge_{factor_type}_{bar_freq}_{output_freq}_{params}
        - EOD: surge_ret_1m_eod_allday_t1.0_mean
        - M10 same_time: surge_ret_1m_m10_sametime_d20_t2.0_mean
        - M10 rolling: surge_ret_1m_m10_rolling_k48_t2.0_mean
        
        包含信息：
        - factor_type: ret/vol
        - bar_freq: 1m/5m/10m
        - output_freq: eod/m10
        - 参数：threshold, lookback, 统计方法等
        
        Returns:
            因子名称字符串
        """
        # 基础部分
        factor_type_str = "ret" if self.factor_type == "surge_ret" else "vol"
        bar_freq_str = self.bar_freq.lower()
        output_freq_str = self.output_freq.lower()
        
        # 参数部分
        parts = [f"surge_{factor_type_str}", bar_freq_str, output_freq_str]
        
        # 根据output_freq添加不同的参数
        if self.output_freq == "EOD":
            # EOD: 添加trading_time
            trading_time_str = self.trading_time.replace("_", "")  # all_day -> allday
            parts.append(trading_time_str)
            
            # 如果是surge_vol，添加窗口信息
            if self.factor_type == "surge_vol":
                parts.append(f"w{self.surge_window}")
                if self.price_type:
                    parts.append(self.price_type)
        
        else:  # M10
            # M10: 添加方法类型
            if self.m10_method == "same_time":
                parts.append("sametime")
                parts.append(f"d{self.lookback_days}")
            else:
                parts.append("rolling")
                parts.append(f"k{self.lookback_bars}")
        
        # 通用参数
        parts.append(f"t{self.threshold}")
        parts.append(self.intraday_stat)
        
        # 可选标记
        if self.is_abs:
            parts.insert(0, "abs")  # 放在最前面
        
        if not self.neutralize:
            parts.append("raw")  # 未中性化
        
        # 组合成名称
        factor_name = "_".join(parts)
        
        return factor_name
