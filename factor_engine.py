"""
factor_engine.py - 因子计算引擎（多线程版）

负责：
1. 管理多个因子配置
2. 计算最大回看天数
3. 多线程并行计算
4. 按因子名拼接结果
"""

import polars as pl
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from datetime import datetime

from dux.cal import bizdays, bizday

from surge_factor import SurgeFactor
from data_loader import DataLoader
from bar_builder import BarBuilder


class FactorEngine:
    """
    因子计算引擎
    
    使用示例：
        # 定义多个因子配置
        factor_configs = [
            {
                "bar_freq": "1m",
                "output_freq": "EOD",
                "factor_type": "surge_ret",
                "trading_time": "all_day",
                "threshold": 1.0,
            },
            {
                "bar_freq": "1m",
                "output_freq": "M10",
                "m10_method": "same_time",
                "lookback_days": 20,
                "threshold": 2.0,
            },
        ]
        
        # 创建引擎并计算
        engine = FactorEngine(factor_configs, n_workers=8)
        results = engine.calculate(settlement_dates=["20220110", "20220111", "20220112"])
        
        # results 是 Dict[factor_name, DataFrame]
        for name, df in results.items():
            print(f"{name}: {len(df)} rows")
    """
    
    def __init__(
        self,
        factor_configs: List[Dict[str, Any]],
        n_workers: int = 8,
        data_path: str = None,
    ):
        """
        初始化因子引擎
        
        Args:
            factor_configs: 因子配置列表，每个配置是 SurgeFactor 的参数字典
            n_workers: 线程数
            data_path: 数据路径
        """
        self.factor_configs = factor_configs
        self.n_workers = n_workers
        self.data_path = data_path
        
        # 线程锁（用于打印）
        self._print_lock = threading.Lock()
        
        # 计算最大回看天数
        self.max_lookback = self._calculate_max_lookback()
        
        self._safe_print(f"✓ FactorEngine 初始化完成")
        self._safe_print(f"  - 因子数量: {len(factor_configs)}")
        self._safe_print(f"  - 最大回看天数: {self.max_lookback}")
        self._safe_print(f"  - 线程数: {n_workers}")
    
    def _safe_print(self, msg: str):
        """线程安全的打印"""
        with self._print_lock:
            print(msg)
    
    def _calculate_max_lookback(self) -> int:
        """
        计算所有因子中最大的回看天数
        
        Returns:
            最大回看天数
        """
        max_lookback = 0
        
        for config in self.factor_configs:
            # 创建临时 SurgeFactor 实例来获取回看天数
            factor = SurgeFactor(**config, data_path=self.data_path)
            lookback = factor.get_lookback_days()
            max_lookback = max(max_lookback, lookback)
        
        return max_lookback
    
    def _calculate_single_settlement_day(
        self,
        settlement_date: str
    ) -> Dict[str, pl.DataFrame]:
        """
        计算单个结算日的所有因子
        
        Args:
            settlement_date: 结算日
        
        Returns:
            Dict[factor_name, DataFrame]
        """
        self._safe_print(f"📅 开始计算 {settlement_date} ...")
        
        try:
            # 1. 计算数据加载范围
            start_date = bizday(settlement_date, -self.max_lookback) if self.max_lookback > 0 else settlement_date
            date_list = bizdays(f"{start_date}-{settlement_date}")
            
            # 2. 加载数据并合成 bar（每个结算日独立加载）
            loader = DataLoader(data_path=self.data_path) if self.data_path else DataLoader()
            builder = BarBuilder(freq=self.factor_configs[0].get("bar_freq", "1m"))
            
            trade_lf = loader.load_trade(
                date_list=date_list,
                columns=["inst_id", "xts", "px", "qty", "amt", "flag"]
            )
            
            bar_data = builder.group_by_bar_trade(
                lf=trade_lf,
                time_col="xts",
                price_col="px",
                qty_col="qty",
                amt_col="amt",
                flag_col="flag",
                filter_valid=True
            )
            
            # 添加 bar_ret
            bar_data = bar_data.with_columns(
                pl.when(pl.col("open") <= 0)
                .then(None)
                .otherwise((pl.col("close") - pl.col("open")) / pl.col("open"))
                .alias("bar_ret")
            )
            
            # 3. 计算所有因子
            results = {}
            
            for config in self.factor_configs:
                factor = SurgeFactor(**config, data_path=self.data_path)
                factor_df = factor.calculate_single_day(
                    settlement_date=settlement_date,
                    bar_data=bar_data
                )
                
                if len(factor_df) > 0:
                    factor_name = factor_df["factor_name"][0]
                    results[factor_name] = factor_df
            
            self._safe_print(f"✓ {settlement_date} 完成，计算了 {len(results)} 个因子")
            return results
            
        except Exception as e:
            self._safe_print(f"❌ {settlement_date} 失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return {}
    
    def calculate(
        self,
        settlement_dates: List[str] = None,
        settlement_range: str = None,
    ) -> Dict[str, pl.DataFrame]:
        """
        并行计算所有结算日的所有因子
        
        Args:
            settlement_dates: 结算日列表，如 ["20220110", "20220111"]
            settlement_range: 结算日范围，如 "20220110-20220120"（与 settlement_dates 二选一）
        
        Returns:
            Dict[factor_name, DataFrame]，每个因子一个完整的 DataFrame
        """
        # 1. 确定结算日列表
        if settlement_dates is None and settlement_range is not None:
            settlement_dates = bizdays(settlement_range)
        elif settlement_dates is None:
            raise ValueError("必须提供 settlement_dates 或 settlement_range")
        
        print(f"\n{'='*60}")
        print(f"开始并行计算因子")
        print(f"{'='*60}")
        print(f"  - 结算日数量: {len(settlement_dates)}")
        print(f"  - 结算日范围: {settlement_dates[0]} ~ {settlement_dates[-1]}")
        print(f"  - 因子数量: {len(self.factor_configs)}")
        print(f"  - 线程数: {self.n_workers}")
        print(f"{'='*60}\n")
        
        # 2. 多线程并行计算
        all_results = []  # List[Dict[factor_name, DataFrame]]
        
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {
                executor.submit(self._calculate_single_settlement_day, date): date
                for date in settlement_dates
            }
            
            for future in as_completed(futures):
                date = futures[future]
                try:
                    result = future.result()
                    if result:
                        all_results.append(result)
                except Exception as e:
                    self._safe_print(f"❌ {date} 异常: {str(e)}")
        
        # 3. 按因子名拼接
        print(f"\n📊 拼接结果...")
        final_results = self._merge_results(all_results)
        
        print(f"\n{'='*60}")
        print(f"✓ 计算完成")
        print(f"{'='*60}")
        for name, df in final_results.items():
            print(f"  - {name}: {len(df)} rows, {df['symbol'].n_unique()} symbols")
        print(f"{'='*60}\n")
        
        return final_results
    
    def _merge_results(
        self,
        all_results: List[Dict[str, pl.DataFrame]]
    ) -> Dict[str, pl.DataFrame]:
        """
        按因子名拼接结果
        
        Args:
            all_results: 每个结算日的结果列表
        
        Returns:
            Dict[factor_name, DataFrame]
        """
        # 收集所有因子名
        all_factor_names = set()
        for result in all_results:
            all_factor_names.update(result.keys())
        
        # 按因子名拼接
        merged = {}
        for factor_name in all_factor_names:
            dfs = [r[factor_name] for r in all_results if factor_name in r]
            if dfs:
                merged_df = pl.concat(dfs)
                # 排序
                sort_cols = ["symbol", "date"]
                if "bar_time" in merged_df.columns:
                    sort_cols.append("bar_time")
                merged[factor_name] = merged_df.sort(sort_cols)
        
        return merged
