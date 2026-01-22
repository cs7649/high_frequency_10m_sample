"""
legion_saver.py - 因子保存到 Legion
"""

import polars as pl
from typing import Dict


def save_factors_to_legion(
    results: Dict[str, pl.DataFrame],
    legion_base_path: str = "/big/share/ctsu/base/cne",
    legion_factor_prefix: str = "ctsu/hf/surge",
):
    """
    将因子结果保存到 Legion
    
    保存路径:
        - EOD 因子: /big/share/ctsu/base/cne/EOD/
        - M10 same_time: /big/share/ctsu/base/cne/M10/Same_Time/
        - M10 rolling: /big/share/ctsu/base/cne/M10/Rolling/
    
    Args:
        results: Dict[factor_name, DataFrame]
        legion_base_path: Legion 基础路径
        legion_factor_prefix: 因子路径前缀
    """
    import legion
    import ajload as ld
    
    print(f"\n{'='*60}")
    print(f"保存因子到 Legion")
    print(f"{'='*60}")
    print(f"  - 基础路径: {legion_base_path}")
    print(f"  - 因子前缀: {legion_factor_prefix}")
    print(f"  - 因子数量: {len(results)}")
    print(f"{'='*60}")
    
    # 分类因子
    eod_results = {}
    m10_same_time_results = {}
    m10_rolling_results = {}
    
    for factor_name, df in results.items():
        name_lower = factor_name.lower()
        if "_eod_" in name_lower:
            eod_results[factor_name] = df
        elif "sametime" in name_lower:
            m10_same_time_results[factor_name] = df
        elif "rolling" in name_lower:
            m10_rolling_results[factor_name] = df
        else:
            print(f"  ⚠️ {factor_name}: 无法识别类型，默认归到 M10/Same_Time")
            m10_same_time_results[factor_name] = df
    
    # 保存 EOD 因子
    if eod_results:
        print(f"\n📁 保存 EOD 因子 ({len(eod_results)} 个)")
        lg = legion.Legion(f"{legion_base_path}/EOD/", freq='EOD', univ='cne', mode='w')
        for factor_name, df in eod_results.items():
            _save_single_factor(df, factor_name, lg, legion_factor_prefix, ld)
    
    # 保存 M10 same_time 因子
    if m10_same_time_results:
        print(f"\n📁 保存 M10/Same_time 因子 ({len(m10_same_time_results)} 个)")
        lg = legion.Legion(f"{legion_base_path}/M10/Same_Time/", freq='M10', univ='cne', mode='w')
        for factor_name, df in m10_same_time_results.items():
            _save_single_factor(df, factor_name, lg, legion_factor_prefix, ld)
    
    # 保存 M10 rolling 因子
    if m10_rolling_results:
        print(f"\n📁 保存 M10/Rolling 因子 ({len(m10_rolling_results)} 个)")
        lg = legion.Legion(f"{legion_base_path}/M10/Rolling/", freq='M10', univ='cne', mode='w')
        for factor_name, df in m10_rolling_results.items():
            _save_single_factor(df, factor_name, lg, legion_factor_prefix, ld)
    
    print(f"\n{'='*60}")
    print(f"✓ 保存完成")
    print(f"{'='*60}\n")


def _save_single_factor(
    df: pl.DataFrame,
    factor_name: str,
    lg,
    factor_prefix: str,
    ld,
):
    """保存单个因子到 Legion"""
    import pandas as pd
    
    # 1. 判断是 EOD 还是 M10
    has_bar_time = "bar_time" in df.columns
    
    # 2. 获取日期范围
    dates = sorted(df["date"].unique().to_list())
    if not isinstance(dates[0], str):
        dates = [str(d) for d in dates]
    start_date = dates[0]
    end_date = dates[-1]
    
    # 3. 转换为宽格式（index 必须是 DatetimeIndex）
    if has_bar_time:
        # M10: index 是 bar_time (datetime)
        wide_pd = (
            df.select(["bar_time", "symbol", "factor_value"])
            .to_pandas()
            .pivot(index="bar_time", columns="symbol", values="factor_value")
            .sort_index()
        )
    else:
        # EOD: 把 date (int) 转成 datetime
        df_with_dt = df.with_columns(
            pl.col("date").cast(pl.Utf8).str.strptime(pl.Datetime, "%Y%m%d").alias("datetime")
        )
        wide_pd = (
            df_with_dt.select(["datetime", "symbol", "factor_value"])
            .to_pandas()
            .pivot(index="datetime", columns="symbol", values="factor_value")
            .sort_index()
        )
    
    # 清理 index/columns 名称
    wide_pd.index.name = None
    wide_pd.columns.name = None
    
    # 4. 保存
    factor_path = f"{factor_prefix}/{factor_name}"
    
    try:
        ld.dk2lg(wide_pd, start_date, end_date, lg, factor_path)
        print(f"  ✓ {factor_name}: {wide_pd.shape}, {start_date} ~ {end_date}")
    except Exception as e:
        print(f"  ❌ {factor_name}: 保存失败 - {str(e)}")
