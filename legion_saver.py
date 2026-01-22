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
        - M10 same_time: /big/share/ctsu/base/cne/M10/Same_time/
        - M10 rolling: /big/share/ctsu/base/cne/M10/Rolling/
    
    Args:
        results: Dict[factor_name, DataFrame]，每个 df 包含 symbol, date, factor_value 列
        legion_base_path: Legion 基础路径
        legion_factor_prefix: 因子路径前缀
    
    使用示例:
        from legion_saver import save_factors_to_legion
        
        results = engine.calculate(settlement_range="20220104-20220131")
        save_factors_to_legion(results)
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
            print(f"  ⚠️ {factor_name}: 无法识别类型，默认归到 M10/Same_time")
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
        lg = legion.Legion(f"{legion_base_path}/M10/Same_time/", freq='M10', univ='cne', mode='w')
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
    # 1. 判断是 EOD 还是 M10
    has_bar_time = "bar_time" in df.columns
    
    # 2. 转换为宽格式（pivot）
    if has_bar_time:
        # M10: 需要 (date, bar_time) 作为索引
        df_pivot = df.with_columns(
            (pl.col("date").cast(pl.Utf8) + "_" + pl.col("bar_time").dt.strftime("%H:%M:%S")).alias("date_time")
        )
        wide_df = df_pivot.pivot(
            values="factor_value",
            index="date_time",
            columns="symbol"
        ).sort("date_time")
        
        wide_pd = wide_df.to_pandas()
        wide_pd = wide_pd.set_index("date_time")
    else:
        # EOD: 只用 date 作为索引
        wide_df = df.pivot(
            values="factor_value",
            index="date",
            columns="symbol"
        ).sort("date")
        
        wide_pd = wide_df.to_pandas()
        wide_pd = wide_pd.set_index("date")
    
    # 3. 获取日期范围
    dates = sorted(df["date"].unique().to_list())
    if not isinstance(dates[0], str):
        dates = [str(d) for d in dates]
    
    start_date = dates[0]
    end_date = dates[-1]
    
    # 4. 保存
    factor_path = f"{factor_prefix}/{factor_name}"
    
    try:
        ld.dk2lg(wide_pd, start_date, end_date, lg, factor_path)
        print(f"  ✓ {factor_name}: {wide_pd.shape}, {start_date} ~ {end_date}")
    except Exception as e:
        print(f"  ❌ {factor_name}: 保存失败 - {str(e)}")
