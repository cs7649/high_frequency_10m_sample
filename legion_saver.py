"""
legion_saver.py - 因子保存到 Legion（修复版）

修复内容：
1. M10因子：使用bar_time列（已经是M10的24个时间点）
2. EOD因子：使用bar_time列（已经是15:00:00.000）
3. 正确处理DataFrame的pivot操作
"""

import polars as pl
from typing import Dict
from datetime import datetime


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
            _save_single_factor_eod(df, factor_name, lg, legion_factor_prefix, ld)
    
    # 保存 M10 same_time 因子
    if m10_same_time_results:
        print(f"\n📁 保存 M10/Same_time 因子 ({len(m10_same_time_results)} 个)")
        lg = legion.Legion(f"{legion_base_path}/M10/Same_Time/", freq='M10', univ='cne', mode='w')
        for factor_name, df in m10_same_time_results.items():
            _save_single_factor_m10(df, factor_name, lg, legion_factor_prefix, ld)
    
    # 保存 M10 rolling 因子
    if m10_rolling_results:
        print(f"\n📁 保存 M10/Rolling 因子 ({len(m10_rolling_results)} 个)")
        lg = legion.Legion(f"{legion_base_path}/M10/Rolling/", freq='M10', univ='cne', mode='w')
        for factor_name, df in m10_rolling_results.items():
            _save_single_factor_m10(df, factor_name, lg, legion_factor_prefix, ld)
    
    print(f"\n{'='*60}")
    print(f"✓ 保存完成")
    print(f"{'='*60}\n")


def _save_single_factor_eod(
    df: pl.DataFrame,
    factor_name: str,
    lg,
    factor_prefix: str,
    ld,
):
    """
    保存EOD因子到Legion
    
    EOD因子的bar_time都是15:00:00.000，直接使用bar_time作为index
    """
    import pandas as pd
    
    # 1. 获取日期范围
    dates = sorted(df["date"].unique().to_list())
    if not isinstance(dates[0], str):
        dates = [str(d) for d in dates]
    start_date = dates[0]
    end_date = dates[-1]
    
    # 2. 转换为宽格式
    # bar_time已经是datetime类型（15:00:00.000）
    wide_pd = (
        df.select(["bar_time", "symbol", "factor_value"])
        .to_pandas()
        .pivot(index="bar_time", columns="symbol", values="factor_value")
        .sort_index()
    )
    
    # 确保index是DatetimeIndex
    if not isinstance(wide_pd.index, pd.DatetimeIndex):
        wide_pd.index = pd.to_datetime(wide_pd.index)
    
    # 清理 index/columns 名称
    wide_pd.index.name = None
    wide_pd.columns.name = None
    
    # 3. 保存
    factor_path = f"{factor_prefix}/{factor_name}"
    
    try:
        ld.dk2lg(wide_pd, start_date, end_date, lg, factor_path)
        print(f"  ✓ {factor_name}: {wide_pd.shape}, {start_date} ~ {end_date}")
    except Exception as e:
        print(f"  ❌ {factor_name}: 保存失败 - {str(e)}")
        import traceback
        traceback.print_exc()


def _save_single_factor_m10(
    df: pl.DataFrame,
    factor_name: str,
    lg,
    factor_prefix: str,
    ld,
):
    """
    保存M10因子到Legion
    
    M10因子的bar_time是每天24个时间点（M10_TIMESTAMPS）
    """
    import pandas as pd
    
    # 1. 获取日期范围
    dates = sorted(df["date"].unique().to_list())
    if not isinstance(dates[0], str):
        dates = [str(d) for d in dates]
    start_date = dates[0]
    end_date = dates[-1]
    
    # 2. 验证bar_time的时间点数量
    unique_times = df.select(pl.col("bar_time").dt.time().unique()).to_series().to_list()
    print(f"    - {factor_name}: {len(unique_times)} 个时间点/天")
    
    # 3. 转换为宽格式
    wide_pd = (
        df.select(["bar_time", "symbol", "factor_value"])
        .to_pandas()
        .pivot(index="bar_time", columns="symbol", values="factor_value")
        .sort_index()
    )
    
    # 确保index是DatetimeIndex
    if not isinstance(wide_pd.index, pd.DatetimeIndex):
        wide_pd.index = pd.to_datetime(wide_pd.index)
    
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
        import traceback
        traceback.print_exc()


def validate_factor_format(df: pl.DataFrame, factor_name: str) -> bool:
    """
    验证因子DataFrame的格式是否正确
    
    检查：
    1. 必需列：symbol, date, bar_time, factor_value
    2. bar_time类型：Datetime
    3. EOD因子：bar_time应该都是15:00:00
    4. M10因子：bar_time应该有24个不同的时间点
    
    Returns:
        True 如果格式正确，False 否则
    """
    required_cols = ["symbol", "date", "bar_time", "factor_value"]
    
    # 检查必需列
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ {factor_name}: 缺少列 {missing_cols}")
        return False
    
    # 检查bar_time类型
    if df["bar_time"].dtype != pl.Datetime:
        print(f"❌ {factor_name}: bar_time类型错误，应为Datetime，实际为{df['bar_time'].dtype}")
        return False
    
    # 检查时间点
    unique_times = df.select(pl.col("bar_time").dt.time().unique()).to_series().to_list()
    name_lower = factor_name.lower()
    
    if "_eod_" in name_lower:
        # EOD应该只有一个时间点（15:00:00）
        from datetime import time
        expected_time = time(15, 0, 0)
        if len(unique_times) != 1:
            print(f"⚠️ {factor_name}: EOD因子应该只有1个时间点，实际有{len(unique_times)}个")
        elif unique_times[0] != expected_time:
            print(f"⚠️ {factor_name}: EOD因子时间点应为15:00:00，实际为{unique_times[0]}")
    else:
        # M10应该有24个时间点
        if len(unique_times) != 24:
            print(f"⚠️ {factor_name}: M10因子应该有24个时间点，实际有{len(unique_times)}个")
    
    print(f"✓ {factor_name}: 格式验证通过")
    return True
