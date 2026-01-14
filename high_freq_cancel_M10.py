import warnings
warnings.filterwarnings('ignore')
import argparse
import re 
import os
import sys
import imp
import logging

from tqdm.contrib.concurrent import process_map
from tqdm import tqdm
import pandas as pd
from datetime import datetime

import legion
from dux.cal import bizday, bizdays

# from ajops import *
# from ajops2 import *
# import ljFuncTool as aj
import ajdata
import ajload as ld
# import fdTools as fd
# import exprTools as et

import polars as pl
import collections # Used in original code's explanation, not directly in this Polars version
import numpy as np   # Used in original code, Polars has its own log10
# import time # 导入 time 模块
from datetime import time


import polars as pl
from pathlib import Path
from typing import Dict, List, Optional, Union


def add_exchange_suffix(df: pl.DataFrame, symbol_col: str = "symbol") -> pl.DataFrame:
    """
    直接在symbol列添加交易所后缀（简化版）
    """
    df = df.with_columns(
        pl.col(symbol_col)
        .cast(pl.Utf8)
        .str.pad_start(6, "0")
        .alias(symbol_col)
    )
    df = df.with_columns(
        pl.when(
            pl.col(symbol_col).str.starts_with("60") | 
            pl.col(symbol_col).str.starts_with("68")
        ).then(pl.col(symbol_col) + ".SH")
        .when(
            pl.col(symbol_col).str.starts_with("00") | 
            pl.col(symbol_col).str.starts_with("30")
        ).then(pl.col(symbol_col) + ".SZ")
        .when(
            pl.col(symbol_col).str.starts_with("8") |
            pl.col(symbol_col).str.starts_with("43") |
            pl.col(symbol_col).str.starts_with("87")
        ).then(pl.col(symbol_col) + ".BJ")
        #.otherwise(pl.col(symbol_col) + ".UNKNOWN")
        .otherwise(pl.col(symbol_col))
        .alias("symbol")
    )
    return df.filter(pl.col('symbol').str.contains('SH|SZ|BJ'))


def high_freq_cancel_m10_single_mkt(exchange: str, date: str, data_path: str = "/local/g04nfs/data/tick/eq/std/"):
    # ── 读数据 ── 
    trade = pl.read_parquet(f"{data_path}trade/{exchange}/{date}")
    quote = pl.read_parquet(f"{data_path}quote/{exchange}/{date}")

    # ── 撤单 & 原委托 ──
    if exchange == "SH":
        cancels = (quote.filter(pl.col('ty') == 68)
                  .filter(pl.col("xts").dt.time() >= time(9,30))
                  .select(["inst_id", "ch", "order_no", pl.col("xts").alias("xts_cancel"), "qty"]))
        orders  = (quote.filter(pl.col('ty') != 68)
                  .select(["inst_id", "ch", "order_no", pl.col("xts").alias("xts_new")])
                  .unique())
    elif exchange == "SZ":
        cancels = (trade.filter(pl.col('flag') == 52)
                  .filter(pl.col("xts").dt.time() >= time(9,30))
                  .with_columns(pl.max_horizontal([pl.col("an"), pl.col("bn")]).alias("order_no"))
                  .select("inst_id", "ch", "order_no", pl.col("xts").alias("xts_cancel"), "qty"))
        orders  = (quote.select(["inst_id", "ch", "order_no", pl.col("xts").alias("xts_new")])
                  .unique())
        
    # ── 撤单关联原委托，计算生命周期 ──
    df = (cancels.join(orders, on=["inst_id", "ch", "order_no"], how="left")
                 .filter(pl.col("xts_new").is_not_null())
                 .with_columns(
                    pl.when(
                        (pl.col("xts_new").dt.time() < time(11, 30)) & (pl.col("xts_cancel").dt.time() >= time(13, 0))
                    )
                    .then((pl.col("xts_cancel") - pl.col("xts_new")) - pl.duration(microseconds=5_400_000_000))    # 剔除午盘
                    .otherwise(pl.col("xts_cancel") - pl.col("xts_new"))
                    .alias("life_us")))
    df = df.sort(["inst_id", "xts_cancel"])
    
    agg = (
        df.group_by_dynamic(                 # 动态窗口
            index_column="xts_cancel",       # 时间戳列
            every="10m",                     # 窗长 10 分钟
            label="right", closed="left",    # 窗标签取右端 ⇒ 09:30~09:40 → 09:40 / 左闭右开
            by="inst_id"                     # 先按股票再做时间窗
        )
        .agg([
            (pl.col("life_us") <= 5_000_000).sum().alias("C5"),
            (pl.col("life_us") <= 30_000_000).sum().alias("C30"),
            (pl.col("life_us") <= 60_000_000).sum().alias("C60"),
            pl.when(pl.col("life_us") <= 5_000_000)
            .then(pl.col("qty")).otherwise(0).sum().alias("V5"),
            pl.when(pl.col("life_us") <= 30_000_000)
            .then(pl.col("qty")).otherwise(0).sum().alias("V30"),
            pl.when(pl.col("life_us") <= 60_000_000)
            .then(pl.col("qty")).otherwise(0).sum().alias("V60"),
        ])
        .with_columns([
            (pl.col("C5") / (pl.col("C30") + 1e-6)).alias("hf_cancel_cnt_5_30"),
            (pl.col("C5") / (pl.col("C60") + 1e-6)).alias("hf_cancel_cnt_5_60"),
            (pl.col("V5") / (pl.col("V30") + 1e-6)).alias("hf_cancel_qty_5_30"),
            (pl.col("V5") / (pl.col("V60") + 1e-6)).alias("hf_cancel_qty_5_60"),
        ])
    )

    # 将13:10行用11:30行forward fill
    agg_no_1310 = agg.filter(pl.col("xts_cancel").dt.time() != time(13, 10))
    agg_1130 = agg.filter(pl.col("xts_cancel").dt.time() == time(11, 30))

    if agg_1130.height > 0:
        agg_1130_1310 = agg_1130.with_columns(
            (pl.col("xts_cancel") + pl.duration(seconds=6000)).alias("xts_cancel")
        )
        agg = pl.concat([agg_no_1310, agg_1130_1310]).sort(["inst_id", "xts_cancel"])
    else:
        agg = agg_no_1310
    
    agg = add_exchange_suffix(agg, "inst_id")
    agg = agg.filter(
        ((pl.col("xts_cancel").dt.time() >= time(9, 40)) & (pl.col("xts_cancel").dt.time() <= time(11, 30))) |
        ((pl.col("xts_cancel").dt.time() >= time(13, 10)) & (pl.col("xts_cancel").dt.time() <= time(15, 0)))
    )
    
    factor_names = ["hf_cancel_cnt_5_30", "hf_cancel_cnt_5_60", 
                    "hf_cancel_qty_5_30", "hf_cancel_qty_5_60"]

    out = {}
    for fname in factor_names:
        df = (agg.select(["xts_cancel", "symbol", fname]).to_pandas()
                .pivot(index="xts_cancel", columns="symbol", values=fname)
                .sort_index())
        df.index.name = None
        df.columns.name = None
        if len(df) < 24:
            morning = pd.date_range(f"{date} 09:40:00", f"{date} 11:30:00", freq="10T")
            afternoon = pd.date_range(f"{date} 13:10:00", f"{date} 15:00:00", freq="10T")
            full_index = morning.append(afternoon)
            df = df.reindex(full_index)
        out[fname] = df

    return out


import os
from functools import wraps

def check_tick_files_exist(base_path='/local/g04nfs/data/tick/eq/std/'):
    def decorator(func):
        @wraps(func)
        def wrapper(date, data_path=base_path, *args, **kwargs):
            required_paths = [
                # f"quote/SH/{date}",
                f"trade/SH/{date}",
                f"quote/SZ/{date}",
                f"trade/SZ/{date}"
            ]
            
            missing_files = []
            for path in required_paths:
                full_path = os.path.join(data_path, path)
                if not os.path.exists(full_path):
                    missing_files.append(path)
            
            if not missing_files:
                return func(date, data_path, *args, **kwargs)
            else:
                print(f"Missing files for date {date}: {missing_files}, skipping.")
                return None
        return wrapper
    return decorator

@check_tick_files_exist()
def high_freq_cancel_m10(date: str, data_path="/local/g04nfs/data/tick/eq/std/"):
    """输出 {factor_name : DataFrame}"""
    sh_dict = high_freq_cancel_m10_single_mkt("SH", date, data_path)
    sz_dict = high_freq_cancel_m10_single_mkt("SZ", date, data_path)

    # 合并同名因子
    merged = {}
    for key in set(sh_dict) | set(sz_dict):
        df_sh = sh_dict.get(key, pd.DataFrame())
        df_sz = sz_dict.get(key, pd.DataFrame())
        merged[key] = pd.concat([df_sh, df_sz], axis=1).sort_index(axis=1)
    
    return merged

def validate_date(date_str: str) -> str:
    """ 验证日期格式是否为YYYYMMDD """
    try:
        datetime.strptime(date_str, "%Y%m%d")
        return date_str
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid date format: {date_str}, expected YYYYMMDD")


if __name__ == "__main__":
    # 命令行参数解析
    parser = argparse.ArgumentParser(description="股票订单数据处理脚本")
    parser.add_argument(
        "-d", "--date",
        type=validate_date,  # 自动校验格式
        required=True,
        help="处理日期（格式：YYYYMMDD，例如20220104）"
    )
    
    parser.add_argument(
        "-p", "--data_path",
        type=str,
        default="/local/g04nfs/data/tick/eq/std/",
        help="数据文件路径（默认：/local/g04nfs/data/tick/eq/std/）"
    )
    
    # 添加其他可选参数（示例）
    parser.add_argument(
        "--cores",
        type=int,
        default=1,
        help="并行处理核心数（默认1，0表示自动检测）"
    )
    
    args = parser.parse_args()
    
    # 调用主函数
    date = str(args.date)
    data_path = args.data_path
    # 确保路径以斜杠结尾
    if not data_path.endswith('/'):
        data_path += '/'
    
    high_freq_cancels = high_freq_cancel_m10(date, data_path)
    import legion
    import ajload as ld
    
    lg_base = legion.Legion('/big/share/zjchen/base/cne/M10/',freq='M10',univ='cne', mode='w')

    for name, df in high_freq_cancels.items():
        ld.dk2lg(df, str(date), str(date), lg_base, f"zjchen/M10/hf/high_freq_cancel/{name}")
