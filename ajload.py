# import dinal as dn
import bottleneck as bn
import numpy as np
import pandas as pd


import seaborn as sn
import matplotlib.pyplot as plt
from tqdm.contrib.concurrent import process_map
import datetime
import time
import os
import abdata
import legion as lg
import ajdata

from legion import KTD
from dux.cal import bizday, bizdays

global LOADER
LOADER = {}
LG_PATH = '/slice/ljs/cne/EOD' 
DN_PATH = '/data/dinal/cne/EOD/'

def init_lgloader(BEGINDATE, ENDDATE, freq='EOD', univ='cne'):
    lg_api = lg.Legion(LG_PATH,freq=freq,univ=univ)
    lg_loader = lg_api.loader(str(BEGINDATE) + '-' + str(ENDDATE), symbolic = True)
    LOADER.update({'lg_loader':lg_loader})
    return lg_loader


def lg2df(x, freq= 'DAILY', univ = None, loader=None):
    if isinstance(x, str) and loader is None:
        loader = LOADER['lg_loader']
        x = loader[x]
    timestamp = ''
    if freq == 'DAILY': 
        timestamp = ajdata.daily_timestamps
        y = x
    if freq == 'M10': 
        timestamp = ajdata.m10_timestamps
        y = x.reshape(x.shape[0], -1)
    # 保证y为2维，避免squeeze导致一维丢失
    y = np.asarray(y)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    df = pd.DataFrame(y, index=x.ks, columns=[str(i) + ' ' + j for i in x.ds for j in timestamp]).T
    df.index = pd.to_datetime([str(i) for i in df.index.values])
    if univ is not None:
        df = df.reindex(index=df.index, columns=df.columns)
    return df

def init_dnloader(begindate, enddate):
    """
    Update loader, sid2K dictionary, and qdi based on the specified date range.

    Args:
    begindate (str): Start date in the format 'YYYYMMDD'.
    enddate (str): End date in the format 'YYYYMMDD'.

    Returns:
    dn.Dinal: Updated loader object.
    dict: Updated sid2K dictionary.
    ndarray: Updated qdi array.
    """
    loader = dn.Dinal(DN_PATH)[f'{begindate}-{enddate}']
    qdate = loader['wdb_fundamental/qdate']
    dates = loader['backoffice/dates']
    tickers = loader['backoffice/tickers']
    sid = loader['backoffice/sid']
    
    data = {}
    
    data['sid2K'] = merge_dicts(dn.utils.gen_keymap(sid, tickers))
    if 553777160 in data['sid2K']:
        data['sid2K'][10847] = data['sid2K'][553777160]
        del data['sid2K'][553777160]
    if 553777161 in data['sid2K']:
        data['sid2K'][10848] = data['sid2K'][553777161]
        del data['sid2K'][553777161]
    if 553777162 in data['sid2K']:
        data['sid2K'][10849] = data['sid2K'][553777162]
        del data['sid2K'][553777162]
    data['K2sid'] = {v: k for k, v in data['sid2K'].items()}
    data['qdate'] = qdate
    data['dates'] = dates
    data['qdi'] = map_dates_to_indices(qdate[:], dates[:])

        
    data['ret'] = loader['md/ret1'][:]
    if 'DAILY' not in ajdata.DATA:
        ajdata.DATA.update({'DAILY':{}})
    ajdata.DATA['DAILY'].update(data)
    ajdata.DATA['DAILY'].update({'sid2K': data['sid2K'],'K2sid': data['K2sid']})
    
    LOADER['dn_loader'] = loader
    descript = pd.read_csv('/data/dinal/cne/EOD/data_descriptions.csv')
    LOADER['descript'] = descript
    LOADER['data'] = data
    
    return loader, descript

def merge_dicts(list_of_dicts):
    """
    Merge a list of dictionaries into a single dictionary.

    Args:
    list_of_dicts (list): List of dictionaries to merge.

    Returns:
    dict: Merged dictionary.
    """
    merged_dict = {}
    for dictionary in list_of_dicts:
        merged_dict.update(dictionary)
    return merged_dict

def map_dates_to_indices(qdates, dates):
    """
    Map dates to indices in the dates array.

    Args:
    qdates (ndarray): Array of dates.
    dates (ndarray): Array of all dates.

    Returns:
    ndarray: Array of indices corresponding to dates.
    """
    date_to_index = {date: index for index, date in enumerate(dates)}
    
    num_samples, num_quarters = qdates.shape
    qdi = np.full((num_samples, num_quarters), np.nan)
    for sample_index in range(num_samples):
        for quarter_index in range(num_quarters):
            date = qdates[sample_index, quarter_index]
            if not np.isnan(date):
                qdi[sample_index, quarter_index] = date_to_index[date]
    return qdi

def map_fq_dates_to_indices(arrfq, data, fill_days=140):
    """
    Map quarterly dates to indices in the daily dates array and fill missing values.

    Args:
    arrfq (ndarray): Array of quarterly dates.
    data (dict): Dictionary containing daily dates and other data.
    fill_days (int): Number of days to fill missing data. Defaults to 140.

    Returns:
    ndarray: Array of mapped indices.
    """
    if 'qdi' not in data:
        raise Exception('Fundamental transformation requires qdi')
    qdi = data['qdi']
    num_k, num_s = data['ret'].shape
    output = np.full_like(data['ret'], np.nan)
    
    for ki in range(num_k):
        arrfq_ki = arrfq[ki, :]
        iqdi = qdi[ki, :]
        valid_fqdate_imask = np.isfinite(iqdi)
        iqdi_valid = iqdi[valid_fqdate_imask].astype(int)
        output[ki, iqdi_valid] = arrfq_ki[valid_fqdate_imask]
        
    if fill_days > 0:
        output = ts_fill(output, fill_days)
    return output


def dn2df(qdata, data=None, freq='DAILY', dk_match=False, loader=None):
    if isinstance(qdata, str) and loader is None:
        loader = LOADER['dn_loader']
        qdata = loader[qdata]
    sids = qdata.index['Ks']
    Ds = qdata.index['Ds']
    timestamp = ''
    if freq == 'DAILY': 
        timestamp = ajdata.daily_timestamps
    if data is None: data = LOADER['data']
    
    output = pd.DataFrame(qdata[:], index=sids, columns= [str(i) + ' ' + j for i in Ds for j in timestamp]).T

    output.index = pd.to_datetime(output.index)
    #output = output[data['sid2K'].keys()].rename(columns = data['sid2K']).sort_index(axis=1)
    output = output.loc[:,output.columns.isin(data['sid2K'].keys())].rename(columns = data['sid2K']).sort_index(axis=1)
    output = output.loc[:,~output.columns.duplicated()]
    if dk_match and ajdata.DATA != {}:
        output = aj.dk_match(output, ajdata.DATA['DAILY']['Ret__bar'])
    return output

def get_dn_varname(varname):
    var, cat = varname.split('__')
    if cat == 'wdb':
        if var.startswith('fi'):
            return 'wdb_fundamental_fi/wdb_'+var
        else:
            return 'wdb_fundamental/wdb_'+var
    elif cat == 'wkq':
        if var.startswith('fi'):
            return 'wkq_fundamental_fi/wkq_'+var
        else:
            return 'wkq_fundamental/wkq_'+var
    elif cat == 'st':
        dnvar = 'suntime_'+var
        descript = DNLOADER['descript']
        dncat = descript[descript.varname==dnvar].groupname.values[0]
        return dncat+'/'+dnvar
    elif cat == 'cor':
        if var.startswith('corr_ret1') or var.startswith('rawcorr_ret1'):
            return 'corr_ret1/'+var
        elif var.startswith('corr_rret1') or var.startswith('rawcorr_rret1'):
            return 'corr_rret1/'+var
    else:            
        raise ValueError

def cumadj(ed, bd='20100101'):
    dinal_api = dn.Dinal(DN_PATH)
    dinal_loader = dinal_api[bd+'-'+ed]
    data = {}
    data['dates'] = dinal_loader['backoffice/dates']
    data['sid2K'] = merge_dicts(dn.utils.gen_keymap(dinal_loader['backoffice/sid'], dinal_loader['backoffice/tickers']))
    if 553777160 in data['sid2K']:
        data['sid2K'][10847] = data['sid2K'][553777160]
        del data['sid2K'][553777160]
    if 553777161 in data['sid2K']:
        data['sid2K'][10848] = data['sid2K'][553777161]
        del data['sid2K'][553777161]
    if 553777162 in data['sid2K']:
        data['sid2K'][10849] = data['sid2K'][553777162]
        del data['sid2K'][553777162]
    prxfac = dn2df(dinal_loader['md/prxfac'],data)
    qtyfac = dn2df(dinal_loader['md/qtyfac'],data)
    cumprxadj = prxfac.cumprod()
    cumqtyadj = qtyfac.cumprod()
    return prxfac, qtyfac, cumprxadj, cumqtyadj

def cumad_lg(ed, bd='20100101'):
    lg_api = lg.Legion(LG_PATH,freq='EOD',univ='cne')
    lg_loader = lg_api.loader(str(bd) + '-' + str(ed), symbolic = True)
    prxadj_lg = lg2df(lg_loader['md/std/Adjf'])
    cumprxadj = prxadj_lg.bfill().iloc[0,:]/prxadj_lg
    return cumprxadj

def alp_freq(alp):
    names = ['M1', 'M5', 'M10','DAILY']
    freqs = [ajdata.m1_timestamps, ajdata.m5_timestamps, ajdata.m10_timestamps, ajdata.daily_timestamps]
    
    timestamps = list(map(lambda x: x.strftime('%H:%M:%S.000', ), np.unique(alp.index.time)))
    return names[freqs.index(timestamps)]

def dk2lg(x,beginDate,endDate,api,alphaName):
    y = x[beginDate: endDate].values
    ds = bizdays(beginDate, endDate)
    ts = time2second(ajdata.get_timestamps(alp_freq(x)))
    z = KTD(ks=list(x.columns.values),ts = ts,ds = ds)
    z[:]= np.swapaxes(y.T.reshape(-1,len(ds),len(ts)),1,2)
    api.save(z, alphaName)
    return z

def time2second(tlist):
    sec_list = []
    for i in tlist:
        t = time.strptime(i,'%H:%M:%S.000')
        sec_list.append(int(datetime.timedelta(hours=t.tm_hour,minutes=t.tm_min,seconds=t.tm_sec).total_seconds()))
    return sec_list

def save_termdict_lg(term_dct, beginDate, endDate, term_path, api, prefix = None, cores=20):
    global save_term_lg
    def save_term_lg(term):
        try:
            saved = dk2lg(term_dct[term], beginDate, endDate, api, os.path.join(term_path, term))
        except Exception as e:
            print(term, ':', str(e))
    if prefix is not None:
        term_dct = dict(zip([prefix+'_'+str(i) for i in range(len(term_dct))], term_dct.values()))
    if cores==1:
        for t in list(term_dct):
            saved = save_term_lg(t)
    else:
        saved = process_map(save_term_lg, [k for k, v in term_dct.items() if v.shape[0] != 0], max_workers=cores)
    return 0

def load_bar_from_abdata(var_name, sd, ed, freq='DAILY', univ='alea', cores=1):
    """
    从 abdata 加载指定时间范围的 bar 数据，并高效处理多个 DataFrame 的合并。

    参数：
    - var_name: 要加载的变量名称
    - sd: 开始日期 (字符串，格式 YYYY-MM-DD)
    - ed: 结束日期 (字符串，格式 YYYY-MM-DD)
    - freq: 数据频率，默认 'DAILY'
    - univ: 数据的 universe，默认 'alea'
    - cores: 并行读取的线程数，默认 1 (单线程)

    返回：
    - pandas.DataFrame，按时间和 symbol 的多索引结构
    """
    import abdata
    import pandas as pd
    import numpy as np
    from concurrent.futures import ProcessPoolExecutor
    from tqdm.contrib.concurrent import process_map

    abdata.init('ck2.ab.cap', 9000)

    # 定义加载单天数据的函数
    global load_bar_by_date
    def load_bar_by_date(date, enddate=None):
        try:
            if enddate is None:
                enddate = date
            df = abdata.bar(univ, freq, [var_name, 'sym'], sd=date, ed=enddate, ret_df=True)
            df.set_index(['D', 'T', 'sym'], inplace=True, drop=True)
            df = df[var_name].unstack('sym')  # 以 symbol 为列
            df.index = pd.to_datetime(df.index.get_level_values(0).astype(str) + ' ' +
                                      df.index.get_level_values(1).astype(str))
            return df
        except Exception as e:
            print(f"Error loading data for date {date}: {e}")
            return pd.DataFrame()  # 返回空 DataFrame

    if cores == 1:
        # 单线程加载所有数据
        return load_bar_by_date(sd, ed)
    else:
        # 多线程并行加载数据
        from dux.cal import bizdays  # 假设 bizdays 已正确初始化
        
        dates = bizdays(sd, ed)  # 获取所有工作日日期
        results = process_map(load_bar_by_date, dates, max_workers=cores)

        # 获取所有 DataFrame 的列名并集
        union_tickers = set()
        for df in results:
            if not df.empty:
                union_tickers.update(df.columns)
        union_tickers = sorted(union_tickers)  # 转换为有序列表

        # 对每个 DataFrame 进行 reindex
        reindexed_results = [
            df.reindex(columns=union_tickers, fill_value=np.nan).values
            for df in results if not df.empty
        ]

        # 使用 numpy 直接拼接所有值
        concatenated_values = np.concatenate(reindexed_results, axis=0)

        # 构造新的 DataFrame
        from functools import reduce
        final_index = reduce(lambda x, y: x.union(y), [df.index for df in results if not df.empty])
        final_df = pd.DataFrame(concatenated_values, index=final_index, columns=union_tickers)

        return final_df
