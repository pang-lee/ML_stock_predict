import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.stattools import adfuller, acf, pacf
from collections import defaultdict
import math, re
import talib

# -------------------------------------- 資料前處理 --------------------------------------

def prepare_data(type_of_data, data_name):
    result = type_of_data.split('/')[0]
    tmp = pd.read_csv(f'./index_data/{type_of_data}/{data_name}.csv')
    if result == 'shioaji':
        tmp['ts'] = pd.to_datetime(tmp['ts'])
        tmp = tmp.rename(columns=lambda x: x.lower())
    else:
        tmp['ts'] = pd.to_datetime(tmp['datetime'])
        tmp = tmp.rename(columns=lambda x: x.lower())
    
    return tmp

df1 = prepare_data('shioaji/2024_0601', '00631L')
df2 = prepare_data('shioaji/2024_0601', 'TXFR1')

# 自定義函數處理 tick_type 分析
def analyze_tick_types(tick_types):
    """
    分析該秒內的成交類型分布
    :param tick_types: Series, 該秒內的 tick_type 列表
    :return: dict, 包括內盤、外盤成交數量及主要成交方向
    """
    tick_types = tick_types.tolist()  # 轉換為列表
    outer_trades = tick_types.count(1)  # 外盤成交數量
    inner_trades = tick_types.count(-1)  # 內盤成交數量
    net_trades = outer_trades - inner_trades  # 淨成交量
    dominant = "外盤" if net_trades > 0 else "內盤" if net_trades < 0 else "均衡"
    return {
        "outer_trades": outer_trades,
        "inner_trades": inner_trades,
        "dominant": dominant
    }

# 資料時間調整
def pre_process(df1, date_start, date_end):
    # 假设你的df已经加载到dataframe
    df1['ts'] = pd.to_datetime(df1['ts'])  # 将ts列转换为datetime类型

    # 按秒分组
    df1 = df1.groupby(df1['ts'].dt.floor('s')).agg({
        'close': 'last', 
        'volume': 'sum', 
        'bid_price': lambda x: tuple(sorted(x)), 
        'bid_volume': 'sum', 
        'ask_price': lambda x: tuple(sorted(x)), 
        'ask_volume': 'sum', 
        'tick_type': lambda x: analyze_tick_types(x)
    }).reset_index()
    
    # 展開內盤、外盤和主要成交方向
    df1['outer_trades'] = df1['tick_type'].apply(lambda x: x['outer_trades'])
    df1['inner_trades'] = df1['tick_type'].apply(lambda x: x['inner_trades'])
    df1['dominant'] = df1['tick_type'].apply(lambda x: x['dominant'])
    
    # 刪除原始的 tick_type 列
    df1.drop(columns=['tick_type'], inplace=True)

    # 创建时间范围从開始到結束天數（或多个天数），每天包含9:00:00到13:30:00之间的每一秒
    time_range = pd.date_range(date_start, date_end, freq='s')

    # 将时间范围转换为DataFrame
    full_time_df = pd.DataFrame(time_range, columns=['ts'])

    # 通过检查时间是否在9:00:00到13:30:00之间来剔除跨天的数据
    valid_time_range = full_time_df['ts'].dt.time.between(pd.to_datetime('09:00:00').time(), pd.to_datetime('13:30:00').time())

    # 筛选出有效时间
    valid_time = full_time_df[valid_time_range]

    # 合并df1和df2的结果，确保它们与mer_ori_data按秒对齐, 首先将df1和df2与mer_ori_data合并，使用'left'连接方式，以保留所有有效时间
    merged_df1 = pd.merge(valid_time, df1, on='ts', how='left')

    # 向后补齐，再向前补齐
    mer_ori_data = merged_df1.ffill()  # 向后补齐
    mer_ori_data = merged_df1.bfill()  # 向后补齐

    # 设置'ts'为index
    mer_ori_data.set_index('ts', inplace=True)

    # 删除含有NaN的行
    mer_ori_data = mer_ori_data.dropna()
    
    return mer_ori_data

mer_ori_data = pre_process(df1, '2024-06-01 9:00:00', '2025-01-22 13:30:00')

df1.set_index('ts', inplace=True)
df2.set_index('ts', inplace=True)

# -------------------------------------- 雙邊交易回測 --------------------------------------

def convert_to_minute_ohlcv(data):
    """
    將秒級別的交易數據轉換為1分鐘OHLCV K線數據。
    
    :param data: pandas DataFrame，必須包含 'close', 'volume' 欄位，索引為時間戳。
    :return: pandas DataFrame 轉換後的1分鐘OHLCV K線數據
    """
    # 確保索引是時間格式
    data = data.copy()
    data.index = pd.to_datetime(data.index)

    # 使用resample進行分組並自訂聚合方式
    minute_ohlcv = data.resample('1min').agg({
        'close': ['first', 'max', 'min', 'last'],  # 開盤, 最高, 最低, 收盤
        'volume': 'sum'                            # 成交量累加
    }).dropna()  # 移除沒有交易的分鐘

    # 重新命名欄位
    minute_ohlcv.columns = ['open', 'high', 'low', 'close', 'volume']

    return minute_ohlcv

def compute_technical_indicators(data, long_window=10, short_window=5, plot_vpfr=False):
    if len(data) < long_window:
        return False  # 如果資料不足，直接返回 False
    
    data['lowest_low'] = data['low'].rolling(window=long_window).min()
    data['highest_high'] = data['high'].rolling(window=long_window).max()
    
    # 計算RSV基於價格
    rsv = ((data['close'] - data['lowest_low']) / 
               (data['highest_high'] - data['lowest_low'])) * 100
    
    # 如果價格沒變動，RSV 設為 0
    rsv = rsv.fillna(0)
    
    # 計算ADX
    adx = talib.ADX(data['high'], data['low'], data['close'], timeperiod=short_window)
    
    # 計算+DI和-DI
    plus_di = talib.PLUS_DI(data['high'], data['low'], data['close'], timeperiod=short_window)
    minus_di = talib.MINUS_DI(data['high'], data['low'], data['close'], timeperiod=short_window)

    # 計算VPFR
    price_bins = pd.interval_range(
        start=data['lowest_low'].iloc[-1], 
        end=data['highest_high'].iloc[-1], 
        freq=(data['highest_high'].iloc[-1] - data['lowest_low'].iloc[-1]) / long_window
    )

    # 使用.copy()防止SettingWithCopyWarning
    recent_data = data.iloc[-long_window:].copy()

    # 使用.loc確保安全賦值
    recent_data.loc[:, 'price_bin'] = pd.cut(recent_data['close'], bins=price_bins)

    # 修正 observed=True 警告
    volume_distribution = recent_data.groupby('price_bin', observed=True)['volume'].sum()
    total_volume = recent_data['volume'].sum()
    vpfr = volume_distribution / total_volume

    # 將VPFR作為字典傳回
    vpfr_data = {str(bin): value for bin, value in vpfr.items()}
    
    # ✅ 可選的VPFR分佈圖繪製
    if plot_vpfr:
        plt.figure(figsize=(12, 6))
        plt.bar([str(bin) for bin in vpfr.index], vpfr.values, color='skyblue')
        plt.xticks(rotation=90)
        plt.xlabel('Price Range')
        plt.ylabel('VPFR Ratio')
        plt.title('VPFR Distribution')
        plt.tight_layout()
        plt.show()
        
        breakpoint()

    return {
        "rsv": rsv.iloc[-1],
        "vpfr": vpfr_data,
        "adx": adx.iloc[-1],  # 最新的ADX值
        "+di": plus_di.iloc[-1],  # 最新的+DI值
        "-di": minus_di.iloc[-1],  # 最新的-DI值
    }

def check_acf_pacf_stationary(data, window=15, max_lag=15, stricter_confidence=0.95, adf_significance=0.05, debug=False):
    """檢查 ACF 和 PACF，並確保滯後期數不超過 window/2"""
    if len(data) < window:
        return False  # 如果資料不足，直接返回 False
    
    recent_data = data[-window:]
    
    # 確保 max_lag 不超過 window 的一半
    max_lag = min(max_lag, len(recent_data) // 2)

    # 計算 ACF 和 PACF
    acf_values = acf(recent_data, nlags=max_lag, fft=False)
    pacf_values = pacf(recent_data, nlags=max_lag)

    # 計算信賴區間 z-score (例如 99% 設為 2.58，97% 設為 1.88，等)
    z_score = stats.norm.ppf(1 - (1 - stricter_confidence) / 2)  # ppf 返回對應的 z-score
    stricter_conf_interval = z_score / np.sqrt(len(recent_data))
    
    # 檢查 ACF 和 PACF 是否快速衰減到更嚴格的信賴區間內
    acf_in_blue_zone = np.all(np.abs(acf_values[1:]) < stricter_conf_interval)
    pacf_in_blue_zone = np.all(np.abs(pacf_values[1:]) < stricter_conf_interval)

    # 執行 ADF 測試
    adf_result = adfuller(recent_data)
    adf_p_value = adf_result[1]  # p-value

    # 設定 ADF 測試門檻
    adf_is_stationary = adf_p_value < adf_significance

    # 记录当前时间戳和 ADF 检验结果
    if debug:
        print(f"Time: {recent_data.index[-1]} | ADF p-value: {adf_p_value}, Is Stationary: 「{adf_is_stationary}」, ACF in blue zone: {acf_in_blue_zone}, PACF in blue zone: {pacf_in_blue_zone}")

    return acf_in_blue_zone and pacf_in_blue_zone and adf_is_stationary

def check_vpfr(vpfr, data, params, is_trend=False, debug=False):
    total_volume_above_threshold = 0
    total_volume_below_threshold = 0

    # 確保 data 是 Pandas DataFrame 格式
    if not isinstance(data, pd.DataFrame):
        print("Error: data 不是 DataFrame 格式!")
        return False

    # 遍歷 DataFrame 中的每一行資料，判斷成交量是否超過閾值
    for _, entry in data.iterrows():
        volume = entry['volume']
        if volume > params.get('volume_threshold'):
            total_volume_above_threshold += 1
        else:
            total_volume_below_threshold += 1

    # 計算成交量超過閾值的比例
    total_data_points = len(data)
    if total_data_points == 0:
        return False  # 若無資料則不進行交易

    volume_above_ratio = total_volume_above_threshold / total_data_points
    
    if debug:
        print(f"成交量超過閾值的比例: {(volume_above_ratio)*100}%")

    # 根據震盪盤或趨勢盤判斷成交量條件
    if is_trend:
        if volume_above_ratio >= params.get('volume_ratio'):  # 超過 60% 是趨勢盤
            if debug:
                print(f"趨勢盤: 大部分成交量大於閾值({params.get('volume_ratio')})")
        else:
            return (False, 0)  # 不符合趨勢盤條件
    
    else:  # 震盪盤: 成交量全部小於閾值
        if volume_above_ratio < params.get('volume_ratio'):  # 所有成交量都小於閾值
            if debug:
                print(f"震盪盤: 所有成交量小於閾值({params.get('volume_ratio')})")
        else:
            return False  # 不符合震盪盤條件
        
    # ------------ VPFR 標準差與集中度計算 --------------
    price_ranges = []
    volumes = []

    # 將 vpfr 解析成數值列表
    for key, volume in vpfr.items():
        lower, upper = map(float, re.findall(r"\d+\.\d+", key))
        price_ranges.append((lower + upper) / 2)  # 中位數作為代表價格
        volumes.append(volume)

    # 計算價格的均值與標準差
    mean_price = np.mean(price_ranges)
    std_dev_price = np.std(price_ranges)

    # 定義高價區與低價區門檻
    high_price_threshold = mean_price + std_dev_price
    low_price_threshold = mean_price - std_dev_price

    # 計算高價與低價的VPFR成交量
    high_price_vpfr = sum(vol for pr, vol in zip(price_ranges, volumes) if pr >= high_price_threshold)
    low_price_vpfr = sum(vol for pr, vol in zip(price_ranges, volumes) if pr <= low_price_threshold)

    if debug:
        print(f"均值價格: {mean_price:.2f}, 標準差: {std_dev_price:.2f}")
        print(f"高價區門檻: {high_price_threshold:.2f}, 低價區門檻: {low_price_threshold:.2f}")
        print(f"高價區成交量: {high_price_vpfr:.4f}, 低價區成交量: {low_price_vpfr:.4f}")

    # 根據 VPFR 判斷盤勢類型
    if is_trend:
        if high_price_vpfr > params.get('vpfr_trend'):
            if debug:
                print("✅ 趨勢盤: 成交量集中於高價位")
            return (True, 1)

        if low_price_vpfr > params.get('vpfr_trend'):
            if debug:
                print("✅ 趨勢盤: 成交量集中於低價位")
            return (True, -1)

        else:
            if debug:
                print("❌ 不符合趨勢盤條件")
            return (False, 0)
    else:
        if high_price_vpfr < params.get('vpfr_oscillation') and low_price_vpfr < params.get('vpfr_oscillation'):
            if debug:
                print("✅ 震盪盤: 成交量分佈均勻")
            return True
        else:
            if debug:
                print("❌ 不符合震盪盤條件")
            return False

def find_support_resistance_based_on_vpfr(vpfr, tick_size, params, debug=False):
    """
    根據VPFR成交量分佈來直接計算支撐位和壓力位。
    
    :param vpfr: VPFR成交量分佈字典
    :return: 支撐位和壓力位
    """
    # 初始化最小和最大價格
    min_price = float('inf')  # 設置一個很大的初始最小價格
    max_price = float('-inf')  # 設置一個很小的初始最大價格
    
    # 遍歷所有區間，更新最小價格和最大價格
    for price_range in vpfr:
        lower_price, upper_price = price_range[1:-1].split(',')  # 提取區間中的價格
        lower_price = float(lower_price)  # 轉換為浮點數
        upper_price = float(upper_price)  # 轉換為浮點數
        
        # 更新最小價格和最大價格
        if lower_price < min_price:
            min_price = lower_price
        if upper_price > max_price:
            max_price = upper_price

    # 計算支撐與壓力區間 (加上 buffer)
    buffer = params.get('oscillation_buffer', 0.1)
    support_range = (round((min_price - buffer) / tick_size) * tick_size, round((min_price + buffer) / tick_size) * tick_size)
    resistance_range = (round((max_price - buffer) / tick_size) * tick_size, round((max_price + buffer) / tick_size) * tick_size)

    # 計算支撐位和壓力位
    price_diff = resistance_range[0] - support_range[1]
    if price_diff <= params.get('price_diff'):
        if debug:
            print(f"價格差 {price_diff:.2f} 需大於閾值 {params.get('price_diff'):.2f}，放棄此次交易判斷")
        return None

    return support_range, resistance_range, min_price, max_price

def trade_oscillation(data1, data2, indicator, params, debug=False):
    """
    震盪判斷並執行交易。
    
    :param data1: df1, data2: df2
    :param indicator: 技術指標字典 (包括rsv和vpfr)
    :param volume_threshold: 成交量閾值
    """
    rsv = indicator['rsv2']
    vpfr1 = indicator['vpfr1']
    vpfr2 = indicator['vpfr2']
    volume_sum = data1['volume'].sum()

    # 成交量過小避免滑價不交易
    if volume_sum <= params.get('volume_slippage'):
        if params.get('check_signal'):
            print(f"成交量 {volume_sum} 小於 {params.get('volume_slippage')}不交易")
        return (False, 0)

    # 使用RSV和平均VPFR共同判斷震盪盤
    if params.get('rsv_low') <= rsv <= params.get('rsv_high'):
        if check_vpfr(vpfr2, data2, is_trend=False, params=params, debug=debug) is True:
            stock_price = data1.iloc[-1]['close']  # 當前股價
            tick_size1 = params.get('tick_size1')

            # 計算支撐位和壓力位
            result = find_support_resistance_based_on_vpfr(vpfr1, tick_size=tick_size1, params=params, debug=debug)

            if result is None:
                return (False, 0)
            
            support_range1, resistance_range1, support_price1, resistance_price1 = result
            if debug:
                print(f"支撐區間: {support_range1}, 壓力區間: {resistance_range1}")

            if debug:
                print(f"支撐位: {support_price1}, 壓力位: {resistance_price1}")
                print(f"支撐範圍: {support_range1}, 壓力範圍: {resistance_range1}")

            # 價格遠離支撐與壓力位, 進入監測
            if not (support_range1[0] < stock_price < support_range1[1] or resistance_range1[0] < stock_price < resistance_range1[1]):
                if debug:
                    print(f"價格偏離支撐與壓力位，進入監控狀態，等待價格接近關鍵區間。")
                return (True, support_range1, resistance_range1)
            
            # 價格接近壓力位, 空頭交易
            if resistance_range1[0] <= stock_price <= resistance_range1[1]:  
                if debug:
                    print(f"價格接近壓力位 {resistance_price1:.2f}，空頭交易")
                return (True, support_range1, resistance_range1)

            # 價格接近支撐位, 多頭交易
            elif support_range1[0] <= stock_price <= support_range1[1]:
                if debug:
                    print(f"價格接近支撐位 {support_price1:.2f}，多頭交易")
                    
                return (True, support_range1, resistance_range1)

    if debug:
        print("震盪盤條件沒過，不交易")

    return (False, 0)

def trade_trend(data2, indicator, params, debug=False):
    """
    判斷趨勢盤條件並執行交易。
    
    :param data1: 價格
    :param indicator: 技術指標字典 (包括rsv和vpfr)
    :param volume_threshold: 成交量閾值
    """
    vpfr = indicator['vpfr2']
    result = check_vpfr(vpfr, data2, is_trend=True, params=params, debug=debug) 

    # 判斷趨勢盤條件
    if result[0] is True:
        adx1 = indicator['adx1']
        di1 = indicator['+di1']
        di2 = indicator['-di1']

        if debug:
            print("趨勢盤確認，判斷是否可以進場")

        # 多頭趨勢
        if result[1] == 1:
            if adx1 > params.get('adx') and di1 > di2:
                if debug:
                    print("多頭趨勢，執行多單交易")
                return (True, 1)

        # 空頭趨勢
        elif result[1] == -1:
            if adx1 > params.get('adx') and di2 > di1:
                if debug:
                    print("空頭趨勢，執行空單交易")
                return (True, -1)
        else:
            return (False, 0)
    
    if debug:
        print("趨勢盤條件沒過，不交易")
        
    return (False, 0)

def analyze(trade_result, is_plot=False):
    total_profit_loss = sum(trade['net_profit_loss'] for trade in trade_result)
    win_count = sum(1 for trade in trade_result if trade['net_profit_loss'] > 0)
    loss_count = sum(1 for trade in trade_result if trade['net_profit_loss'] < 0)
    win_rate = win_count / len(trade_result) if trade_result else 0
    total_trades = len(trade_result)
    capital = trade_result[-1]['capital'] if trade_result else 0

    # 計算資本曲線與最大回撤
    equity_curve = [trade['capital'] for trade in trade_result]
    peak = equity_curve[0] if equity_curve else 0
    max_drawdown = 0
    drawdown_curve = []
    for value in equity_curve:
        peak = max(peak, value)
        drawdown = peak - value
        drawdown_curve.append(drawdown)
        max_drawdown = max(max_drawdown, drawdown)

    # 年化回報
    annualized_return = total_profit_loss / len(trade_result) * 252 if trade_result else 0

    # 夏普比率
    returns = [trade['net_profit_loss'] for trade in trade_result]
    avg_return = sum(returns) / len(returns) if returns else 0
    return_std = (sum((x - avg_return) ** 2 for x in returns) / len(returns)) ** 0.5 if returns else 0
    sharpe_ratio = (annualized_return / (return_std + 1e-9)) if return_std > 0 else 0

    # 風暴比
    sterling_ratio = (annualized_return / (max_drawdown + 1e-9)) if max_drawdown > 0 else 0

    # 按 type 計算績效
    type_performance = defaultdict(lambda: {
        'total_trades': 0,
        'win_count': 0,
        'loss_count': 0,
        'win_rate': 0,
        'total_profit_loss': 0,
        'profit_timestamps': [],  # 賺錢的時間點
        'loss_timestamps': []     # 虧錢的時間點
    })
    
    for trade in trade_result:
        trade_type = trade['type']
        timestamp = trade['timestamp'].strftime('%Y-%m-%d %H:%M:%S')  # 假設每筆交易有 'timestamp'
        
        type_performance[trade_type]['total_trades'] += 1
        type_performance[trade_type]['total_profit_loss'] += trade['net_profit_loss']
        
        # 記錄賺錢或虧錢的時間
        if trade['net_profit_loss'] > 0:
            type_performance[trade_type]['win_count'] += 1
            type_performance[trade_type]['profit_timestamps'].append(timestamp)
        elif trade['net_profit_loss'] < 0:
            type_performance[trade_type]['loss_count'] += 1
            type_performance[trade_type]['loss_timestamps'].append(timestamp)

    # 計算每個盤面類型的勝率
    for trade_type, performance in type_performance.items():
        performance['win_rate'] = (
            performance['win_count'] / performance['total_trades']
            if performance['total_trades'] > 0 else 0
        )

    if is_plot is True:
        fig, axs = plt.subplots(3, 1, figsize=(12, 6))

        # 資金曲線圖
        axs[0].plot(equity_curve, label="Equity (Capital) Curve", color="green")
        axs[0].set_title("Equity Curve (capital curve)")
        axs[0].set_ylabel("Capital")
        axs[0].legend()

        # 最大回撤圖
        axs[1].plot(drawdown_curve, label="Max Drawdown", color="red")
        axs[1].set_title("Max Drawdown")
        axs[1].set_ylabel("Drawdown")
        axs[1].legend()

        # 累積盈虧 (累計盈虧視覺化)
        cumulative_returns = [sum(returns[:i+1]) for i in range(len(returns))]
        axs[2].plot(cumulative_returns, label="Cumulative P&L", color="blue")
        axs[2].set_title("Cumulative Profit & Loss")
        axs[2].set_ylabel("Cumulative Return")
        axs[2].legend()

        plt.tight_layout()
        plt.show()

    return {
        'net_profit_loss': total_profit_loss,
        'capital': capital,
        'win_count': win_count,
        'loss_count': loss_count,
        'win_rate': win_rate,
        'max_drawdown': max_drawdown,
        'annualized_return': annualized_return,
        'sharpe_ratio': sharpe_ratio,
        'sterling_ratio': sterling_ratio,
        'total_trades': total_trades,
        'type_performance': dict(type_performance)
    }

def bilateral_strategy( **params):
    signals, trade_results = [], []  # 儲存交易訊號與紀錄
    indicators_data = {'time': [], 'rsv': []}  # 用來存儲所有的技術指標數據
    monitoring_oscillation, monitoring_trend = False, False  # 初始化監控狀態
    support_range1, resistance_range1 = None, None
    trend_signal1 = None
    position_stock1, position_stock2 = 0, 0
    stop_loss_stock1, stop_loss_stock2 = 0, 0
    profit_stop_stock1, profit_stop_stock2 = 0, 0
    total_fee = 0
    
    data = params.get('data')
    start_date, end_date = params.get('start_date'), params.get('end_date')
    long_window, short_window = params.get('long_window'), params.get('short_window')
    df1, df2 = convert_to_minute_ohlcv(params.get('df1')), params.get('df2')
    entry_price_stock1, entry_price_stock2 = 0, 0
    capital = params.get('capital')
    tick_size1, tick_size2 = params.get('tick_size1'), params.get('tick_size2')
    shares_per_trade1 = params.get('shares_per_trade1')
    tax1 = params.get('tax1')
    commission1 = params.get('commission1')
    stop_ratio1, profit_ratio1 = params.get('stop_ratio1'), params.get('profit_ratio1')
    dynamic_profit_ratio1 = params.get('dynamic_profit_ratio1')
    dynamic_stop_ratio1 = params.get('dynamic_stop_ratio1')
    bid_ask_slippage = params.get('bid_ask_slippage')
    min_bid_ask_volume = params.get('min_bid_ask_volume')

    # 根據給定的start_date與end_date進行篩選
    if start_date and end_date:
        t_d = data[(data.index.date >= pd.to_datetime(start_date).date()) & 
                           (data.index.date <= pd.to_datetime(end_date).date())]
    elif start_date:
        t_d = data[data.index.date >= pd.to_datetime(start_date).date()]
    elif end_date:
        t_d = data[data.index.date <= pd.to_datetime(end_date).date()]
    else:
        t_d = data  # 如果沒有提供日期範圍，則使用全部數據

    # 遍歷每秒數據，但每分鐘檢查一次斜率
    last_checked_minute = None  # 追蹤最後一次檢查的分鐘
    base_time = t_d.index[0]
    last_date = t_d.index[0].date() 
    
    for i in range(1, len(t_d)):
        if capital <= 0: break
        
        current_time = t_d.index[i]
        current_minute = current_time.floor('min')  # 取得分鐘時間戳
        current_date = current_time.date()

        # 檢查是否跨天
        if current_date != last_date:
            # 恢復變數至預設值
            monitoring_oscillation, monitoring_trend = False, False
            support_range1, resistance_range1 = None, None
            trend_signal1 = None
            position_stock1, position_stock2 = 0, 0
            stop_loss_stock1, stop_loss_stock2 = 0, 0
            profit_stop_stock1, profit_stop_stock2 = 0, 0
            entry_price_stock1, entry_price_stock2 = 0, 0
            last_checked_minute = None
            base_time = t_d.index[i]
            
            last_date = current_date  # 更新為新的日期

        # 检查当前时间是否在 13:00 超過就平倉
        if current_time.time() >= pd.Timestamp('13:00:00').time():
            # **強制平倉邏輯**
            if position_stock1 > 0:  # 多倉
                sell_price_stock1 = current_time_price_data['bid_price'][0]  # 平倉價格（賣出價格）
                sell_fee_stock1 = math.ceil(sell_price_stock1 * position_stock1 * commission1)
                sell_tax_stock1 = math.ceil(sell_price_stock1 * position_stock1 * tax1)

                # 計算平倉盈虧
                profit_loss_stock1 = ((sell_price_stock1 - entry_price_stock1) * position_stock1) - (sell_fee_stock1 + sell_tax_stock1)

                # 更新資金
                capital += profit_loss_stock1  # 加回盈虧
                total_fee += sell_fee_stock1 + sell_tax_stock1

                print(f"{current_time}: 強制平倉多倉, 平倉價={sell_price_stock1}, 盈虧={profit_loss_stock1:.2f}, 總資金={capital:.2f}")

                # 記錄交易結果
                trade_results.append({
                    'entry_price_stock1': entry_price_stock1,
                    'exit_price_stock1': sell_price_stock1,
                    'position_stock1': position_stock1,
                    'profit_loss_stock1': profit_loss_stock1,
                    'total_fees': sell_fee_stock1 + sell_tax_stock1,
                    'net_profit_loss': profit_loss_stock1,
                    'capital': capital,
                    'timestamp': current_time,
                    'type': 'close'
                })

                # 清空多倉
                position_stock1 = 0
                entry_price_stock1 = 0

            elif position_stock1 < 0:  # 空倉
                buy_price_stock1 = current_time_price_data['ask_price'][0]  # 平倉價格（買入價格）
                buy_fee_stock1 = math.ceil(buy_price_stock1 * abs(position_stock1) * commission1)

                # 計算平倉盈虧
                profit_loss_stock1 = ((entry_price_stock1 - buy_price_stock1) * abs(position_stock1)) - buy_fee_stock1

                # 更新資金
                capital += profit_loss_stock1  # 加回盈虧
                total_fee += buy_fee_stock1

                print(f"{current_time}: 強制平倉空倉, 平倉價={buy_price_stock1}, 盈虧={profit_loss_stock1:.2f}, 總資金={capital:.2f}")

                # 記錄交易結果
                trade_results.append({
                    'entry_price_stock1': entry_price_stock1,
                    'exit_price_stock1': buy_price_stock1,
                    'position_stock1': position_stock1,
                    'profit_loss_stock1': profit_loss_stock1,
                    'total_fees': buy_fee_stock1,
                    'net_profit_loss': profit_loss_stock1,
                    'capital': capital,
                    'timestamp': current_time,
                    'type': 'close'
                })

                # 清空空倉
                position_stock1 = 0
                entry_price_stock1 = 0

            # **平倉後退出當前迴圈**
            continue
        
        # ✅ 進入監控狀態時僅檢查趨勢
        if monitoring_trend:
            stock_price1 = t_d.iloc[i]['close']
            current_time_price_data = t_d.iloc[i]

            # 計算委買比例和委賣比例
            total_volume = t_d.iloc[i]['bid_volume'] + t_d.iloc[i]['ask_volume']
            if total_volume > 0:
                bid_ratio = t_d.iloc[i]['bid_volume'] / total_volume
                ask_ratio = t_d.iloc[i]['ask_volume'] / total_volume
            else:
                bid_ratio = 0  # 或設置為其他合理值，例如 NaN
                ask_ratio = 0  # 或設置為其他合理值，例如 NaN

            # 如果當前持有多倉
            if position_stock1 > 0:
                if stock_price1 >= profit_stop_stock1:  # 如果價格達到止盈價
                    profit_stop_stock1 = round((stock_price1 * (1 + dynamic_profit_ratio1)) / tick_size1) * tick_size1
                    stop_loss_stock1 = round((stock_price1 * (1 - dynamic_stop_ratio1)) / tick_size1) * tick_size1
                    print(f"{current_time}: 動態止盈調整, 當前價格={stock_price1}, 新止盈價={profit_stop_stock1:.2f}, 新止損價={stop_loss_stock1:.2f}")

                if stock_price1 <= stop_loss_stock1:  # 判斷是否止損
                    sell_price_stock1 = current_time_price_data['bid_price'][0]
                    sell_fee_stock1 = math.ceil(sell_price_stock1 * shares_per_trade1 * commission1)
                    sell_tax_stock1 = math.ceil(sell_price_stock1 * shares_per_trade1 * tax1)

                    profit_loss_stock1 =  ((sell_price_stock1 - entry_price_stock1) * position_stock1) - (sell_fee_stock1 + sell_tax_stock1)
                    capital += profit_loss_stock1
                    total_fee += sell_fee_stock1 + sell_tax_stock1

                    print(f"{current_time}: 多倉止損, 進場價={entry_price_stock1}, 平倉價={sell_price_stock1}, 手續費={sell_fee_stock1:.2f}, 稅金={sell_tax_stock1:.2f}, 盈虧={profit_loss_stock1:.2f}, 總資金={capital:.2f}")
                    signals.append((current_time, 'STOP LOSS LONG'))

                    trade_results.append({
                        'entry_price_stock1': entry_price_stock1,
                        'exit_price_stock1': sell_price_stock1,
                        'position_stock1': position_stock1,
                        'profit_loss_stock1': profit_loss_stock1,
                        'total_fees': total_fee,
                        'net_profit_loss': profit_loss_stock1,
                        'capital': capital,
                        'timestamp': current_time,
                        'type': 'trend'
                    })

                    position_stock1 = 0
                    entry_price_stock1 = 0
                    monitoring_trend = False
                    
                continue

            # 如果當前持有空倉
            elif position_stock1 < 0:
                if stock_price1 <= profit_stop_stock1:  # 如果價格達到止盈價
                    profit_stop_stock1 = round((stock_price1 * (1 - dynamic_profit_ratio1)) / tick_size1) * tick_size1
                    stop_loss_stock1 = round((stock_price1 * (1 + dynamic_stop_ratio1)) / tick_size1) * tick_size1
                    print(f"{current_time}: 動態止盈調整, 當前價格={stock_price1}, 新止盈價={profit_stop_stock1:.2f}, 新止損價={stop_loss_stock1:.2f}")

                if stock_price1 >= stop_loss_stock1:  # 判斷是否止損
                    buy_price_stock1 = current_time_price_data['ask_price'][0]
                    buy_fee_stock1 = math.ceil(buy_price_stock1 * shares_per_trade1 * commission1)

                    profit_loss_stock1 = ((entry_price_stock1 - buy_price_stock1) * abs(position_stock1)) - (buy_fee_stock1)
                    capital += profit_loss_stock1
                    total_fee += buy_fee_stock1

                    print(f"{current_time}: 空倉止損, 進場價={entry_price_stock1}, 平倉價={buy_price_stock1}, 手續費={buy_fee_stock1:.2f}, 盈虧={profit_loss_stock1:.2f}, 總資金={capital:.2f}")
                    signals.append((current_time, 'STOP LOSS SHORT'))

                    trade_results.append({
                        'entry_price_stock1': entry_price_stock1,
                        'exit_price_stock1': buy_price_stock1,
                        'position_stock1': position_stock1,
                        'profit_loss_stock1': profit_loss_stock1,
                        'total_fees': total_fee,
                        'net_profit_loss': profit_loss_stock1,
                        'capital': capital,
                        'timestamp': current_time,
                        'type': 'trend'
                    })

                    position_stock1 = 0
                    entry_price_stock1 = 0  
                    monitoring_trend = False
                    
                continue

            # 如果有交易信號要建立多倉
            elif trend_signal1 == 1 and position_stock1 == 0:
                # 判斷當前的內外盤比
                if t_d.iloc[i]['dominant'] != '內盤' or bid_ratio < params.get('volume_ratio'):
                    if params.get('check_signal'):
                        print(f"當前是 {t_d.iloc[i]['dominant']}成交, 或 {bid_ratio} 小於 {params.get('volume_ratio')}")
                    continue

                buy_price_stock1 = current_time_price_data['ask_price'][0]
                buy_fee_stock1 = math.ceil(buy_price_stock1 * shares_per_trade1 * commission1)

                stop_loss_stock1 = round((buy_price_stock1 * (1 - dynamic_stop_ratio1)) / tick_size1) * tick_size1  # 設定止損
                profit_stop_stock1 = round((buy_price_stock1 * (1 + dynamic_stop_ratio1)) / tick_size1) * tick_size1  # 設定止盈

                entry_price_stock1 = buy_price_stock1
                total_cost = buy_fee_stock1
                total_fee += buy_fee_stock1
                capital -= total_cost
                position_stock1 += shares_per_trade1

                print(f"{current_time}: 建立多倉, 進場價={buy_price_stock1}, 止損價={stop_loss_stock1:.2f}, 止盈價={profit_stop_stock1:.2f}, 手續費={buy_fee_stock1:.2f}, 總資金={capital:.2f}")
                signals.append((current_time, 'BUY'))

            # 如果有交易信號要建立空倉
            elif trend_signal1 == -1 and position_stock1 == 0:
                # 判斷當前的外內盤比
                if t_d.iloc[i]['dominant'] != '外盤' or ask_ratio < params.get('volume_ratio'):
                    if params.get('check_signal'):
                        print(f"當前是 {t_d.iloc[i]['dominant']}成交, 或 {ask_ratio} 小於 {params.get('volume_ratio')}")
                    continue

                sell_price_stock1 = current_time_price_data['bid_price'][0]
                sell_fee_stock1 = math.ceil(sell_price_stock1 * shares_per_trade1 * commission1)
                sell_tax_stock1 = math.ceil(sell_price_stock1 * shares_per_trade1 * tax1)

                stop_loss_stock1 = round((sell_price_stock1 * (1 + dynamic_stop_ratio1)) / tick_size1) * tick_size1  # 設定止損
                profit_stop_stock1 =  round((sell_price_stock1 * (1 - dynamic_stop_ratio1)) / tick_size1) * tick_size1    # 設定止盈

                entry_price_stock1 = sell_price_stock1
                total_cost =  (sell_fee_stock1 + sell_tax_stock1)
                total_fee += sell_fee_stock1 + sell_tax_stock1
                capital -= total_cost
                position_stock1 -= shares_per_trade1

                print(f"{current_time}: 建立空倉, 進場價={sell_price_stock1}, 止損價={stop_loss_stock1:.2f}, 止盈價={profit_stop_stock1:.2f}, 手續費={sell_fee_stock1:.2f}, 稅金={sell_tax_stock1:.2f}, 總資金={capital:.2f}")
                signals.append((current_time, 'SELL'))
        
            continue
        
        # ✅ 進入監控狀態時僅檢查突破支撐壓力
        elif monitoring_oscillation:
            stock_price1 = t_d.iloc[i]['close']
            current_time_price_data = t_d.iloc[i]
            
            max_bid_price = max(t_d.iloc[i]['bid_price'])
            min_ask_price = min(t_d.iloc[i]['ask_price'])
            spread = min_ask_price - max_bid_price
            
            # 取得委買與委賣的成交量
            total_bid_volume = t_d.iloc[i]['bid_volume']  # 當前所有委買的總量
            total_ask_volume = t_d.iloc[i]['ask_volume']  # 當前所有委賣的總量

            # ✅ 價格超出範圍 => 重新檢測
            if position_stock1 == 0 and (stock_price1 < support_range1[0] or stock_price1 > resistance_range1[1]):
                if params.get('check_signal'):
                    print(f"{current_time}: 當前價格 {stock_price1} 超出支撐與壓力範圍，重置監控狀態")

                monitoring_oscillation = False
                continue
                        
            # ✅ 委買委賣超出範圍 => 可能滑價不交易繼續等待
            if position_stock1 == 0 and spread > bid_ask_slippage:
                if params.get('check_signal'):
                    print(f"{current_time}: 當前最高委買 {max_bid_price} 與最低委賣 {min_ask_price} 間距大於 {bid_ask_slippage}")

                continue

            # ✅ 沒有倉位則進場
            if position_stock1 == 0:
                if params.get('check_signal'):
                    print(f'\n{current_time}, 檢測中當前價錢', stock_price1, "支撐範圍", support_range1, "壓力範圍", resistance_range1)

                # ✅ 當前價格在支撐與壓力範圍內 => A股票做空
                if resistance_range1[0] <= stock_price1  <= resistance_range1[1]:
                    if total_bid_volume < min_bid_ask_volume:  # 買方成交量不足
                        print(f"{current_time}: 當前總委買量 {total_bid_volume} 不足，可能出現滑價，繼續等待")
                        monitoring_oscillation = False
                        continue
                    
                    sell_price_stock1 = current_time_price_data['bid_price'][0]
                    sell_fee_stock1 = math.ceil(sell_price_stock1 * shares_per_trade1 * commission1)
                    sell_tax_stock1 = math.ceil(sell_price_stock1 * shares_per_trade1 * tax1)

                    # 計算止損價止盈價
                    stop_loss_stock1 = round((sell_price_stock1 * (1 + stop_ratio1)) / tick_size1) * tick_size1
                    profit_stop_stock1 = round((sell_price_stock1 * (1 - profit_ratio1)) / tick_size1) * tick_size1

                    # ✅ 建立空倉 (並紀錄進場價格)
                    entry_price_stock1 = sell_price_stock1
                    total_cost = (sell_fee_stock1 + sell_tax_stock1)
                    total_fee += sell_fee_stock1 + sell_tax_stock1
                    capital -= total_cost
                    position_stock1 -= shares_per_trade1
                    print(f"{current_time}: stock1建立空倉, 進場價={sell_price_stock1}, 止損價={stop_loss_stock1:.2f}, 手續費={sell_fee_stock1:.2f}, 稅金={sell_tax_stock1:.2f}, 總資金={capital:.2f}")
                    signals.append((current_time, 'SELL'))
                    continue
                
                # ✅ 當前價格在支撐與壓力範圍內 => A股票做多
                elif support_range1[0] <= stock_price1 <= support_range1[1]:
                    if total_ask_volume < min_bid_ask_volume:  # 賣方成交量不足
                        print(f"{current_time}: 當前總委賣量 {total_ask_volume} 不足，可能出現滑價，繼續等待")
                        monitoring_oscillation = False
                        continue
                    
                    buy_price_stock1 = current_time_price_data['ask_price'][0]
                    buy_fee_stock1 = math.ceil(buy_price_stock1 * shares_per_trade1 * commission1)

                    # 計算止損價止盈價
                    stop_loss_stock1 = round((buy_price_stock1 * (1 - stop_ratio1)) / tick_size1) * tick_size1
                    profit_stop_stock1 = round((buy_price_stock1 * (1 + profit_ratio1)) / tick_size1) * tick_size1

                    # ✅ 建立多倉 (並紀錄進場價格)
                    entry_price_stock1 = buy_price_stock1
                    total_cost = buy_fee_stock1
                    total_fee += buy_fee_stock1
                    capital -= total_cost
                    position_stock1 += shares_per_trade1
                    print(f"{current_time}: stock1建立多倉, 進場價={buy_price_stock1}, 止損價={stop_loss_stock1:.2f}, 手續費={(buy_fee_stock1):.2f}, 總資金={capital:.2f}")
                    signals.append((current_time, 'BUY'))
                    continue
                
                # 條件不符合繼續等待
                else:
                    if params.get('check_signal'):
                        print(f"{current_time}: 繼續監控中，等待價格突破關鍵區間...")
                    continue

            # ✅ 如果已經有持倉 => 檢查平倉條件
            elif position_stock1 > 0:  # 持有多倉
                # 判斷是否止損
                if stock_price1 <= stop_loss_stock1:
                    sell_price_stock1 = current_time_price_data['bid_price'][0]
                    sell_fee_stock1 = math.ceil(sell_price_stock1 * shares_per_trade1 * commission1)
                    sell_tax_stock1 = math.ceil(sell_price_stock1 * shares_per_trade1 * tax1)

                    profit_loss_stock1 =  ((sell_price_stock1 - entry_price_stock1) * position_stock1) - (sell_fee_stock1 + sell_tax_stock1)
                    capital += profit_loss_stock1
                    total_fee += sell_fee_stock1 + sell_tax_stock1
                    print(f"{current_time}: 多倉止損, 進場價={entry_price_stock1}, 平倉價={sell_price_stock1}, 手續費={sell_fee_stock1:.2f}, 稅金={sell_tax_stock1:.2f}, 盈虧={profit_loss_stock1:.2f}, 總資金={capital:.2f}")
                    signals.append((current_time, 'STOP LOSS LONG'))

                    # ✅ 記錄交易結果
                    trade_results.append({
                        'entry_price_stock1': entry_price_stock1,
                        'exit_price_stock1': sell_price_stock1,
                        'position_stock1': position_stock1,
                        'profit_loss_stock1': profit_loss_stock1,
                        'total_fees': total_fee,
                        'net_profit_loss': profit_loss_stock1,
                        'capital': capital,
                        'timestamp': current_time,
                        'type': 'oscillation'
                    })

                    position_stock1 = 0
                    entry_price_stock1 = 0
                    monitoring_oscillation = False
                    continue

                # 判斷是否止盈
                if stock_price1 >= profit_stop_stock1:
                    # 調整止盈價位，向下移動
                    profit_stop_stock1 = round((profit_stop_stock1 * (1 + profit_ratio1)) / tick_size1) * tick_size1
                    # 調整止損價位，跟隨價格
                    stop_loss_stock1 = round((profit_stop_stock1 * (1 - stop_ratio1)) / tick_size1) * tick_size1

                    print(f"{current_time}: stock1動態調整止盈與止損, 新止盈價={profit_stop_stock1:.2f}, 新止損價={stop_loss_stock1:.2f}")                    
    
                # 沒有止損, 判斷壓力區間
                if resistance_range1[0] <= stock_price1  <= resistance_range1[1]:
                    sell_price_stock1 = current_time_price_data['bid_price'][0]
                    sell_fee_stock1 = math.ceil(sell_price_stock1 * shares_per_trade1 * commission1)
                    sell_tax_stock1 = math.ceil(sell_price_stock1 * shares_per_trade1 * tax1)
                    
                    # ✅ 平多倉邏輯 (計算實際平倉盈虧)
                    profit_loss_stock1 = ((sell_price_stock1 - entry_price_stock1) * position_stock1) - (sell_fee_stock1 + sell_tax_stock1)
                    capital += profit_loss_stock1
                    total_fee += sell_fee_stock1 + sell_tax_stock1
                    
                    print(f"{current_time}: stock1平多倉, 進場價={entry_price_stock1}, 平倉價={sell_price_stock1}, 手續費={sell_fee_stock1:.2f}, 稅金={sell_tax_stock1:.2f}, 獲利={profit_loss_stock1:.2f}, 總資金={capital:.2f}")

                    # ✅ 記錄交易結果
                    trade_results.append({
                        'entry_price_stock1': entry_price_stock1,
                        'entry_price_stock2': entry_price_stock2 if 'entry_price_stock2' in locals() else None,
                        'exit_price_stock1': sell_price_stock1,
                        'exit_price_stock2': 0,
                        'position_stock1': position_stock1,
                        'position_stock2': position_stock2 if 'position_stock2' in locals() else 0,
                        'profit_loss_stock1': profit_loss_stock1,
                        'profit_loss_stock2': 0,
                        'total_fees': total_fee,
                        'net_profit_loss': profit_loss_stock1,
                        'capital': capital,
                        'timestamp': current_time,
                        'type': 'oscillation'
                    })
                    
                    position_stock1, entry_price_stock1 = 0, 0
                    signals.append((current_time, 'CLOSE LONG'))
            
                    # ✅ 建立空倉 (並紀錄進場價格)
                    if total_bid_volume < min_bid_ask_volume:  # 買方成交量不足
                        print(f"{current_time}: 當前總委買量 {total_bid_volume} 不足，可能出現滑價，繼續等待")
                        continue
                    
                    entry_price_stock1 = sell_price_stock1
                    total_cost = (sell_fee_stock1 + sell_tax_stock1)
                    total_fee += sell_fee_stock1 + sell_tax_stock1
                    capital -= total_cost
                    position_stock1 -= shares_per_trade1
                    # 計算止損價
                    stop_loss_stock1 = round((sell_price_stock1 * (1 + stop_ratio1)) / tick_size1) * tick_size1
                    
                    print(f"{current_time}: stock1建立空倉, 進場價={sell_price_stock1}, 止損價={stop_loss_stock1:.2f}, 手續費={sell_fee_stock1:.2f}, 稅金={sell_tax_stock1:.2f}, 總資金={capital:.2f}")
                    signals.append((current_time, 'SELL'))
                    continue

            elif position_stock1 < 0:  # 持有空倉
                # 判斷是否止損
                if stock_price1 >= stop_loss_stock1:
                    buy_price_stock1 = current_time_price_data['ask_price'][0]
                    buy_fee_stock1 = math.ceil(buy_price_stock1 * shares_per_trade1 * commission1)

                    profit_loss_stock1 =((entry_price_stock1 - buy_price_stock1) * abs(position_stock1)) - (buy_fee_stock1)
                    capital += profit_loss_stock1
                    total_fee += buy_fee_stock1
                    print(f"{current_time}: stock1空倉止損, 進場價={entry_price_stock1}, 平倉價={buy_price_stock1}, 手續費={buy_fee_stock1:.2f}, 盈虧={profit_loss_stock1:.2f}, 總資金={capital:.2f}")
                    signals.append((current_time, 'STOP LOSS SHORT'))

                    # ✅ 記錄交易結果
                    trade_results.append({
                        'entry_price_stock1': entry_price_stock1,
                        'exit_price_stock1': buy_price_stock1,
                        'position_stock1': position_stock1,
                        'profit_loss_stock1': profit_loss_stock1,
                        'total_fees': total_fee,
                        'net_profit_loss': profit_loss_stock1,
                        'capital': capital,
                        'timestamp': current_time,
                        'type': 'oscillation'
                    })

                    position_stock1 = 0
                    entry_price_stock1 = 0
                    monitoring_oscillation = False
                    continue
                
                # 判斷是否止盈
                if stock_price1 <= profit_stop_stock1:
                    # 調整止盈價位，向下移動
                    profit_stop_stock1 = round((profit_stop_stock1 * (1 - profit_ratio1)) / tick_size1) * tick_size1
                    # 調整止損價位，跟隨價格
                    stop_loss_stock1 = round((profit_stop_stock1 * (1 + stop_ratio1)) / tick_size1) * tick_size1

                    print(f"{current_time}: stock1動態調整止盈與止損, 新止盈價={profit_stop_stock1:.2f}, 新止損價={stop_loss_stock1:.2f}")
                
                # 沒有止損, 判斷支撐區間
                if support_range1[0] <= stock_price1 <= support_range1[1]:
                    buy_price_stock1 = current_time_price_data['ask_price'][0]
                    buy_fee_stock1 = math.ceil(buy_price_stock1 * shares_per_trade1 * commission1)
                    
                    # ✅ 平空倉邏輯 (計算實際平倉盈虧)
                    profit_loss_stock1 = ((entry_price_stock1 - buy_price_stock1) * abs(position_stock1)) - buy_fee_stock1
                    capital += profit_loss_stock1
                    total_fee += buy_fee_stock1
                    
                    print(f"{current_time}: stock1平空倉, 進場價={entry_price_stock1}, 平倉價={buy_price_stock1}, 手續費={buy_fee_stock1:.2f}, 獲利={profit_loss_stock1:.2f}, 總資金={capital:.2f}")

                    # ✅ 記錄交易結果
                    trade_results.append({
                        'entry_price_stock1': entry_price_stock1,
                        'entry_price_stock2': entry_price_stock2 if 'entry_price_stock2' in locals() else None,
                        'exit_price_stock1': sell_price_stock1,
                        'exit_price_stock2': 0,
                        'position_stock1': position_stock1,
                        'position_stock2': position_stock2 if 'position_stock2' in locals() else 0,
                        'profit_loss_stock1': (entry_price_stock1 - buy_price_stock1) * abs(position_stock1),
                        'profit_loss_stock2': 0,
                        'total_fees': total_fee,
                        'net_profit_loss': profit_loss_stock1,
                        'capital': capital,
                        'timestamp': current_time,
                        'type': 'oscillation'
                    })
                    
                    position_stock1, entry_price_stock1 = 0, 0
                    signals.append((current_time, 'CLOSE SHORT'))
            
                    # ✅ 建立多倉 (並紀錄進場價格)
                    if total_ask_volume < min_bid_ask_volume:  # 賣方成交量不足
                        print(f"{current_time}: 當前總委賣量 {total_ask_volume} 不足，可能出現滑價，繼續等待")
                        continue
                    
                    entry_price_stock1 = buy_price_stock1
                    total_cost = buy_fee_stock1
                    total_fee += buy_fee_stock1
                    capital -= total_cost
                    position_stock1 += shares_per_trade1
                    
                    # 計算止損價
                    stop_loss_stock1 = round((buy_price_stock1 * (1 - stop_ratio1)) / tick_size1) * tick_size1
                    
                    print(f"{current_time}: stock1建立多倉, 進場價={buy_price_stock1}, 止損價={stop_loss_stock1:.2f}, 手續費={buy_price_stock1:.2f}, 總資金={capital:.2f}")
                    signals.append((current_time, 'BUY'))
                    continue
            
            # 震盪檢測跳過後續檢驗
            continue
        
        # 僅在分鐘變化時檢查一次斜率
        if current_minute != last_checked_minute:
            last_checked_minute = current_minute
            
            # 取得最近 window 的分鐘級斜率數據(df1: 股票資料, df2: 指數資料)
            try:
                recent_price1 = df1.loc[base_time:current_minute].iloc[-long_window:]
                recent_price2 = df2.loc[base_time:current_minute].iloc[-long_window:]
            except IndexError:
                continue  # 如果資料不足，跳過這一輪檢查
            
            print('\n---------------', current_time, '---------------')

            indicator1 = compute_technical_indicators(recent_price1, long_window=long_window, short_window=short_window, plot_vpfr=False)
            indicator2 = compute_technical_indicators(recent_price2, long_window=long_window, short_window=short_window, plot_vpfr=False)

            if indicator1 is False or indicator2 is False: # 如果資料不足，跳過這一輪檢查
                continue

            # 繪製matplotlib圖表需要的資料(僅記錄指數)
            indicators_data['time'].append(current_time)
            indicators_data['rsv'].append(indicator2['rsv'])
            
            combined_indicators = {
                'rsv2': indicator2['rsv'],
                'vpfr1': indicator1['vpfr'],
                'vpfr2': indicator2['vpfr'],
                'adx1': indicator1['adx'],
                '+di1': indicator1['+di'],
                '-di1': indicator1['-di']
            }

            is_stationary_long = bool(check_acf_pacf_stationary(recent_price2['close'], window=long_window, max_lag=long_window, debug=False))
            is_stationary_short = bool(check_acf_pacf_stationary(recent_price2['close'], window=short_window, max_lag=short_window, debug=False))
            
            # 震盪盤
            if is_stationary_long:
                if params.get('check_signal'):
                    if not is_stationary_short:
                        print("震盪盤，但目前測試壓力位/支撐位")
                    else:
                        print("穩定震盪")
                
                trade_signal = trade_oscillation(recent_price1, recent_price2, combined_indicators, params=params, debug=False)
                
                # 需要監測價錢是否有接近支撐與壓力
                if trade_signal[0] is True:
                    monitoring_trend = False
                    monitoring_oscillation = True
                    support_range1 = trade_signal[1]
                    resistance_range1 = trade_signal[2]
                    signals.append((current_time, "MONITOR"))
                    continue

            # 趨勢盤    
            else:
                if params.get('check_signal'):
                    if is_stationary_short:
                        print("趨勢盤，但短期波動減弱")
                    else:
                        print("強烈趨勢")
                    
                trade_signal = trade_trend(recent_price2, combined_indicators, params=params, debug=False)
                
                # 趨勢盤進場判斷
                if trade_signal[0] is True:
                    monitoring_trend = True
                    trend_signal1 = trade_signal[1]
                    signals.append((current_time, "MONITOR"))
                    continue

    # 績效檢測
    if trade_results:
        performance = analyze(trade_result=trade_results, is_plot=params.get('plot3'))
    
        # 打印分析结果
        print(f"\n净盈亏：{performance['net_profit_loss']:.2f}")
        print(f"交易後總資產：{performance['capital']:.2f}")
        print(f"獲利次數：{performance['win_count']:.2f}")
        print(f"虧損次數：{performance['loss_count']:.2f}")
        print(f"胜率：{performance['win_rate']*100:.2f}%")
        print(f"最大回撤：{performance['max_drawdown']:.2f}")
        print(f"年化回报：{performance['annualized_return']:.2f}")
        print(f"夏普比率：{performance['sharpe_ratio']:.2f}")
        print(f"风暴比：{performance['sterling_ratio']:.2f}")
        print(f"总交易次数：{performance['total_trades']}")
        print(f"总手续费和税费：{total_fee:.2f}")
        print(f"最终总资产：{capital:.2f}\n")
        
        # 打印每个交易类型的分析结果
        print("按盘面类型分析结果：")
        for trade_type, stats in performance['type_performance'].items():
            print(f"\n类型：{trade_type}")
            print(f" - 总交易次数：{stats['total_trades']}")
            print(f" - 净盈亏：{stats['total_profit_loss']:.2f}")
            print(f" - 获利次数：{stats['win_count']}")
            print(f" - 亏损次数：{stats['loss_count']}")
            print(f" - 胜率：{stats['win_rate']*100:.2f}%")
            print(f" - 获利时间点：{stats['profit_timestamps']}")
            print(f" - 亏损时间点：{stats['loss_timestamps']}")
            
    else:
        print("\n沒有交易結果可分析。")

    # ✅ 更新繪圖部分
    if params.get('plot1') is True:
        # 提取交易信號時間點
        buy_signals = [signal[0] for signal in signals if signal[1] == 'BUY']
        sell_signals = [signal[0] for signal in signals if signal[1] == 'SELL']
        no_trade_signals = [signal[0] for signal in signals if signal[1] == 'NO_TRADE']
        monitor_signals = [signal[0] for signal in signals if signal[1] == 'MONITOR']
        close_long_signals = [signal[0] for signal in signals if signal[1] == 'CLOSE LONG']
        close_short_signals = [signal[0] for signal in signals if signal[1] == 'CLOSE SHORT']

        # ✅ 繪製價格圖
        plt.figure(figsize=(12, 6))
        plt.plot(t_d.index, t_d['close'], label='Price', color='blue', alpha=0.6)

        # ✅ 標註各種信號點
        plt.scatter(buy_signals, t_d.loc[buy_signals, 'close'], marker='^', color='red', label='BUY Signal', alpha=1)
        plt.scatter(sell_signals, t_d.loc[sell_signals, 'close'], marker='v', color='green', label='SELL Signal', alpha=1)
        plt.scatter(no_trade_signals, t_d.loc[no_trade_signals, 'close'], marker='o', color='orange', label='NO_TRADE Signal', alpha=1)
        plt.scatter(monitor_signals, t_d.loc[monitor_signals, 'close'], marker='s', color='purple', label='Monitor Signal', alpha=0.8)
        plt.scatter(close_long_signals, t_d.loc[close_long_signals, 'close'], marker='^', color='blue', label='CLOSE LONG Signal', alpha=1)
        plt.scatter(close_short_signals, t_d.loc[close_short_signals, 'close'], marker='v', color='brown', label='CLOSE SHORT Signal', alpha=1)

        # ✅ 設定標題與標籤
        plt.title('Price Chart with Buy/Sell/Close Signals and Monitoring Signals')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()

        # ✅ 顯示圖表
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    if params.get('plot2') is True:
        # 將指標數據轉換為 DataFrame
        indicators_df = pd.DataFrame(indicators_data)

        plt.figure(figsize=(12, 12))

        # 價格圖
        plt.subplot(4, 2, 1)
        plt.plot(t_d.index, t_d['close'], label='Price', color='blue', alpha=0.6)
        plt.title('Price Chart')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()

        # RSV圖
        plt.subplot(4, 2, 3)
        plt.plot(indicators_df['time'], indicators_df['rsv'], label='RSV', color='orange', alpha=0.7)
        plt.title('RSV')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()

        # 增加間距並旋轉 x 軸標籤
        plt.subplots_adjust(hspace=0.6, wspace=0.4)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    return signals

# ✅ 執行策略
params = {
    'data': mer_ori_data,
    'df1': df1, # 股票價格
    'df2': df2, # 指數
    'price_diff': 0.5, # 當震盪價差小於多少, 則放棄此次震盪交易
    'vpfr_trend': 0.4,
    'vpfr_oscillation': 0.6,
    'rsv_low': 20,
    'rsv_high': 80,
    'adx': 60,
    'oscillation_buffer': 0.05, # 震盪緩衝(支撐壓力的價差區間)
    'volume_ratio': 0.6, # 趨勢盤內外盤比
    'volume_threshold': 600, # 指數成交量門檻(口)
    'volume_slippage': 100, # 時間段內成交量小於指定數值不交易(震盪盤滑價)
    'bid_ask_slippage': 0.15, # 委買賣滑價價差(tick)
    'min_bid_ask_volume': 10, # 委買賣成交量最低門檻(震盪盤使用)
    'long_window': 10,
    'short_window': 5,
    # 'start_date': None,
    # 'end_date': None,
    'start_date': '2024-06-05',
    'end_date': '2024-06-05',
    'tick_size1': 0.05,
    'tick_size2': 0,
    'shares_per_trade1': 1000,
    'capital': 100000,
    'stop_ratio1': 0.002, # 震盪盤止損
    'profit_ratio1': 0.001, # 震盪盤止損
    'dynamic_stop_ratio1': 0.003, # 趨勢盤止損
    'dynamic_profit_ratio1': 0.001, # 趨勢盤止盈
    'commission1': 0.001425 * 0.28,
    'tax1': 0.001,
    'plot1': False,
    'plot2': False,
    'plot3': False,
    'check_signal': True
}

signals = bilateral_strategy(**params)
