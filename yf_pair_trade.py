import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from arch.unitroot import KPSS
from hurst import compute_Hc

# 定義股票代碼和日期範圍
a, b, c = '00685L.TW', '00686R.TW', '^TWII'
tickers = [a, b, c]
start_date = '2025-01-15'
end_date = '2025-03-14'

# # 獲取資料
data = []
# data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)['Close']

# # 將資料放入 pandas DataFrame
# df = pd.DataFrame(data)

# # 計算 ^TWII 的對數收益率
# df['return'] = np.log(df[c] / df[c].shift(1))
# df = df.dropna(subset=['return'])

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

df1 = prepare_data('shioaji/2025_0103', 'TSE001')
df2 = prepare_data('shioaji/2025_0103', 'TXFR1')
df3 = prepare_data('shioaji/2025_0103', 'SRFR1')
df4 = prepare_data('shioaji/2025_0103', '00675L')
df1.set_index('ts', inplace=True)
df2.set_index('ts', inplace=True)
df3.set_index('ts', inplace=True)
df4.set_index('ts', inplace=True)
df1 = df1.rename(columns=lambda x: f'{x}_TWSE' if x != 'ts' else x)
df2 = df2.rename(columns=lambda x: f'{x}_TXFR1' if x != 'ts' else x)
df3 = df3.rename(columns=lambda x: f'{x}_SRFR1' if x != 'ts' else x)
df4 = df4.rename(columns=lambda x: f'{x}_00675L' if x != 'ts' else x)

def transfer(start_date, end_date):
    # 創建完整的時間序列
    time_range = pd.date_range(start=start_date, end=end_date, freq='B')
    full_time_range = pd.DatetimeIndex([])

    for single_date in time_range:
        daily_range = pd.date_range(
            start=single_date.replace(hour=9, minute=0, second=0),
            end=single_date.replace(hour=13, minute=30, second=0),
            freq='min'
        )
        full_time_range = full_time_range.union(daily_range)

    # 初始化總表
    df_total = pd.DataFrame(index=full_time_range)

    # 初始化總表
    df_total = pd.DataFrame(index=full_time_range)

    # 直接使用外部的 df1~df4 合併到總表
    df_total = df_total.join(df1, how='left')
    df_total = df_total.join(df2, how='left')
    df_total = df_total.join(df3, how='left')
    df_total = df_total.join(df4, how='left')
    # 填充缺失值
    df_total = df_total.bfill().ffill()

    # 設置索引名稱
    df_total.index.name = 'ts'

    return df_total

df = transfer(start_date, end_date)

# df = df.resample('30min').agg({
#     # TWSE相關欄位
#     'open_TWSE': 'first',   # 開盤價取第一個值
#     'high_TWSE': 'max',     # 最高價取最大值
#     'low_TWSE': 'min',      # 最低價取最小值
#     'close_TWSE': 'last',   # 收盤價取最後一個值
#     'volume_TWSE': 'sum',   # 成交量取總和
#     'amount_TWSE': 'sum',   # 成交額取總和
#     # TXFR1相關欄位
#     'open_TXFR1': 'first',
#     'high_TXFR1': 'max',
#     'low_TXFR1': 'min',
#     'close_TXFR1': 'last',
#     'volume_TXFR1': 'sum',
#     'amount_TXFR1': 'sum',
#     # SRFR1相關欄位
#     'open_SRFR1': 'first',
#     'high_SRFR1': 'max',
#     'low_SRFR1': 'min',
#     'close_SRFR1': 'last',
#     'volume_SRFR1': 'sum',
#     'amount_SRFR1': 'sum',
#     # 00675L相關欄位
#     'open_00675L': 'first',
#     'high_00675L': 'max',
#     'low_00675L': 'min',
#     'close_00675L': 'last',
#     'volume_00675L': 'sum',
#     'amount_00675L': 'sum'
# })

# 計算價差
df['return'] = (df['close_TWSE'] - df['close_TXFR1']).pct_change(fill_method=None)
df = df.dropna(subset=['return'])

def find_optimal_timeframe(spread_series, min_window=10, max_window=100, base_T=15, k=0.7):
    """
    根據輸入的時間範圍，找出最適合的交易週期。
    
    :param spread_series: 價差數據（pd.Series）
    :param min_window: 最小時間窗口（分鐘）
    :param max_window: 最大時間窗口（分鐘）
    :param base_T: 基準週期（分鐘）
    :param k: 波動率調節係數
    :return: 最佳時間窗口（分鐘）、動態週期（分鐘）和結果列表
    """
    # 初始化結果存儲
    results = []

    # 遍歷所有時間窗口
    for window in range(min_window, max_window + 1):
        # 計算滾動波動率（標準差）
        spread_vol = spread_series.rolling(window=window).std()
        
        print(spread_vol.head(15))
        print('-----')
        
        # 計算波動率分位數（過去21天的數據）
        lookback_period = 21 * 1440  # 21天（1天=1440分鐘）
        spread_vol_rank = spread_vol.rolling(window=lookback_period).rank(pct=True)
        
        # 去除NaN值
        spread_vol_rank_cleaned = spread_vol_rank.dropna()
        
        # 如果沒有有效數據，跳過該窗口
        if len(spread_vol_rank_cleaned) == 0:
            continue
        
        # 計算平均波動率分位數
        avg_vol_rank = spread_vol_rank_cleaned.mean()
        
        # 動態調整交易週期（加入base_T和k的應用）
        dynamic_T = base_T * (1 / (avg_vol_rank + 0.01)) ** k  # 避免分位數為0
        
        # 存儲結果
        results.append({
            'window': window,
            'avg_vol_rank': avg_vol_rank,
            'dynamic_T': dynamic_T
        })
    
    # 如果沒有有效結果，返回默認值
    if len(results) == 0:
        print("警告：沒有找到有效的時間窗口，請檢查輸入數據。")
        return None, None, []
    
    # 將結果轉為DataFrame（僅用於方便查看）
    results_df = pd.DataFrame(results)
    
    # 根據波動率分位數選擇最佳時間窗口
    avg_vol_rank_mean = results_df['avg_vol_rank'].mean()
    
    # 檢查是否有有效的 avg_vol_rank
    if results_df['avg_vol_rank'].isna().all():
        print("警告：所有波動率分位數均為 NaN，請檢查輸入數據。")
        return None, None, results
    
    # 選擇最佳時間窗口
    if avg_vol_rank_mean < 0.3:  # 低波動，選擇更短週期
        optimal_window = results_df.loc[results_df['window'].idxmin(), 'window']
    else:  # 正常或高波動，選擇波動率分位數最低的窗口
        optimal_window = results_df.loc[results_df['avg_vol_rank'].idxmin(), 'window']
    
    # 獲取對應的動態週期
    optimal_dynamic_T = results_df.loc[results_df['window'] == optimal_window, 'dynamic_T'].values[0]
    
    # 在函數內部打印結果
    print(f"最佳時間窗口：{optimal_window} 分鐘")
    print(f"對應的動態週期：{optimal_dynamic_T:.2f} 分鐘")
    
    return optimal_window, optimal_dynamic_T, results

# 執行函數
optimal_window, optimal_dynamic_T, results = find_optimal_timeframe(df['return'], min_window=10, max_window=100)
# 打印結果列表
for result in results:
    print(result)

def is_stationary(series):
    # ADF 檢驗
    adf_result = adfuller(series.dropna())
    adf_p_value = adf_result[1]
    adf_stationary = adf_p_value < 0.05  # p < 0.05 代表拒絕 H0，序列是平穩的
    adf_conclusion = "時間序列是平穩的。" if adf_stationary else "時間序列不是平穩的。"
    
    # KPSS 檢驗
    kpss_result = KPSS(series.dropna())
    kpss_p_value = kpss_result.pvalue
    kpss_stationary = kpss_p_value > 0.05  # p > 0.05 代表無法拒絕 H0，序列是平穩的
    kpss_conclusion = "時間序列是平穩的。" if kpss_stationary else "時間序列不是平穩的。"
    
    # Hurst 指數
    hurst_value = compute_Hc(series.dropna(), kind='change', simplified=False, min_window=200)[0]
    hurst_stationary = hurst_value < 0.5  # H < 0.5 代表是均值回歸型（平穩）
    hurst_conclusion = "時間序列是平穩的（均值回歸型）。" if hurst_stationary else "時間序列不是平穩的（持續趨勢型）。"
    
    # 顯示檢驗結果
    adf_result_str = f"ADF 檢驗 p 值: {adf_p_value:.4f}. 結論: {adf_conclusion}"
    kpss_result_str = f"KPSS 檢驗 p 值: {kpss_p_value:.4f}. 結論: {kpss_conclusion}"
    hurst_result_str = f"Hurst 指數: {hurst_value:.4f}. 結論: {hurst_conclusion}"
    
    # 統一結果
    if adf_stationary and kpss_stationary and hurst_stationary:
        final_conclusion = "✅ 是平穩時間序列，可以使用均值回歸策略。"
    elif adf_stationary and not kpss_stationary:
        final_conclusion = "⚠️ 可能是趨勢型時間序列，需要差分處理。"
    else:
        final_conclusion = "❌ 不是平穩時間序列，可能是隨機漫步，不適合均值回歸策略。"
    
    # 回傳結果
    print(f"{adf_result_str}\n{kpss_result_str}\n{hurst_result_str}\n\n綜合判斷: {final_conclusion}")

# 測試是否平穩
# is_stationary(df['return'])

def plot_acf_pacf(series):
    # 畫出 ACF 和 PACF 圖
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    plot_acf(series, lags=20, ax=axes[0])
    plot_pacf(series, lags=20, ax=axes[1])

    plt.show()

# plot_acf_pacf(df['return'])

def half_life(series):
    # 計算自相關 (Auto-correlation)
    autocorr = series.autocorr(lag=35)

    # 這裡的 beta 就是滯後1期的自相關係數
    beta = autocorr

    # 檢查 beta 是否有效
    if beta <= 0 or beta >= 1:
        print("Beta 值無效，無法計算半衰期。")
    else:
        # 計算半衰期：半衰期公式 T1/2 = ln(2) / -ln(beta)
        half_life = np.log(2) / -np.log(beta)
        print(f"半衰期: {half_life:.2f} 天")
    
    print(f'自相關係數: {autocorr:.4f}')

# half_life(df['return'])

def plot_two_trend():
    # 绘图
    plt.figure(figsize=(10, 6))

    # 绘制每只股票的价格
    for ticker in tickers:
        plt.plot(data[ticker], label=ticker)

    plt.title(f'Stock Prices of {tickers} ({start_date} - {end_date})')
    plt.xlabel('Date')
    plt.ylabel('Adjusted Close Price')
    plt.legend()
    plt.grid()
    plt.show()
    
# plot_two_trend()

def plot_mean(timeseries, diff_col, label):
    plt.figure(figsize=(12, 6))

    # 计算 z-score： (值 - 均值) / 标准差
    diff_zscore = (diff_col - np.mean(diff_col)) / np.std(diff_col)

    # 绘制 z-score 标准化后的时间序列数据
    plt.plot(timeseries.index, diff_zscore, label=label, alpha=0.7, color=np.random.rand(3,))

    # 绘制均值为0的水平线
    plt.axhline(y=0, color='black', label='Mean = 0')

    # z-score 标准化后的数据标准差为1
    std_val = 1

    # 添加标准差范围的水平线
    plt.axhline(0 - 1 * std_val, color='g', linestyle='--', label=f'Mean ± 1 * STD: {round(std_val, 2)}')
    plt.axhline(0 + 1 * std_val, color='g', linestyle='--')

    plt.axhline(0 - 2 * std_val, color='orange', label=f'Mean ± 2 * STD: {round(2 * std_val, 2)}')
    plt.axhline(0 + 2 * std_val, color='orange')

    plt.axhline(0 - 3 * std_val, color='red', label=f'Mean ± 3 * STD: {round(3 * std_val, 2)}')
    plt.axhline(0 + 3 * std_val, color='red')

    plt.legend()
    plt.title('Z-score Normalization of Price Difference (a - b)')
    plt.xlabel('Date')
    plt.ylabel('Z-score')
    plt.show()
    
    # 统计 z-score 分布
    bins = [-np.inf, -3, -2, -1, 1, 2, 3, np.inf]
    labels = ['大於-3', '-3到-2', '-2到-1', '-1到1', '1到2', '2到3', '大於3']
    
    # 计算各个区间的频数
    counts = pd.cut(diff_zscore, bins=bins, labels=labels).value_counts()

    # 打印统计结果
    print("Z-score 分布统计:")
    for label in labels:
        print(f"{label}: {counts.get(label, 0)} 个数据点")

# 使用函數繪圖，傳入價差的 z-score
# plot_mean(df, df['return'], f'return of data')

def pair_trade(data, col, a, b, threshold=1, **params):
    spread = data[col]

    # 计算Z-score
    mean_spread = spread.mean()
    std_spread = spread.std()
    z_score = (spread - mean_spread) / std_spread

    # 初始化交易信号
    position = 0  # 0: 无持仓, 1: 空A多B, -1: 多A空B
    entry_time = None
    entry_type = None
    entry_price_A = None
    entry_price_B = None
    trades = []  # 记录交易

    # 遍历Z-score数据
    for i in range(1, len(z_score)):
        current_z = z_score.iloc[i]
        price_A = round(data[a].iloc[i], 2)  # 目前 A 股票的价格
        price_B = round(data[b].iloc[i], 2)  # 目前 B 股票的价格
        
        if position == 0:  # 目前无持仓，等待入场
            if current_z < -threshold:  # Z-score < -1，多A空B
                position = 1
                entry_time = z_score.index[i]
                entry_type = "多A空B"
                entry_price_A = price_A
                entry_price_B = price_B
            elif current_z > threshold:  # Z-score > 1，空A多B
                position = -1
                entry_time = z_score.index[i]
                entry_type = "空A多B"
                entry_price_A = price_A
                entry_price_B = price_B

        elif position == 1 and current_z >= 0:  # 持有多A空B仓位，Z-score 回归 0 平仓
            exit_time = z_score.index[i]

            # 计算每只股票的价格变动
            price_diff_A = price_A - entry_price_A
            price_diff_B = price_B - entry_price_B

            # 计算绝对值的总盈亏
            total_profit_loss = abs(price_diff_A) + abs(price_diff_B)

            # 亏损判断
            loss_flag = (price_A < entry_price_A) or (price_B > entry_price_B)

            # 如果是亏损，则设为负数
            total_profit_loss = -total_profit_loss if loss_flag else total_profit_loss

            trades.append({
                "entry_time": entry_time,
                "exit_time": exit_time,
                "trade_type": entry_type,
                "entry_price_A": entry_price_A,
                "exit_price_A": price_A,
                "price_diff_A": price_diff_A,
                "entry_price_B": entry_price_B,
                "exit_price_B": price_B,
                "price_diff_B": price_diff_B,
                "total_profit_loss": total_profit_loss,
                "loss_flag": "亏损" if loss_flag else "盈利"
            })

            position = 0  # 平仓
            entry_time = None
            entry_type = None
            entry_price_A = None
            entry_price_B = None

        elif position == -1 and current_z <= 0:  # 持有空A多B仓位，Z-score 回归 0 平仓
            exit_time = z_score.index[i]

            # 计算每只股票的价格变动
            price_diff_A = price_A - entry_price_A
            price_diff_B = price_B - entry_price_B

            # 计算绝对值的总盈亏
            total_profit_loss = abs(price_diff_A) + abs(price_diff_B)

            # 亏损判断
            loss_flag = (price_A > entry_price_A) or (price_B < entry_price_B)

            # 如果是亏损，则设为负数
            total_profit_loss = -total_profit_loss if loss_flag else total_profit_loss

            trades.append({
                "entry_time": entry_time,
                "exit_time": exit_time,
                "trade_type": entry_type,
                "entry_price_A": entry_price_A,
                "exit_price_A": price_A,
                "price_diff_A": price_diff_A,
                "entry_price_B": entry_price_B,
                "exit_price_B": price_B,
                "price_diff_B": price_diff_B,
                "total_profit_loss": total_profit_loss,
                "loss_flag": "亏损" if loss_flag else "盈利"
            })

            position = 0  # 平仓
            entry_time = None
            entry_type = None
            entry_price_A = None
            entry_price_B = None

    data = pd.DataFrame(trades)  # 返回交易记录
    print(data)
    
    # 统计盈利和亏损的次数
    win_loss_counts = data["loss_flag"].value_counts()

    # 获取盈利和亏损的次数
    win_count = win_loss_counts.get("盈利", 0)
    loss_count = win_loss_counts.get("亏损", 0)

    # 打印统计结果
    print(f"盈利次数: {win_count}")
    print(f"亏损次数: {loss_count}")
    
    # 计算 total_profit_loss 列的总和
    total_profit_loss_sum = data["total_profit_loss"].sum()

    # 打印总盈亏
    print(f"总盈亏(未扣手續費): {total_profit_loss_sum:.2f}")

params = {
    'initial_capital': 100000,
    'commission': 0.001
}

# pair_trade(df, 'return', 'close_SRFR1', 'close_00675L', threshold=1, **params)