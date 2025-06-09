import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint
from sklearn.preprocessing import MinMaxScaler
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import math

# -------------------------------------- 資料準備與前處理 --------------------------------------

def prepare_data(type_of_data, data_name):
    result = type_of_data.split('/')[0]
    tmp = pd.read_csv(f'./index_data/{type_of_data}/{data_name}.csv')
    if result == 'shioaji':
        tmp['ts'] = pd.to_datetime(tmp['ts'])
    else:
        tmp['ts'] = pd.to_datetime(tmp['datetime'])

    return tmp

df1 = prepare_data('shioaji/2024_1201', '2353') # 小
df2 = prepare_data('shioaji/2024_1201', '6841') # 大

# 資料時間調整
def pre_process(df1, df2, date_start, date_end):
    # 假设你的df已经加载到dataframe
    df1['ts'] = pd.to_datetime(df1['ts'])  # 将ts列转换为datetime类型
    df2['ts'] = pd.to_datetime(df2['ts'])  # 将ts列转换为datetime类型

    # 按秒分组
    df1 = df1.groupby(df1['ts'].dt.floor('s')).agg({
        'close': 'last', 
        'volume': 'sum', 
        'bid_price': 'last', 
        'bid_volume': 'sum', 
        'ask_price': 'last', 
        'ask_volume': 'sum', 
        'tick_type': 'last'
    }).reset_index()

    # 按秒分组
    df2 = df2.groupby(df1['ts'].dt.floor('s')).agg({
        'close': 'last', 
        'volume': 'sum', 
        'bid_price': 'last', 
        'bid_volume': 'sum', 
        'ask_price': 'last', 
        'ask_volume': 'sum', 
        'tick_type': 'last'
    }).reset_index()

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
    merged_df2 = pd.merge(valid_time, df2, on='ts', how='left')

    # 如果需要，你可以将merged_df1和merged_df2合并为一个表格
    mer_ori_data = pd.merge(merged_df1, merged_df2, on='ts', how='left', suffixes=('_stock1', '_stock2'))

    # 向后补齐，再向前补齐
    mer_ori_data = mer_ori_data.ffill()  # 向后补齐
    mer_ori_data = mer_ori_data.bfill()  # 向后补齐

    # 设置'ts'为index
    mer_ori_data.set_index('ts', inplace=True)

    # 删除含有NaN的行
    mer_ori_data = mer_ori_data.dropna()
    
    return mer_ori_data

mer_ori_data = pre_process(df1, df2, '2024-12-02 9:00:00', '2025-01-02 13:30:00')

# 斜率轉換
def pre_slope(df):
    # 將CSV讀入DataFrame
    df['ts'] = pd.to_datetime(df['ts'])  # 確保時間欄位為datetime格式
    df.set_index('ts', inplace=True)
    
    # 重新調整為每分鐘的數據
    resampled_data = df.resample('1min').last().dropna()
    
    # 將索引轉換為列以便計算
    resampled_data = resampled_data.reset_index()
    resampled_data['timestamp'] = resampled_data['ts'].astype('int64') // 1_000_000_000  # 秒級
    
    # 計算每分鐘的斜率
    resampled_data['slope'] = resampled_data['close'].diff() / resampled_data['timestamp'].diff()
    
    # 將NaN填充為0
    resampled_data.loc[:, 'slope'] = resampled_data['slope'].fillna(0)
    
    # 移除第一個 NaN 值
    resampled_data.dropna(inplace=True)
    resampled_data.set_index('ts', inplace=True)

    return resampled_data

df1_slope = pre_slope(df1)
df2_slope = pre_slope(df2)

# -------------------------------------- 繪圖檢驗 --------------------------------------
threshold1, threshold2 = 2, -2

def plot_trend(col1, col2, col3, date=None):
    # 如果有指定日期，篩選數據
    if date:
        date = pd.to_datetime(date)
        filtered_data = mer_ori_data[mer_ori_data.index.date == date.date()]
    else:
        filtered_data = mer_ori_data  # 如果沒有指定日期，使用完整數據

    # 繪製走勢圖
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)  # Create two subplots

    # Subplot for the trend lines of col1 and col2
    axs[0].plot(filtered_data.index, filtered_data[col1], label=f'{col1}', color='blue', alpha=0.7)
    axs[0].plot(filtered_data.index, filtered_data[col2], label=f'{col2}', color='red', alpha=0.7)
    axs[0].set_title('Stock Price Trend')
    axs[0].set_ylabel('Price')
    axs[0].legend()

    # Subplot for the diff (col3)
    axs[1].plot(filtered_data.index, filtered_data[col3], label=f'{col3}', color='green', linestyle='--')
    axs[1].set_title('Price Difference')
    axs[1].set_ylabel('Price Diff')
    axs[1].set_xlabel('Date')
    axs[1].legend()

    # 旋轉日期刻度，避免重疊
    plt.xticks(rotation=45)

    # 顯示圖表
    plt.tight_layout()  # 自動調整佈局以避免標籤被截斷
    plt.show()

# plot_trend(col1='close_stock1', col2='close_stock2', col3='diff', date='2024-12-02')

def plot_mean(data, start_date=None, end_date=None, col=None):
    # 確保日期是字符串並轉換為 datetime
    if start_date:
        start_date = pd.to_datetime(start_date)
    if end_date:
        end_date = pd.to_datetime(end_date)
    
    # 如果未指定 end_date，預設為 start_date 的下一天
    if start_date and not end_date:
        end_date = start_date + pd.Timedelta(days=1)
    
    # 根據日期範圍篩選數據
    filtered_data = data.copy()
    if start_date and end_date:
        filtered_data = filtered_data[(filtered_data.index >= start_date) & (filtered_data.index <= end_date)]
    elif start_date:
        filtered_data = filtered_data[filtered_data.index >= start_date]
    elif end_date:
        filtered_data = filtered_data[filtered_data.index <= end_date]

    # 繪製 Z-score
    plt.figure(figsize=(12, 6))
    plt.plot(filtered_data.index, filtered_data[col], label=f'{col}', color='blue')

    # 添加水平線表示阈值
    plt.axhline(y=0, color='black')
    plt.axhline(y=threshold2, color='red', linestyle='--', label='Negative Threshold')
    plt.axhline(y=threshold1, color='green', linestyle='--', label='Positive Threshold')

    # 設置圖表屬性
    plt.title(f'{col} of Mean')
    plt.xlabel('Time')
    plt.ylabel(f'{col}')
    plt.legend()
    plt.grid()
    plt.show()

# plot_mean(d1, start_date='2024-12-02', col='slope')

def plot_dist(col=None):
    # 設定畫布和大小
    plt.figure(figsize=(10, 6))

    # 使用 seaborn 繪製直方圖
    sns.histplot(mer_ori_data[col], kde=True, stat="density", bins=50, color='blue', label=f'{col}')

    # 計算 `diff` 的平均值和標準差
    mean_diff = mer_ori_data[col].mean()
    std_diff = mer_ori_data[col].std()

    # 創建正態分佈的數據點
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = (1 / (std_diff * np.sqrt(2 * np.pi))) * np.exp(-(x - mean_diff)**2 / (2 * std_diff**2))
    
    # D'Agostino and Pearson's Test
    stat, p_value = stats.normaltest(mer_ori_data[col].dropna())

    print(f"D'Agostino and Pearson's Test p-value: {p_value}")
    if p_value > 0.05:
        print("資料是常態分佈")
    else:
        print("資料不是常態分佈")

    # 在圖上繪製正態分佈曲線
    plt.plot(x, p, 'k', linewidth=2, label='Normal Distribution')

    # 標題與標籤
    plt.title('Distribution of Diff with Normal Curve')
    plt.xlabel('Diff')
    plt.ylabel('Density')

    # 顯示圖例
    plt.legend()

    # 顯示圖形
    plt.show()

# plot_dist(col='zscore')

def test_co(timeseries1, timeseries2):
    # 确保两个时间序列的长度相同
    timeseries1, timeseries2 = timeseries1.align(timeseries2, join='inner')
    
    # 进行协整检验
    coint_test = coint(timeseries1, timeseries2)

    # 获取结果
    t_stat = coint_test[0]
    p_value = coint_test[1]
    critical_values = coint_test[2]

    print(f"协整检验统计量 (t-statistic): {t_stat}")
    print(f"p 值: {p_value}")
    print(f"临界值 (Critical values): {critical_values}\n")
    
    # 设置条件判断并给出总结
    if p_value < 0.05:
        print("基于 p 值：存在协整关系，意味着它们具有长期均衡关系。")
    elif p_value < 0.10:
        print("基于 p 值：存在协整关系，但在 10% 显著性水平下。")
    else:
        print("基于 p 值：不存在协整关系。")

    # 也可以基于 t_stat 和 critical_values 进行判断
    if t_stat < critical_values[0]:  # 比如使用 1% 的置信水平
        print("基于1% 置信水平：协整关系非常显著。")
    elif t_stat < critical_values[1]:  # 5% 的置信水平
        print("基于5% 置信水平：协整关系显著。")
    elif t_stat < critical_values[2]:  # 10% 的置信水平
        print("基于10% 置信水平：协整关系较为显著。")
    else:
        print("基于t檢驗: 不存在显著的协整关系。")
        
    # 2. 计算相关性
    correlation = timeseries1.corr(timeseries2)

    # 输出相关性结果
    print(f"\n相关性系数: {correlation:.2f}")

    # 相关性总结
    if correlation > 0.8:
        print("存在很强的正相关性。")
    elif correlation > 0.5:
        print("存在较强的正相关性。")
    elif correlation > 0:
        print("存在弱正相关性。")
    elif correlation == 0:
        print("没有相关性。")
    elif correlation > -0.5:
        print("存在弱负相关性。")
    elif correlation > -0.8:
        print("存在较强的负相关性。")
    else:
        print("存在很强的负相关性。")

# test_co(mer_ori_data['close_stock1'].loc['2024-12-02'], mer_ori_data['close_stock2'].loc['2024-12-02'])

# -------------------------------------- 訊號生成 --------------------------------------

# 正規化z-socre
def regular_signal(data, window1, window2, col1, col2):
    # 计算两个标的物之间的差异
    spread = data[col1] / data[col2]
    
    # 进行 Min-Max 标准化，将数据缩放到 [0, 10] 范围
    scaler = MinMaxScaler(feature_range=(0, 10))
    spread_normalized = scaler.fit_transform(spread.values.reshape(-1, 1))
    
    # 将标准化后的数据转换回 DataFrame
    spread_normalized_df = pd.DataFrame(spread_normalized, index=spread.index, columns=['spread_normalized'])
    
    # 使用滚动窗口计算 Z-Score
    rolling_mean1 = spread_normalized_df['spread_normalized'].rolling(window=window1).mean()
    rolling_mean2 = spread_normalized_df['spread_normalized'].rolling(window=window2).mean()
    rolling_std2 = spread_normalized_df['spread_normalized'].rolling(window=window2).std()
    
    # 计算滚动 Z-Score
    zscore = (rolling_mean1 - rolling_mean2) / rolling_std2
    
    print(zscore)
    
    return zscore

mer_ori_data['zscore'] = regular_signal(mer_ori_data, 60, 900, 'close_stock1', 'close_stock2')

plot_mean(mer_ori_data, start_date='2024-12-02', col='zscore')

# 生成交易訊號
def generate_signal(ori_data, col1, col2, col3, threshold1, threshold2, desired_profit1=1, desired_profit2=1, debug=0):
    data = ori_data.copy()
    
    # 動態計算 threshold，每一行都根據當前的 close 重新計算
    data['threshold1'] = (5.022e-5 * ((117.0 * data[col2] + 20.0) / data[col2]) * desired_profit1)
    
    # 動態計算 threshold，每一行都根據當前的 close 重新計算
    data['threshold2'] = (5.022e-5 * ((117.0 * data[col3] + 20.0) / data[col3]) * desired_profit2)
    
    # Step 1: 信號生成邏輯：同時滿足 slope_df3 > threshold 和 col 閾值條件
    data['signal1'] = np.where(
        # (abs(data['slope_df2']) >= data['threshold1']) & (abs(data['slope_df1']) >= data['threshold2']) & (data[col1] <= threshold2), -1,   # 賣出信號
         (data[col1] <= threshold2), 1,   # 賣出信號
        np.where(
            # (abs(data['slope_df2']) >= data['threshold1']) & (abs(data['slope_df1']) >= data['threshold2']) & (data[col1] >= threshold1), 1, # 買入信號
            (data[col1] >= threshold1), -1, # 買入信號
            0  # 其他情況
        )
    )
    
    # Step 2: 初始化 signal2 為 0（預設無平倉訊號）
    data['signal2'] = 0
    
    # 出場邏輯 (反向訊號)
    data['signal2'] = np.where(
        # ((abs(data['slope_df2']) >= data['threshold1']) & (abs(data['slope_df1']) >= data['threshold2']) & (data[col1] <= threshold2)), 2,
         (data['signal1'] == 1) & (data[col1] <= threshold2), 2,
        np.where(
            # ((abs(data['slope_df2']) >= data['threshold2']) & (abs(data['slope_df1']) >= data['threshold2']) & (data[col1] >= threshold1)), 2,
            (data['signal1'] == -1)  & (data[col1] >= threshold1), 2,
            0
        )
    )
    
    # Step 5: 檢查結果
    if debug != 0:
        print("符合條件的前五筆資料：")
        selected_columns = ['zscore', 'signal1', 'signal2'] 
        print(data[data['signal1'].isin([1, -1]) | (data['signal2'] == 2)].loc[:, selected_columns].head(debug))

    return data[['signal1', 'signal2', 'threshold1', 'threshold2']]

# 整分鐘訊號檢查
def check_signal(df, start_time, end_time):
    # 將索引設置為時間戳並保證時間排序
    df = df.sort_index()

    # 轉換 start_time 和 end_time 為 pandas datetime 並將秒數設為0，這樣會得到整分鐘的範圍
    start_time = pd.to_datetime(start_time).replace(second=0, microsecond=0)
    end_time = pd.to_datetime(end_time).replace(second=0, microsecond=0)
    
    # 使用 floor 函數將資料的時間戳精確到分鐘，並只保留那些剛好在整分鐘的資料
    df['minute'] = df.index.floor('min')

    # 篩選條件：在範圍內的整分鐘資料，並且剛好為整分鐘
    filtered_df = df[(df['minute'] >= start_time) & (df['minute'] <= end_time)]
    
    # 排除秒數不是0的資料（即不需要的資料）
    filtered_df = filtered_df[filtered_df.index.second == 0]

    # 顯示結果
    print(filtered_df[['slope_df1', 'slope_df2', 'zscore', 'signal1', 'signal2', 'threshold1', 'threshold2', 'close_stock1', 'close_stock2']], '\n')

# 呼叫並處理資料
mer_ori_data['slope_df1'] = df1_slope['slope'].reindex(mer_ori_data.index, fill_value=0)
mer_ori_data['slope_df2'] = df2_slope['slope'].reindex(mer_ori_data.index, fill_value=0)

# spread = (index_slope['slope'] - df3_slope['slope']).fillna(0)
# mer_ori_data['zscore'] = spread.reindex(mer_ori_data.index, fill_value=0)  # 填充缺失值，讓長度一致

generate_data = generate_signal(mer_ori_data, desired_profit1=1, desired_profit2=1, debug=5, col1='zscore', col2='close_stock2', col3='close_stock1', threshold1=threshold1, threshold2=threshold2)
mer_ori_data[['signal1', 'signal2', 'threshold1', 'threshold2']] = generate_data

# check_signal(mer_ori_data, '2025-01-09 10:20:00', '2025-01-09 10:25:00')

# -------------------------------------- 回測交易 --------------------------------------

# 計算績效與分析
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
        'total_trades': total_trades
    }

# 配對交易回測
def pair_trade(**params):
    data = params.get('data')
    start_date = params.get('start_date')
    end_date = params.get('end_date')
    
    # 初始化交易相關變量
    buy1_signal, buy2_signal = [], []
    sell1_signal, sell2_signal = [], []
    entry_price_stock1, entry_price_stock2 = None, None
    stop_loss_stock1, stop_loss_stock2 = None, None
    position_stock1, position_stock2 = 0, 0
    can_reenter_trade = True  # 控制是否允许重新入场
    trade_results = []
    
    shares_per_trade1, shares_per_trade2 = params.get('shares_per_trade1'), params.get('shares_per_trade2')  # 商品交易的數量(股數)
    capital = params.get('capital')  # 初始資金
    stop_ratio1, stop_ratio2 = params.get('stop_loss1'), params.get('stop_loss2')  # 停損比例
    tick_size1, tick_size2 = params.get('tick_size1'), params.get('tick_size2')
    commission = params.get('commission')
    tax = params.get('tax')
    total_fees = 0  # 交易中产生的手续费和税费

    # 挑選測試日期, 確保索引為datetime格式
    data.index = pd.to_datetime(data.index)

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

    # 遍历时间序列
    for i in range(1, len(t_d)):
        if capital <= 0: break
        
        current_timestamp = t_d.index[i]
        current_time = current_timestamp.time()
        
        # 检查当前时间是否在 13:00 超過就平倉
        if current_time >= pd.Timestamp('13:00:00').time():
            if position_stock1 != 0 or position_stock2 != 0:
                print(f"超过指定時間，开始平仓：时间={current_timestamp}")

                # 对于stock1的仓位类型处理
                if position_stock1 > 0:  # 如果是多仓位
                    sell_price_stock1 = t_d['bid_price_stock1'].iloc[i]  # 卖出stock1的价格
                    # 计算多仓位平仓的手续费
                    sell_fee_stock1 = math.ceil(sell_price_stock1 * abs(position_stock1) * commission)
                    sell_tax_stock1 = math.ceil(sell_price_stock1 * abs(position_stock1) * tax)
                    total_fees += sell_fee_stock1 + sell_tax_stock1

                    # 计算盈亏
                    profit_loss_stock1 = (sell_price_stock1 - entry_price_stock1) * abs(position_stock1)
                    capital += profit_loss_stock1 - (sell_fee_stock1 + sell_tax_stock1) # 更新资本
                    sell1_signal.append(i)
                    print(f"平仓多仓：stock1，卖出价格={sell_price_stock1:.2f}, 盈亏={profit_loss_stock1:.2f}, 手續費={sell_fee_stock1:.2f}, 稅费={sell_tax_stock1:.2f}")

                elif position_stock1 < 0:  # 如果是空仓位
                    buy_price_stock1 = t_d['ask_price_stock1'].iloc[i]  # 买回stock1的价格
                    
                    # 计算空仓位平仓的手续费
                    buy_fee_stock1 = math.ceil(buy_price_stock1 * abs(position_stock1) * commission)
                    total_fees += buy_fee_stock1

                    # 计算盈亏
                    profit_loss_stock1 = (entry_price_stock1 - buy_price_stock1) * abs(position_stock1)  # 空仓平仓是买回价
                    capital += profit_loss_stock1 - buy_fee_stock1 # 更新资本
                    buy1_signal.append(i)
                    print(f"平仓空仓：stock1，买回价格={buy_price_stock1:.2f}, 盈亏={profit_loss_stock1:.2f}, 手續費={buy_fee_stock1:.2f}")

                # 对于stock2的仓位类型处理
                if position_stock2 > 0:  # 如果是多仓位
                    sell_price_stock2 = t_d['bid_price_stock2'].iloc[i]  # 卖出stock2的价格
                    # 计算多仓位平仓的手续费
                    sell_fee_stock2 = math.ceil(sell_price_stock2 * abs(position_stock2) * commission)
                    sell_tax_stock2 = math.ceil(sell_price_stock2 * abs(position_stock2) * tax)
                    total_fees += sell_fee_stock2 + sell_tax_stock2

                    # 计算盈亏
                    profit_loss_stock2 = (sell_price_stock2 - entry_price_stock2) * abs(position_stock2)
                    capital += profit_loss_stock2 - (sell_fee_stock2 + sell_tax_stock2)  # 更新资本
                    sell2_signal.append(i)
                    print(f"平仓多仓：stock2，卖出价格={sell_price_stock2:.2f}, 盈亏={profit_loss_stock2:.2f}, 手續費={sell_fee_stock2:.2f}, 稅费={sell_tax_stock2:.2f}")

                elif position_stock2 < 0:  # 如果是空仓位
                    buy_price_stock2 = t_d['ask_price_stock2'].iloc[i]  # 买回stock2的价格
                    
                    # 计算空仓位平仓的手续费
                    buy_fee_stock2 = math.ceil(buy_price_stock2 * abs(position_stock2) * commission)
                    total_fees += buy_fee_stock2

                    # 计算盈亏
                    profit_loss_stock2 = (entry_price_stock2 - buy_price_stock2) * abs(position_stock2)  # 空仓平仓是买回价
                    capital += profit_loss_stock2 - buy_fee_stock2 # 更新资本
                    buy2_signal.append(i)
                    print(f"平仓空仓：stock2，买回价格={buy_price_stock2:.2f}, 盈亏={profit_loss_stock2:.2f}, 手續費={buy_fee_stock1:.2f}")

                # 计算总盈亏
                net_profit_loss = profit_loss_stock1 + profit_loss_stock2 - total_fees
                trade_results.append({
                    'entry_price_stock1': entry_price_stock1,
                    'entry_price_stock2': entry_price_stock2,
                    'exit_price_stock1': sell_price_stock1 if position_stock1 > 0 else buy_price_stock1,
                    'exit_price_stock2': sell_price_stock2 if position_stock2 > 0 else buy_price_stock2,
                    'position_stock1': position_stock1,
                    'position_stock2': position_stock2,
                    'profit_loss_stock1': profit_loss_stock1,
                    'profit_loss_stock2': profit_loss_stock2,
                    'total_fees': total_fees,
                    'net_profit_loss': net_profit_loss,
                    'capital': capital,
                    'timestamp': current_timestamp
                })

                print(f"平仓后总资产：{capital:.2f}, 总手续费和税费：{total_fees:.2f}")

                # 清空仓位
                entry_price_stock1, entry_price_stock2 = None, None
                stop_loss_stock1, stop_loss_stock2 = None, None
                position_stock1, position_stock2 = 0, 0
                
                # 可以重新入场
                can_reenter_trade = True
                
                # 记录信号
                if net_profit_loss > 0:
                    print(f"盈利：净盈亏={net_profit_loss:.2f}\n")
                else:
                    print(f"亏损：净盈亏={net_profit_loss:.2f}\n")

            continue  # 超过13:00就跳过交易
    
        # 確保觸發止損後, 不會頻繁入場
        if not can_reenter_trade:
            if t_d['signal1'].iloc[i] == 0:
                print(f"signal回歸{t_d['signal1'].iloc[i]}, 可重新入場")
                can_reenter_trade = True
            else:
                continue

        # 平仓逻辑
        if position_stock1 != 0 and position_stock2 != 0:
            if position_stock1 > 0 and position_stock2 < 0: # 多:stock1, 空:stock2
                if t_d['close_stock1'].iloc[i] <= stop_loss_stock1 or t_d['close_stock2'].iloc[i] >= stop_loss_stock2:
                    # 打印當前消息
                    if t_d['close_stock1'].iloc[i] <= stop_loss_stock1:
                        print(f"止损多倉觸發：stock1，当前价格={t_d['close_stock1'].iloc[i]}, 止损价格={stop_loss_stock1:.2f}, 當前時間: {current_timestamp}")
                    elif t_d['close_stock2'].iloc[i] >= stop_loss_stock2:
                        print(f"止损空倉觸發：stock2，当前价格={t_d['close_stock2'].iloc[i]}, 止损价格={stop_loss_stock2:.2f}, 當前時間: {current_timestamp}")
                    
                    # 执行平仓 stock1 賣出, stock2 買入
                    sell_price_stock1 = t_d['ask_price_stock1'].iloc[i]
                    buy_price_stock2 = t_d['bid_price_stock2'].iloc[i]
                    
                    sell_fee_stock1 = math.ceil(sell_price_stock1 * abs(position_stock1) * commission)
                    sell_tax_stock1 = math.ceil(sell_price_stock1 * abs(position_stock1) * tax)
                    buy_fee_stock2 = math.ceil(buy_price_stock2 * abs(position_stock2) * commission)
                    total_fees += sell_fee_stock1 + sell_tax_stock1 + buy_fee_stock2

                    # 计算stock1的盈亏
                    profit_loss_stock1 = (sell_price_stock1 - entry_price_stock1) * abs(position_stock1)
                    capital += profit_loss_stock1 - (sell_fee_stock1 + sell_tax_stock1)  # 更新资本
                    sell1_signal.append(i)

                    # 计算stock2的盈亏
                    profit_loss_stock2 = (entry_price_stock2 - buy_price_stock2) * abs(position_stock2)
                    capital += profit_loss_stock2 - buy_fee_stock2  # 更新资本
                    buy2_signal.append(i)

                    print(f"止損多倉：stock1，卖出价格={sell_price_stock1:.2f}, 盈亏={profit_loss_stock1:.2f}, 手续费={sell_fee_stock1:.2f}, 税费={sell_tax_stock1:.2f}")
                    print(f"止損空倉：stock2，买回价格={buy_price_stock2:.2f}, 盈亏={profit_loss_stock2:.2f}, 手续费={buy_fee_stock2:.2f}")

                    # 清空仓位
                    entry_price_stock1, entry_price_stock2 = None, None
                    stop_loss_stock1, stop_loss_stock2 = None, None
                    position_stock1, position_stock2 = 0, 0

                    # 禁止重新入场
                    can_reenter_trade = False

                    # 计算总盈亏
                    net_profit_loss = profit_loss_stock1 + profit_loss_stock2 - total_fees
                    trade_results.append({
                        'entry_price_stock1': entry_price_stock1,
                        'entry_price_stock2': entry_price_stock2,
                        'exit_price_stock1': sell_price_stock1 if position_stock1 > 0 else buy_price_stock1,
                        'exit_price_stock2': sell_price_stock2 if position_stock2 > 0 else buy_price_stock2,
                        'position_stock1': position_stock1,
                        'position_stock2': position_stock2,
                        'profit_loss_stock1': profit_loss_stock1,
                        'profit_loss_stock2': profit_loss_stock2,
                        'total_fees': total_fees,
                        'net_profit_loss': net_profit_loss,
                        'capital': capital,
                        'timestamp': current_timestamp
                    })

                    print(f"平仓后总资产：{capital:.2f}, 总手续费和税费：{total_fees:.2f}")

                    # 记录信号
                    if net_profit_loss > 0:
                        print(f"盈利：净盈亏={net_profit_loss:.2f}\n")
                    else:
                        print(f"亏损：净盈亏={net_profit_loss:.2f}\n")
                        
            elif position_stock1 < 0 and position_stock2 > 0: # 空:stock1, 多:stock2
                if t_d['close_stock1'].iloc[i] >= stop_loss_stock1 or t_d['close_stock2'].iloc[i] <= stop_loss_stock2:
                    # 打印當前消息
                    if t_d['close_stock2'].iloc[i] <= stop_loss_stock2:
                        print(f"止损多倉觸發：stock2，当前价格={t_d['close_stock2'].iloc[i]}, 止损价格={stop_loss_stock2:.2f}, 當前時間: {current_timestamp}")
                    elif t_d['close_stock1'].iloc[i] >= stop_loss_stock1:
                        print(f"止损空倉觸發：stock1，当前价格={t_d['close_stock1'].iloc[i]}, 止损价格={stop_loss_stock1:.2f}, 當前時間: {current_timestamp}")

                    # 执行平仓 stock1 買回, stock2 賣出
                    buy_price_stock1 = t_d['bid_price_stock1'].iloc[i]
                    sell_price_stock2 = t_d['ask_price_stock2'].iloc[i]
                    
                    buy_fee_stock1 = math.ceil(buy_price_stock1 * abs(position_stock1) * commission)
                    sell_fee_stock2 = math.ceil(sell_price_stock2 * abs(position_stock2) * commission)
                    sell_tax_stock2 = math.ceil(sell_price_stock2 * abs(position_stock1) * tax)
                    total_fees += buy_fee_stock1 + sell_fee_stock2 + sell_tax_stock2

                    # 计算stock1的盈亏
                    profit_loss_stock1 = (entry_price_stock1 - buy_price_stock1) * abs(position_stock1)
                    capital += profit_loss_stock1 - buy_fee_stock1  # 更新资本
                    sell1_signal.append(i)

                    # 计算stock2的盈亏
                    profit_loss_stock2 = (sell_price_stock2 - entry_price_stock2) * abs(position_stock2)
                    capital += profit_loss_stock2 - (sell_fee_stock2 + sell_tax_stock2) # 更新资本
                    buy2_signal.append(i)
                    
                    print(f"止損空倉：stock1，买回价格={buy_price_stock1:.2f}, 盈亏={profit_loss_stock1:.2f}, 手续费={buy_fee_stock1:.2f}")
                    print(f"止損多倉：stock2，卖出价格={sell_price_stock2:.2f}, 盈亏={profit_loss_stock2:.2f}, 手续费={sell_fee_stock2:.2f}, 税费={sell_tax_stock2:.2f}")

                    # 清空仓位
                    entry_price_stock1, entry_price_stock2 = None, None
                    stop_loss_stock1, stop_loss_stock2 = None, None
                    position_stock1, position_stock2 = 0, 0
                    
                    # 禁止重新入场
                    can_reenter_trade = False
                
                    # 计算总盈亏
                    net_profit_loss = profit_loss_stock1 + profit_loss_stock2 - total_fees
                    trade_results.append({
                        'entry_price_stock1': entry_price_stock1,
                        'entry_price_stock2': entry_price_stock2,
                        'exit_price_stock1': sell_price_stock1 if position_stock1 > 0 else buy_price_stock1,
                        'exit_price_stock2': sell_price_stock2 if position_stock2 > 0 else buy_price_stock2,
                        'position_stock1': position_stock1,
                        'position_stock2': position_stock2,
                        'profit_loss_stock1': profit_loss_stock1,
                        'profit_loss_stock2': profit_loss_stock2,
                        'total_fees': total_fees,
                        'net_profit_loss': net_profit_loss,
                        'capital': capital,
                        'timestamp': current_timestamp
                    })

                    print(f"平仓后总资产：{capital:.2f}, 总手续费和税费：{total_fees:.2f}")

                    # 记录信号
                    if net_profit_loss > 0:
                        print(f"盈利：净盈亏={net_profit_loss:.2f}\n")
                    else:
                        print(f"亏损：净盈亏={net_profit_loss:.2f}\n")
                
            # 确保在仓位打开后检查zscore回归到0的情况
            if t_d['signal2'].iloc[i] == 2:
                print(f"Z分数回归，执行平仓：时间={current_timestamp}")

                # 对于stock1的仓位类型处理
                if position_stock1 > 0:  # 如果是多仓位
                    sell_price_stock1 = t_d['bid_price_stock1'].iloc[i]  # 卖出stock1的价格
                    # 计算多仓位平仓的手续费
                    sell_fee_stock1 = math.ceil(sell_price_stock1 * abs(position_stock1) * commission)
                    sell_tax_stock1 = math.ceil(sell_price_stock1 * abs(position_stock1) * tax)
                    total_fees += sell_fee_stock1 + sell_tax_stock1

                    # 计算盈亏
                    profit_loss_stock1 = (sell_price_stock1 - entry_price_stock1) * abs(position_stock1)
                    capital += profit_loss_stock1 - (sell_fee_stock1 + sell_tax_stock1) # 更新资本
                    sell1_signal.append(i)
                    print(f"平仓多仓：stock1，卖出价格={sell_price_stock1:.2f}, 盈亏={profit_loss_stock1:.2f}, 手續費={sell_fee_stock1:.2f}, 稅费={sell_tax_stock1:.2f}")

                elif position_stock1 < 0:  # 如果是空仓位
                    buy_price_stock1 = t_d['ask_price_stock1'].iloc[i]  # 买回stock1的价格
                    
                    # 计算空仓位平仓的手续费
                    buy_fee_stock1 = math.ceil(buy_price_stock1 * abs(position_stock1) * commission)
                    total_fees += buy_fee_stock1

                    # 计算盈亏
                    profit_loss_stock1 = (entry_price_stock1 - buy_price_stock1) * abs(position_stock1)  # 空仓平仓是买回价
                    capital += profit_loss_stock1 - buy_fee_stock1  # 更新资本
                    buy1_signal.append(i)
                    print(f"平仓空仓：stock1，买回价格={buy_price_stock1:.2f}, 盈亏={profit_loss_stock1:.2f}, 手續費={buy_fee_stock1:.2f}")

                # 对于stock2的仓位类型处理
                if position_stock2 > 0:  # 如果是多仓位
                    sell_price_stock2 = t_d['bid_price_stock2'].iloc[i]  # 卖出stock2的价格
                    # 计算多仓位平仓的手续费
                    sell_fee_stock2 = math.ceil(sell_price_stock2 * abs(position_stock2) * commission)
                    sell_tax_stock2 = math.ceil(sell_price_stock2 * abs(position_stock2) * tax)
                    total_fees += sell_fee_stock2 + sell_tax_stock2

                    # 计算盈亏
                    profit_loss_stock2 = (sell_price_stock2 - entry_price_stock2) * abs(position_stock2)
                    capital += profit_loss_stock2 - (sell_fee_stock2 + sell_tax_stock2)  # 更新资本
                    sell2_signal.append(i)
                    print(f"平仓多仓：stock2，卖出价格={sell_price_stock2:.2f}, 盈亏={profit_loss_stock2:.2f}, 手續費={sell_fee_stock2:.2f}, 稅费={sell_tax_stock2:.2f}")

                elif position_stock2 < 0:  # 如果是空仓位
                    buy_price_stock2 = t_d['ask_price_stock2'].iloc[i]  # 买回stock2的价格
                    
                    # 计算空仓位平仓的手续费
                    buy_fee_stock2 = math.ceil(buy_price_stock2 * abs(position_stock2) * commission)
                    total_fees += buy_fee_stock2

                    # 计算盈亏
                    profit_loss_stock2 = (entry_price_stock2 - buy_price_stock2) * abs(position_stock2)  # 空仓平仓是买回价
                    capital += profit_loss_stock2 - buy_fee_stock2  # 更新资本
                    buy2_signal.append(i)
                    print(f"平仓空仓：stock2，买回价格={buy_price_stock2:.2f}, 盈亏={profit_loss_stock2:.2f}, 手續費={buy_fee_stock2:.2f}")

                # 计算总盈亏
                net_profit_loss = profit_loss_stock1 + profit_loss_stock2 - total_fees
                trade_results.append({
                    'entry_price_stock1': entry_price_stock1,
                    'entry_price_stock2': entry_price_stock2,
                    'exit_price_stock1': sell_price_stock1 if position_stock1 > 0 else buy_price_stock1,
                    'exit_price_stock2': sell_price_stock2 if position_stock2 > 0 else buy_price_stock2,
                    'position_stock1': position_stock1,
                    'position_stock2': position_stock2,
                    'profit_loss_stock1': profit_loss_stock1,
                    'profit_loss_stock2': profit_loss_stock2,
                    'total_fees': total_fees,
                    'net_profit_loss': net_profit_loss,
                    'capital': capital,
                    'timestamp': current_timestamp
                })

                print(f"平仓后总资产：{capital:.2f}, 总手续费和税费：{total_fees:.2f}")

                # 清空仓位
                entry_price_stock1, entry_price_stock2 = None, None
                stop_loss_stock1, stop_loss_stock2 = None, None
                position_stock1, position_stock2 = 0, 0

                # 记录信号
                if net_profit_loss > 0:
                    print(f"盈利：净盈亏={net_profit_loss:.2f}\n")
                else:
                    print(f"亏损：净盈亏={net_profit_loss:.2f}\n")

        # 配对交易信号：signal為-1，做多stock1，做空stock2
        if t_d['signal1'].iloc[i] == -1 and position_stock1 == 0 and position_stock2 == 0:
            buy_price_stock1 = t_d['ask_price_stock1'].iloc[i]
            sell_price_stock2 = t_d['bid_price_stock2'].iloc[i]
            
            # 计算买入/卖出手续费
            buy_fee_stock1 = math.ceil(buy_price_stock1 * shares_per_trade1 * commission)
            sell_fee_stock2 = math.ceil(sell_price_stock2 * shares_per_trade2 * commission)
            sell_tax_stock2 = math.ceil(sell_price_stock2 * shares_per_trade2 * tax)
            total_fees += buy_fee_stock1 + sell_fee_stock2 + sell_tax_stock2
            entry_price_stock1 = buy_price_stock1
            entry_price_stock2 = sell_price_stock2

            # 止盈止损
            stop_loss_stock1 = round((buy_price_stock1 * (1 - stop_ratio1)) / tick_size1) * tick_size1
            # take_profit_stock1 = round((buy_price_stock1 * 1.01) / tick_size1) * tick_size1
            stop_loss_stock2 = round((sell_price_stock2 * (1 + stop_ratio2)) / tick_size2) * tick_size2
            # take_profit_stock2 = round((sell_price_stock2 * 0.99) / tick_size2) * tick_size2
            
            # 更新仓位與總資金
            capital -= buy_fee_stock1 + sell_fee_stock2 + sell_tax_stock2
            position_stock1 += shares_per_trade1  # 做多stock1，增加仓位
            position_stock2 -= shares_per_trade2  # 做空stock2，减少仓位
            buy1_signal.append(i)
            sell2_signal.append(i)

            print(f"配对交易做多stock1，做空stock2：时间={current_timestamp}, Z分数={t_d['zscore'].iloc[i]:.2f}")
            print(f"stock1: 买入价格={buy_price_stock1}, 止损={stop_loss_stock1:.2f}, 手續費={buy_fee_stock1:.2f}")
            print(f"stock2: 卖出价格={sell_price_stock2}, 止损={stop_loss_stock2:.2f}, 手續費={sell_fee_stock2:.2f}, 稅费={sell_tax_stock2:.2f}\n")
            # print(f"stock1: 买入价格={buy_price_stock1}, 止盈={take_profit_stock1}, 止损={stop_loss_stock1}")
            # print(f"stock2: 卖出价格={sell_price_stock2}, 止盈={take_profit_stock2}, 止损={stop_loss_stock2}\n")

        # 配对交易信号：signal為1，做空stock1，做多stock2
        elif t_d['signal1'].iloc[i] == 1 and position_stock1 == 0 and position_stock2 == 0:
            sell_price_stock1 = t_d['bid_price_stock1'].iloc[i]
            buy_price_stock2 = t_d['ask_price_stock2'].iloc[i]
            
            # 计算买入/卖出手续费
            sell_fee_stock1 = math.ceil(sell_price_stock1 * shares_per_trade1 * commission)
            sell_tax_stock1 = math.ceil(sell_price_stock1 * shares_per_trade1 * tax)
            buy_fee_stock2 = math.ceil(buy_price_stock2 * shares_per_trade2 * commission)
            total_fees += sell_fee_stock1 + buy_fee_stock2 + sell_tax_stock1
            entry_price_stock1 = sell_price_stock1
            entry_price_stock2 = buy_price_stock2

            # 止盈止损
            stop_loss_stock1 = round((sell_price_stock1 * (1 + stop_ratio1)) / tick_size1) * tick_size1
            # take_profit_stock1 = round((sell_price_stock1 * 0.99) / tick_size1) * tick_size1
            stop_loss_stock2 = round((buy_price_stock2 * (1 - stop_ratio2)) / tick_size2) * tick_size2
            # take_profit_stock2 = round((buy_price_stock2 * 1.01) / tick_size2) * tick_size2

            # 更新仓位
            capital -= sell_fee_stock1 + buy_fee_stock2 + sell_tax_stock1
            position_stock1 -= shares_per_trade1  # 做空stock1，减少仓位
            position_stock2 += shares_per_trade2  # 做多stock2，增加仓位
            sell1_signal.append(i)
            buy2_signal.append(i)

            print(f"配对交易做空stock1，做多stock2：时间={current_timestamp}, Z分数={t_d['zscore'].iloc[i]:.2f}")
            print(f"stock1: 卖出价格={sell_price_stock1}, 止损={stop_loss_stock1:.2f}, 手續費={sell_fee_stock1:.2f}, 稅费={sell_tax_stock1:.2f}")
            print(f"stock2: 买入价格={buy_price_stock2}, 止损={stop_loss_stock2:.2f}, 手續費={buy_fee_stock2:.2f}\n")
            # print(f"stock1: 卖出价格={sell_price_stock1}, 止盈={take_profit_stock1}, 止损={stop_loss_stock1}")
            # print(f"stock2: 买入价格={buy_price_stock2}, 止盈={take_profit_stock2}, 止损={stop_loss_stock2}\n")

    if trade_results:
        performance = analyze(trade_result=trade_results, is_plot=params.get('plot2'))
    
        # 打印分析结果
        print(f"净盈亏：{performance['net_profit_loss']:.2f}")
        print(f"交易後總資產：{performance['capital']:.2f}")
        print(f"獲利次數：{performance['win_count']:.2f}")
        print(f"虧損次數：{performance['loss_count']:.2f}")
        print(f"胜率：{performance['win_rate']*100:.2f}%")
        print(f"最大回撤：{performance['max_drawdown']:.2f}")
        print(f"年化回报：{performance['annualized_return']:.2f}")
        print(f"夏普比率：{performance['sharpe_ratio']:.2f}")
        print(f"风暴比：{performance['sterling_ratio']:.2f}")
        print(f"总交易次数：{performance['total_trades']}")
        print(f"总手续费和税费：{total_fees:.2f}")
        print(f"最终总资产：{capital:.2f}\n")
    else:
        print("沒有交易結果可分析。")

    if params.get('plot1') is True or params.get('detail') is True:
        if params.get('detail') is True: # 输出每次交易的盈亏结果
            print("\n【詳細交易盈虧結果】")
            for idx, trade  in enumerate(trade_results):
                print(f"第 {idx + 1} 次交易 - 凈盈虧: {trade['net_profit_loss']:.2f}, "
                  f"資本: {trade['capital']:.2f}")
        
        if params.get('plot1') is True:
            # 创建一个 figure 和 ax 对象
            fig, ax1 = plt.subplots(figsize=(12, 6))

            # 左侧价格轴：绘制商品1和商品2的价格
            ax1.plot(t_d.index, t_d['close_stock1'], label='Stock 1 Price', color='blue')
            ax1.plot(t_d.index, t_d['close_stock2'], label='Stock 2 Price', color='orange')

            # 设置左轴标签
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Price', color='black')
            ax1.tick_params(axis='y', labelcolor='black')

            # 在左侧价格轴上标记商品1和商品2的进出场信号
            ax1.scatter(t_d.index[buy1_signal], t_d['close_stock1'].iloc[buy1_signal], marker='^', color='red', label='Buy Signal (Stock 1)', s=100)
            ax1.scatter(t_d.index[sell1_signal], t_d['close_stock1'].iloc[sell1_signal], marker='v', color='green', label='Sell Signal (Stock 1)', s=100)
            ax1.scatter(t_d.index[buy2_signal], t_d['close_stock2'].iloc[buy2_signal], marker='^', color='red', label='Buy Signal (Stock 2)', s=100)
            ax1.scatter(t_d.index[sell2_signal], t_d['close_stock2'].iloc[sell2_signal], marker='v', color='green', label='Sell Signal (Stock 2)', s=100)

            # 右侧z-score轴：创建一个新的轴，分享x轴
            ax2 = ax1.twinx()
            ax2.plot(t_d.index, t_d['zscore'], label='Z-Score', color='purple', alpha=0.3)

            # 设置右轴标签
            ax2.set_ylabel('Z-Score', color='purple')
            ax2.tick_params(axis='y', labelcolor='purple')

            # 添加水平线，表示阈值
            ax2.axhline(y=0, color='black')
            ax2.axhline(y=threshold2, color='#8B0000', linestyle='--', label='Negative Threshold')
            ax2.axhline(y=threshold1, color='#B8860B', linestyle='--', label='Positive Threshold')

            # 设置图标标题和图例
            plt.title('Price and Z-Score with Trading Signals')
            ax1.legend(loc='upper left')
            ax2.legend(loc='upper right')

            # 显示网格
            ax1.grid(True)

            # 显示图表
            plt.show()

params = {
    'data': mer_ori_data,
    'start_date': '2024-12-02',
    'end_date': '2024-12-02',
    'tick_size1': 0.05,
    'tick_size2': 0.05,
    'shares_per_trade1':1000,
    'shares_per_trade2':1000,
    'capital': 100000,
    'stop_loss1': 0.01,
    'stop_loss2': 0.01,
    'commission': 0.001425,
    'tax': 0.003,
    'plot1': False,
    'plot2': False,
    'detail': False
}

# pair_trade(**params)

