import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint
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

df1 = prepare_data('shioaji/2024_1201', '2426')
df2 = prepare_data('shioaji/2024_1201', '3024')

def pre_process(df1, df2):
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
    time_range = pd.date_range('2024-12-02 09:00:00', '2025-01-02 13:30:00', freq='s')

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

mer_ori_data = pre_process(df1, df2)

# -------------------------------------- 訊號生成 --------------------------------------
threshold1, threshold2 = 2, -2

# 赫斯特檢驗
def hurst_exponent(ts, max_lag=100):
    ts = ts.dropna()
    
    lags = range(2, max_lag)
    epsilon = 1e-10  # 避免0標準差
    tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) + epsilon for lag in lags]

    hurst = np.polyfit(np.log(lags), np.log(tau), 1)[0]
    return hurst

# 正規化z-socre
def regular_signal(data, col1, col2, window1=1, window2=3, plot=False):
    # 確保索引為日期時間格式
    data = data.copy()
    data.index = pd.to_datetime(data.index)
    
    # 建立一個新的欄位用來儲存日期
    data['date'] = data.index.date

    # 初始化一個列表存放 Z-Score
    zscore_list = []

    # 針對每一天分組計算
    for date, group in data.groupby('date'):
        # 計算 spread
        spread = (group[col1] - group[col2]) / (group[col1] + group[col2])
        
        # 計算 Hurst 指數
        hurst_val = hurst_exponent(spread)

        # 如果 Hurst 指數 > 0.5，代表趨勢性過強，不適合均值回歸
        if hurst_val > 0.5:
            zscore_list.append(pd.Series([0] * len(group), index=group.index))
            continue
        
        # 計算短期與長期的均值
        spread_mean_short = spread.rolling(window=window1, min_periods=1).mean()
        spread_mean_long = spread.rolling(window=window2, min_periods=1).mean()
        spread_std_long = spread.rolling(window=window2, min_periods=1).std()
        
        # 計算Z-Score
        zscore = (spread_mean_short - spread_mean_long) / spread_std_long
        zscore_list.append(zscore)

    # 將分組結果合併回原始DataFrame
    data['zscore'] = pd.concat(zscore_list)

    # 如果需要繪圖
    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(data['zscore'])
        plt.title('Daily Z-Score')
        plt.show()
    
    return data['zscore']

zdata = regular_signal(mer_ori_data, 'close_stock1', 'close_stock2', window1=60, window2=1800, plot=False)

# OLS回歸z-score
def ols_signal(data, x, y, plot=False):
    data1 = data[x]
    data2 = data[y]
    
    print(data1, data2)
    breakpoint()
    
    # 在 X 中加入常數項（截距）以進行回歸
    X = sm.add_constant(data1)

    # 計算回歸
    model = sm.OLS(data2, X)  # OLS 最小二乘法回歸
    results = model.fit()

    # 取得回歸結果中的係數（alpha 和 beta）
    alpha, beta = results.params
    
    print(results.summary())
    print(f"Alpha (截距): {alpha}, Beta (斜率): {beta}")
    
    
    # # 計算 Spread
    # spread = data2 - (alpha + beta * X[x])  # P_A - (alpha + beta * P_B)

    # # 計算 Spread 的均值和標準差
    # spread_mean = spread.mean()
    # spread_std = spread.std()

    # # 計算 Z-Score
    # zscore = (spread - spread_mean) / spread_std

    # 如果需要繪圖，可以在這裡使用 plot 庫進行繪圖
    # if plot:
    #     plt.plot(spread)
    #     plt.title('Spread')
    #     plt.show()

    # return zscore

# zdata = ols_signal(mer_ori_data, 'close_stock2', 'close_stock1', plot=False)

mer_ori_data['zscore'] = zdata
mer_ori_data['signal'] = np.where(mer_ori_data['zscore'] < threshold2, -1, 
                            np.where(mer_ori_data['zscore'] > threshold1, 1,
                            np.where(mer_ori_data['zscore'] == 0, 2, 0)))

# 檢查1, -1, 2訊號
# print(mer_ori_data[mer_ori_data['signal'].isin([1, -1])].head(5), mer_ori_data[mer_ori_data['signal'] == 2].head(5)) 

# -------------------------------------- 繪圖檢驗 --------------------------------------

def plot_scatter(timeseries1, label1, timeseries2, label2):
    plt.scatter(timeseries1, timeseries2)
    plt.xlabel(label1)
    plt.ylabel(label2)
    plt.title(f'{label1} vs {label2}')
    plt.show()

# plot_scatter(mer_ori_data['close_stock1'], 'close_stock1', mer_ori_data['close_stock2'], 'close_stock2')


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

# mer_ori_data['spread'] = mer_ori_data['close_stock1'] - mer_ori_data['close_stock2']
# plot_trend(col1='close_stock1', col2='close_stock2', col3='spread', date='2024-12-02')

def plot_mean(start_date=None, end_date=None, col=None):
    # 確保日期是字符串並轉換為 datetime
    if start_date:
        start_date = pd.to_datetime(start_date)
    if end_date:
        end_date = pd.to_datetime(end_date)
    
    # 如果未指定 end_date，預設為 start_date 的下一天
    if start_date and not end_date:
        end_date = start_date + pd.Timedelta(days=1)
    
    # 根據日期範圍篩選數據
    filtered_data = mer_ori_data.copy()
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
    
      # 設置y軸範圍以顯示所有數據
    plt.ylim([filtered_data[col].min() - 0.5, filtered_data[col].max() + 0.5])

    # 設置圖表屬性
    plt.title(f'{col} of Mean')
    plt.xlabel('Time')
    plt.ylabel(f'{col}')
    plt.legend()
    plt.grid()
    plt.show()

# plot_mean(start_date='2024-12-02', col='zscore')

def plot_dist(data, col=None):
    # 設定畫布和大小
    plt.figure(figsize=(10, 6))

    # 使用 seaborn 繪製直方圖
    sns.histplot(data[col], kde=True, stat="density", bins=50, color='blue', label=f'{col}')

    # 計算 `diff` 的平均值和標準差
    mean_diff = mer_ori_data[col].mean()
    std_diff = mer_ori_data[col].std()

    # 創建正態分佈的數據點
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = (1 / (std_diff * np.sqrt(2 * np.pi))) * np.exp(-(x - mean_diff)**2 / (2 * std_diff**2))
    
    # D'Agostino and Pearson's Test
    stat, p_value = stats.normaltest(data[col].dropna())

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

# plot_dist(data=mer_ori_data, col='zscore')

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
def pair_trade(data, start_date=None, end_date=None, tick_size1=0.05, tick_size2=0.05, plot1=True, plot2=True, detail=False):
    # 初始化交易相關變量
    buy1_signal, buy2_signal = [], []
    sell1_signal, sell2_signal = [], []
    entry_price_stock1, entry_price_stock2 = None, None
    stop_loss_stock1, stop_loss_stock2 = None, None
    position_stock1, position_stock2 = 0, 0
    can_reenter_trade = True  # 控制是否允许重新入场
    trade_results = []
    
    shares_per_trade1, shares_per_trade2 = 1000, 1000  # 商品交易的數量(股數)
    capital = 100000  # 初始資金
    stop_ratio1, stop_ratio2 = 0.01, 0.01
    commission = 0.001425
    tax = 0.003
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
        # 已經無資金
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
            if t_d['signal'].iloc[i] == 0:
                print(f"signal回歸{t_d['signal'].iloc[i]}, 可重新入場")
                can_reenter_trade = True
            else:
                continue
                  
        # 配对交易信号：signal為-1，做多stock1，做空stock2
        if t_d['signal'].iloc[i] == -1 and position_stock1 == 0 and position_stock2 == 0:
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
        elif t_d['signal'].iloc[i] == 1 and position_stock1 == 0 and position_stock2 == 0:
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
            elif t_d['signal'].iloc[i] == 2:  # 判断zscore是否为0或碰到止損, 符合条件时平仓
                print(f"Z分数回归到0，执行平仓：时间={current_timestamp}")

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

    # 分析報告
    if trade_results:
        performance = analyze(trade_result=trade_results, is_plot=plot2)
    
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

    if plot1 is True or detail is True:
        if detail is True: # 输出每次交易的盈亏结果
            print("\n【詳細交易盈虧結果】")
            for idx, trade  in enumerate(trade_results):
                print(f"第 {idx + 1} 次交易 - 凈盈虧: {trade['net_profit_loss']:.2f}, "
                  f"資本: {trade['capital']:.2f}")
        
        if plot1 is True:
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

# pair_trade(data=mer_ori_data, start_date='2024-12-02', end_date='2024-12-07', plot1=False, plot2=False, detail=False)

