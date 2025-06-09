import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta, datetime
import math, os
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf
import statsmodels.api as sm

def prepare_data(type_of_data, data_name):
    result = type_of_data.split('/')[0]
    tmp = pd.read_csv(f'./index_data/{type_of_data}/{data_name}.csv')
    if result == 'shioaji':
        tmp['ts'] = pd.to_datetime(tmp['ts'])
    else:
        tmp['ts'] = pd.to_datetime(tmp['datetime'])

    return tmp

df1 = prepare_data('shioaji/2024_1201', '00631L')
df2 = prepare_data('shioaji/2024_1201', '00675L')

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

# 計算兩個標的物之間的差異
window1, window2 = 5, 60
threshold1, threshold2 = 3, -3
mer_ori_data['diff'] = mer_ori_data['close_stock1'] - mer_ori_data['close_stock2']
mer_ori_data['zscore'] = (mer_ori_data['diff'].rolling(window=window1).mean() - mer_ori_data['diff'].rolling(window=window2).mean()) / mer_ori_data['diff'].rolling(window=window2).std()
mer_ori_data['singal'] = np.where(mer_ori_data['zscore'] < threshold2, -1, np.where(mer_ori_data['zscore'] > threshold1, 1, 0))
# print(mer_ori_data[mer_ori_data['singal'].isin([1, -1])]) # 檢查是1, -1訊號

def plot_trend():
    # 繪製走勢圖
    plt.figure(figsize=(12, 6))

    # 繪製 close_stock1 和 close_stock2 的走勢圖
    plt.plot(mer_ori_data.index, mer_ori_data['close_stock1'], label='Close Stock 1', color='blue', alpha=0.7)
    plt.plot(mer_ori_data.index, mer_ori_data['close_stock2'], label='Close Stock 2', color='red', alpha=0.7)

    # 繪製 price_diff 的走勢圖
    plt.plot(mer_ori_data.index, mer_ori_data['diff'], label='Price Diff (Stock 1 - Stock 2)', color='green', linestyle='--')

    # 添加標題和標籤
    plt.title('Stock Price and Price Difference Trend')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.xticks(rotation=45)  # 旋轉日期刻度，避免重疊
    plt.legend()  # 顯示圖例

    # 顯示圖表
    plt.tight_layout()  # 自動調整佈局以避免標籤被截斷
    plt.show()

# plot_trend()

def plot_mean(start_date=None, end_date=None):
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
    plt.plot(filtered_data.index, filtered_data['zscore'], label='Z-score', color='blue')

    # 添加水平線表示阈值
    plt.axhline(y=threshold2, color='red', linestyle='--', label='Negative Threshold')
    plt.axhline(y=threshold1, color='green', linestyle='--', label='Positive Threshold')

    # 設置圖表屬性
    plt.title('Z-score of Close Price Ratio')
    plt.xlabel('Date')
    plt.ylabel('Z-score')
    plt.legend()
    plt.grid()
    plt.show()

# plot_mean(start_date='2024-12-03')

# 配對交易回測
def pair_trade(specific_date, tick_size1=0.05, tick_size2=0.05, plot=False):
    # 初始化买卖信号列表
    buy_signals = []
    sell_signals = []
    trade_results = []

    capital = 100000  # 初始資金
    position_stock1 = 0  # stock1 的仓位（0表示无仓位）
    position_stock2 = 0  # stock2 的仓位（0表示无仓位）
    shares_per_trade = 1000  # 每次交易1000股

    # 初始化费用记录
    total_fees = 0  # 交易中产生的手续费和税费
    
    t_d = mer_ori_data[mer_ori_data.index.date == pd.to_datetime(specific_date).date()]

    # 確保從第二個索引開始（避免初始的 NaN 值）
    for i in range(1, len(t_d)):
        current_timestamp = t_d.index[i]
        current_time = current_timestamp.time()
        price_stock1 = t_d['close_stock1'].iloc[i]
        price_stock2 = t_d['close_stock2'].iloc[i]
        
        # 检查当前时间是否在 13:00 超過就平倉
        if current_time >= pd.Timestamp('13:00:00').time():
            if position_stock1 != 0 or position_stock2 != 0:
                print(f"超过13:00，开始平仓：时间={current_timestamp}")

                # 对于stock1的仓位类型处理
                if position_stock1 > 0:  # 如果是多仓位
                    sell_price_stock1 = t_d['ask_price_stock1'].iloc[i]  # 卖出stock1的价格
                    # 计算多仓位平仓的手续费
                    sell_fee_stock1 = math.ceil(sell_price_stock1 * abs(position_stock1) * 0.001425)
                    sell_tax_stock1 = math.ceil(sell_price_stock1 * abs(position_stock1) * 0.003)
                    total_fees += sell_fee_stock1 + sell_tax_stock1

                    # 计算盈亏
                    profit_loss_stock1 = (sell_price_stock1 - price_stock1) * abs(position_stock1)
                    capital += profit_loss_stock1  # 更新资本
                    print(f"平仓多仓：stock1，卖出价格={sell_price_stock1}, 盈亏={profit_loss_stock1}")

                elif position_stock1 < 0:  # 如果是空仓位
                    buy_price_stock1 = t_d['bid_price_stock1'].iloc[i]  # 买回stock1的价格
                    # 计算空仓位平仓的手续费
                    buy_fee_stock1 = math.ceil(buy_price_stock1 * abs(position_stock1) * 0.001425)
                    buy_tax_stock1 = math.ceil(buy_price_stock1 * abs(position_stock1) * 0.003)
                    total_fees += buy_fee_stock1 + buy_tax_stock1

                    # 计算盈亏
                    profit_loss_stock1 = (price_stock1 - buy_price_stock1) * abs(position_stock1)  # 空仓平仓是买回价
                    capital += profit_loss_stock1  # 更新资本
                    print(f"平仓空仓：stock1，买回价格={buy_price_stock1}, 盈亏={profit_loss_stock1}")

                # 对于stock2的仓位类型处理
                if position_stock2 > 0:  # 如果是多仓位
                    sell_price_stock2 = t_d['ask_price_stock2'].iloc[i]  # 卖出stock2的价格
                    # 计算多仓位平仓的手续费
                    sell_fee_stock2 = math.ceil(sell_price_stock2 * abs(position_stock2) * 0.001425)
                    sell_tax_stock2 = math.ceil(sell_price_stock2 * abs(position_stock2) * 0.003)
                    total_fees += sell_fee_stock2 + sell_tax_stock2

                    # 计算盈亏
                    profit_loss_stock2 = (sell_price_stock2 - price_stock2) * abs(position_stock2)
                    capital += profit_loss_stock2  # 更新资本
                    print(f"平仓多仓：stock2，卖出价格={sell_price_stock2}, 盈亏={profit_loss_stock2}")

                elif position_stock2 < 0:  # 如果是空仓位
                    buy_price_stock2 = t_d['bid_price_stock2'].iloc[i]  # 买回stock2的价格
                    # 计算空仓位平仓的手续费
                    buy_fee_stock2 = math.ceil(buy_price_stock2 * abs(position_stock2) * 0.001425)
                    buy_tax_stock2 = math.ceil(buy_price_stock2 * abs(position_stock2) * 0.003)
                    total_fees += buy_fee_stock2 + buy_tax_stock2

                    # 计算盈亏
                    profit_loss_stock2 = (price_stock2 - buy_price_stock2) * abs(position_stock2)  # 空仓平仓是买回价
                    capital += profit_loss_stock2  # 更新资本
                    print(f"平仓空仓：stock2，买回价格={buy_price_stock2}, 盈亏={profit_loss_stock2}")

                # 计算总盈亏
                net_profit_loss = profit_loss_stock1 + profit_loss_stock2 - total_fees
                trade_results.append(net_profit_loss)

                print(f"平仓后总资产：{capital:.2f}")

                # 清空仓位
                position_stock1 = position_stock2 = 0

                # 记录信号
                if net_profit_loss > 0:
                    print(f"盈利：净盈亏={net_profit_loss:.2f}")
                else:
                    print(f"亏损：净盈亏={net_profit_loss:.2f}\n")

                buy_signals.append(i)  # 记录买入信号
                sell_signals.append(i)  # 记录卖出信号

            continue  # 超过13:00就跳过交易
    
        
        # 配对交易信号：signal為-1，做多stock1，做空stock2
        if t_d['singal'].iloc[i] == -1 and position_stock1 == 0 and position_stock2 == 0:
            buy_price_stock1 = t_d['bid_price_stock1'].iloc[i]
            sell_price_stock2 = t_d['ask_price_stock2'].iloc[i]
            
            # 计算买入/卖出手续费
            buy_fee_stock1 = math.ceil(buy_price_stock1 * shares_per_trade * 0.001425)
            sell_fee_stock2 = math.ceil(sell_price_stock2 * shares_per_trade * 0.001425)
            total_fees += buy_fee_stock1 + sell_fee_stock2

            # 止盈止损
            stop_loss_stock1 = round((buy_price_stock1 * 0.99) / tick_size1) * tick_size1
            # take_profit_stock1 = round((buy_price_stock1 * 1.01) / tick_size1) * tick_size1
            stop_loss_stock2 = round((sell_price_stock2 * 1.01) / tick_size2) * tick_size2
            # take_profit_stock2 = round((sell_price_stock2 * 0.99) / tick_size2) * tick_size2
            
            # 更新仓位
            position_stock1 += shares_per_trade  # 做多stock1，增加仓位
            position_stock2 -= shares_per_trade  # 做空stock2，减少仓位

            print(f"配对交易做多stock1，做空stock2：时间={current_timestamp}, Z分数={t_d['zscore'].iloc[i]:.2f}")
            print(f"stock1: 买入价格={buy_price_stock1}, 止损={stop_loss_stock1:.2f}")
            print(f"stock2: 卖出价格={sell_price_stock2},  止损={stop_loss_stock2:.2f}\n")
            # print(f"stock1: 买入价格={buy_price_stock1}, 止盈={take_profit_stock1}, 止损={stop_loss_stock1}")
            # print(f"stock2: 卖出价格={sell_price_stock2}, 止盈={take_profit_stock2}, 止损={stop_loss_stock2}\n")

        # 配对交易信号：signal為1，做空stock1，做多stock2
        elif t_d['singal'].iloc[i] == 1 and position_stock1 == 0 and position_stock2 == 0:
            sell_price_stock1 = t_d['ask_price_stock1'].iloc[i]
            buy_price_stock2 = t_d['bid_price_stock2'].iloc[i]
            
            # 计算买入/卖出手续费
            sell_fee_stock1 = math.ceil(sell_price_stock1 * shares_per_trade * 0.001425)
            buy_fee_stock2 = math.ceil(buy_price_stock2 * shares_per_trade * 0.001425)
            total_fees += sell_fee_stock1 + buy_fee_stock2

            # 止盈止损
            stop_loss_stock1 = round((sell_price_stock1 * 1.01) / tick_size1) * tick_size1
            # take_profit_stock1 = round((sell_price_stock1 * 0.99) / tick_size1) * tick_size1
            stop_loss_stock2 = round((buy_price_stock2 * 0.99) / tick_size2) * tick_size2
            # take_profit_stock2 = round((buy_price_stock2 * 1.01) / tick_size2) * tick_size2

            # 更新仓位
            position_stock1 -= shares_per_trade  # 做空stock1，减少仓位
            position_stock2 += shares_per_trade  # 做多stock2，增加仓位

            print(f"配对交易做空stock1，做多stock2：时间={current_timestamp}, Z分数={t_d['zscore'].iloc[i]:.2f}")
            print(f"stock1: 卖出价格={sell_price_stock1}, 止损={stop_loss_stock1:.2f}")
            print(f"stock2: 买入价格={buy_price_stock2}, 止损={stop_loss_stock2:.2f}\n")
            # print(f"stock1: 卖出价格={sell_price_stock1}, 止盈={take_profit_stock1}, 止损={stop_loss_stock1}")
            # print(f"stock2: 买入价格={buy_price_stock2}, 止盈={take_profit_stock2}, 止损={stop_loss_stock2}\n")

        # 平仓逻辑
        if position_stock1 != 0 or position_stock2 != 0:
            # 确保在仓位打开后检查zscore回归到0的情况
            if t_d['zscore'].iloc[i] == 0:  # 判断zscore是否为0，符合条件时平仓
                print(f"Z分数回归到0，执行平仓：时间={current_timestamp}")

                # 对于stock1的仓位类型处理
                if position_stock1 > 0:  # 如果是多仓位
                    sell_price_stock1 = t_d['ask_price_stock1'].iloc[i]  # 卖出stock1的价格
                    # 计算多仓位平仓的手续费
                    sell_fee_stock1 = math.ceil(sell_price_stock1 * abs(position_stock1) * 0.001425)
                    sell_tax_stock1 = math.ceil(sell_price_stock1 * abs(position_stock1) * 0.003)
                    total_fees += sell_fee_stock1 + sell_tax_stock1

                    # 计算盈亏
                    profit_loss_stock1 = (sell_price_stock1 - price_stock1) * abs(position_stock1)
                    capital += profit_loss_stock1  # 更新资本
                    print(f"平仓多仓：stock1，卖出价格={sell_price_stock1}, 盈亏={profit_loss_stock1:.2f}")

                elif position_stock1 < 0:  # 如果是空仓位
                    buy_price_stock1 = t_d['bid_price_stock1'].iloc[i]  # 买回stock1的价格
                    # 计算空仓位平仓的手续费
                    buy_fee_stock1 = math.ceil(buy_price_stock1 * abs(position_stock1) * 0.001425)
                    buy_tax_stock1 = math.ceil(buy_price_stock1 * abs(position_stock1) * 0.003)
                    total_fees += buy_fee_stock1 + buy_tax_stock1

                    # 计算盈亏
                    profit_loss_stock1 = (price_stock1 - buy_price_stock1) * abs(position_stock1)  # 空仓平仓是买回价
                    capital += profit_loss_stock1  # 更新资本
                    print(f"平仓空仓：stock1，买回价格={buy_price_stock1}, 盈亏={profit_loss_stock1:.2f}")

                # 对于stock2的仓位类型处理
                if position_stock2 > 0:  # 如果是多仓位
                    sell_price_stock2 = t_d['ask_price_stock2'].iloc[i]  # 卖出stock2的价格
                    # 计算多仓位平仓的手续费
                    sell_fee_stock2 = math.ceil(sell_price_stock2 * abs(position_stock2) * 0.001425)
                    sell_tax_stock2 = math.ceil(sell_price_stock2 * abs(position_stock2) * 0.003)
                    total_fees += sell_fee_stock2 + sell_tax_stock2

                    # 计算盈亏
                    profit_loss_stock2 = (sell_price_stock2 - price_stock2) * abs(position_stock2)
                    capital += profit_loss_stock2  # 更新资本
                    print(f"平仓多仓：stock2，卖出价格={sell_price_stock2}, 盈亏={profit_loss_stock2:.2f}")

                elif position_stock2 < 0:  # 如果是空仓位
                    buy_price_stock2 = t_d['bid_price_stock2'].iloc[i]  # 买回stock2的价格
                    # 计算空仓位平仓的手续费
                    buy_fee_stock2 = math.ceil(buy_price_stock2 * abs(position_stock2) * 0.001425)
                    buy_tax_stock2 = math.ceil(buy_price_stock2 * abs(position_stock2) * 0.003)
                    total_fees += buy_fee_stock2 + buy_tax_stock2

                    # 计算盈亏
                    profit_loss_stock2 = (price_stock2 - buy_price_stock2) * abs(position_stock2)  # 空仓平仓是买回价
                    capital += profit_loss_stock2  # 更新资本
                    print(f"平仓空仓：stock2，买回价格={buy_price_stock2}, 盈亏={profit_loss_stock2:.2f}")

                # 计算总盈亏
                net_profit_loss = profit_loss_stock1 + profit_loss_stock2 - total_fees
                trade_results.append(net_profit_loss)

                print(f"平仓后总资产：{capital:.2f}, 总手续费和税费：{total_fees:.2f}")

                # 清空仓位
                position_stock1 = position_stock2 = 0

                # 记录信号
                if net_profit_loss > 0:
                    print(f"盈利：净盈亏={net_profit_loss:.2f}")
                else:
                    print(f"亏损：净盈亏={net_profit_loss:.2f}\n")

                buy_signals.append(i)  # 记录买入信号
                sell_signals.append(i)  # 记录卖出信号

                
    # 输出总手续费和税费
    print(f"总手续费和税费：{total_fees:.2f}")
    print(f"最终总资产：{capital - total_fees:.2f}\n")

    # 输出每次交易的盈亏结果
    for idx, result in enumerate(trade_results):
        print(f"第 {idx + 1} 次交易净盈亏：{result:.2f}")

    if plot is True:
        # 可视化实际结果
        plt.figure(figsize=(12, 6))
        # 绘制实际数据
        plt.plot(t_d.index, t_d['zscore'], label='Actual Data', color='blue', alpha=0.3)
        # 绘制进出场信号
        plt.scatter(t_d.index[buy_signals], t_d['zscore'].iloc[buy_signals], 
                    marker='^', color='red', label='Buy Signal', s=100)  # 进场信号
        plt.scatter(t_d.index[sell_signals], t_d['zscore'].iloc[sell_signals], 
                    marker='v', color='green', label='Sell Signal', s=100)  # 出场信号
        
        # 添加水平線表示阈值
        plt.axhline(y=threshold2, color='#FF8C00', linestyle='--', label='Negetive Threshold')
        plt.axhline(y=threshold1, color='#8A2BE2', linestyle='--', label='Positive Threshold')
        
        plt.title('Actual Data with Trading Signals')
        plt.xlabel('Time')
        plt.ylabel('z-score')
        plt.legend()
        plt.grid()
        plt.show()

pair_trade(specific_date='2024-12-02', plot=True)



# ------------------------------------------------ 參考代碼 ------------------------------------------------------------

# --------------------------------- 多天回測 ---------------------------------

def oneday_backtest(df, date, pre_date, capital):
    test_data = df.loc[date.strftime('%Y-%m-%d')]  # 获取当天的数据
    pre_data = df.loc[pre_date.strftime('%Y-%m-%d')]  # 获取前一天的数据
    stock_data = test_data.copy()  # 拷貝一份當天數據
    stock_data['pct_change_actual'] = stock_data['close_stock1'].pct_change()    

    # 计算前一天的ACF
    acf_values, _ = acf(pre_data['close_stock1'], nlags=25, alpha=0.05)
    n = len(pre_data['close_stock1'])
    se = 1.96 / np.sqrt(n)
    significant_lags = np.where((acf_values > se) | (acf_values < -se))[0]
    first_lag_in_confidence = significant_lags[-1] if len(significant_lags) > 0 else 0

    # 根據滯後數量決策策略
    if first_lag_in_confidence >= 24:
        strategy = 'trend'
        print(f"自相關程度高({first_lag_in_confidence}), 使用「趨勢策略」\n")
    else:
        strategy = 'mean'
        print(f"自相關程度低({first_lag_in_confidence})，使用「均值回歸策略」")
    
        # r×P×1000−(P×1000×0.001425+(P+r×P)×1000×0.001425+(P+r×P)×1000×0.003)≥1.0(由此公式推導出, 固定購買股數1000股)
        # 設定變化百分比閾值(P:股票價錢, desired_profit:目標利潤=>1代表要賺到1塊錢時, 需要設定多少%數)
        threshold = (lambda P, desired_profit=1.5: 5.022e-5 * ((117.0 * P + 20.0) / P) * (desired_profit / 1.0))(stock_data['close_stock1'].iloc[0], 1.5)
        print(f'均值回歸交易的變化率閾值：{(threshold * 100):.2f}%\n')
        
    # 使用策略回测
    buy_signals = []
    sell_signals = []
    trade_results = []

    position = 0  # 持倉狀態
    shares_per_trade = 1000  # 每次交易1000股
    
    # 初始化费用记录
    total_fees = 0  # 交易中产生的手续费和税费
    
    # 根據變化率來判斷進出場
    for i in range(1, len(stock_data)):
        current_timestamp = stock_data.index[i]
        price = stock_data['close_stock1'].iloc[i]
        actual_change = stock_data['pct_change_actual'].iloc[i]
        # 股票價錢tick級距
        tick_size = 0.05
        
        if strategy == 'mean':
            # 判斷做空信號：當變化率大於threshold，做空
            if actual_change > threshold and position == 0:
                # 买入成本计算：手续费
                buy_fee = math.ceil(price * shares_per_trade * 0.001425)
                total_fees += buy_fee  # 累加手续费
                print(f"做空手续费：{buy_fee:.2f}")

                # 做空头寸
                position = -shares_per_trade  # 持仓为负表示做空
                buy_price = price  # 记录买入价格
                sell_stop_loss = round((buy_price * 1.01) / tick_size) * tick_size # 止损价(1%亏损)
                sell_take_profit = round((buy_price * 0.97) / tick_size) * tick_size  # 止盈价(3%盈利)
                buy_signals.append(i)
                print(f"做空：时间={current_timestamp}, 價格={price}, 止盈={sell_take_profit:.2f}, 止损={sell_stop_loss:.2f}, 變化率={actual_change:.2%}, 持倉={position}股\n")

            # 判斷做多信號：當變化率小於threshold，做多
            elif actual_change < -threshold and position == 0:
                # 买入成本计算：手续费
                buy_fee = math.ceil(price * shares_per_trade * 0.001425)
                total_fees += buy_fee  # 累加手续费
                print(f"做多手续费：{buy_fee:.2f}")

                # 做多头寸
                position = shares_per_trade  # 持仓为正表示做多
                buy_price = price  # 记录买入价格
                buy_stop_loss = round((buy_price * 0.99) / tick_size) * tick_size  # 止损价(1%亏损)
                buy_take_profit = round((buy_price * 1.03) / tick_size) * tick_size  # 止盈价(3%盈利)
                buy_signals.append(i)
                print(f"做多：时间={current_timestamp}, 價格={price}, 止盈={buy_take_profit:.2f}, 止损={buy_stop_loss:.2f}, 變化率={actual_change:.2%}, 持倉={position}股\n")
                
        elif strategy == 'trend':
            # 過往三分鐘的收盤價
            close1 = stock_data['close_stock1'].iloc[i-2]
            close2 = stock_data['close_stock1'].iloc[i-1]
            close3 = stock_data['close_stock1'].iloc[i]

            # 做多: 連續三分鐘收盤斜率為正
            if close2 > close1 and close3 > close2 and position == 0:
                open = stock_data['open_stock1'].iloc[i-1]
                
                # 檢查第二根的是否是長紅K
                if close2 > open * 1.002:
                    # 檢查第三根的收盤是否低於第二根的收盤的一半
                    if close3 >= (open + close2) / 2:
                        price = close3
                        
                        # 买入成本计算：手续费
                        buy_fee = math.ceil(price * shares_per_trade * 0.001425)
                        total_fees += buy_fee  # 累加手续费
                        print(f"做多手续费：{buy_fee:.2f}")

                        # 做多头寸
                        position = shares_per_trade  # 持仓为正表示做多
                        buy_price = price  # 记录买入价格
                        buy_stop_loss = round((buy_price * 0.99) / tick_size) * tick_size  # 止损价(1%亏损)
                        buy_take_profit = round((buy_price * 1.03) / tick_size) * tick_size  # 止盈价(3%盈利)
                        buy_signals.append(i)
                        print(f"做多：时间={current_timestamp}, 價格={price}, 止盈={buy_take_profit:.2f}, 止损={buy_stop_loss:.2f}, 持倉={position}股\n")

            # 做空: 連續三分鐘收盤斜率為負
            elif close2 < close1 and close3 < close2 and position == 0:
                open = stock_data['open_stock1'].iloc[i-1]
                
                # 檢查第二根的是否是長綠K
                if close2 < open * 0.9975:
                    # 檢查第三根的收盤是否高於第二根的收盤的一半
                    if close3 <= (open + close2) / 2:
                        price = close3
                                                
                        # 买入成本计算：手续费
                        buy_fee = math.ceil(price * shares_per_trade * 0.001425)
                        total_fees += buy_fee  # 累加手续费
                        print(f"做空手续费：{buy_fee:.2f}")

                        # 做空头寸
                        position = -shares_per_trade  # 持仓为负表示做空
                        buy_price = price  # 记录买入价格
                        sell_stop_loss = round((buy_price * 1.01) / tick_size) * tick_size  # 止损价(1%亏损)
                        sell_take_profit = round((buy_price * 0.97) / tick_size) * tick_size  # 止盈价(3%盈利)
                        sell_signals.append(i)
                        print(f"做空：时间={current_timestamp}, 價格={price}, 止盈={sell_take_profit:.2f}, 止损={sell_stop_loss:.2f}, 持倉={position}股\n")
        
        # 仓位控制
        if position > 0:  # 如果是做多仓位
            if price <= buy_stop_loss:  # 达到止损，直接平仓
                print(f"做多平仓：时间={current_timestamp}, 价格 {price:.2f} 达到止损 {buy_stop_loss:.2f}")
                sell_fee = math.ceil(price * shares_per_trade * 0.001425)
                sell_tax = math.ceil(price * shares_per_trade * 0.003)
                total_fees += sell_fee + sell_tax

                # 计算盈亏
                profit_loss = (price - buy_price) * position
                net_profit_loss = profit_loss - (buy_fee + sell_fee + sell_tax)
                trade_results.append(net_profit_loss)
                capital += net_profit_loss
                
                print(
                    f"平仓：时间={current_timestamp}, 價格={price}, 策略={'均值回归' if strategy == 'mean' else '趋势'}, "
                    f"變化率={round((actual_change * 100), 2) if actual_change is not None else '無'}%, "
                    f"交易含稅結果={'盈利' if net_profit_loss > 0 else '亏损'}, 本次(手續費+稅)={(buy_fee + sell_fee + sell_tax)}, "
                    f"盈虧(未扣稅+手續費)={profit_loss:.2f}，净盈亏={net_profit_loss:.2f}\n"
                )
                print(f"平倉後總資產：{capital:.2f}\n")
                
                position = 0  # 清空仓位
                
                if capital < 0:
                    buy_signals.append(i)  # 做空的平仓记录为买入信号
                else:
                    sell_signals.append(i)  # 做多的平仓记录为卖出信号

            elif price >= buy_take_profit:  # 达到止盈，不平仓，更新止盈止损
                print(f"做多未平仓：时间={current_timestamp}, 价格 {price:.2f} 达到止盈 {buy_take_profit:.2f}, 更新止盈止损")
                # 以当前价格重新设定新的止盈止损
                buy_take_profit = price * 1.03  # 新的止盈价为当前价格上方3%
                buy_stop_loss = price * 0.99  # 新的止损价为当前价格下方1%
                print(f"新的止盈价格设定为 {buy_take_profit:.2f}, 新的止损价格设定为 {buy_stop_loss:.2f}\n")

        elif position < 0:  # 如果是做空仓位
            if price >= sell_stop_loss:  # 达到止损，直接平仓
                print(f"做空平仓：时间={current_timestamp}, 价格 {price:.2f} 达到止损 {sell_stop_loss:.2f}")
                sell_fee = math.ceil(price * shares_per_trade * 0.001425)
                sell_tax = math.ceil(price * shares_per_trade * 0.003)
                total_fees += sell_fee + sell_tax

                # 计算盈亏
                profit_loss = (buy_price - price) * -position
                net_profit_loss = profit_loss - (buy_fee + sell_fee + sell_tax)
                trade_results.append(net_profit_loss)
                capital += net_profit_loss
                
                print(
                    f"平仓：时间={current_timestamp}, 價格={price}, 策略={'均值回归' if strategy == 'mean' else '趋势'}, "
                    f"變化率={round((actual_change * 100), 2) if actual_change is not None else '無'}%, "
                    f"交易含稅結果={'盈利' if net_profit_loss > 0 else '亏损'}, 本次(手續費+稅)={(buy_fee + sell_fee + sell_tax)}, "
                    f"盈虧(未扣稅+手續費)={profit_loss:.2f}，净盈亏={net_profit_loss:.2f}\n"
                )
                print(f"平倉後總資產：{capital:.2f}\n")
                
                position = 0  # 清空仓位
                
                if capital < 0:
                    buy_signals.append(i)  # 做空的平仓记录为买入信号
                else:
                    sell_signals.append(i)  # 做多的平仓记录为卖出信号

            elif price <= sell_take_profit:  # 达到止盈，不平仓，更新止盈止损
                print(f"做空未平仓：时间={current_timestamp}, 价格 {price:.2f} 达到止盈 {sell_take_profit:.2f}, 更新止盈止损")
                # 以当前价格重新设定新的止盈止损
                sell_take_profit = price * 0.97  # 新的止盈价为当前价格下方3%
                sell_stop_loss = price * 1.01  # 新的止损价为当前价格上方1%
                print(f"新的止盈价格设定为 {sell_take_profit:.2f}, 新的止损价格设定为 {sell_stop_loss:.2f}\n")


    # 如果还持有头寸，到最后时平仓
    if position != 0:
        price = stock_data['close_stock1'].iloc[-1]

        # 卖出成本计算：手续费和税, 累加手续费和税费
        sell_fee = math.ceil(price * shares_per_trade * 0.001425)
        sell_tax = math.ceil(price * shares_per_trade * 0.003)
        total_fees += sell_fee + sell_tax

        # 计算盈亏（不包括手续费和税费）
        profit_loss = (price - buy_price) * position

        # 计算净盈亏（扣除手续费和税费）
        net_profit_loss = profit_loss - (buy_fee + sell_fee + sell_tax)
        trade_results.append(net_profit_loss)
        capital += net_profit_loss
        
        if capital < 0:
            buy_signals.append(i)  # 做空的平仓记录为买入信号
        else:
            sell_signals.append(i)  # 做多的平仓记录为卖出信号

        print(f"最后平仓：时间={current_timestamp}, 价格={price}, 最终资产={round(capital)}")
        print(f"最后交易净盈亏：{net_profit_loss:.2f}\n")

    # 输出总手续费和税费
    print(f"交易日: {date}, 檢驗日: {pre_date}")
    print(f"总手续费和税费：{total_fees:.2f}")
    print(f"最终总资产：{capital - total_fees:.2f}\n")

    # 输出每次交易的盈亏结果
    for idx, result in enumerate(trade_results):
        print(f"第 {idx + 1} 次交易净盈亏：{result:.2f}")
    
    return round(capital)

def full_backtest(df, start_date, end_date, initial_capital=100000):
    current_capital = initial_capital
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')  # 遍历工作日
    capital_over_time = []  # 用来记录每天的资本变化
    last_valid_date = None  # 用来跟踪最近的有效日期

    i = 1  # 从第二个日期开始
    while i < len(date_range):
        print("-----------------------------------------------------")
        current_date = date_range[i].date()  # 获取当前日期的 date 类型     
           
        # 检查当前日期是否有数据, 如果当前日期有数据，获取前一天的日期
        if current_date in df.index.date:
            previous_date = date_range[i - 1].date() if last_valid_date is None else last_valid_date

            # 更新最近有效日期
            last_valid_date = current_date

            # 调用单天回测，获取更新后的资金
            current_capital = oneday_backtest(df, current_date, previous_date, current_capital)
            capital_over_time.append(current_capital)

            print(f"Current: {current_date}, Capital: {current_capital}\n")
            i += 1  # 移动到下一个日期
            
        else: # 如果当前日期没有数据，则跳过该日期, 继续到下一个日期
            print(f"Skipping current date: {current_date} (no data)\n")
            i += 1 

    # 返回最后的资金结果和每日资金记录
    return current_capital, capital_over_time

# 运行完整的回测，假设df包含了2024年1月2日到2024年9月30日的所有数据
# final_capital, capital_history = full_backtest(mer_ori_data, '2024-01-02', '2024-01-30')
# print(f"回測結束最終資產: {final_capital}")

# --------------------------------- 單天回測 ---------------------------------

def backtest(df, date, plot=False):
    # 提取特定日期的数据
    stock_data = df.loc[date]

    # 筛选该日期从 9:02 到 13:24 的数据
    stock_data = stock_data.between_time('09:02', '13:24')

    # 删除含有NaN的行
    stock_data.dropna(inplace=True)
    
    # r×P×1000−(P×1000×0.001425+(P+r×P)×1000×0.001425+(P+r×P)×1000×0.003)≥1.0(由此公式推導出, 固定購買股數1000股)
    # 設定變化百分比閾值(P:股票價錢, desired_profit:目標利潤=>1代表要賺到1塊錢時, 需要設定多少%數)
    threshold = (lambda P, desired_profit=1.5: 5.022e-5 * ((117.0 * P + 20.0) / P) * (desired_profit / 1.0))(stock_data['close_stock1'].iloc[0], 1.5)
    print(f'均值回歸交易的變化率閾值：{(threshold * 100):.2f}%\n')
        
    # 在策略判断之前，计算变化率, 繪圖可看到預測結果, 可再進行調整
    stock_data['pct_change_actual'] = stock_data['close_stock1'].pct_change()
  
    # 初始化买卖信号列表
    buy_signals = []
    sell_signals = []
    trade_results = []

    capital = 100000  # 初始資金
    position = 0  # 持倉狀態
    shares_per_trade = 1000  # 每次交易1000股
    
    # 初始化费用记录
    total_fees = 0  # 交易中产生的手续费和税费
    
    # 根據變化率來判斷進出場
    for i in range(1, len(stock_data)):
        current_timestamp = stock_data.index[i]
        
        # 股票價錢tick級距
        tick_size = 0.05
        
        actual_change = stock_data['pct_change_actual'].iloc[i]
        price = stock_data['close_stock1'].iloc[i]

        # 判斷做空信號：當變化率大於threshold，做空
        if actual_change > threshold and position == 0:
            # 买入成本计算：手续费
            buy_fee = math.ceil(price * shares_per_trade * 0.001425)
            total_fees += buy_fee  # 累加手续费
            print(f"做空手续费：{buy_fee:.2f}")

            # 做空头寸
            position = -shares_per_trade  # 持仓为负表示做空
            buy_price = price  # 记录买入价格
            sell_stop_loss = round((buy_price * 1.01) / tick_size) * tick_size # 止损价(1%亏损)
            sell_take_profit = round((buy_price * 0.99) / tick_size) * tick_size  # 止盈价(1%盈利)
            buy_signals.append(i)
            print(f"做空：时间={current_timestamp}, 價格={price}, 止盈={sell_take_profit}, 止损={sell_stop_loss}, 變化率={actual_change:.2%}, 持倉={position}股\n")

        # 判斷做多信號：當變化率小於threshold，做多
        elif actual_change < -threshold and position == 0:
            # 买入成本计算：手续费
            buy_fee = math.ceil(price * shares_per_trade * 0.001425)
            total_fees += buy_fee  # 累加手续费
            print(f"做多手续费：{buy_fee:.2f}")

            # 做多头寸
            position = shares_per_trade  # 持仓为正表示做多
            buy_price = price  # 记录买入价格
            buy_stop_loss = round((buy_price * 0.99) / tick_size) * tick_size  # 止损价(1%亏损)
            buy_take_profit = round((buy_price * 1.01) / tick_size) * tick_size  # 止盈价(1%盈利)
            buy_signals.append(i)
            print(f"做多：时间={current_timestamp}, 價格={price}, 止盈={buy_take_profit}, 止损={buy_stop_loss}, 變化率={actual_change:.2%}, 持倉={position}股\n")
               
        # 仓位控制
        if position > 0:  # 如果是做多仓位
            if price <= buy_stop_loss:  # 达到止损，直接平仓
                print(f"做多平仓：时间={current_timestamp}, 达到止损价格 {price}")
                sell_fee = math.ceil(price * shares_per_trade * 0.001425)
                sell_tax = math.ceil(price * shares_per_trade * 0.003)
                total_fees += sell_fee + sell_tax

                # 计算盈亏
                profit_loss = (price - buy_price) * position
                net_profit_loss = profit_loss - (buy_fee + sell_fee + sell_tax)
                trade_results.append(net_profit_loss)
                capital += net_profit_loss
                
                print(f"平仓：时间={current_timestamp}, 價格={price}, 策略=均值回归, 变化率: {actual_change:.2 if actual_change is not None else '无'}%, 交易含稅結果{'盈利' if net_profit_loss > 0 else '亏损'}, 本次(手續費+稅){(buy_fee + sell_fee + sell_tax)}, 盈虧(未扣稅+手續費){profit_loss:.2f}，净盈亏={net_profit_loss:.2f}\n")
                print(f"平倉後總資產：{capital:.2f}\n")
                
                position = 0  # 清空仓位
                
                if capital < 0:
                    buy_signals.append(i)  # 做空的平仓记录为买入信号
                else:
                    sell_signals.append(i)  # 做多的平仓记录为卖出信号

            elif price >= buy_take_profit:  # 达到止盈，不平仓，更新止盈止损
                print(f"做多未平仓：时间={current_timestamp}, 达到止盈价格 {price}, 更新止盈止损")
                # 以当前价格重新设定新的止盈止损
                buy_take_profit = price * 1.03  # 新的止盈价为当前价格上方3%
                buy_stop_loss = price * 0.99  # 新的止损价为当前价格下方1%
                print(f"新的止盈价格设定为 {buy_take_profit:.2f}, 新的止损价格设定为 {buy_stop_loss:.2f}\n")

        elif position < 0:  # 如果是做空仓位
            if price >= sell_stop_loss:  # 达到止损，直接平仓
                print(f"做空平仓：时间={current_timestamp}, 达到止损价格 {price}")
                sell_fee = math.ceil(price * shares_per_trade * 0.001425)
                sell_tax = math.ceil(price * shares_per_trade * 0.003)
                total_fees += sell_fee + sell_tax

                # 计算盈亏
                profit_loss = (buy_price - price) * -position
                net_profit_loss = profit_loss - (buy_fee + sell_fee + sell_tax)
                trade_results.append(net_profit_loss)
                capital += net_profit_loss
                print(f"平仓：时间={current_timestamp}, 價格={price}, 策略=均值回归, 变化率: {actual_change:.2 if actual_change is not None else '无'}%, 交易含稅結果{'盈利' if net_profit_loss > 0 else '亏损'}, 本次(手續費+稅){(buy_fee + sell_fee + sell_tax)}, 盈虧(未扣稅+手續費){profit_loss:.2f}，净盈亏={net_profit_loss:.2f}\n")
                print(f"平倉後總資產：{capital:.2f}\n")
                
                position = 0  # 清空仓位
                
                if capital < 0:
                    buy_signals.append(i)  # 做空的平仓记录为买入信号
                else:
                    sell_signals.append(i)  # 做多的平仓记录为卖出信号

            elif price <= sell_take_profit:  # 达到止盈，不平仓，更新止盈止损
                print(f"做空未平仓：时间={current_timestamp}, 达到止盈价格 {price}, 更新止盈止损")
                # 以当前价格重新设定新的止盈止损
                sell_take_profit = price * 0.97  # 新的止盈价为当前价格下方3%
                sell_stop_loss = price * 1.01  # 新的止损价为当前价格上方1%
                print(f"新的止盈价格设定为 {sell_take_profit:.2f}, 新的止损价格设定为 {sell_stop_loss:.2f}\n")


    # 如果还持有头寸，到最后时平仓
    if position != 0:
        price = stock_data['close_stock1'].iloc[-1]

        # 卖出成本计算：手续费和税, 累加手续费和税费
        sell_fee = price * shares_per_trade * 0.001425
        sell_tax = price * shares_per_trade * 0.003
        total_fees += sell_fee + sell_tax

        # 计算盈亏（不包括手续费和税费）
        profit_loss = (price - buy_price) * position

        # 计算净盈亏（扣除手续费和税费）
        net_profit_loss = profit_loss - (buy_fee + sell_fee + sell_tax)
        trade_results.append(net_profit_loss)
        capital += net_profit_loss
        
        if capital < 0:
            buy_signals.append(i)  # 做空的平仓记录为买入信号
        else:
            sell_signals.append(i)  # 做多的平仓记录为卖出信号

        print(f"最后平仓：时间={current_timestamp}, 价格={price}, 最终资产={capital}")
        print(f"最后交易净盈亏：{net_profit_loss:.2f}\n")

    # 输出总手续费和税费
    print(f"总手续费和税费：{total_fees:.2f}")
    print(f"最终总资产：{capital - total_fees:.2f}\n")

    # 输出每次交易的盈亏结果
    for idx, result in enumerate(trade_results):
        print(f"第 {idx + 1} 次交易净盈亏：{result:.2f}")
    
    if plot is True:
        # 可视化实际结果、ARIMA预测和GARCH预测波动性
        plt.figure(figsize=(12, 6))

         # 绘制实际数据
        plt.plot(stock_data.index, stock_data['close_stock1'], label='Actual Data', color='blue')

     # 绘制进出场信号
        plt.scatter(stock_data.index[buy_signals], stock_data['close_stock1'].iloc[buy_signals], 
                marker='^', color='red', label='Buy Signal', s=100)  # 进场信号

        plt.scatter(stock_data.index[sell_signals], stock_data['close_stock1'].iloc[sell_signals], 
                marker='v', color='green', label='Sell Signal', s=100)  # 出场信号

        plt.title('Actual Data with Trading Signals')
        plt.xlabel('Time')
        plt.ylabel('Log Price / Volatility')
        plt.legend()
        plt.grid()
        plt.show()

# backtest(mer_ori_data, '2024-09-19', '2024-09-16', plot=True)

# --------------------------------- 平均滞后計算 ---------------------------------

# # 将数据按日期进行分组
# mer_ori_data['date'] = mer_ori_data.index.date
# grouped_data = mer_ori_data.groupby('date')

# # 初始化变量来记录第一组的起始日期和最后一组的结束日期, 用于存储每个交易日的滞后数 
# first_start_date = None
# last_end_date = None
# lag_list = []

# # 遍历每个交易日的数据
# for date, pre_data in grouped_data:
#     # 如果是第一组数据，记录起始日期
#     if first_start_date is None:  
#         first_start_date = pre_data.index[0].date()  
#     last_end_date = pre_data.index[-1].date()  
    
#     acf_values, confint = acf(pre_data['close_stock1'], nlags=25, alpha=0.05)
#     # 计算标准误差
#     n = len(pre_data['close_stock1'])
#     se = 1.96 / np.sqrt(n)
#     # 找到显著的滞后值（ACF 超出标准误差范围的滞后值）
#     significant_lags = np.where((acf_values > se) | (acf_values < -se))[0]
#     # 如果有显著滞后值，计算第一个回到置信区间内的滞后数
#     first_lag_in_confidence = significant_lags[-1] if len(significant_lags) > 0 else 0
#     lag_list.append(first_lag_in_confidence)

# # 循环结束后，打印第一组和最后一组的日期范围, 计算平均滞后数
# print(f"第一组数据起始日期：{first_start_date}")
# print(f"最后一组数据结束日期：{last_end_date}")
# print(f"平均滞后数: {round(np.mean(lag_list))}")

# --------------------------------- ARIMA均值回歸 ---------------------------------

# def mean_backtest(df, date, pre_date, plot=False, mean_plot=False):
#     # 提取特定日期的数据
#     stock_data = df.loc[date]
#     pre_data = df.loc[pre_date]

#     # 筛选该日期从 9:02 到 13:24 的数据
#     stock_data = stock_data.between_time('09:02', '13:24')
#     pre_data = pre_data.between_time('09:02', '13:24')

#     # 删除含有NaN的行
#     stock_data.dropna(inplace=True)
#     pre_data.dropna(inplace=True)
    
#     # 确认时间序列的频率信息
#     pre_data = pre_data.asfreq('min')  # 再次明确频率
    
#     # 拟合 AR(1) 模型
#     model = sm.tsa.ARIMA(pre_data['close_stock1'], order=(1, 0, 4), exog=pre_data['close_stock3'])
#     results = model.fit()

#     # 获取自回归系数 beta
#     beta = results.params['ar.L1']

#     # 计算半衰期
#     half_life = -np.log(2) / np.log(beta)
#     print(f"半衰期: {round(half_life)} 个时间单位")
    
#     # 使用大约 2 倍的半衰期作为均值窗口
#     mean_window = int(2 * half_life)

#     # 计算 spread
#     stock_data['spread'] = stock_data['close_stock1'] - stock_data['close_stock3'] * beta

#     # 計算 spread 的滾動均值和標準差
#     stock_data['rolling_mean'] = stock_data['spread'].rolling(window=mean_window).mean()
#     stock_data['rolling_std'] = stock_data['spread'].rolling(window=mean_window).std()

#     # 計算 z-score 標準化的 spread
#     stock_data['z_score'] = (stock_data['spread'] - stock_data['rolling_mean']) / stock_data['rolling_std']

#     # r×P×1000−(P×1000×0.001425+(P+r×P)×1000×0.001425+(P+r×P)×1000×0.003)≥1.0(由此公式推導出, 固定購買股數1000股)
#     # 設定變化百分比閾值(P:股票價錢, desired_profit:目標利潤=>1代表要賺到1塊錢時, 需要設定多少%數)
#     threshold = (lambda P, desired_profit=1.5: 5.022e-5 * ((117.0 * P + 20.0) / P) * (desired_profit / 1.0))(stock_data['spread'].iloc[0], 1.5)
#     print(f'均值回歸交易的變化率閾值：{(threshold * 100):.2f}%\n')

#     # 設置信號列
#     stock_data['signal'] = 0

#     # 設置進場信號：同時滿足變化率超過 threshold 且 z-score 超過 ±3
#     # 做多信號
#     stock_data.loc[stock_data['z_score'] < -3, 'signal'] = 1
#     # 做空信號
#     stock_data.loc[stock_data['z_score'] > 3, 'signal'] = -1

#     # 設置出場信號
#     # 做多時從 3 到 -3 出場, 平倉做多
#     stock_data.loc[(stock_data['signal'].shift(1) == 1) & (stock_data['z_score'] >= 3), 'signal'] = 0
#     # 做空時從 -3 到 3 出場, 平倉做空
#     stock_data.loc[(stock_data['signal'].shift(1) == -1) & (stock_data['z_score'] <= -3), 'signal'] = 0
    
#     # 初始化买卖信号列表
#     buy_signals = []
#     sell_signals = []
#     trade_results = []

#     capital = 100000  # 初始資金
#     position = 0  # 持倉狀態
#     shares_per_trade = 1000  # 每次交易1000股
    
#     # 初始化费用记录
#     total_fees = 0  # 交易中产生的手续费和税费

#     # 每次迭代检查滯後時間和趨勢信号来决定策略
#     for i in range(1, len(stock_data)):
#         current_timestamp = stock_data.index[i]
#         tick_size = 0.05
#         price = stock_data['close_stock1'].iloc[i]
#         signal = stock_data['signal'].iloc[i]
        
#         # 做多信号
#         if signal == 1 and position == 0:
#             buy_fee = math.ceil(price * shares_per_trade * 0.001425)
#             total_fees += buy_fee
#             position = shares_per_trade
#             buy_price = price
#             buy_stop_loss = round((buy_price * 0.99) / tick_size) * tick_size
#             buy_take_profit = round((buy_price * 1.03) / tick_size) * tick_size
#             buy_signals.append(i)
#             print(f"做多：时间={current_timestamp}, 價格={price}, 止盈={buy_take_profit}, 止损={buy_stop_loss}, 持倉={position}股\n")

#         # 做空信号
#         elif signal == -1 and position == 0:
#             buy_fee = math.ceil(price * shares_per_trade * 0.001425)
#             total_fees += buy_fee
#             position = -shares_per_trade
#             buy_price = price
#             sell_stop_loss = round((buy_price * 1.01) / tick_size) * tick_size
#             sell_take_profit = round((buy_price * 0.97) / tick_size) * tick_size
#             sell_signals.append(i)
#             print(f"做空：时间={current_timestamp}, 價格={price}, 止盈={sell_take_profit}, 止损={sell_stop_loss}, 持倉={position}股\n")
        
        
#         # 仓位控制
#         if position > 0:  # 如果是做多仓位
#             if price <= buy_stop_loss:  # 达到止损，直接平仓
#                 print(f"做多平仓：时间={current_timestamp}, 达到止损价格 {price}")
#                 sell_fee = math.ceil(price * shares_per_trade * 0.001425)
#                 sell_tax = math.ceil(price * shares_per_trade * 0.003)
#                 total_fees += sell_fee + sell_tax

#                 # 计算盈亏
#                 profit_loss = (price - buy_price) * position
#                 net_profit_loss = profit_loss - (buy_fee + sell_fee + sell_tax)
#                 trade_results.append(net_profit_loss)
#                 capital += net_profit_loss
                
#                 print(f"平仓：时间={current_timestamp}, 價格={price}, 交易含稅結果{'「盈利」' if net_profit_loss > 0 else '「亏损」'}, 本次(手續費+稅){(buy_fee + sell_fee + sell_tax)}, 盈虧(未扣稅+手續費){profit_loss:.2f}，净盈亏={net_profit_loss:.2f}\n")
#                 print(f"平倉後總資產：{capital:.2f}\n")
                
#                 position = 0  # 清空仓位
                
#                 if capital < 0:
#                     buy_signals.append(i)  # 做空的平仓记录为买入信号
#                 else:
#                     sell_signals.append(i)  # 做多的平仓记录为卖出信号

#             elif price >= buy_take_profit:  # 达到止盈，不平仓，更新止盈止损
#                 print(f"做多未平仓：时间={current_timestamp}, 达到止盈价格 {price}, 更新止盈止损")
#                 # 以当前价格重新设定新的止盈止损
#                 buy_take_profit = price * 1.03  # 新的止盈价为当前价格上方3%
#                 buy_stop_loss = price * 0.99  # 新的止损价为当前价格下方1%
#                 print(f"新的止盈价格设定为 {buy_take_profit:.2f}, 新的止损价格设定为 {buy_stop_loss:.2f}\n")
                
#             elif signal == 0:  # 根據z-score平倉
#                 print(f"多頭平倉：时间={current_timestamp}, z-score 回到均值附近，平倉")
#                 sell_fee = math.ceil(price * shares_per_trade * 0.001425)
#                 sell_tax = math.ceil(price * shares_per_trade * 0.003)
#                 total_fees += sell_fee + sell_tax
            
#                 profit_loss = (price - buy_price) * position
#                 net_profit_loss = profit_loss - (buy_fee + sell_fee + sell_tax)
#                 trade_results.append(net_profit_loss)
#                 capital += net_profit_loss

#                 print(f"平仓：时间={current_timestamp}, 價格={price}, 交易含稅結果{'「盈利」' if net_profit_loss > 0 else '「亏损」'}, 本次(手續費+稅){(buy_fee + sell_fee + sell_tax)}, 盈虧(未扣稅+手續費){profit_loss:.2f}，净盈亏={net_profit_loss:.2f}\n")
#                 print(f"平倉後總資產：{capital:.2f}\n")

#                 position = 0  # 清空仓位    
                            
#                 if capital < 0:
#                     buy_signals.append(i)  # 做空的平仓记录为买入信号
#                 else:
#                     sell_signals.append(i)  # 做多的平仓记录为卖出信号


#         elif position < 0:  # 如果是做空仓位
#             if price >= sell_stop_loss:  # 达到止损，直接平仓
#                 print(f"做空平仓：时间={current_timestamp}, 达到止损价格 {price}")
#                 sell_fee = math.ceil(price * shares_per_trade * 0.001425)
#                 sell_tax = math.ceil(price * shares_per_trade * 0.003)
#                 total_fees += sell_fee + sell_tax

#                 # 计算盈亏
#                 profit_loss = (buy_price - price) * -position
#                 net_profit_loss = profit_loss - (buy_fee + sell_fee + sell_tax)
#                 trade_results.append(net_profit_loss)
#                 capital += net_profit_loss

#                 print(f"平仓：时间={current_timestamp}, 價格={price},  交易含稅結果{'「盈利」' if net_profit_loss > 0 else '「亏损」'}, 本次(手續費+稅){(buy_fee + sell_fee + sell_tax)}, 盈虧(未扣稅+手續費){profit_loss:.2f}，净盈亏={net_profit_loss:.2f}\n")
#                 print(f"平倉後總資產：{capital:.2f}\n")
                
#                 position = 0  # 清空仓位
                
#                 if capital < 0:
#                     buy_signals.append(i)  # 做空的平仓记录为买入信号
#                 else:
#                     sell_signals.append(i)  # 做多的平仓记录为卖出信号

#             elif price <= sell_take_profit:  # 达到止盈，不平仓，更新止盈止损
#                 print(f"做空未平仓：时间={current_timestamp}, 达到止盈价格 {price}, 更新止盈止损")
#                 # 以当前价格重新设定新的止盈止损
#                 sell_take_profit = price * 0.97  # 新的止盈价为当前价格下方3%
#                 sell_stop_loss = price * 1.01  # 新的止损价为当前价格上方1%
#                 print(f"新的止盈价格设定为 {sell_take_profit:.2f}, 新的止损价格设定为 {sell_stop_loss:.2f}\n")

#             elif signal == 0:  # 根據z-score平倉
#                 print(f"空頭平倉：时间={current_timestamp}, z-score 回到均值附近，平倉")
#                 sell_fee = math.ceil(price * abs(position) * 0.001425)
#                 sell_tax = math.ceil(price * abs(position) * 0.003)
#                 total_fees += buy_fee + sell_tax

#                 profit_loss = (buy_price - price) * abs(position)
#                 net_profit_loss = profit_loss - (buy_fee + sell_fee + sell_tax)
#                 trade_results.append(net_profit_loss)
#                 capital += net_profit_loss

#                 print(f"平仓：时间={current_timestamp}, 價格={price}, 交易含稅結果{'「盈利」' if net_profit_loss > 0 else '「亏损」'}, 本次(手續費+稅){(buy_fee + sell_fee + sell_tax)}, 盈虧(未扣稅+手續費){profit_loss:.2f}，净盈亏={net_profit_loss:.2f}\n")
#                 print(f"平倉後總資產：{capital:.2f}\n")

#                 position = 0  # 清空仓位
                
#                 if capital < 0:
#                     buy_signals.append(i)  # 做空的平仓记录为买入信号
#                 else:
#                     sell_signals.append(i)  # 做多的平仓记录为卖出信号


#     # 如果还持有头寸，到最后时平仓
#     if position != 0:
#         price = stock_data['close_stock1'].iloc[-1]

#         # 卖出成本计算：手续费和税, 累加手续费和税费
#         sell_fee = price * shares_per_trade * 0.001425
#         sell_tax = price * shares_per_trade * 0.003
#         total_fees += sell_fee + sell_tax

#         # 计算盈亏（不包括手续费和税费）
#         profit_loss = (price - buy_price) * position

#         # 计算净盈亏（扣除手续费和税费）
#         net_profit_loss = profit_loss - (buy_fee + sell_fee + sell_tax)
#         trade_results.append(net_profit_loss)
#         capital += net_profit_loss
        
#         if capital < 0:
#             buy_signals.append(i)  # 做空的平仓记录为买入信号
#         else:
#             sell_signals.append(i)  # 做多的平仓记录为卖出信号

#         print(f"最后平仓：时间={current_timestamp}, 价格={price}, 最终资产={capital}")
#         print(f"最后交易净盈亏：{net_profit_loss:.2f}\n")

#     # 输出总手续费和税费
#     print(f"总手续费和税费：{total_fees:.2f}")
#     print(f"最终总资产：{capital - total_fees:.2f}\n")

#     # 输出每次交易的盈亏结果
#     for idx, result in enumerate(trade_results):
#         print(f"第 {idx + 1} 次交易净盈亏：{result:.2f}")
    
#     if plot is True:
#         if mean_plot is True:
#             # 創建一個圖形
#             fig, ax1 = plt.subplots(figsize=(12, 6))

#             # 繪製滾動均值在左Y軸
#             color = np.random.rand(3,)
#             ax1.set_ylabel('Rolling Mean', color=color)
#             ax1.plot(stock_data.index, stock_data['rolling_mean'], label='Rolling Mean', color=color, linestyle='--', alpha=0.5)
#             ax1.tick_params(axis='y', labelcolor=color)

#             # 創建第二個Y軸
#             ax2 = ax1.twinx()
#             ax2.set_ylabel('Z-Score', color='blue')
#             ax2.plot(stock_data.index, stock_data['z_score'], label='Z-Score', color='blue')
#             ax2.tick_params(axis='y', labelcolor='blue')

#             # 繪製 z-score 的上下界線
#             ax2.axhline(y=0, color='black', label='Mean = 0')
#             ax2.axhline(1, color='g', label='Mean ± 1 * STD')
#             ax2.axhline(-1, color='g')
#             ax2.axhline(2, color='orange', label='Mean ± 2 * STD')
#             ax2.axhline(-2, color='orange')
#             ax2.axhline(3, color='red', label='Mean ± 3 * STD')
#             ax2.axhline(-3, color='red')

#             # 標記進場、出場和平倉信號
#             for i in range(len(stock_data)):
#                 if stock_data['signal'].iloc[i] == 1:  # 多頭信號
#                     ax2.scatter(stock_data.index[i], stock_data['z_score'].iloc[i], marker='^', color='r', label='Buy Signal' if 'Buy Signal' not in ax2.get_legend_handles_labels()[1] else "")
#                 elif stock_data['signal'].iloc[i] == -1:  # 空頭信號
#                     ax2.scatter(stock_data.index[i], stock_data['z_score'].iloc[i], marker='v', color='g', label='Sell Signal' if 'Sell Signal' not in ax2.get_legend_handles_labels()[1] else "")

#             # 添加標題
#             plt.title('Z-Score and Rolling Mean with Buy/Sell Signals')

#             # 顯示圖例
#             fig.tight_layout()  # 確保布局不重疊
#             ax1.legend(loc='upper left')
#             ax2.legend(loc='upper right')

#             # 顯示圖形
#             plt.grid()
#             plt.show()
        
#         plt.figure(figsize=(12, 6))

#          # 绘制实际数据
#         plt.plot(stock_data.index, stock_data['close_stock1'], label='Actual Data', color='blue')

#         # 绘制进出场信号
#         plt.scatter(stock_data.index[buy_signals], stock_data['close_stock1'].iloc[buy_signals], 
#                 marker='^', color='red', label='Buy Signal', s=100)  # 进场信号

#         plt.scatter(stock_data.index[sell_signals], stock_data['close_stock1'].iloc[sell_signals], 
#                 marker='v', color='green', label='Sell Signal', s=100)  # 出场信号

#         plt.title('Actual Data with Trading Signals')
#         plt.xlabel('Time')
#         plt.ylabel('Price')
#         plt.legend()
#         plt.grid()
#         plt.show()

# mean_backtest(mer_ori_data, '2024-09-19', '2024-09-16', plot=True, mean_plot=True)
