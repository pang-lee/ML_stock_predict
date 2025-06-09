# coding: utf-8
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
load_dotenv()
import os
import shioaji as sj
from fugle_marketdata import WebSocketClient, RestClient
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from statsmodels.tsa.stattools import coint, adfuller
import numpy as np

# https://ithelp.ithome.com.tw/articles/10280898 教學參考
def get_shioaji_index_data(begin, end, id, type='1k'):
    # 將 begin 和 end 轉換為所需的格式 2024_0708
    begin_month = begin[:7].replace('-', '')  # 取得2024-07部分，變成202407
    end_month = end[:7].replace('-', '')      # 取得2024-08部分，變成202408
    folder_name = f"{begin[:4]}_{begin_month[4:6]}{end_month[4:6]}"  # 2024_0708
    
    # 定義路徑
    folder_path = f"./index_data/shioaji/{folder_name}/"
    
    # 如果資料夾不存在，則創建
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    # 定義CSV檔案的名稱
    file_name = f"{id}.csv"
    file_path = os.path.join(folder_path, file_name)
    
    api = sj.Shioaji()
    api.login(
        api_key=os.getenv('API_KEY'),
        secret_key=os.getenv('SECRET_KEY'),
        fetch_contract=False,
    )
    api.fetch_contracts(contract_download=True)
    #如果要查看合約代號可以直接print(api.Contracts), 查看特定的股票可以print(api.Contracts.Stocks.TSE['2330'])
    
    # 創建一個空的 DataFrame 來存儲所有數據
    all_ticks = pd.DataFrame()

    if type == '1k':
        print('Remain usage API usage:', api.usage())
            
        # 調用 API 獲取當天的數據
        kbars = api.kbars(
            contract=api.Contracts.Stocks.TSE[id],
            start=begin,
            end=end,
        )
            
        df = pd.DataFrame({**kbars})
        all_ticks = pd.concat([all_ticks, df], ignore_index=True)
        
        print(f"Processed data from {begin} to {end}")
        
        all_ticks['ts'] = pd.to_datetime(all_ticks['ts'])

        all_ticks.to_csv(file_path, index=False)
        print(f'Data saved to {file_path}')
        
    elif type == 'tick':
        # 將 begin 和 end 轉換為 datetime 對象
        start_date = datetime.strptime(begin, "%Y-%m-%d")
        end_date = datetime.strptime(end, "%Y-%m-%d")
    
        # 循環日期範圍
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            print(f"Fetching data for {date_str}..., Remain usage API usage: {api.usage()}")
            
            try:
                # 調用 API 獲取當日數據
                ticks = api.ticks(
                    contract=api.Contracts.Stocks[id],
                    date=date_str
                )
                df = pd.DataFrame({**ticks})
            
                if not df.empty:
                    # 合併當天數據到總表
                    all_ticks = pd.concat([all_ticks, df], ignore_index=True)
                    print(f"Data for {date_str} processed.")
                else:
                    print(f"No data available for {date_str}.")
        
            except Exception as e:
                print(f"Error fetching data for {date_str}: {e}")
        
            # 日期增加一天
            current_date += timedelta(days=1)   
        
        all_ticks['ts'] = pd.to_datetime(all_ticks['ts'])
        all_ticks.to_csv(file_path, index=False)
        print(f"Data saved to {file_path}")
        
    elif type == 'future-k':
        # 循環日期範圍
        print(f"Fetching data for {begin}, {end}..., Remain usage API usage: {api.usage()}")
        
        try:
            # 調用 API 獲取當日數據
            ticks = api.kbars(
                contract=api.Contracts.Futures.TXF[id],
                start=begin,
                end=end
            )
            df = pd.DataFrame({**ticks})
            
            if not df.empty:
                # 合併當天數據到總表
                all_ticks = pd.concat([all_ticks, df], ignore_index=True)
                print(f"Data for {begin}, {end} processed.")
            else:
                print(f"No data available for {begin}, {end}.")
        
        except Exception as e:
            print(f"Error fetching data for {begin}, {end}: {e}")
        
        all_ticks['ts'] = pd.to_datetime(all_ticks['ts'])
        all_ticks.to_csv(file_path, index=False)
        print(f"Data saved to {file_path}")
   
    elif type == 'future-t':
        # 將 begin 和 end 轉換為 datetime 對象
        start_date = datetime.strptime(begin, "%Y-%m-%d")
        end_date = datetime.strptime(end, "%Y-%m-%d")
    
        # 循環日期範圍
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            print(f"Fetching data for {date_str}..., Remain usage API usage: {api.usage()}")
            
            try:
                # 調用 API 獲取當日數據
                ticks = api.ticks(
                    contract=api.Contracts.Futures[id],
                    date=date_str
                )
                df = pd.DataFrame({**ticks})
            
                if not df.empty:
                    # 合併當天數據到總表
                    all_ticks = pd.concat([all_ticks, df], ignore_index=True)
                    print(f"Data for {date_str} processed.")
                else:
                    print(f"No data available for {date_str}.")
        
            except Exception as e:
                print(f"Error fetching data for {date_str}: {e}")
        
            # 日期增加一天
            current_date += timedelta(days=1)
        
        all_ticks['ts'] = pd.to_datetime(all_ticks['ts'])
        all_ticks.to_csv(file_path, index=False)
        print(f"Data saved to {file_path}")

    elif type == 'index':
        # 設置開始日期和結束日期
        start_date, end_date = datetime.strptime(begin, "%Y-%m-%d"), datetime.strptime(end, "%Y-%m-%d")
        
        # 遍歷日期範圍，按天調用 API
        current_date = start_date
        
        while current_date <= end_date:
            print('Remain usage API usage:', api.usage())

            # 格式化日期為字符串
            date_str = current_date.strftime("%Y-%m-%d")

            # 調用 API 獲取當天的數據 TSE001(大盤), TSE099(台灣50)
            ticks = api.ticks(
                contract=api.Contracts.Indexs.TSE.TSE001,
                date=date_str
            )

            df = pd.DataFrame({**ticks})
            all_ticks = pd.concat([all_ticks, df], ignore_index=True)

            print(f"Processed data for {date_str}")

            current_date += timedelta(days=1)

        all_ticks = all_ticks[['ts', 'close']]
        all_ticks['ts'] = pd.to_datetime(all_ticks['ts'])

        all_ticks.to_csv(file_path, index=False)
        print(f'Data saved to {file_path}')

# get_shioaji_index_data('2024-12-02', '2025-01-02', '2353', type='tick')
# get_shioaji_index_data('2024-06-01', '2025-01-22', '00631L', type='tick')
# get_shioaji_index_data('2024-06-01', '2025-01-22', 'TXFR1', type='future-k')
# get_shioaji_index_data('2025-04-01', '2025-04-11', 'MXFR1', type='future-t')
get_shioaji_index_data('2025-04-21', '2025-04-28', 'TMFR1', type='future-t')


#使用Fugle API獲得資料
def get_fugle_data(stock_list, data_folder):
    folder = f'./index_data/fugle/{data_folder}'

    client = RestClient(api_key = os.getenv('FUGLE'))
    
    # 遍历股票代码列表
    for stock_code in stock_list:
        stock = client.stock.historical.candles(**{"symbol": f'{stock_code}',  "timeframe": 1})
    
        if 'symbol' not in stock:
            raise ValueError(stock['message'])

        data = pd.DataFrame(stock['data'])
        data.rename(columns={'date': 'datetime'}, inplace=True)
        data['datetime'] = pd.to_datetime(data['datetime']).dt.strftime('%Y-%m-%d %H:%M:%S')
        data.set_index('datetime', inplace=True)
        reversed_df = data.iloc[::-1]
        reversed_df.to_csv(f'{folder}/{data_folder}/{stock_code}.csv')
        
    print(f'Data All safet to {folder}/{data_folder}')

# 獲取ETF
stock_list = ["00631L", "00632R", "00663L", "00664R", "00675L", "00676R", "00685L", "00686R", 
              "00633L", "00634R", "00637L", "00638R", "00655L", "00656R", "00753L", "00650L",
              "00651R", "00665L", "00666R", "00640L", "00641R", "00654R", "00647L", "00648R",
              "00852L", "00669R", "00670L", "00671R", "00683L", "00684R", "00706L", "00707R",
              "00708L", "00674R", "00673R", "00715L", "00680L", "00681R", "00688L", "00689R",
              "2330"
            ]


#每個月要更換一次月份代號(2024_0708 => 2024_0809)
# get_fugle_data(stock_list=stock_list, data_folder='2024_0809')


# 獲取熱力圖
def get_stock_data(stock_list, start, end):
    data = {}
    for stock in stock_list:
        print(f"Attempting to fetch {stock} (TW)")
        stock_data = yf.download(stock, start=start, end=end, auto_adjust=True)[['Close', 'Volume']]
        
        if stock_data.empty:
            print(f"Data for {stock} (TW) is empty, retrying with .TWO")
            new_stock = stock.replace('.TW', '.TWO')
            stock_data = yf.download(new_stock, start=start, end=end, auto_adjust=True)[['Close', 'Volume']]
            
            if stock_data.empty:
                print(f"Failed to fetch data for {new_stock} (TWO), skipping.")
                continue
            
        data[stock] = stock_data
    return data

def get_heatmap(html, stock_ids, start, end, type=1, idx_start=0, idx_end=-1):
    if html and not stock_ids:
        soup = BeautifulSoup(html, 'html.parser')
        if type == 1:  # 獲取一般的選股網頁
            stock_ids = [span.text for span in soup.find_all('span', {'c-model': 'id'})]
        elif type == 2:  # 獲取成交值網頁
            a_tags = soup.find_all('a')
            stock_ids = [a['href'].split('/')[-1] for a in a_tags if 'href' in a.attrs and a['href'].startswith('/stock/')]

        if idx_end == -1 or idx_end > len(stock_ids):
            idx_end = len(stock_ids)
        stock_ids = stock_ids[idx_start:idx_end]

    stock_list = [stock_id + '.TW' for stock_id in stock_ids]
    # 使用get_stock_data函數下載數據（包含收盤價與成交量）
    raw_data = get_stock_data(stock_list, start, end)
    
    # 將收盤價與成交量拆成兩個 DataFrame
    close_data = {k: v['Close'] for k, v in raw_data.items()}
    
    # 移除缺失數據
    close_data = {k: v for k, v in close_data.items() if not v.isnull().all().all()}
    close_data = {k: v.squeeze() for k, v in close_data.items()}

    if not close_data:
        print("No valid stock data found.")
        return

    # 將收盤價轉換為 DataFrame
    close_data = pd.DataFrame(close_data)

    # 初始化協整矩陣與 ADF 平穩性結果
    num_stocks = len(close_data.columns)
    coint_matrix = np.ones((num_stocks, num_stocks))
    adf_results = {}
    mean_reverting_stocks = []
    cointegrated_pairs = []

    # 進行ADF檢驗
    for stock in close_data.columns:
        adf_pvalue = adfuller(close_data[stock].dropna())[1]
        adf_results[stock] = adf_pvalue
        if adf_pvalue <= 0.05:
            mean_reverting_stocks.append(stock)

    # 計算協整檢定 (僅在 ADF 不顯著時進行)
    for i in range(num_stocks):
        for j in range(i + 1, num_stocks):
            stock1, stock2 = close_data.columns[i], close_data.columns[j]
            stock1_series = close_data[stock1].dropna()
            stock2_series = close_data[stock2].dropna()
            
            # 如果任一股票平穩，則跳過協整檢定
            if adf_results[stock1] <= 0.05 or adf_results[stock2] <= 0.05:
                continue

            # 協整檢定
            common_index = stock1_series.index.intersection(stock2_series.index)
            if len(common_index) > 1:
                score, p_value, _ = coint(stock1_series[common_index], stock2_series[common_index])
                coint_matrix[i, j] = p_value
                coint_matrix[j, i] = p_value

                # 如果協整檢定通過，並且都是非平穩的，記錄下統計套利的標的對
                if p_value <= 0.05:
                    cointegrated_pairs.append((stock1, stock2, p_value))

    # 將協整結果轉為 DataFrame 並過濾
    coint_df = pd.DataFrame(coint_matrix, index=close_data.columns, columns=close_data.columns)
    coint_df = coint_df.where(coint_df <= 0.05)  # 只保留通過檢定的協整關係

    # 可視化熱力圖
    plt.figure(figsize=(10, 6))
    sns.heatmap(coint_df, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Cointegration Heatmap (p-value)')
    plt.xticks(rotation=45)
    plt.show()

    # 在這裡將成交量、收盤價（最後一筆）與平均成交量一併列印
    print("\n====== ✅ 可以單獨使用均值回歸交易的標的 (平穩序列) ✅ ======")
    for stock in mean_reverting_stocks:
        last_close = close_data[stock].iloc[-1]  # 最後一筆收盤價
        print(f"{stock} (p-value={adf_results[stock]:.4f}, Last Close={last_close:.2f})")

    print("\n====== 📈 適合統計套利的標的對 (非平穩但協整) 📈 ======")
    for pair in cointegrated_pairs:
        stock1, stock2, p_value = pair
        last_close1 = close_data[pair[0]].iloc[-1]
        last_close2 = close_data[pair[1]].iloc[-1]
        print(f"{stock1} (Last Close={last_close1:.2f}) & {stock2} (Last Close={last_close2:.2f}), Cointegration p-value={p_value:.4f}")
        print('------------------')

    # 返回篩選結果以供進一步分析
    return {
        "mean_reverting_stocks": mean_reverting_stocks,
        "cointegrated_pairs": cointegrated_pairs
    }

# 到玩股網中 https://www.wantgoo.com/index/overview , 挑選選股清單後, 複製main > container > table (外部複製)
h = '''



'''

stock_list = []

# 取table的所有資料
# type 1: 一般類股網頁(https://www.wantgoo.com/index/overview#index-7)
# type 2: 有單獨的代號的網頁(https://www.wantgoo.com/stock/ranking/turnover)
# get_heatmap(h, stock_list, '2024-01-01', '2025-03-11', type=2, idx_start=120, idx_end=160)
