

import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
import pandas as pd
import numpy as np
import requests
import time, json
from tqdm import tqdm
# BASE_URL_V3 = "https://api.binance.cc/api/v3"
BASE_URL_V3 = "https://api.binance.com/api/v3"
import math

# 爬k线价格
def get_k_data(coin_type, interval, limit_nums=1000):
    url = f"{BASE_URL_V3}/klines?symbol={coin_type.upper()}&interval={interval}&limit={limit_nums}"
    return requests.get(url, timeout=5, verify=True).json()

def get_history_k_data(coin_type, interval =None, start_time='2017-01-01', end_time='2020-01-01'):
    df = pd.read_hdf('D:\\Q\\xbx-coin-2020_part3\\data\\%s_5m.h5' % coin_type, key='df')
    # 任何原始数据读入都进行一下排序、去重，以防万一
    df.sort_values(by=['candle_begin_time'], inplace=True)
    df.drop_duplicates(subset=['candle_begin_time'], inplace=True)
    df.reset_index(inplace=True, drop=True)

    # =====转换为其他分钟数据
    rule_type = '15T'
    # period_df = df.resample(rule=rule_type, on='candle_begin_time', label='left', closed='left').agg(
    #     {'open': 'first',
    #      'high': 'max',
    #      'low': 'min',
    #      'close': 'last',
    #      'volume': 'sum',
    #      'quote_volume': 'sum',
    #      })
    period_df = df
    period_df.dropna(subset=['open'], inplace=True)  # 去除一天都没有交易的周期
    period_df = period_df[period_df['volume'] > 0]  # 去除成交量为0的交易周期
    period_df.reset_index(inplace=True)
    df = period_df[['candle_begin_time', 'open', 'high', 'low', 'close', 'volume', ]]
    df = df[df['candle_begin_time'] >= pd.to_datetime(start_time)] #'2017-01-01'
    df = df[df['candle_begin_time'] <= pd.to_datetime(end_time)]  # '2017-01-01'
    df.reset_index(inplace=True, drop=True)
    df = df[['open', 'high', 'low', 'close']]
    result = df.values.tolist()
    # print(result)
    return result

class BackTest:
    def __init__(self, coin_type, my_T, M, money=10000):
        self.coin_type = coin_type.upper()
        self.T = my_T
        self.M = M
        self.num = len(M)
        self.result = {'币种': coin_type}
        self.ini_money = money

    # 回测
    def back_cal(self, base_prob=0.35,  low_line = 30,row_prob = 0.025):
        """

        :param low_limit:
        :param high_limit:
        :param low_line: 底价
        :param row_prob:  每格利润比例
        :return:
        """
        # 计算收益/回撤
        if 'm' == self.T[-1].lower():
            t = int(self.T[:-1])
        elif 'h' == self.T[-1].lower():
            t = int(self.T[:-1]) * 60
        elif 'd' == self.T[-1].lower():
            t = int(self.T[:-1]) * 60 * 24
        elif 'w' == self.T[-1].lower():
            t = int(self.T[:-1]) * 60 * 24 * 7
        else:
            t = int(self.T[:-1]) * 60 * 24 * 30

        # --------------------------------现货---------------------------
        def xianhuo(base_prob, low_line, row_prob):
            """

            :param a: 底仓的比例
            :return:
            """
            # 无限网格，保持持仓部分总值不变
            #
            dd_qty = dd_price = 6
            price = [i[3] for i in self.M] # 收盘价 [开盘价,最高价,最低价,收盘价]
            init_money = self.ini_money
            # 买币资金
            buy_money = round(init_money * base_prob, dd_price)
            # 初始币量
            base_coins = round(buy_money / price[0] / (1 + 0.001), dd_qty)  # 8.12
            coins = base_coins
            #剩余资金
            invest_money = init_money - buy_money
            # 计算下线有几格
            row_num = math.log(( low_line / price[0]),1-row_prob)
            # price[0]  *  (1-row_prob )^ row_num = low_line
            # 计算买入格子价格线
            # buy_lines = [price[0]* (1-row_prob)**i for i in range(1, row_num)]
            # 套利次数
            taoli_list = []

            # 计算卖出格子线

            # 买入一格的钱
            row_money = invest_money / row_num

            #base_value = price[0] * coins  # 底仓价值，以此为基准加减仓 615*8.12 =4995

            bot_money_ls = [invest_money]  # 可用+币值变动列表 #[10000]
            # 先买a比例的底仓
            #money = round(money * (1 - a), dd_price)  # 可用资金剩余
            #row_price = (price[0] - low_line) * row_prob # 每格利润


            break_idx_buy, break_idx_sell = [], []
            currect_line = price[0] # 当前价格基准线
            buy_line = currect_line * (1-row_prob) # 买入线
            #  sell_line =  currect_line * (1+row_prob)
            sell_line_list = []
            for i in range(1, self.num):
                if price[i] < buy_line:  # 低于基准，买
                    if  price[i] >= low_line:
                        break_idx_buy.append(i)
                        # 更新买卖线和基准线
                        currect_line = buy_line
                        buy_line = currect_line * (1 - row_prob)
                        sell_line = currect_line * (1 + row_prob)
                        sell_line_list.append(sell_line)
                        sell_line_list.sort()
                        invest_money -= round(row_money, dd_price)
                        coins += round(round( row_money, dd_price) / price[i] , dd_qty)
                        # (1 + 0.001) 手续费
                elif len(sell_line_list) != 0 and price[i] > sell_line_list[0]:  # 高于基准，卖
                    sell_id = 0
                    for  sell_p in  sell_line_list:
                        if coins > base_coins and price[i] > sell_p:
                            # 更新买卖线和基准线
                            sell_line = sell_p
                            currect_line = sell_line
                            buy_line = currect_line * (1 - row_prob)


                            break_idx_sell.append(i)
                            invest_money += round(row_money * (1 + row_prob), dd_price)
                            coins -= round(round(row_money * (1 + row_prob), dd_price) / price[i] / (1 + 0.001),
                                               dd_qty)
                            sell_id += 1
                            taoli_list.append(round(row_money * ( row_prob), dd_price))
                        else:
                            break
                    sell_line_list = sell_line_list[sell_id:]

                bot_money_ls.append(invest_money + coins * price[i])
            coin_last_profit = coins * price[-1] - base_coins * price[0]
            return [break_idx_buy, break_idx_sell], coin_last_profit, taoli_list, bot_money_ls

        max_rate = -100
        max_arg = []
        # 循环只是找到最优底仓仓位而已，自定义底仓就不用循环了

        res = xianhuo(base_prob, low_line, row_prob)

        # res = max_arg[-1]
        #self.result['总收益率(%)'] = 100 * (res[-1][-1] / self.ini_money - 1)
        self.result['网格收益率(%)'] = 100 * ((sum(res[-2]) + self.ini_money) / self.ini_money - 1)
        self.result['套利次数'] = len(res[-2])
        self.result['网格日化(%)'] = round(
            100 * (10 ** (np.log10(self.result['网格收益率(%)'] / 100 + 1) / (t * self.num / 24 / 60)) - 1), 2)
        self.result['网格年化(%)'] = round(self.result['网格日化(%)'] * 365, 2)
        self.result['策略总收益率(%)'] = 100 * ((sum(res[-2]) + res[-3] + self.ini_money) / self.ini_money - 1)
        self.result['策略日化(%)'] = round(
            100 * (10 ** (np.log10(self.result['策略总收益率(%)'] / 100 + 1) / (t * self.num / 24 / 60)) - 1), 2)
        self.result['策略年化(%)'] = round(self.result['策略日化(%)'] * 365, 2)

        self.result['现货日化(%)'] = round(
            100 * (10 ** (np.log10(self.M[-1][3] / self.M[0][3]) / (t * self.num / 24 / 60)) - 1), 2)
        self.result['现货年化(%)'] = round(self.result['现货日化(%)'] * 365, 2)
        max_back = 0
        max_back_cash = 0
        for i in tqdm(range(len(res[-1]) - 2)):
            if res[-1][i] > res[-1][i + 1]:
                max_back = max(1 - min([j for j in res[-1][i + 1:-1]]) / res[-1][i], max_back)
            if self.M[i][-1] > self.M[i + 1][-1]:
                max_back_cash = max(1 - min([j[-1] for j in self.M[i + 1:-1]]) / self.M[i][-1], max_back_cash)
        self.result['策略最大回撤(%)'] = round(100 * max_back, 2)
        # max_back_cash = 0
        # {'币种': 'DOGE-USDT', '网格收益率(%)': 12.220878402903734, '套利次数': 480, '网格日化(%)': 0.24, '网格年化(%)': 87.6, '策略总收益率(%)': 21.277700524700506, '策略日化(%)': 0.4, '策略年化(%)': 146.0, '现货日化(%)': -1.95, '现货年化(%)': -711.75, '策略最大回撤(%)': 39.59, '现货最大回撤(%)': 77.66, '底仓比例': 0.35, '收益波动值': 983.3, '夏普率': 0.07}

        # {'币种': 'BTC-USDT', '策略收益率(%)': -6.284980536480145, '策略日化(%)': -0.07, '现货日化(%)': -0.3, '策略最大回撤(%)': 25.49, '现货最大回撤(%)': 47.27, '底仓比例': 0.5, '收益波动值': 10053.31, '夏普率': -0.0}
        # for i in range(self.num - 2):
        #     if self.M[i][-1] > self.M[i + 1][-1]:
        #         max_back_cash = max(1 - min([j[-1] for j in self.M[i + 1:-1]]) / self.M[i][-1], max_back_cash)
        self.result['现货最大回撤(%)'] = round(100 * max_back_cash, 2)
        self.result['底仓比例'] = base_prob
        # n期收益率均值
        miu = sum([res[-1][i + 1] / res[-1][i] - 1 for i in range(len(res[-1]) - 1)]) / (len(res[-1]) - 1)
        # n期收益率方差
        theta = (sum([(i - miu) ** 2 for i in res[-1][1:]]) / (len(res[-1]) - 1)) ** 0.5
        self.result['收益波动值'] = round(theta, 2)
        try:
            self.result['夏普率'] = round(100000 * 100 * miu / theta, 2)
        except ZeroDivisionError:
            self.result['夏普率'] = 0
        return res

    def draw_fig(self, point_ls, money_ls):
        plt.subplot(211)
        plt.plot([i[3] for i in self.M], color='grey', label='时间粒度=%s' % self.T)
        plt.scatter(point_ls[0], [self.M[i][3] for i in point_ls[0]], c='green', label='买点')#,s=0.5)
        plt.scatter(point_ls[1], [self.M[i][3] for i in point_ls[1]], c='red', label='卖点')#,s=0.5)
        plt.legend()
        plt.subplot(212)
        plt.plot([i for i in range(len(money_ls))], [i / self.ini_money - 1 for i in money_ls], label='无限网格')
        plt.plot([self.M[i][3] / self.M[0][3] - 1 for i in range(self.num)], label='拿现货')
        plt.ylabel('倍数')
        plt.xlim(0, self.num)
        plt.legend()
        plt.suptitle('币种：%s，最大回撤=%.2f%%，夏普率=%.2f' % (self.coin_type, self.result['策略最大回撤(%)'], self.result['夏普率']))
        plt.show()


if __name__ == '__main__':
    for coin in ['ETH-USDT']:#['DOGE-USDT']: #, 'ETH-USDT'
    #for coin in ['DOGE-USDT']:
        while 1:
            #try:
            #s = get_k_data(coin, '15m', 1500)
            # M_15 = [list(map(eval, i[1:5])) for i in get_history_k_data(coin)]  # [开盘价,最高价,最低价,收盘价]
            M_15 =  get_history_k_data(coin,None,'2020-05-08','2021-06-25')  # [开盘价,最高价,最低价,收盘价]
            bt = BackTest(coin, '5m', M_15, 1102)  # 本金为可选参数，默认10000，
            p_ls, c_pro, b_ls,m_ls = bt.back_cal(0.35, 0.136, 0.025) # 0.05 网格年化(%)': 7.3 0.15 网格年化(%)': 10.95
            # p_ls, c_pro, b_ls, m_ls = bt.back_cal(0.35, 1000, 0.025)  # 0.05 网格年化(%)': 7.3 0.15 网格年化(%)': 10.95
            print(bt.result)
            bt.draw_fig(p_ls, m_ls)
            time.sleep(0.5)
            break
            # ETH-USDT'2020-01-01','2021-01-01'
            # {'币种': 'ETH-USDT', '策略收益率(%)': 109.57499531703854, '策略日化(%)': 0.2, '现货日化(%)': 0.48, '策略最大回撤(%)': 39.01, '现货最大回撤(%)': 69.32, '底仓比例': 0.5, '收益波动值': 15364.2, '夏普率': 0.01}
            # except Exception as e:
            #     print(e)
            #     time.sleep(1)
