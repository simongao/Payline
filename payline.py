#!/usr/env/python python
# _*_ coding: utf-8 _*_

import argparse
import tushare as ts
from datetime import datetime
import numpy as np
import pandas as pd
import operator
import matplotlib.pyplot as plt


# 股票代码归类
def asset_category(code):
    category = 'UNKNOWN'
    if (len(code) == 6): category = 'A'
    if (len(code) == 5): category = 'HK'
    if (len(code) == 6 and (code.startswith('43') or code.startswith('83'))): category = 'XSB'
    if (len(code) == 6 and (code.startswith('15') or code.startswith('12') or code.startswith('18') or code.startswith(
        '11'))): category = 'FUND_TRADED'
    if (len(code) == 6 and (code.startswith('16'))): category = 'FUND_UNTRADED'
    if (len(code) == 6 and (code.startswith('204') or code.startswith('131'))): category = 'REPO'
    if (len(code) == 6 and (code.startswith('7') or code.startswith('07'))): category = 'IPO'
    if (code == '000300' or code == '399300'): category = 'INDEX'
    return category

# 查询股票价格
def get_close_price(code, start, end=datetime.today(), freq='D'):
    if (asset_category(code) == 'FUND_TRADED' or asset_category(code) == 'FUND_UNTRADED'):
        try:
            f = fp_prefix + code + '.csv'
            if (asset_category(code) == 'FUND_TRADED'): col_name = '收盘价'
            if (asset_category(code) == 'FUND_UNTRADED'): col_name = '净值'
            close_price = pd.read_csv(f, header=0, sep=',',
                                      converters={'交易时间': lambda x: datetime.strptime(x, '%Y-%m-%d')}, index_col='交易时间',
                                      usecols=['交易时间', col_name])
        except:
            print('Cannot find {}.\n'.format(f))
    else:
        if (asset_category(code) == 'A'): asset = 'E'
        if (asset_category(code) == 'HK'): asset = 'X'
        if (asset_category(code) == 'XSB'): asset = 'X'
        if (asset_category(code) == 'INDEX'): asset = 'INDEX'
        try:
            cons = ts.get_apis()
            s = ts.bar(code, conn=cons, freq='D', start_date=start, end_date=end, asset=asset)
            close_price = s['close']
        except:
            print('Error when get price of {}.\n'.format(code))

    close_price = close_price.sort_index(ascending='ascending')

    if (freq == 'Y' or freq == 'END'):
        close_price = close_price.iloc[[0, -1]]

    return close_price

# 如果是流动性较差的新三板股票，为防止出席期初无交易价格的情况，用后续第一个交易价格填补
def backfill_xsb(c):
    if (asset_category(c.name)=='XSB'):
        first_nonzero_loc = c.nonzero()[0][0]
        first_nonzero_value = c[first_nonzero_loc]
        c.ix[0:first_nonzero_loc] = first_nonzero_value
    return c


# 计算最大回撤
def drawdown(timeseries):
    # 回撤结束时间点
    i = np.argmax(np.maximum.accumulate(timeseries) - timeseries)
    # 回撤开始的时间点
    j = np.argmax(timeseries[:i])
    return (float(timeseries[i]) / timeseries[j]) - 1.0


# 计算夏普比率 (日均收益率*250-3%) / (日均波动率*squareroot(250))
def sharpe(rets, rf=0.0, ann=252):
    return (rets.mean()*ann - rf) / (rets.std() * np.sqrt(ann))


# 计算回报率
def total_returns(s):
    return s.iloc[-1] / s.iloc[0] - 1.0


# 计算收益率曲线和最终数值
def calc_return(ts):
    ret = ts / ts.shift(1)
    ret = ret.replace([np.inf, -np.inf], np.nan).fillna(1.0)
    ret = ret.cumprod()
    ret_value = ret[-1] - 1.0
    return [ret, ret_value]


# 定义xirr函数
def xirr(transactions):
    years = [(ta[0] - transactions[0][0]).days / 365.0 for ta in transactions]
    residual = 1
    step = 0.05
    guess = 0.05
    epsilon = 0.0001
    limit = 10000
    while abs(residual) > epsilon and limit > 0:
        limit -= 1
        residual = 0.0
        for i, ta in enumerate(transactions):
            residual += ta[1] / pow(guess, years[i])
        if abs(residual) > epsilon:
            if residual > 0:
                guess += step
            else:
                guess -= step
                step /= 2.0
    return guess - 1

# 业务名称字典以适应各个券商的不同业务名称
name_map = {'中信证券' : \
                {'融资债务项目' : ['融资借款', '卖券偿还融资负债', '直接偿还融资负债'], \
                '现金存取项目' : ['银行转存','银行转取','质押回购拆出','拆出质押购回'], \
                '交易记录项目' : ['证券买入','证券卖出','融资买入','还款卖出','红股入帐','ETF现金替代退款','基金合并','基金分拆','货币ETF现金替代划入','托管转入','托管转出','开放基金赎回','开放基金赎回返款','担保品划出','担保品划入','ETF赎回基金过户'], \
                '股息红利项目' : ['股息入帐','利息归本','股息红利税补缴'], \
                '融资利息项目' : ['卖券偿还融资利息','卖券偿还融资费用','直接偿还融资费用'], \
                '新股申购项目' : ['新股申购', '申购返款'], \
                '基金赎回项目' : ['开放基金赎回'], \
                '港股通收费项目' : ['港股通组合费收取'], \
                '发生金额' : '发生金额', \
                '证券代码' : '证券代码', \
                '发生日期' : '发生日期', \
                '证券名称' : '证券名称', \
                '股东代码' : '股东代码', \
                '成交价格' : '成交价格', \
                '成交数量' : '成交数量', \
                '成交金额' : '成交金额', \
                '股份余额' : '股份余额', \
                '资金帐号' : '资金帐号', \
                '手续费' : '手续费', \
                '印花税' : '印花税', \
                '过户费' : '过户费', \
                '交易所清算费' : '交易所清算费', \
                '资金本次余额' : '资金本次余额', \
                '反号业务项目' : [], \
                'date_format' : "%Y%m%d"}, \
            '平安证券' : \
                {'融资债务项目' : ['融资借款', '卖券偿还融资负债', '直接偿还融资负债'], \
                '现金存取项目' : ['银证转入','银证转出','OTC业务资金上账','OTC业务资金下账','基金申购','基金赎回','质押回购拆出','拆出质押购回'], \
                '交易记录项目' : ['证券买入清算','证券卖出清算','融资买入','还款卖出','红股入帐','ETF赎回现金替代','基金合并','基金分拆','ETF现金退补','托管转入','托管转出','基金赎回清算','担保品划出','担保品划入','ETF赎回基金过户'], \
                '股息红利项目' : ['红利发放','红股派息','深圳市场股息红利个人所得税扣款','上海市场股息红利个人所得税扣款'], \
                '融资利息项目' : ['卖券偿还融资利息','卖券偿还融资费用','直接偿还融资费用'], \
                '新股申购项目' : ['新股申购', '申购返款'], \
                '基金赎回项目' : [], \
                '港股通收费项目' : ['深港通组合费'], \
                '发生金额' : '发生金额', \
                '证券代码' : '证券代码', \
                '发生日期' : '发生日期', \
                '证券名称' : '证券名称', \
                '股东代码' : '股东代码', \
                '成交价格' : '成交均价', \
                '成交数量' : '成交数量', \
                '成交金额' : '成交金额', \
                '股份余额' : '股份余额', \
                '资金帐号' : '资金帐号', \
                '手续费' : '手续费', \
                '印花税' : '印花税', \
                '过户费' : '过户费', \
                '交易所清算费' : '其他费', \
                '资金本次余额' : '资金余额', \
                '反号业务项目' : ['证券卖出清算','基金赎回清算'], \
                'date_format' : "%Y-%m-%d" }
            }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="payline.py", fromfile_prefix_chars='@')

    parser.add_argument('--start', action='store', type=(lambda s: datetime.strptime(s, '%Y-%m-%d')),
                        help='start date for example 2017-01-01')
    parser.add_argument('--end', action='store', type=(lambda s: datetime.strptime(s, '%Y-%m-%d')),
                        help='end date for example 2017-12-31')
    # parser.add_argument('--year', action='store', type=int, help='year')
    parser.add_argument('--local', action='store_true', default=False, help='read stock price data from local storage')
    parser.add_argument('--fp_prefix', action='store', default='./', help='file path prefix')
    parser.add_argument('--broker', action='store', default='中信证券', help='broker')

    args = parser.parse_args()

    start_date = args.start
    end_date = args.end
    LOAD_FROM_LOCAL = args.local
    fp_prefix = args.fp_prefix
    broker = name_map[args.broker]
    duration = start_date.strftime('%Y%m%d') + '_' + end_date.strftime('%Y%m%d')

    #fp = fp_prefix + 'Log_' + duration + '.csv'
    #log = pd.read_csv(fp, header=0, sep=',', converters={broker['发生日期'] : lambda x:datetime.strptime(x,broker['date_format']), broker['证券代码'] : lambda x:str(x)})
    fp = fp_prefix + 'Log_' + duration + '.xlsx'
    log = pd.read_excel(fp, header=0, converters={broker['发生日期'] : lambda x:datetime.strptime(str(x),broker['date_format']), broker['证券代码'] : lambda x:str(x)})

    # 计算红利及利息
    dividend = log[log.业务名称.isin(broker['股息红利项目'])].发生金额.sum()
    interest = log[log.业务名称.isin(broker['融资利息项目'])].发生金额.sum()

    # 计算现金头寸
    cash_activities = log[log.业务名称.isin(broker['现金存取项目'])]
    cash_in = cash_activities.发生金额.sum()
    cash_begin = log[broker['资金本次余额']][0] - log[broker['发生金额']][0]
    cash_end = log[broker['资金本次余额']].iloc[-1]

    cash_report = cash_activities.groupby([broker['发生日期']]).agg({broker['发生金额']:sum})
    cash_report = cash_report[cash_report.发生金额 != 0]

    # 提取证券交易数据
    # 剔除新股以及无证券代码的记录
    trades = log[log.业务名称.isin(broker['交易记录项目'])]
    trades = trades[trades[broker['证券代码']] != '']
    trades.资产类别 = trades[broker['证券代码']].map(asset_category)
    trades = trades[trades.资产类别 != 'IPO']
    sign = trades.业务名称.isin(broker['反号业务项目']).map({True:-1, False:1})
    trades[broker['成交数量']] = trades[broker['成交数量']].mul(sign)

    # 提取证券名称
    stock_name = trades.groupby(broker['证券代码'])[broker['证券名称']].unique()
    stock_name = stock_name.map(lambda x: [a for a in x if str(a) != 'nan'])
    stock_name = stock_name.map(lambda x: ''.join(x))

    # 计算交易金额
    stock_transaction = trades.groupby([broker['股东代码'],broker['证券代码']]).agg({broker['发生金额']:sum})

    # 判断是否为港股
    trades.ishk = (trades.证券代码.str.len() == 5)
    trades_hk = trades[trades.ishk == True]
    trades_nonhk = trades[trades.ishk == False]

    # 计算期初期末股票持仓量
    stock_begin_nonhk = trades_nonhk.groupby([broker['股东代码'],broker['证券代码']]).first()[broker['股份余额']] - trades_nonhk.groupby([broker['股东代码'],broker['证券代码']]).first()['成交数量']
    grouped_trades_hk = trades_hk.groupby([broker['股东代码'],broker['证券代码'],broker['发生日期']]).agg({broker['股份余额']:'first', broker['成交数量']:sum})
    grouped_trades_hk = grouped_trades_hk.groupby(level=[broker['股东代码'],broker['证券代码']]).agg({broker['股份余额']:'first', broker['成交数量']:'first'})
    stock_begin_hk = grouped_trades_hk.股份余额 - grouped_trades_hk.成交数量
    stock_begin = pd.concat([stock_begin_hk, stock_begin_nonhk])
    stock_end = trades.groupby([broker['股东代码'],broker['证券代码']]).last()[broker['股份余额']]
    stock_amount = pd.DataFrame({'期初持仓':stock_begin,'期末持仓':stock_end})

    # 获取比较基准的价格走势
    hs300 = get_close_price('000300', start_date, end_date)
    benchmark_index = hs300.index

    stock_list = list(trades[broker['证券代码']].unique())

    fp = fp_prefix + duration + '_close_price.csv'

    # 远程查询组合收盘价
    if (LOAD_FROM_LOCAL==False):
        stock_close_price = pd.DataFrame(index=benchmark_index, columns=stock_list)
        for code in stock_list:
            if(code != '') :
                print("正在查询%s的行情数据......" % code)
                stock_close_price[code] = get_close_price(code, start_date, end_date)
        stock_close_price = stock_close_price.sort_index(ascending='ascending')
        stock_close_price = stock_close_price.ffill().fillna(0.0)
        stock_close_price = stock_close_price.reindex(benchmark_index, fill_value=0, method='ffill')
        stock_close_price.apply(backfill_xsb)
        stock_close_price.to_csv(fp, float_format='%8.3f', sep=",", encoding='utf-8')

    # 本地读取期组合收盘价
    if (LOAD_FROM_LOCAL==True):
        print("正在从%s读取行情数据......" % fp)
        stock_close_price = pd.read_csv(fp, header=0, sep=',', converters={'datetime' : lambda x:datetime.strptime(x,"%Y-%m-%d")}, index_col='datetime')
        stock_close_price = stock_close_price.sort_index(ascending='ascending')
        stock_close_price = stock_close_price.ffill().fillna(0.0)
        stock_close_price = stock_close_price.reindex(benchmark_index, fill_value=0, method='ffill')


    # 提取期初和期末股票价格
    stock_price = stock_close_price.T
    stock_price = stock_price[[0,-1]]
    stock_price.index.rename('证券代码', inplace=True)
    stock_price.rename(columns={stock_price.columns[0]:'期初价格', stock_price.columns[1]:'期末价格'}, inplace=True)


    # 汇总证券交易数据
    stocks = stock_transaction
    stocks = pd.merge(stocks, stock_amount, left_index=True, right_index=True)
    stocks = pd.merge(stocks, stock_price, left_index=True, right_index=True)

    # 计算股票交易盈亏
    stocks['盈亏'] = stocks['期末持仓'] * stocks['期末价格'] + stocks['发生金额'] - stocks['期初持仓'] * stocks['期初价格']
    stocks.reset_index(level=0, drop=True, inplace=True)
    stocks.reset_index(level=0, drop=False, inplace=True)
    stocks = stocks.groupby('证券代码').agg({'发生金额':sum, '期末持仓':sum, '期初持仓':sum, '盈亏':sum, '期初价格':"first", '期末价格':"first"})
    name = pd.DataFrame({'证券名称':stock_name})
    stocks = pd.merge(stocks, name, left_index=True, right_index=True)
    stocks = stocks[['证券名称', '发生金额', '期初持仓', '期末持仓', '期初价格', '期末价格', '盈亏']]
    stocks = stocks.round({'发生金额':0, '期初持仓':0, '期末持仓':0, '期初价格':2, '期末价格':2, '盈亏':0})

    # 计算交易费用，包括手续费、印花税、过户费、交易所清算费、融资利息
    trade_cost = trades[broker['手续费']].sum() + trades[broker['印花税']].sum() + trades[broker['过户费']].sum() + trades[broker['交易所清算费']].sum()
    trade_cost = trade_cost - trades[trades.业务名称.isin(broker['港股通收费项目'])].发生金额.sum()

    # 计算摘牌市值
    # unlisted = log[log.业务名称.isin(['托管转出'])]
    # unlisted_value = (unlisted.成交价格 * unlisted.成交数量).sum()

    # 计算投入资金和盈亏比例
    capital = cash_begin + cash_in + (stocks.期初持仓*stocks.期初价格).sum()
    pnl = stocks.盈亏.sum() + dividend + interest
    pnl_percentage = pnl / capital

    # 计算Money Weighted Return Rate
    value_begin = cash_begin + (stocks.期初持仓*stocks.期初价格).sum()
    value_end = cash_end + (stocks.期末持仓*stocks.期末价格).sum()
    cash_log = cash_report
    cash_log.reset_index(drop=False, inplace=True)
    cash_log = cash_log.append({'发生日期':start_date, '发生金额':value_begin}, ignore_index=True)
    cash_log = cash_log.append({'发生日期':end_date, '发生金额':-value_end}, ignore_index=True)
    cash_log.sort_values(by=['发生日期'], inplace=True)
    cash_log.发生金额 = -cash_log.发生金额

    mwrr = xirr(cash_log.as_matrix(columns=None))

    # 计算每天的股票持仓
    stock_position_begin = pd.DataFrame(stocks['期初持仓'])
    stock_position_begin = stock_position_begin.T
    stock_position_begin.insert(0, column='发生日期', value=start_date)
    stock_position_begin = stock_position_begin.set_index('发生日期')

    stock_position = trades.groupby([broker['发生日期'],broker['证券代码'],broker['股东代码']]).agg({broker['成交数量']:sum}).unstack(fill_value=0).sum(axis=1)
    stock_position = stock_position.unstack(fill_value=0)
    stock_position = pd.concat([stock_position_begin, stock_position])
    stock_position = stock_position.cumsum(axis=0)
    stock_position = stock_position.reindex(benchmark_index, fill_value=0, method='ffill')

    # 计算每天的资金余额
    cash_balance_begin = pd.Series({start_date:cash_begin})
    cash_balance = log.groupby([broker['发生日期'],broker['资金帐号']]).last()[broker['资金本次余额']].unstack()
    cash_balance = cash_balance.ffill().fillna(0.0).sum(axis=1)
    cash_balance = pd.concat([cash_balance_begin, cash_balance])
    cash_balance = cash_balance.reindex(benchmark_index, fill_value=0, method='ffill')


    # 计算新股申购占用资金，因市值中并未包含新股申购占用资金，后续需要加回来
    def fill_ipo_value(c):
        index1 = c[c < 0].index.tolist()[0]
        index2 = c[c > 0].index.tolist()[0]
        v = c[c < 0].tolist()[0]
        c.ix[index1:index2] = v
        c.ix[index2] = 0.0
        return c


    ipo = log[log.业务名称.isin(broker['新股申购项目'])]
    ipo_value = ipo.groupby([broker['发生日期'], broker['证券代码']]).agg({broker['发生金额']: sum})
    ipo_value = ipo_value.unstack()
    ipo_value = ipo_value.reindex(benchmark_index)
    ipo_value = ipo_value.apply(fill_ipo_value)
    ipo_value = ipo_value.sum(axis=1)

    # 计算融资净债务
    debt = log[log.业务名称.isin(broker['融资债务项目'])]
    debt_value = debt.groupby(broker['发生日期']).agg({broker['发生金额']: sum})
    debt_value['债务余额'] = debt_value['发生金额'].cumsum()
    debt_value = debt_value.reindex(benchmark_index, fill_value=0.0, method='ffill')
    debt_value = debt_value['债务余额']

    # 计算基金赎回
    fund = log[log.业务名称.isin(broker['基金赎回项目'])]
    fund_value = fund.groupby(broker['发生日期']).agg({broker['成交金额']: sum})
    fund_value = fund_value.squeeze().reindex(benchmark_index, fill_value=0.0)

    # 计算每天的持仓市值
    market_value = stock_close_price * stock_position
    market_value = market_value.sum(axis=1)
    market_value = market_value.reindex(benchmark_index, fill_value=0.0, method='ffill')
    net_value = market_value + cash_balance - ipo_value - debt_value + fund_value
    net_value = net_value.sort_index(ascending='ascending')

    # 调整现金进出的影响
    cash_report = cash_activities.groupby(['发生日期']).agg({'发生金额':sum})
    cash_report = cash_report[cash_report.发生金额 != 0]
    cash_deposit = cash_report.squeeze().reindex(index=net_value.index, fill_value=0.0)
    net_value_adj = net_value - cash_deposit

    # 计算投资组合的时间加权回报率
    twrr_portfolio_daily_gain = net_value_adj / net_value.shift(1)
    twrr_portfolio_daily_gain = twrr_portfolio_daily_gain.replace([np.inf, -np.inf], np.nan).fillna(1.0)
    low_value = net_value_adj<5000
    twrr_portfolio_daily_gain.loc[low_value] = 1.0
    twrr_portfolio = twrr_portfolio_daily_gain.cumprod()
    twrr_portfolio_value = twrr_portfolio[-1] - 1.0

    # 计算比较基准的时间加权回报率
    twrr_benchmark = hs300 / hs300.shift(1)
    twrr_benchmark = twrr_benchmark.replace([np.inf, -np.inf], np.nan).fillna(1.0)
    twrr_benchmark = twrr_benchmark.cumprod()
    twrr_benchmark_value = twrr_benchmark[-1] - 1.0

    # 计算最大回撤
    drawdown_portfolio = drawdown(twrr_portfolio)
    drawdown_benchmark = drawdown(twrr_benchmark)

    # 计算夏普比率
    sharpe_portfolio = sharpe(twrr_portfolio, rf=0.03)

    # 绘制时间加权收益曲线图
    returns = pd.DataFrame({'Portfolio':twrr_portfolio,'Benchmark':twrr_benchmark})
    ax = returns.plot(kind='line', color=['steelblue','darkorange'])
    ax.set_title('时间加权收益率', fontname='Simhei', fontsize=18)
    ax.annotate('{:.2f}'.format(twrr_benchmark[-1]), xy=(returns.index[-1], twrr_benchmark[-1]))
    ax.annotate('{:.2f}'.format(twrr_portfolio[-1]), xy=(returns.index[-1], twrr_portfolio[-1]))
    ax.set_xlabel('')
    plt.savefig(fp_prefix+'TWRR_vs_Benchmark_'+duration+'.jpg')

    # 存储业绩回报数据
    fp = fp_prefix + 'Performance_Data_' + duration + '.csv'
    returns.to_csv(fp, float_format='%8.3f', sep=",", encoding='utf-8')

    # 计算年度或月度回报，两年内计算月度收益率，超过2年计算年度收益率
    if ((twrr_portfolio.index.year[-1] - twrr_portfolio.index.year[0]) > 1) :
        breakdown_period = [twrr_portfolio.index.year]
    else:
        breakdown_period = [twrr_portfolio.index.year, twrr_portfolio.index.month]
    return_breakdown_portfolio = twrr_portfolio.groupby(breakdown_period).apply(total_returns) * 100
    return_breakdown_benchmark = twrr_benchmark.groupby(breakdown_period).apply(total_returns) * 100
    return_breakdown = pd.DataFrame({'Portfolio':return_breakdown_portfolio,'Benchmark':return_breakdown_benchmark})
    ax = return_breakdown.plot(kind='bar', color=['steelblue','darkorange'], edgecolor='none')
    ax.set_title('各期收益率(%)', fontname='Simhei', fontsize=18)
    plt.axhline(0, color='k')
    plt.savefig(fp_prefix+'monthly_return_'+duration+'.jpg')

    # 验算TWRR
    r = 1.0
    net_value_prev = net_value_adj[0]
    for i in range(0,len(cash_report)):
        p = cash_report.index[i]
        r = r * ((net_value.loc[p] - cash_report.iloc[i]) / net_value_prev)
        net_value_prev = net_value.loc[p]
    r = r * (net_value[-1]) / net_value_prev

    # 检验是否有数据异常
    t = twrr_portfolio_daily_gain
    abnormal_data = pd.concat([t[t>1.1], t[t<0.9]]).sort_index()
    if (len(abnormal_data)>0):
        print("发现异常数据点：")
        print(abnormal_data.head(100))


    # 生成Word格式报告
    from docxtpl import DocxTemplate, InlineImage
    from docx.shared import Mm, Inches, Pt

    tpl=DocxTemplate(fp_prefix+'投资业绩分析.docx')

    s = stocks.reset_index()
    portfolio = s.T.to_dict().values()
    portfolio = sorted(portfolio, key=operator.itemgetter('盈亏'), reverse=True)

    context = {
        '期初市值' : '{:12,.0f}'.format(value_begin),
        '存入现金' : '{:12,.0f}'.format(cash_in),
        '期末市值' : '{:12,.0f}'.format(value_end),
        '交易费用' : '{:12,.0f}'.format(trade_cost),
        '税后红利' : '{:12,.0f}'.format(dividend),
        '支付利息' : '{:12,.0f}'.format(interest),
        '盈亏总额' : '{:12,.0f}'.format(pnl),
        '投入资金' : '{:12,.0f}'.format(capital),
        'twrr' : '{:.2f}%'.format(twrr_portfolio_value*100),
        'mwrr' : '{:.2f}%'.format(mwrr*100),
        '基准盈亏比例' : '{:.2f}%'.format(twrr_benchmark_value*100),
        '夏普比率' : '{:.2f}'.format(sharpe_portfolio),
        '最大回撤' : '{:.2f}%'.format(drawdown_portfolio*100),
        '基准最大回撤' : '{:.2f}%'.format(drawdown_benchmark*100),
        'start_date' : start_date.strftime("%Y-%m-%d"),
        'end_date' : end_date.strftime("%Y-%m-%d"),
        'twrr_vs_benchmark_img' : InlineImage(tpl,fp_prefix+'TWRR_vs_Benchmark_'+duration+'.jpg', width=Mm(100)),
        'monthly_return_img' : InlineImage(tpl,fp_prefix+'monthly_return_'+duration+'.jpg', width=Mm(100)),
        'portfolio' : portfolio
    }
    tpl.render(context)
    tpl.save(fp_prefix+'投资业绩分析_'+duration+'.docx')

    print('成功生成投资分析报告: '+fp_prefix+'投资业绩分析_'+duration+'.docx')

