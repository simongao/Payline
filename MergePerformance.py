#!/usr/env/python python
# _*_ coding: utf-8 _*_

import argparse
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog="MergePerformance.py", fromfile_prefix_chars='@')

    parser.add_argument('--infiles', action='store', nargs='+', required=True, help='input files')
    parser.add_argument('--outfile', action='store', required=True, help='output file')
    parser.add_argument('--fp_prefix', action='store', default='./', help='file path prefix')

    args = parser.parse_args()

    in_files = args.infiles
    outfile = args.outfile
    fp_prefix = args.fp_prefix

    all = pd.DataFrame({'datetime':[], 'Benchmark': [], 'Portfolio': []})
    multiplier = {'Benchmark':1.0, 'Portfolio':1.0}
    for filename in in_files:
        fp = fp_prefix + filename
        current = pd.read_csv(fp, header=0, sep=',',
                          converters={'datetime': lambda x: datetime.strptime(x, "%Y-%m-%d")})
        current.Benchmark = current.Benchmark.mul(multiplier['Benchmark'])
        current.Portfolio = current.Portfolio.mul(multiplier['Portfolio'])
        all = pd.concat([all, current])
        multiplier = all.iloc[-1].to_dict()

    # 绘制合并后的时间加权收益曲线图
    all.set_index('datetime', inplace=True)
    ax = all.plot(kind='line', color=['steelblue', 'darkorange'])
    ax.set_title('时间加权收益率', fontname='Simhei', fontsize=18)
    ax.set_xlabel('')
    ax.annotate('{:.2f}'.format(multiplier['Benchmark']), xy=(all.index[-1], multiplier['Benchmark']))
    ax.annotate('{:.2f}'.format(multiplier['Portfolio']), xy=(all.index[-1], multiplier['Portfolio']))
    plt.show()

    # 存储合并后的数据
    fp = fp_prefix + outfile
    all.to_csv(fp, float_format='%8.3f', sep=",", encoding='utf-8')
