![Logo](resource/logo.png)

# Stock Optimizer 1.0
[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/kaiCbs/StockOpt/blob/master/README.md)

## Table of Contents

- [Background](#background)
- [Install](#install)
- [Usage](#usage)
- [Maintainers](#maintainers)
- [Contributing](#contributing)
- [License](#license)

## Background

The purpose of this package is to solve the portfolio optimization problem, which is essentially a mixed integer programming problem. Based on [pulp](https://pypi.org/project/PuLP/), we have achieved a rapid dynamic adjustment based on yesterday's position, which maximize the weighted alpha scores minus the transaction costs.


## Install


Download the source code from github:

    $ git clone git@github.com:kaiCbs/StockOpt.git
    
## Usage

```
from Optimizer import *
para = pd.read_csv("parameters.csv", index_col=0) 
stock = pd.read_csv("input_example.csv", index_col=0)

# solve the optimization problem
weights = execute(stock, para, time_limit=5)  
```

Which will return the optimal weights:

```
Status: Optimal

[Before Adjust] Score: 0.038014    Percentile: 0.2604
[~After Adjust] Score: 0.039680    Percentile: 0.1207

In:116	 Out:61	 Adjust:130	

>> Turnover:    
 Buy in  0.4160   Adjust 0.0028   Total 0.4188

>> market_value:
 Before  0.7840   After 0.7597   Target 0.8000   Δ -0.0403 

>> trend:
 Before  0.6159   After 0.5826   Target 0.8000   Δ -0.2174 

>> holders:
 Before  0.7274   After 0.6891   Target 0.8000   Δ -0.1109 

>> sector:

              before     after    target         Δ
sector                                           
801080.SI  0.110066  0.134175  0.068116  0.066059
801050.SI  0.078530  0.068918  0.040184  0.028734
801790.SI  0.012024  0.044847  0.024875  0.019972
801780.SI  0.000000  0.024907  0.007659  0.017247
...             ...       ...       ...       ...
801200.SI  0.011622  0.015900  0.031879 -0.015979
801180.SI  0.040183  0.040114  0.057571 -0.017457
801040.SI  0.001627  0.001373  0.023442 -0.022069
801170.SI  0.015564  0.012755  0.038993 -0.026237

[28 rows x 4 columns]
...
```

For detailed simulation results, it will also generate a [log file](resource/Fri_Jul_3_2020.log) that contains the weight changes of all target stocks.


## Maintainers

[@kaiCbs](https://github.com/kaiCbs).


## License

[GNU General Public License](resource/GNU.txt)