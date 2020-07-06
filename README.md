![Logo](resource/logo.png)

# Stock Optimizer 1.0
[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)

## Table of Contents

- [Background](#background)
- [Install](#install)
- [Usage](#usage)
- [Maintainers](#maintainers)
- [Contributing](#contributing)
- [License](#license)

## Background

The purpose of this package is to solve the portfolio optimization problem, which is essentially a mixed integer programming problem. Based on [pulp](https://pypi.org/project/PuLP/), we have achieved a rapid dynamic adjustment based on yesterday's position, which maximize the weighted alpha scores minus the transaction costs.


`aggregate` module aims to format data input, this we can retrieve a comprehensive data frame that is valid for the Solver by just providing the date. Of course, you need to set everything up, like the folders contain the raw data etc.

## Install


## Usage

```
import yaml
from Aggregate import *
from Optimizer import *

rules = yaml.safe_load(open("constraints.yaml").read())
         
for n, date in enumerate(SIM_DATES):
    stocks = genInputSimple(date)
    pool = Portfolio(stocks)
    solver = Solver(pool, rules=rules)
    print(">> Processing {} ...\n".format(date))
    %time solver.solve()
    save_path = os.path.join(foldpath, "weight", "Weight.{}.csv".format(date))
    solver.evaluate(save_as = save_path)
    print("\n" + "="* 55, "\n\n")
```

Which will give the trading instructions:

```
>> Processing 20200401 ...

Status: Optimal
CPU times: user 3.28 s, sys: 63.7 ms, total: 3.34 s
Wall time: 6.63 s

[Before Adjust] Score: 0.000009    Percentile: 0.5299
[~After Adjust] Score: 0.002125    Percentile: 0.0686

>> Turnover:    
 Buy in 0.673316   Adjust 0.0570

>> Market Value:
 Before 0.356444   After 0.3968   Target 0.3200   Δ  0.0768 

>> Trend:       
 Before 0.043475   After 0.0301   Target 0.0500   Δ -0.0199

Section exposure:

              before     after   target         Δ
sect                                            
801080.SI  0.114026  0.176908  0.11728  0.059628
801750.SI  0.087057  0.136540  0.08575  0.050790
801770.SI  0.049779  0.063462  0.02498  0.038482
801760.SI  0.038560  0.089871  0.05414  0.035731
...             ...       ...      ...       ...
801170.SI  0.022181  0.000000  0.03111 -0.031110
801790.SI  0.041160  0.002816  0.03734 -0.034524
801180.SI  0.024516  0.000000  0.03521 -0.035210
801150.SI  0.101028  0.075629  0.12211 -0.046481

[28 rows x 4 columns]

======================================================= 


>> Processing 20200402 ...

Status: Optimal
CPU times: user 3.6 s, sys: 27.8 ms, total: 3.63 s
Wall time: 7.99 s

[Before Adjust] Score: 0.001510    Percentile: 0.1082
[~After Adjust] Score: 0.001708    Percentile: 0.0736

>> Turnover:    
 Buy in 0.127055   Adjust 0.0574

>> Market Value:
 Before 0.396790   After 0.3932   Target 0.3200   Δ  0.0732 

>> Trend:       
 Before 0.030103   After 0.0317   Target 0.0500   Δ -0.0183

Section exposure:

              before     after   target         Δ
sect                                            
801080.SI  0.176908  0.180074  0.11728  0.062794
801750.SI  0.136540  0.136587  0.08575  0.050837
801760.SI  0.089871  0.094734  0.05414  0.040594
801770.SI  0.063462  0.063123  0.02498  0.038143
...             ...       ...      ...       ...
801170.SI  0.000000  0.000000  0.03111 -0.031110
801160.SI  0.002158  0.000000  0.03180 -0.031800
801790.SI  0.002816  0.004349  0.03734 -0.032991
801180.SI  0.000000  0.000000  0.03521 -0.035210

[28 rows x 4 columns]

======================================================= 

...
```

For detailed simulation results, it will also generate a log file that contains the weight changes of all target stocks.


## Maintainers

[@kaiCbs](https://github.com/kaiCbs).


## License

[GNU General Public License](resource/GNU.txt)