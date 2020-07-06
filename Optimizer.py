import os
import time
import numpy as np
import pandas as pd
from pulp import *

weights_save_path = "/mnt/shared/public/for_kai/stockOpt/weights_pred/"
logs_save_path = "/mnt/shared/public/for_kai/stockOpt/logs/" 


class Portfolio:
    def __init__(self, df, parameter):
        df = df.copy()
        df.index = [int(x.split(".")[0]) for x in df.index]
        mapping = {k:v for k,v in parameter.alias.to_dict().items() if not pd.isnull(v)}
        df.columns = [mapping.get(c, c) for c in df.columns]      
        self.scores = (df.scores - df.scores.min()).to_dict()
        self.weights_ystd = df.weights_ystd.to_dict()
        self.stock_pool = df.index
        
        tcost_factor = parameter.set_index("alias").val.to_dict()["tcost"]
        tcost_enable = parameter.set_index("alias").enable.to_dict()["tcost"]
        tcost_base = parameter.set_index("alias").val.to_dict()["tcost_base"]
        
        self.max_position = df.Max.to_dict()
        self.scores_adj = (df.scores - df.scores.min() + (df.weights_ystd > 0) * (tcost_base + tcost_factor * tcost_enable * df.tcost)).to_dict()
        self.threshold = parameter.set_index("alias").val.to_dict()["threshold"]
        self.weights_tdy = self.assign()
        
        self.continuous = list(parameter.index[parameter.type == 2])
        self.descrete = list(parameter.index[parameter.type == 3])
        self.to_rank = list(parameter.index[parameter.to_rank == 1])
        
        for var in self.descrete:
            setattr(self,"{}_weights".format(var), df.groupby(var).weights_ystd.sum().sort_values(ascending=False))
            setattr(self, "{}_list".format(var), df[var].unique())
            setattr(self, "{}_constraint".format(var), eval(parameter.loc[var,"content"]))
            setattr(self, "{}_enable".format(var), parameter.loc[var,"enable"])
            setattr(self, "{}_tol".format(var), parameter.loc[var,"tol"])
            setattr(self, var, df[var])
        
        for var in self.continuous:
            setattr(self, var, df[var])
            setattr(self, "{}_constraint".format(var), parameter.loc[var,"val"])
            setattr(self, "{}_enable".format(var), parameter.loc[var,"enable"])
            setattr(self, "{}_tol".format(var), parameter.loc[var,"tol"])
        
        
        for var in self.to_rank:
            setattr(self, var, getattr(self, var).rank()/getattr(self, var).rank().max())
        
        # self.mv = 0.5 + 2 * abs(df.mv.rank() / df.mv.shape[0] - 0.5) * (df.mv.rank() / df.mv.shape[0] - 0.5)
        # self.trend = df.trend.to_dict()
        self.df = df
        self.para = parameter


    def assign(self):
        def func(score):
            """
            Method that decide how many weights we want to assgin given its scores.  ???
            """
            return max(score * 500 - self.threshold, 0) 

        return {k: func(v) for k, v in self.scores_adj.items()}


class Solver:
    def __init__(self, portfolio):
        self.portfolio = portfolio
        self.solver = LpProblem("Portfolio_Optimization", LpMaximize)
        self.holdStatus = LpVariable.dicts("Status",
                                           self.portfolio.stock_pool,
                                           lowBound=0,
                                           upBound=1,
                                           cat=LpInteger)
        self.weights_raw = {
            i: LpVariable("Weights_{}".format(i), lowBound=0, upBound=v)
            for i, v in self.portfolio.max_position.items()
        }
        
        self.para = self.portfolio.para.set_index("alias")
        self.trim_tol, self.trim_enable = self.para.loc["trim","val"], self.para.loc["trim","enable"]
        self.stock_num = int(self.para.loc["stock_num", "val"])
        self.tcost_base  = self.para.loc["tcost_base", "val"]
        



    def is_this_category(self, stock, cat, var):
        return getattr(self.portfolio, var)[stock] == cat


    def add_obj_func(self):
        self.solver += lpSum([
            self.holdStatus[s] * self.portfolio.weights_tdy[s] *
            self.portfolio.scores_adj[s] for s in self.portfolio.stock_pool 
        ]), "max alpha score"

        # industry

    def add_constrains(self):
        self.solver += lpSum(self.holdStatus) == self.stock_num, "Pick Stocks"

        
        
        for var in self.portfolio.descrete:
            if getattr(self.portfolio, "{}_enable".format(var)):
                for sc in getattr(self.portfolio, "{}_list".format(var)):
                    self.solver += lpSum([
                        self.holdStatus[s] * self.portfolio.weights_tdy[s] *
                        self.is_this_category(s, sc, var) 
                        for s in self.portfolio.stock_pool
                    ]) <= (getattr(self.portfolio, "{}_constraint".format(var))[sc]
                           + getattr(self.portfolio, "{}_tol".format(var))) * lpSum([
                        self.holdStatus[s] * self.portfolio.weights_tdy[s]
                        for s in self.portfolio.stock_pool
                    ]), "{} {} upper bound".format(var, sc)
                    
                    self.solver += lpSum([
                        self.holdStatus[s] * self.portfolio.weights_tdy[s] *
                        self.is_this_category(s, sc, var) 
                        for s in self.portfolio.stock_pool
                    ]) >= ( getattr(self.portfolio, "{}_constraint".format(var))[sc]
                           - getattr(self.portfolio, "{}_tol".format(var))) * lpSum([
                        self.holdStatus[s] * self.portfolio.weights_tdy[s]
                        for s in self.portfolio.stock_pool
                    ]), "{} {} lower bound".format(var, sc)

        
        for var in self.portfolio.continuous:
            if getattr(self.portfolio, "{}_enable".format(var)):
                self.solver += lpSum([
                    self.holdStatus[s] * self.portfolio.weights_tdy[s] *
                    getattr(self.portfolio, var)[s] for s in self.portfolio.stock_pool
                ]) >= (getattr(self.portfolio, "{}_constraint".format(var))
                               - getattr(self.portfolio, "{}_tol".format(var))) * lpSum([
                    self.holdStatus[s] * self.portfolio.weights_tdy[s]
                    for s in self.portfolio.stock_pool
                ]), "{} lower bound".format(var)

                self.solver += lpSum([
                    self.holdStatus[s] * self.portfolio.weights_tdy[s] *
                    getattr(self.portfolio, var)[s] for s in self.portfolio.stock_pool
                ]) <= (getattr(self.portfolio, "{}_constraint".format(var))
                               + getattr(self.portfolio, "{}_tol".format(var))) * lpSum([
                    self.holdStatus[s] * self.portfolio.weights_tdy[s] for s in self.portfolio.stock_pool
                ]), "{} upper bound".format(var)


    def solve(self, time_limit = 30):
        self.add_obj_func()
        self.add_constrains()
        self.solver.solve(PULP_CBC_CMD(maxSeconds=time_limit, msg=1, fracGap=0))
        print("Status:", LpStatus[self.solver.status])
        self.weights_normalize()
        if self.trim_enable:
            self.trim()


    def weights_normalize(self):
        weights_raw = {
            int(v.name.split("_")[1]):
            self.portfolio.weights_tdy.get(int(v.name.split("_")[1]), 0)
            for v in self.solver.variables() if v.varValue
        }

        w = {k: v / sum(weights_raw.values()) for k, v in weights_raw.items()}

        reach_max = {
            k: self.portfolio.max_position.get(k, float("inf"))
            for k, v in w.items()
            if v >= self.portfolio.max_position.get(k, float("inf"))
        }
        
        reach_min = {
        }  # {k:min_limit.get(k, float("-inf")) for k,v in w.items() if v<min_limit.get(k, float("-inf"))}
        
        fmax, fmin = reach_max, reach_min
        
        while len(fmax) or len(fmin):
            reach_max, reach_min = {
                k: self.portfolio.max_position.get(k, float("inf"))
                for k, v in w.items()
                if v >= self.portfolio.max_position.get(k, float("inf"))
            }, {}
                      
            normal = {
                k: v
                for k, v in w.items() if k not in reach_max and k not in reach_min
            }
            normal = {
                k: (1 - sum(reach_min.values()) - sum(reach_max.values())) * v /
                sum(normal.values())
                for k, v in w.items() if k not in reach_max and k not in reach_min
            }
            
            normal.update(reach_max)
            normal.update(reach_min)
            
            w = normal
            
            fmax, fmin = {
                k: self.portfolio.max_position.get(k, float("inf"))
                for k, v in w.items()
                if v > self.portfolio.max_position.get(k, float("inf"))
            }, {}   
            
        
        self.reach_max = reach_max
        self.hold_weights = normal
        
        
        
    
    def trim(self):
        self.w_trim = {}
        for s in self.hold_weights:
            if s in self.reach_max:
                self.w_trim[s] = self.reach_max[s]
            elif abs(self.hold_weights[s]-self.portfolio.weights_ystd[s]) < self.trim_tol:
                self.w_trim[s] = self.portfolio.weights_ystd[s]
        
        if len(self.w_trim) == len(self.hold_weights):
            print("Total position might not be 100%")
            self.hold_weights = self.w_trim
            return 
        
        
        
        disfac = (1-sum(self.w_trim.values())) / sum(
                  [self.hold_weights[s] for s in self.hold_weights if s not in self.w_trim]) 


        for s in self.hold_weights:
            if s not in self.w_trim:
                self.w_trim[s] = self.hold_weights[s] * disfac
                
        self.hold_weights = self.w_trim

    def score_rank(self, score):
        return sum([1 for s in self.portfolio.scores.values() if s > score
                    ]) / len(self.portfolio.scores.values())

    def evaluate(self, save_as = None):
        if save_as:
            log = open(logs_save_path+"{}.log".format(save_as), "w")
        else:
            log = open("temp_{}.log".format(time.ctime()), "w")
            
        ystd_score = sum([
            self.portfolio.scores.get(s, 0) *
            self.portfolio.weights_ystd.get(s, 0)
            for s in self.portfolio.stock_pool
        ])
        
        tdy_score = sum([
            self.portfolio.scores.get(s, 0) * self.hold_weights.get(s, 0)
            for s in self.hold_weights
        ])
        
        
        print()
        print("[Before Adjust] Score: {:.6f}    Percentile: {:.4f}".format(
            ystd_score, self.score_rank(ystd_score)))
        print("[~After Adjust] Score: {:.6f}    Percentile: {:.4f}".format(
            tdy_score, self.score_rank(tdy_score)))

        self.ystd_score_rank = self.score_rank(ystd_score)
        self.tdy_score_rank = self.score_rank(tdy_score)
        # 
        print("[Before Adjust] Score: {:.6f}    Percentile: {:.4f}".format(
            ystd_score, self.score_rank(ystd_score)), file=log)
        print("[~After Adjust] Score: {:.6f}    Percentile: {:.4f}".format(
            tdy_score, self.score_rank(tdy_score)), file=log)
        
        ysd_holding = {
            k: v
            for k, v in self.portfolio.weights_ystd.items() if v > 0
        }
        tdy_holding = {
            int(v.name.split("_")[1]): v.varValue
            for v in self.solver.variables() if v.varValue
        }
        stock_out = set(ysd_holding) - set(tdy_holding)
        stock_in = set(tdy_holding) - set(ysd_holding)
        adjust = set(tdy_holding) & set(ysd_holding)

        # print("\nIn:{}\t Out:{}\t Adjust:{}\t".format(len(stock_in), len(stock_out), len(adjust)))
        
        adjust = [s for s in adjust if abs(self.portfolio.weights_ystd.get(s, 0)-self.hold_weights.get(s, 0)) > 1e-6]
        
        print("\nIn:{}\t Out:{}\t Adjust:{}\t".format(len(stock_in), len(stock_out), len(adjust)))
        print("\nIn:{}\t Out:{}\t Adjust:{}\t".format(len(stock_in), len(stock_out), len(adjust)), file=log)
        
        buyin_tvr = sum([self.hold_weights.get(s, 0) for s in stock_in])
        adj_tvr = sum([
            max(
                self.hold_weights.get(s, 0) -
                self.portfolio.weights_ystd.get(s, 0), 0) for s in adjust
        ])
        
        
        print("\n>> Turnover:    \n Buy in {:7.4f}   Adjust {:.4f}   Total {:.4f}".format(
            buyin_tvr, adj_tvr, buyin_tvr+adj_tvr))
        
        print("\n>> Turnover:    \n Buy in {:7.4f}   Adjust {:.4f}   Total {:.4f}".format(
            buyin_tvr, adj_tvr, buyin_tvr+adj_tvr), file=log)

        self.total_tvr = buyin_tvr + adj_tvr
        
        
        tdy_values, ystd_values = {}, {}
        
        for var in self.portfolio.continuous:
            tdy_values[var] = sum([
            getattr(self.portfolio, var)[s] * self.hold_weights.get(s, 0)
            for s in self.hold_weights
        ])
            ystd_values[var] = sum([
            getattr(self.portfolio, var)[s] * self.portfolio.weights_ystd.get(s, 0)
            for s in self.portfolio.stock_pool
        ])
        
        
            print(
            "\n>> {}:\n Before {:7.4f}   After {:.4f}   Target {:.4f}   Δ {:7.4f} "
            .format(var,
                    ystd_values[var], 
                    tdy_values[var], 
                    getattr(self.portfolio, "{}_constraint".format(var)),
                    tdy_values[var] - getattr(self.portfolio, "{}_constraint".format(var))))
        
        
            print(
                "\n>> {}:\n Before {:7.4f}   After {:.4f}   Target {:.4f}   Δ {:7.4f} "
                .format(var, 
                        ystd_values[var], 
                        tdy_values[var], 
                        getattr(self.portfolio, "{}_constraint".format(var)),
                        tdy_values[var] - getattr(self.portfolio, "{}_constraint".format(var))), 
                                file=log)
        
        df_info = {}
        for var in self.portfolio.descrete:
            df_info[var] = pd.DataFrame({"before": 
                                         getattr(self.portfolio, "{}_weights".format(var))})

 
            after = {
                sc: sum([
                    w * self.is_this_category(s, sc, var) for s, w in self.hold_weights.items()
                ])
                for sc in getattr(self.portfolio, "{}_list".format(var))
            }

            df_info[var]["after"] = [
                after.get(x) for x in df_info[var].index
            ]
        
            df_info[var]["target"] = [getattr(self.portfolio, "{}_constraint".format(var)).get(x) 
                                      for x in df_info[var].index
             ]
            
            df_info[var]["Δ"] = df_info[var]["after"] - df_info[var]["target"]


        
            with pd.option_context('display.max_rows', 8):
                print("\n>> {}:\n\n".format(var), df_info[var].sort_values(by="Δ", ascending=False))
                print("\n>> {}:\n\n".format(var), df_info[var].sort_values(by="Δ", ascending=False), file=log)

        
        print("\n\nNew buy in:\n", file=log)
        for i, s in enumerate(stock_in):
            print("  {:0>6}   {}% -> {:.2f}% {}".format(
                s, 0, 100 * self.hold_weights.get(s, 0),
                [" ", "*"][s in self.reach_max]),
                  end=["    |   ", "\n"][(i + 1) % 3 == 0], file=log)

        print("\n\nSell:\n", file=log)
        for i, s in enumerate(stock_out):
            print("  {:0>6}   {:.2f}% -> {}%".format(
                s, 100 * self.portfolio.weights_ystd.get(s, 0), 0),
                  end=["      |   ", "\n"][(i + 1) % 3 == 0], file=log)

        print("\n\nAdjust:\n", file=log)
        
        
        for i, s in enumerate(adjust):
            print("  {:0>6}   {:.2f}% -> {:.2f}%".format(
                s, 100 * self.portfolio.weights_ystd.get(s, 0),
                100 * self.hold_weights.get(s, 0)),
                  end=["    | ", "\n"][(i + 1) % 3 == 0], file=log)

        self.result = pd.Series(
            {s: self.hold_weights.get(s, 0)
             for s in tdy_holding},
            name="weights")
                    
        self.result.index.name = "Symbol"
        
        if save_as:
            self.result.sort_index().to_csv(weights_save_path+"{}.csv".format(save_as))
        
        log.close()
        
        return self.result.sort_index()

    
def execute(stock, para, time_limit = 30, save_as=None):
    p = Portfolio(stock, para)
    solver = Solver(p)
    solver.solve(time_limit)
    return solver.evaluate(save_as)

    
def test(stock, para, save_as=None):
    print(stock.index[0:4])
    print(stock.head(3))
    print(para.index)
    print(para.head(3))
    return 