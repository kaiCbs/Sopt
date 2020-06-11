import os
import numpy as np
import pandas as pd
from pulp import *


class Portfolio:
    def __init__(self, df):
        self.sect_weights = df.groupby("sect").weights_ystd.sum().sort_values(
            ascending=False)
        self.sect_list = df.sect.unique()
        self.sect = df.sect.to_dict()
        self.stock_pool = df.index
        self.scores = df.scores.to_dict()
        self.weights_ystd = df.weights_ystd.to_dict()
        self.tcosts = df.tcost.to_dict()
        self.max_position = df.Max.to_dict()
        self.scores_adj = (df.scores + (df.weights_ystd > 0) * 0.005).to_dict()
        self.weights_tdy = self.assign()
        self.mv = (1 - df.mv.rank() / df.mv.rank().max()).to_dict()
        self.trend = df.trend.to_dict()
        self.df = df

    def assign(self):
        def func(score):
            """
            Method that decide how many weights we want to assgin given its scores.  ???
            """
            return max(score * 500, 0) + 0.1

        return {k: func(v) for k, v in self.scores.items()}


class Solver:
    def __init__(self, portfolio, rules):
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
        self.stock_num = rules["stock_num"]
        self.section_constrain, self.sect_tol = rules["section"]
        self.mv_constrain, self.mv_tol = rules["mv"]
        self.trend_constrain, self.trend_tol = rules["trend"]
        self.tcost = rules["tcost"]
        self.portfolio.scores_adj = (
            self.portfolio.df.scores +
            (self.portfolio.df.weights_ystd > 0) * self.tcost).to_dict()

    def is_sect(self, stock, sect):
        return self.portfolio.sect[stock] == sect

    def add_obj_func(self):
        self.solver += lpSum([
            self.holdStatus[s] * self.portfolio.weights_tdy[s] *
            self.portfolio.scores_adj[s] for s in self.portfolio.stock_pool
        ]), "max alpha score"

        # industry

    def add_constrains(self):
        self.solver += lpSum(self.holdStatus) == self.stock_num, "Pick Stocks"

        # section
        for sc in self.portfolio.sect_list:
            self.solver += lpSum([
                self.holdStatus[s] * self.portfolio.weights_tdy[s] *
                self.is_sect(s, sc) for s in self.portfolio.stock_pool
            ]) <= (self.section_constrain[sc] + self.sect_tol) * lpSum([
                self.holdStatus[s] * self.portfolio.weights_tdy[s]
                for s in self.portfolio.stock_pool
            ])
            self.solver += lpSum([
                self.holdStatus[s] * self.portfolio.weights_tdy[s] *
                self.is_sect(s, sc) for s in self.portfolio.stock_pool
            ]) >= (self.section_constrain[sc] - self.sect_tol) * lpSum([
                self.holdStatus[s] * self.portfolio.weights_tdy[s]
                for s in self.portfolio.stock_pool
            ])

        # market value
        self.solver += lpSum([
            self.holdStatus[s] * self.portfolio.weights_tdy[s] *
            self.portfolio.mv[s] for s in self.portfolio.stock_pool
        ]) >= (self.mv_constrain - self.mv_tol) * lpSum([
            self.holdStatus[s] * self.portfolio.weights_tdy[s]
            for s in self.portfolio.stock_pool
        ])

        self.solver += lpSum([
            self.holdStatus[s] * self.portfolio.weights_tdy[s] *
            self.portfolio.mv[s] for s in self.portfolio.stock_pool
        ]) <= (self.mv_constrain + self.mv_tol) * lpSum([
            self.holdStatus[s] * self.portfolio.weights_tdy[s]
            for s in self.portfolio.stock_pool
        ])

        # trend
        self.solver += lpSum([
            self.holdStatus[s] * self.portfolio.weights_tdy[s] *
            self.portfolio.trend[s] for s in self.portfolio.stock_pool
        ]) >= (self.trend_constrain - self.trend_tol) * lpSum([
            self.holdStatus[s] * self.portfolio.weights_tdy[s]
            for s in self.portfolio.stock_pool
        ])

        self.solver += lpSum([
            self.holdStatus[s] * self.portfolio.weights_tdy[s] *
            self.portfolio.trend[s] for s in self.portfolio.stock_pool
        ]) <= (self.trend_constrain + self.trend_tol) * lpSum([
            self.holdStatus[s] * self.portfolio.weights_tdy[s]
            for s in self.portfolio.stock_pool
        ])

    def solve(self):
        self.add_obj_func()
        self.add_constrains()
        self.solver.solve()
        print("Status:", LpStatus[self.solver.status])
        self.weights_normalize()

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
        self.reach_max = reach_max
        self.hold_weights = normal
        # return normal

    def score_rank(self, score):
        return sum([1 for s in self.portfolio.scores.values() if s > score
                    ]) / len(self.portfolio.scores.values())

    def evaluate(self):

        ystd_score = sum([
            self.portfolio.scores.get(s, 0) *
            self.portfolio.weights_ystd.get(s, 0)
            for s in self.portfolio.stock_pool
        ])
        tdy_score = sum([
            self.portfolio.scores.get(s, 0) * self.hold_weights.get(s, 0)
            for s in self.hold_weights
        ])
        print("[Before Adjust] Score: {:.6f}    Percentile: {:.4f}".format(
            ystd_score, self.score_rank(ystd_score)))
        print("[~After Adjust] Score: {:.6f}    Percentile: {:.4f}".format(
            tdy_score, self.score_rank(tdy_score)))

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

        buyin_tvr = sum([self.hold_weights.get(s, 0) for s in stock_in])
        adj_tvr = sum([
            max(
                self.hold_weights.get(s, 0) -
                self.portfolio.weights_ystd.get(s, 0), 0) for s in adjust
        ])
        print("\n>> Turnover:    \n Buy in {:4f}   Adjust {:.4f}".format(
            buyin_tvr, adj_tvr))

        mv_tdy = sum([
            self.portfolio.mv.get(s, 0) * self.hold_weights.get(s, 0)
            for s in self.hold_weights
        ])
        mv_ystd = sum([
            self.portfolio.mv.get(s, 0) *
            self.portfolio.weights_ystd.get(s, 0)
            for s in self.portfolio.stock_pool
        ])
        print(
            "\n>> Market Value:\n Before {:4f}   After {:.4f}   Target {:.4f}   Δ {:7.4f} "
            .format(mv_ystd, mv_tdy, self.mv_constrain,
                    mv_tdy - self.mv_constrain))

        trend_tdy = sum([
            self.portfolio.trend.get(s, 0) * self.hold_weights.get(s, 0)
            for s in self.hold_weights
        ])
        trend_ystd = sum([
            self.portfolio.trend.get(s, 0) *
            self.portfolio.weights_ystd.get(s, 0)
            for s in self.portfolio.stock_pool
        ])
        print(
            "\n>> Trend:       \n Before {:4f}   After {:.4f}   Target {:.4f}   Δ {:7.4f}"
            .format(trend_ystd, trend_tdy, self.trend_constrain,
                    trend_tdy - self.trend_constrain))

        sect_exposure = pd.DataFrame({"before": self.portfolio.sect_weights})
        after_sec = {
            sc: sum([
                w * self.is_sect(s, sc) for s, w in self.hold_weights.items()
            ])
            for sc in self.portfolio.sect_list
        }

        sect_exposure["after"] = [
            after_sec.get(x) for x in sect_exposure.index
        ]
        sect_exposure["target"] = [
            self.section_constrain.get(x) for x in sect_exposure.index
        ]
        sect_exposure["Δ"] = sect_exposure["after"] - sect_exposure["target"]

        print("\nSection exposure:\n\n", sect_exposure)

        print("\n\nNew buy in:\n")
        for i, s in enumerate(stock_in):
            print("  {:0>6}   {}% -> {:.2f}% {}".format(
                s, 0, 100 * self.hold_weights.get(s, 0),
                [" ", "*"][s in self.reach_max]),
                  end=["    |   ", "\n"][(i + 1) % 3 == 0])

        print("\n\nSell:\n")
        for i, s in enumerate(stock_out):
            print("  {:0>6}   {:.2f}% -> {}%".format(
                s, 100 * self.portfolio.weights_ystd.get(s, 0), 0),
                  end=["      |   ", "\n"][(i + 1) % 3 == 0])

        print("\n\nAdjust:\n")
        for i, s in enumerate(adjust):
            print("  {:0>6}   {:.2f}% -> {:.2f}%".format(
                s, 100 * self.portfolio.weights_ystd.get(s, 0),
                100 * self.hold_weights.get(s, 0)),
                  end=["    | ", "\n"][(i + 1) % 3 == 0])

        self.result = pd.Series(
            {s: self.hold_weights.get(s, 0)
             for s in tdy_holding},
            name="weights")
        self.result.sort_index().to_csv("weights_tdy.csv")