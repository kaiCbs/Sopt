import os
import numpy as np
import pandas as pd

foldpath = "/mnt/shared/public/for_kai/"

SIM_DATES = sorted([i.split(".")[1] for i in os.listdir(foldpath+"score") if i.endswith(".csv")])

date = "20200601"

def yesterday(date):
    return SIM_DATES[SIM_DATES.index(date)-1]


def genInput(date="20200601"):
    stocks_path = os.path.join(foldpath, "stocks", date, "stocks.csv")
    hold_path = os.path.join(foldpath, "hold", date, "hold.csv")
    sect_path = os.path.join(foldpath, "para",date, "level_Sect.csv")
    scores_path = os.path.join(foldpath, "score", "Score.{}.csv".format(date))
    init_weights_path = os.path.join(foldpath, "weight", "Weight.{}.csv".format(yesterday(date)))
    mv_path = os.path.join(foldpath, "para",date, "num_Mkt.csv")
    trend_path = os.path.join(foldpath, "para",date, "num_Trend.csv")
    sect_cond_path = os.path.join(foldpath, "condition", date, "level_Sect.csv")
    score = pd.read_csv(scores_path)
    score["Symbol"] = score.Ticker 
    score.drop(["Ticker"],axis=1, inplace=True)
    weights = pd.read_csv(init_weights_path)
    if "symbol" in weights.columns:
        weights["Symbol"] = weights.symbol
        weights.drop(["symbol"],axis=1, inplace=True)
    sect = pd.read_csv(sect_path)
    sect["Symbol"] = sect.Symbol.apply(lambda x: int(x.split(".")[0]))
    mv = pd.read_csv(mv_path)
    mv["Symbol"] = mv.Symbol.apply(lambda x: int(x.split(".")[0]))
    trend = pd.read_csv(trend_path)
    trend["Symbol"] = trend.Symbol.apply(lambda x: int(x.split(".")[0]))
    stocks = pd.read_csv(stocks_path).sort_values("Symbol").reset_index(drop=True)
    stocks.Symbol = stocks.Symbol.str.split(".").apply(lambda x: int(x[0]))
    stocks = pd.merge(stocks, score, on="Symbol", how="left")
    stocks = pd.merge(stocks, weights, on="Symbol", how="left")
    stocks = pd.merge(stocks, sect, on="Symbol", how="left")
    stocks = pd.merge(stocks, mv, on="Symbol", how="left")
    stocks = pd.merge(stocks, trend, on="Symbol", how="left")
    stocks.set_index("Symbol", inplace=True)
    stocks.columns = ['Max', 'tcost', 'scores', 'weights_ystd', 'sect', 'mv', 'trend']
    stocks.trend.fillna(stocks.trend.mean(),inplace=True)
    stocks.mv.fillna(stocks.mv.mean(),inplace=True)
    stocks.fillna(0, inplace=True)
    return stocks


def init_weights(date):
    ystd_weights_path = os.path.join(foldpath, "weight", "Weight.init.csv")
    try:
        df = pd.read_csv(ystd_weights_path)
    except FileNotFoundError:
        scores_path = os.path.join(foldpath, "score", "Score.{}.csv".format(date))
        df = pd.read_csv(scores_path).sort_values(by="Score", ascending=False)
        df["Score"] = np.where(df.Score >0 , df.Score, 0) * 500 + 0.1
        df.columns  = ["symbol","weight"]
        df = df.reset_index(drop=True).sort_values("symbol")
        df.set_index("symbol").to_csv(ystd_weights_path)
    return df


def genInputSimple(sim_date="20200601"):
    default_date = "20200601"
    stocks_path = os.path.join(foldpath, "stocks", default_date, "stocks.csv")
    hold_path = os.path.join(foldpath, "hold", default_date, "hold.csv")
    sect_path = os.path.join(foldpath, "para",default_date, "level_Sect.csv")
    scores_path = os.path.join(foldpath, "score", "Score.{}.csv".format(sim_date))
    ystd_weights_path = os.path.join(foldpath, "weight", "Weight.{}.csv".format(yesterday(sim_date)))
    mv_path = os.path.join(foldpath, "para", default_date, "num_Mkt.csv")
    trend_path = os.path.join(foldpath, "para",default_date, "num_Trend.csv")
    sect_cond_path = os.path.join(foldpath, "condition", default_date, "level_Sect.csv")
    score = pd.read_csv(scores_path)
    score["Symbol"] = score.Ticker 
    score.drop(["Ticker"],axis=1, inplace=True)
    try:
        weights = pd.read_csv(ystd_weights_path)
    except FileNotFoundError:
        weights = init_weights(sim_date)
        
    if "symbol" in weights.columns:
        weights["Symbol"] = weights.symbol
        weights.drop(["symbol"],axis=1, inplace=True)
    sect = pd.read_csv(sect_path)
    sect["Symbol"] = sect.Symbol.apply(lambda x: int(x.split(".")[0]))
    mv = pd.read_csv(mv_path)
    mv["Symbol"] = mv.Symbol.apply(lambda x: int(x.split(".")[0]))
    trend = pd.read_csv(trend_path)
    trend["Symbol"] = trend.Symbol.apply(lambda x: int(x.split(".")[0]))
    stocks = pd.read_csv(stocks_path).sort_values("Symbol").reset_index(drop=True)
    stocks.Symbol = stocks.Symbol.str.split(".").apply(lambda x: int(x[0]))
    stocks = pd.merge(stocks, score, on="Symbol", how="left")
    stocks = pd.merge(stocks, weights, on="Symbol", how="left")
    stocks = pd.merge(stocks, sect, on="Symbol", how="left")
    stocks = pd.merge(stocks, mv, on="Symbol", how="left")
    stocks = pd.merge(stocks, trend, on="Symbol", how="left")
    stocks.set_index("Symbol", inplace=True)
    stocks.columns = ['Max', 'tcost', 'scores', 'weights_ystd', 'sect', 'mv', 'trend']
    stocks["weights_ystd"] = stocks.weights_ystd / stocks.weights_ystd.sum()
    stocks.trend.fillna(stocks.trend.mean(),inplace=True)
    stocks.mv.fillna(stocks.mv.mean(),inplace=True)
    stocks.fillna(0, inplace=True)
    return stocks