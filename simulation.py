import yaml
from aggregate import *
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