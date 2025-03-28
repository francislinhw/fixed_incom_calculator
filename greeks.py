import numpy as np
import matplotlib.pyplot as plt
import math
from fixed_income_clculators.core import call_delta, put_delta, gamma, vega

# ---------------------------
# 參數設定
# ---------------------------
K = 100.0  # 履約價
r = 0.05  # 無風險利率
sigma = 0.2  # 波動度
T = 1.0  # 距到期時間 (1 年)
S_min = 50
S_max = 150
n_points = 101  # 取多少點來畫圖

stock_prices = np.linspace(S_min, S_max, n_points)

# 計算各 Greek
call_deltas = []
put_deltas = []
gammas = []
vegas = []

for S in stock_prices:
    call_deltas.append(call_delta(S, K, r, sigma, T))
    put_deltas.append(put_delta(S, K, r, sigma, T))
    gammas.append(gamma(S, K, r, sigma, T))
    vegas.append(vega(S, K, r, sigma, T))

# ---------------------------
# 繪製 Delta 圖 (各自一張)
# ---------------------------
plt.figure()
plt.plot(stock_prices, call_deltas, label="Call Delta")
plt.plot(stock_prices, put_deltas, label="Put Delta")
plt.xlabel("Stock Price")
plt.ylabel("Delta")
plt.title("Delta vs. Stock Price")
plt.legend()
plt.show()

# ---------------------------
# 繪製 Gamma 圖
# ---------------------------
plt.figure()
plt.plot(stock_prices, gammas, label="Gamma (Call=Put)")
plt.xlabel("Stock Price")
plt.ylabel("Gamma")
plt.title("Gamma vs. Stock Price")
plt.legend()
plt.show()

# ---------------------------
# 繪製 Vega 圖
# ---------------------------
plt.figure()
plt.plot(stock_prices, vegas, label="Vega (Call=Put)")
plt.xlabel("Stock Price")
plt.ylabel("Vega")
plt.title("Vega vs. Stock Price")
plt.legend()
plt.show()
