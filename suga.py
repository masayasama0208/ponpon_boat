import numpy as np
import matplotlib.pyplot as plt

# ====== シミュレーションパラメータ ======
dt = 0.01              # 時間刻み [秒]
T_total = 20           # 総シミュレーション時間 [秒]
time = np.arange(0, T_total, dt)  # 時間配列

# ====== 各種変数の初期化 ======
P = np.zeros_like(time)   # ボイラー圧力 [Pa]
V = np.zeros_like(time)   # 船の速度 [m/s]
X = np.zeros_like(time)   # 船の位置 [m]
F = np.zeros_like(time)   # 推進力 [N]

P0 = 101325     # 大気圧（基準圧力）[Pa]
P[0] = P0       # 初期圧力

# ====== 定数（物理パラメータ） ======
heat_input = 30                 # 熱入力 [W]
boiler_volume = 5e-6           # ボイラー容量 [m^3]（5 mL）
R = 461.5                      # 水蒸気の気体定数 [J/kg/K]
T_steam = 373.15               # 蒸気温度 [K]（100℃）
mass_flow_coeff = 1e-7         # 蒸気の流出係数 [kg/s/Pa]
nozzle_area = np.pi * (0.005 / 2) ** 2  # ノズル面積（直径5 mm）[m^2]
vaporization_energy = 2260e3   # 蒸発潜熱 [J/kg]
mass = 0                       # 蒸気の質量 [kg]

# ====== メインループ ======
for i in range(1, len(time)):
    # --- 蒸気の生成量（熱量 ÷ 蒸発潜熱） ---
    dm_gen = (heat_input * dt) / vaporization_energy

    # --- 蒸気の流出量（圧力差に比例） ---
    dm_out = mass_flow_coeff * (P[i-1] - P0) * dt
    dm_out = max(dm_out, 0)  # マイナスにならないように補正

    # --- 蒸気質量の更新 ---
    mass += dm_gen - dm_out
    mass = max(mass, 1e-9)  # ゼロ割防止（最小質量）

    # --- 圧力の更新（理想気体の状態方程式） ---
    P[i] = (mass * R * T_steam) / boiler_volume

    # --- 推進力（圧力差 × ノズル面積） ---
    F[i] = max((P[i] - P0) * nozzle_area, 0)

    # --- 船の加速度 = 力 / 質量（仮に0.2kg） ---
    V[i] = V[i-1] + (F[i] / 0.2) * dt
    X[i] = X[i-1] + V[i] * dt

# ====== 結果をグラフで出力 ======
fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

axs[0].plot(time, P)
axs[0].set_ylabel("Pressure [Pa]")
axs[0].set_title("Boiler Pressure vs Time")

axs[1].plot(time, F)
axs[1].set_ylabel("Thrust Force [N]")
axs[1].set_title("Thrust Force vs Time")

axs[2].plot(time, V)
axs[2].set_ylabel("Velocity [m/s]")
axs[2].set_title("Velocity vs Time")

axs[3].plot(time, X)
axs[3].set_ylabel("Position [m]")
axs[3].set_xlabel("Time [s]")
axs[3].set_title("Position vs Time")

plt.tight_layout()
plt.show()