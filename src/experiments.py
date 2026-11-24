"""
Robbins–Monro 실험 A~F (실험설계 보고서 5절).
실행:  python src/experiments.py    → results/ 폴더에 그림(PNG)과 summary.txt 저장
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

from rm_core import (SCHEDULES, Oracle, b_n, isotonic_fit, load_data, pav,
                     robbins_monro, root)

plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]  # 한글 + 수학기호(−, ≈, ̂) 대체 글꼴
plt.rcParams["axes.unicode_minus"] = False

OUT = Path(__file__).resolve().parent.parent / "results"
OUT.mkdir(exist_ok=True)
SEED, R, N = 2026, 200, 3000
ALPHA, X1, C = 0.05, 100.0, 2000.0
WEAR = "Tool wear [min]"
TORQUE = "Torque [Nm]"
EPS = 5.0  # 확률수렴 확인용 ε (분)

lines = []


def log(s=""):
    print(s)
    lines.append(s)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)


def summarize(path, theta):
    xN = path[-1]
    return dict(mean=xN.mean(), sd=xN.std(), rmse=np.sqrt(np.mean((xN - theta) ** 2)),
                p_far=np.mean(np.abs(xN - theta) > EPS))


def fmt(s):
    return f"θ̂={s['mean']:.1f}±{s['sd']:.1f}, RMSE={s['rmse']:.2f}, P(|x_N−θ*|>{EPS:g})={s['p_far']:.3f}"


# ---------------- 0. 준비: 오라클과 참 θ* ----------------
df = load_data()
wear = Oracle(df, WEAR, half_width=5, lo=0, hi=230, step=1)
theta = {a: root(wear.grid, pav(wear.p, wear.count), a) for a in (0.01, 0.05, 0.10)}
c_raw, m_raw = isotonic_fit(df, WEAR, 0, 240, 10)

log("=== 0. 준비 ===")
log(f"오라클 기준 θ* (등온회귀): α=1% → {theta[0.01]}, α=5% → {theta[0.05]:.2f}, α=10% → {theta[0.10]:.2f}")
log(f"원자료 10분 구간 등온회귀 θ (α=5%): {root(c_raw, m_raw, 0.05):.2f}")
TH = theta[ALPHA]

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(wear.grid, wear.p, color="0.7", label="국소 고장률 p̂(x) (±5분)")
ax.plot(wear.grid, pav(wear.p, wear.count), color="C0", lw=2, label="등온회귀 M̂(x) (단조)")
ax.axhline(ALPHA, color="C3", ls="--", lw=1, label=f"α = {ALPHA:.0%}")
ax.axvline(TH, color="C3", ls=":", lw=1)
ax.annotate(f"θ* ≈ {TH:.1f}분", (TH, ALPHA), xytext=(120, 0.15), arrowprops=dict(arrowstyle="->"))
ax.set(xlabel="공구 마모 시간 x [min]", ylabel="고장확률 M(x)", title="0. 조건 점검: M(x)는 단조증가하고 M(θ)=α인 θ가 존재")
ax.legend(); save(fig, "0_M_curve.png")

rng = np.random.default_rng(SEED)

# ---------------- A. 걸음 크기 a_n ----------------
log("\n=== A. 걸음 크기 (Σa_n = ∞, Σa_n² < ∞) ===")
c_of = {k: v for k, v in zip(SCHEDULES, [2000, 300, 100, 2, 2000])}
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
NA = 30000  # 조건 위반(상수, c/√n)의 오차 정체가 드러나도록 길게
nsA = np.arange(1, NA + 2)
for k, f in SCHEDULES.items():
    c = c_of[k]
    pathA = robbins_monro(wear, ALPHA, lambda n: f(n, c), X1, NA, R, rng)
    ax1.loglog(nsA, b_n(pathA, TH), label=f"{k}, c={c:g}")
    log(f"{k:28s} c={c:<5g} [N={NA}] {fmt(summarize(pathA, TH))}")
    path = pathA[: N + 1]
    ax2.plot(np.arange(1, N + 2), path[:, 0], lw=0.8, label=k)
    log(f"{'':28s}         [N={N}] {fmt(summarize(path, TH))}")
ns = np.arange(1, N + 2)
ax1.set(xlabel="n", ylabel="b_n = E(x_n − θ*)²", title="A. 평균제곱오차 b_n")
ax2.axhline(TH, color="k", ls=":", lw=1)
ax2.set(xlabel="n", ylabel="x_n [min]", title="A. 대표 경로 1개", ylim=(0, 235))
ax1.legend(fontsize=8); ax2.legend(fontsize=8); save(fig, "A_step_size.png")

# A'. 상수 c
log("\n=== A'. a_n = c/n 의 상수 c ===")
fig, ax = plt.subplots(figsize=(8, 4.5))
for c in (50, 200, 1000, 2000, 5000):
    path = robbins_monro(wear, ALPHA, lambda n: c / n, X1, N, R, rng)
    ax.loglog(ns, b_n(path, TH), label=f"c={c}")
    log(f"c={c:<5} {fmt(summarize(path, TH))}")
ax.set(xlabel="n", ylabel="b_n", title="A′. 1/n형 안에서도 c에 따라 속도가 다름"); ax.legend()
save(fig, "A2_constant_c.png")

# ---------------- B. 초기값 ----------------
log("\n=== B. 초기값 x_1 (수렴은 x_1과 무관) ===")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
for x1 in (0.0, 100.0, 230.0):
    path = robbins_monro(wear, ALPHA, lambda n: C / n, x1, N, R, rng)
    ax1.loglog(ns, b_n(path, TH), label=f"x₁={x1:g}")
    ax2.plot(ns, np.median(path, axis=1), label=f"x₁={x1:g} (중앙값)")
    log(f"x1={x1:<5g} {fmt(summarize(path, TH))}")
ax2.axhline(TH, color="k", ls=":", lw=1)
ax1.set(xlabel="n", ylabel="b_n", title="B. 초기값별 b_n"); ax2.set(xlabel="n", ylabel="x_n", xscale="log", title="B. 초기값별 경로")
ax1.legend(); ax2.legend(); save(fig, "B_initial_value.png")

# ---------------- C. 단조성 위반 (토크) ----------------
log("\n=== C. 단조성 위반: 입력 = 토크, α = 10% ===")
tq = Oracle(df, TORQUE, half_width=2, lo=3.8, hi=76.6, step=0.1)
tq_hi = Oracle(df[df[TORQUE] >= 40], TORQUE, half_width=2, lo=40, hi=76.6, step=0.1)
th_tq = root(tq_hi.grid, pav(tq_hi.p, tq_hi.count), 0.10)
log(f"단조 구간 [40, 76.6]의 θ* (등온회귀) = {th_tq:.2f} Nm")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
ax1.plot(tq.grid, tq.p, color="C0"); ax1.axhline(0.10, color="C3", ls="--")
ax1.set(xlabel="토크 [Nm]", ylabel="고장률", title="C. 토크–고장률은 U자형 (θ 후보 2개)")
C_TQ = 60  # ≈ 1/M'(θ). 너무 크면 첫 걸음이 하한 경계(왼쪽 꼬리)로 튀어 갇힌다
for x1 in (10.0, 30.0, 60.0):
    full = robbins_monro(tq, 0.10, lambda n: C_TQ / n, x1, N, R, rng)
    ax2.plot(ns, np.median(full, axis=1), label=f"전 구간, x₁={x1:g}")
    log(f"전 구간  x1={x1:<4g} 최종 평균 {full[-1].mean():.2f} Nm, θ*={th_tq:.1f} 근처(±2) 비율 {np.mean(np.abs(full[-1]-th_tq)<=2):.2f}")
restr = robbins_monro(tq_hi, 0.10, lambda n: C_TQ / n, 45.0, N, R, rng)
ax2.plot(ns, np.median(restr, axis=1), "k--", label="[40,76.6]로 제한, x₁=45")
log(f"[40,76.6] x1=45  {fmt(summarize(restr, th_tq))}")
ax2.set(xlabel="n", ylabel="x_n (중앙값)", xscale="log", title="C. 초기값에 따라 결과가 갈림"); ax2.legend(fontsize=8)
save(fig, "C_monotonicity.png")

# C'. θ 부재
log("\n=== C'. θ가 존재하지 않는 α = 1% ===")
path = robbins_monro(wear, 0.01, lambda n: C / n, X1, N, R, rng)
log(f"α=1%: 최종 x_N 평균 {path[-1].mean():.1f}분, 경계 0 근처(≤5분) 비율 {np.mean(path[-1] <= 5):.2f}  (θ 없음 → 경계로 밀려남)")

# ---------------- D. 묶음 관측 (식 49–50) ----------------
log("\n=== D. r개 묶음 관측, 총 관측 수 6000 고정 ===")
fig, ax = plt.subplots(figsize=(8, 4.5))
for r in (1, 5, 20):
    steps = 6000 // r
    path = robbins_monro(wear, ALPHA, lambda n: C / n, X1, steps, R, rng, r=r)
    ax.loglog(np.arange(0, steps + 1) * r + 1, b_n(path, TH), label=f"r={r} ({steps}스텝)")
    log(f"r={r:<3} {fmt(summarize(path, TH))}")
ax.set(xlabel="누적 관측 수", ylabel="b_n", title="D. 묶음 관측: 같은 관측 수에서의 오차"); ax.legend()
save(fig, "D_batch.png")

# ---------------- E. 등급별 θ ----------------
log("\n=== E. 제품 등급별 θ (α = 5%) ===")
for t in ("L", "M", "H"):
    o = Oracle(load_data(t), WEAR, half_width=5, lo=0, hi=230, step=1)
    th = root(o.grid, pav(o.p, o.count), ALPHA)
    path = robbins_monro(o, ALPHA, lambda n: C / n, X1, N, R, rng)
    log(f"Type {t}: θ*={th if th is None else round(th, 1)}, R–M {fmt(summarize(path, th if th else 0))}")

# ---------------- F. 같은 관측 예산에서 방법 비교 ----------------
log("\n=== F. 관측 3000회 예산: R–M vs 균등 격자 + 등온회귀 vs 로지스틱 회귀 ===")
est = {"Robbins–Monro": [], "균등 격자 + 등온회귀": [], "균등 격자 + 로지스틱": []}
rm_path = robbins_monro(wear, ALPHA, lambda n: C / n, X1, N, R, rng)
est["Robbins–Monro"] = list(rm_path[-1])
for _ in range(R):
    xs = rng.uniform(0, 230, N)
    ys = wear(xs, rng)
    c_, m_ = [], []
    edges = np.arange(0, 240, 10)
    for lo_ in edges[:-1]:
        sel = (xs >= lo_) & (xs < lo_ + 10)
        c_.append(lo_ + 5); m_.append(ys[sel].mean() if sel.any() else 0)
    w_ = [((xs >= e) & (xs < e + 10)).sum() for e in edges[:-1]]
    th_iso = root(np.array(c_), pav(np.array(m_), np.array(w_)), ALPHA)
    est["균등 격자 + 등온회귀"].append(th_iso if th_iso is not None else np.nan)
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(xs[:, None], ys)
    b0, b1 = lr.intercept_[0], lr.coef_[0, 0]
    est["균등 격자 + 로지스틱"].append((np.log(ALPHA / (1 - ALPHA)) - b0) / b1)
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.boxplot([np.array(v)[~np.isnan(v)] for v in est.values()], tick_labels=list(est), vert=False)
ax.axvline(TH, color="C3", ls=":", label=f"θ* = {TH:.1f}")
ax.set(xlabel="θ 추정값 [min]", title="F. 같은 관측 예산(3000회)에서 추정값 분포"); ax.legend()
save(fig, "F_baselines.png")
for k, v in est.items():
    v = np.array(v, float); ok = ~np.isnan(v)
    log(f"{k:20s} 평균 {v[ok].mean():.1f}, 표준편차 {v[ok].std():.2f}, RMSE {np.sqrt(np.mean((v[ok]-TH)**2)):.2f}, 실패 {(~ok).sum()}회")

# ---------------- 식 (15) 점검: b_{n+1} = b_1 + Σa_j²e_j − 2Σa_j d_j ----------------
N15, R15, N0 = 20000, 400, 100
log(f"\n=== 식 (15)~(18) 점검 (a_n = c/(n+{N0}), 1/n형: c/((1+{N0})n) ≤ a_n ≤ c/n, N={N15}, R={R15}) ===")
x1s = rng.uniform(0, 230, R15)  # b_1 > 0 이 되도록 초기값을 퍼뜨림
# 식 (15)는 사영(자르기)이 없는 원래 갱신식에서 성립하므로 이 점검에서는 경계를 없앤다.
# (구간 밖의 M은 양 끝값으로 일정하게 연장되어 단조성은 유지됨)
# 경계가 없으면 a_1=c 같은 큰 첫 걸음이 x를 멀리 날려 보내므로, 첫 걸음이 작은 a_n=c/(n+N0)를 쓴다.
path, d, e = robbins_monro(wear, ALPHA, lambda n: C / (n + N0), x1s, N15, R15, rng,
                           bounds=(-1e9, 1e9), theta=TH, track=True)
N = N15
a = C / (np.arange(1, N + 1) + N0)
S1, S2 = np.cumsum(a ** 2 * e), np.cumsum(a * d)
b = b_n(path, TH)
rhs = b[0] + S1 - 2 * S2
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
ax1.plot(np.arange(2, N + 2), b[1:], label="b_{n+1} (시뮬레이션)")
ax1.plot(np.arange(2, N + 2), rhs, "--", label="b₁ + Σa_j²e_j − 2Σa_j d_j")
ax1.set(xscale="log", xlabel="n", title="식 (15): 양변 비교 (기댓값 등식, R개 평균)"); ax1.legend()
ax2.plot(S1, label="Σ a_j² e_j  (노이즈 누적, 식 6 덕분에 수렴)")
ax2.plot(2 * S2, label="2Σ a_j d_j  (θ로 당기는 힘, 식 17)")
ax2.set(xscale="log", xlabel="n", title="식 (16)~(18): 두 양항급수의 부분합"); ax2.legend()
save(fig, "Eq15_series.png")
for k in (10, 100, 1000, N):
    log(f"n={k:<6} b_(n+1)={b[k]:10.1f}   우변 b_1+Σa²e−2Σad={rhs[k-1]:10.1f}")
log(f"Σa²e → {S1[-1]:.1f} (수렴),  2Σad → {2*S2[-1]:.1f} (수렴),  b_1={b[0]:.1f}")
log("※ 식 (15)는 기댓값의 등식이므로 R개 실행 평균으로 계산한 양변에는 몬테카를로 오차가 남는다.")
log(f"d_n ≥ 0 인 비율 (식 5의 결과): {np.mean(d >= -1e-12):.3f}")

(OUT / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
print(f"\n저장: {OUT}")
