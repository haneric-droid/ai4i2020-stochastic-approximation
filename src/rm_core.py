"""
Robbins–Monro(1951) 확률근사 실험의 공통 도구.

- 오라클 O(x): ai4i2020.csv에서 "입력 x 근처 행 하나를 무작위로 뽑아 고장 여부(0/1)를 돌려주는" 모의 현장.
  근처 행을 복원추출하는 것은 Bernoulli(p̂(x))를 뽑는 것과 같으므로, 미리 계산한 국소 고장률 p̂로 벡터화한다.
  → E[O(x)] = p̂(x) ≈ M(x),  |O(x)| ≤ 1  (논문 식 4: 유계성)
- 참 θ*: 등온회귀(PAV)로 단조 함수 M̂(x)를 적합하고 M̂(θ*) = α 를 푼다.
- robbins_monro(): x_{n+1} = x_n + a_n (α − ȳ_n)  (논문 식 7, 45, 50) 을 R개 독립 실행으로 동시에 돌린다.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "초기제공자료" / "ai4i2020.csv"


def load_data(type_filter=None):
    d = pd.read_csv(DATA, encoding="utf-8-sig")
    if type_filter is not None:
        d = d[d["Type"] == type_filter]
    return d


class Oracle:
    """입력 변수 col의 값 x를 받으면 |w − x| ≤ half_width 인 행 하나의 고장 여부를 돌려준다."""

    def __init__(self, df, col, half_width, lo, hi, step):
        self.col, self.lo, self.hi, self.step = col, lo, hi, step
        self.grid = np.arange(lo, hi + step / 2, step)
        w = df[col].to_numpy()
        f = df["Machine failure"].to_numpy()
        p, m = [], []
        for g in self.grid:
            mask = np.abs(w - g) <= half_width
            if not mask.any():  # 근처 행이 없으면 가장 가까운 행
                mask = np.abs(w - g) == np.abs(w - g).min()
            p.append(f[mask].mean())
            m.append(mask.sum())
        self.p = np.array(p)      # 국소 고장률 p̂(grid)
        self.count = np.array(m)  # 국소 표본 수

    def idx(self, x):
        return np.clip(np.rint((x - self.lo) / self.step).astype(int), 0, len(self.grid) - 1)

    def M(self, x):
        """오라클의 기댓값 E[O(x)] (실험 내부의 '참' 회귀함수)."""
        return self.p[self.idx(x)]

    def __call__(self, x, rng, r=1):
        """r번 관측한 평균 ȳ (r=1이면 0/1)."""
        return rng.binomial(r, self.M(x)) / r


def pav(y, w):
    """가중 등온회귀 (Pool-Adjacent-Violators): 비감소 수열 중 가중제곱오차 최소."""
    vals, wts, sizes = [], [], []
    for yi, wi in zip(y, w):
        vals.append(yi); wts.append(wi); sizes.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v2, w2, s2 = vals.pop(), wts.pop(), sizes.pop()
            v1, w1, s1 = vals.pop(), wts.pop(), sizes.pop()
            vals.append((v1 * w1 + v2 * w2) / (w1 + w2)); wts.append(w1 + w2); sizes.append(s1 + s2)
    return np.repeat(vals, sizes)


def isotonic_fit(df, col, lo, hi, bin_width):
    """구간별 고장률에 PAV를 적용한 단조 M̂ (구간 중심값, M̂값)."""
    edges = np.arange(lo, hi + bin_width, bin_width)
    b = pd.cut(df[col], edges, include_lowest=True)
    g = df.groupby(b, observed=True)["Machine failure"].agg(["mean", "count"])
    centers = np.array([iv.mid for iv in g.index])
    return centers, pav(g["mean"].to_numpy(), g["count"].to_numpy())


def root(xs, ms, alpha):
    """단조 계단/선형보간 함수에서 M̂(θ) = α 인 θ (없으면 None: 중간값정리의 전제 실패)."""
    if alpha <= ms[0] or alpha > ms[-1]:
        return None
    k = np.argmax(ms >= alpha)
    x0, x1, m0, m1 = xs[k - 1], xs[k], ms[k - 1], ms[k]
    return x0 + (alpha - m0) * (x1 - x0) / (m1 - m0) if m1 > m0 else x1


# ---------- 걸음 크기 수열 a_n ----------
SCHEDULES = {
    "c/n  (1/n형, 정리 조건 충족)": lambda n, c: c / n,
    "c/n^0.6  (Σa=∞, Σa²<∞ 충족)": lambda n, c: c / n ** 0.6,
    "c/√n  (Σa²=∞ 위반)": lambda n, c: c / np.sqrt(n),
    "상수 c  (Σa²=∞ 위반)": lambda n, c: c * np.ones_like(n, dtype=float),
    "c/n²  (Σa=∞ 위반)": lambda n, c: c / n ** 2,
}


def robbins_monro(oracle, alpha, a_fn, x1, N, R, rng, r=1, bounds=None, theta=None, track=False):
    """
    R개 독립 실행을 동시에 N스텝 진행.
    반환: x 궤적 (N+1, R)  [x_1 ... x_{N+1}],
          track=True면 논문 식 (12)(13)의 d_n, e_n 표본평균도 함께.
    """
    lo, hi = bounds if bounds else (oracle.lo, oracle.hi)
    x = np.full(R, float(x1)) if np.isscalar(x1) else np.asarray(x1, float).copy()
    path = np.empty((N + 1, R)); path[0] = x
    d = np.empty(N); e = np.empty(N)
    for n in range(1, N + 1):
        if track:
            M = oracle.M(x)
            d[n - 1] = np.mean((x - theta) * (M - alpha))                 # d_n = E[(x_n−θ)(M(x_n)−α)]
            e[n - 1] = np.mean(M * (1 - alpha) ** 2 + (1 - M) * alpha ** 2)  # e_n = E[(y−α)²]
        y = oracle(x, rng, r)
        x = np.clip(x + a_fn(n) * (alpha - y), lo, hi)  # 사영: 유계 구간으로 자름
        path[n] = x
    return (path, d, e) if track else path


def b_n(path, theta):
    """b_n = E(x_n − θ)²  (R개 실행 평균)."""
    return np.mean((path - theta) ** 2, axis=1)
