"""Direct multi-output ridge forecaster (Phase 5 reference learner).

λ is chosen on an internal patient-level holdout of the *training* records, using one
eigendecomposition for the whole λ grid, then the model is refit on all training
records. The validation fold is never seen during fitting.
"""

from __future__ import annotations

import numpy as np
from scipy import linalg

LAMBDAS = np.logspace(-1, 6, 15)


class RidgeForecaster:
    def __init__(self, lambdas: np.ndarray = LAMBDAS, holdout: float = 0.2, seed: int = 0):
        self.lambdas, self.holdout, self.seed = np.asarray(lambdas, float), holdout, seed

    def fit(self, x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> RidgeForecaster:
        """``x`` (n, p), ``y`` (n, q); ``groups`` = patient IDs for the internal holdout."""
        x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
        patients = np.unique(groups)
        rng = np.random.default_rng(self.seed)
        held = np.isin(
            groups, rng.choice(patients, round(self.holdout * len(patients)), replace=False)
        )
        tr = ~held
        xm, ym = x[tr].mean(0), y[tr].mean(0)
        xc = x[tr] - xm
        s, v = linalg.eigh(xc.T @ xc)
        vb = v.T @ (xc.T @ (y[tr] - ym))
        xv = (x[held] - xm) @ v
        lambdas = list(self.lambdas)
        maes = [self._holdout_mae(xv, vb, s, ym, y[held], lam) for lam in lambdas]
        # An optimum at either edge of the grid is not a real optimum: extend the grid in that
        # direction (cheap: the eigendecomposition is reused) until it is interior.
        for _ in range(4):
            k = int(np.argmin(maes))
            if 0 < k < len(lambdas) - 1:
                break
            step = lambdas[1] / lambdas[0]
            new = (
                [lambdas[-1] * step**i for i in (1, 2, 3)]
                if k == len(lambdas) - 1
                else [lambdas[0] / step**i for i in (1, 2, 3)]
            )
            new_maes = [self._holdout_mae(xv, vb, s, ym, y[held], lam) for lam in new]
            if k == len(lambdas) - 1:
                lambdas, maes = lambdas + new, maes + new_maes
            else:
                lambdas, maes = new[::-1] + lambdas, new_maes[::-1] + maes
        self.lambdas_tried, self.holdout_mae = np.array(lambdas), maes
        self.lam = float(lambdas[int(np.argmin(maes))])
        self.x_mean, self.y_mean = x.mean(0), y.mean(0)
        xc = x - self.x_mean
        gram = xc.T @ xc
        gram[np.diag_indices_from(gram)] += self.lam
        self.coef = linalg.solve(gram, xc.T @ (y - self.y_mean), assume_a="pos")
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, np.float64) - self.x_mean) @ self.coef + self.y_mean

    @staticmethod
    def _holdout_mae(xv, vb, s, ym, y_held, lam) -> float:
        return float(np.abs(xv @ (vb / (s + lam)[:, None]) + ym - y_held).mean())
