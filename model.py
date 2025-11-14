import numpy as np
from typing import Optional, Tuple

class MulticlassPerceptron:
    
    def __init__(self, lr: float = 1.0, n_epochs: int = 40, shuffle: bool = True, random_state: Optional[int] = None):
        self.lr, self.n_epochs, self.shuffle = lr, n_epochs, shuffle
        self.rng = np.random.default_rng(random_state)
        self.W = None  # shape: (n_classes, d+1)
        self.history_ = {"train_acc": [], "test_acc": [], "loss": []}

    @staticmethod
    def _augment(X: np.ndarray) -> np.ndarray:
        return np.hstack([X, np.ones((X.shape[0], 1), dtype=X.dtype)])

    def _scores(self, Xa: np.ndarray) -> np.ndarray:
        return Xa @ self.W.T  # (n, K)

    def fit(self, X: np.ndarray, y: np.ndarray, eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        n, d = X.shape
        K = int(np.max(y)) + 1
        Xa = self._augment(X)

        if self.W is None:
            self.W = np.zeros((K, d + 1), dtype=float)

        Xte = yte = None
        if eval_set is not None:
            Xte = np.asarray(eval_set[0], dtype=float)
            yte = np.asarray(eval_set[1], dtype=int)

        self.history_ = {"train_acc": [], "test_acc": [], "loss": []}

        print("\nEpoch | Train Acc | Test Acc | Loss")
        print("-"*40)

        for epoch in range(1, self.n_epochs + 1):
            idx = np.arange(n)
            if self.shuffle:
                self.rng.shuffle(idx)

            # online updates
            for i in idx:
                s = self.W @ Xa[i]
                y_true = y[i]
                y_pred = int(np.argmax(s))
                if y_pred != y_true:
                    self.W[y_true] += self.lr * Xa[i]
                    self.W[y_pred] -= self.lr * Xa[i]

            # compute metrics
            train_acc = self.score(X, y)
            loss = self.per_epoch_loss(X, y)
            self.history_["train_acc"].append(train_acc)
            self.history_["loss"].append(loss)
            test_acc = None
            if Xte is not None:
                test_acc = self.score(Xte, yte)
                self.history_["test_acc"].append(test_acc)

            # live logging
            if test_acc is not None:
                print(f"{epoch:5d} | {train_acc:10.4f} | {test_acc:9.4f} | {loss:7.4f}")
            else:
                print(f"{epoch:5d} | {train_acc:10.4f} | {'-':>9} | {loss:7.4f}")

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        Xa = self._augment(np.asarray(X, dtype=float))
        return self._scores(Xa)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.decision_function(X), axis=1)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return np.mean(self.predict(X) == np.asarray(y, dtype=int))

    def per_epoch_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        Xa = self._augment(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=int)
        S = self._scores(Xa)                     # (n, K)
        s_true = S[np.arange(X.shape[0]), y]     # (n,)
        S[np.arange(X.shape[0]), y] = -np.inf
        s_pred = np.max(S, axis=1)
        hinge = np.maximum(0.0, 1.0 + s_pred - s_true)
        return float(np.mean(hinge))