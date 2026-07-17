from collections.abc import Sequence

import numpy as np


class DetectionCalculator:
    def __init__(self, prediction, target, thresholds: Sequence[float]) -> None:

        assert prediction.shape == target.shape, (
            f"Prediction {prediction.shape} and label {target.shape} are not in the same shape!"
        )

        self.hits = np.zeros(len(thresholds))
        self.misses = np.zeros(len(thresholds))
        self.false_alarms = np.zeros(len(thresholds))
        self.correct_rejections = np.zeros(len(thresholds))

        for i, threshold in enumerate(thresholds):
            self.hits[i] = np.sum(prediction[target > threshold] > threshold)
            self.misses[i] = np.sum(prediction[target > threshold] <= threshold)
            self.false_alarms[i] = np.sum(prediction[target <= threshold] > threshold)
            self.correct_rejections[i] = np.sum(
                prediction[target <= threshold] <= threshold
            )

    def calculate_CSI(
        self,
        hits: np.ndarray | None = None,
        misses: np.ndarray | None = None,
        false_alarms: np.ndarray | None = None,
    ) -> np.ndarray:
        hits = hits if hits is not None else self.hits
        misses = misses if misses is not None else self.misses
        false_alarms = false_alarms if false_alarms is not None else self.false_alarms

        return hits / (hits + misses + false_alarms)

    def calculate_SR(
        self, hits: np.ndarray | None = None, false_alarms: np.ndarray | None = None
    ) -> np.ndarray:
        hits = hits if hits is not None else self.hits
        false_alarms = false_alarms if false_alarms is not None else self.false_alarms

        return 1 - (false_alarms / (hits + false_alarms))

    def calculate_POD(
        self, hits: np.ndarray | None = None, misses: np.ndarray | None = None
    ) -> np.ndarray:
        hits = hits if hits is not None else self.hits
        misses = misses if misses is not None else self.misses

        return hits / (hits + misses)

    def calculate_FAR(
        self,
        false_alarms: np.ndarray | None = None,
        correct_rejections: np.ndarray | None = None,
    ) -> np.ndarray:
        false_alarms = false_alarms if false_alarms is not None else self.false_alarms
        correct_rejections = (
            correct_rejections
            if correct_rejections is not None
            else self.correct_rejections
        )

        return false_alarms / (false_alarms + correct_rejections)

    def calculate_bias(
        self,
        hits: np.ndarray | None = None,
        misses: np.ndarray | None = None,
        false_alarms: np.ndarray | None = None,
    ) -> np.ndarray:
        hits = hits if hits is not None else self.hits
        misses = misses if misses is not None else self.misses
        false_alarms = false_alarms if false_alarms is not None else self.false_alarms

        return self.calculate_POD(hits, misses) / self.calculate_SR(hits, false_alarms)

    def calculate_bcCSI(
        self,
        hits: np.ndarray | None = None,
        misses: np.ndarray | None = None,
        false_alarms: np.ndarray | None = None,
    ) -> np.ndarray:

        hits = hits if hits is not None else self.hits
        misses = misses if misses is not None else self.misses
        false_alarms = false_alarms if false_alarms is not None else self.false_alarms

        CSI = self.calculate_CSI(hits, misses, false_alarms)
        bias = self.calculate_bias(hits, misses, false_alarms)
        bcCSI = np.where(bias < 1, CSI * bias, CSI / bias)

        return bcCSI
