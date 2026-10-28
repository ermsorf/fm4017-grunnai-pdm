class HealthMonitor:

    def __init__(
        self,
        warning_threshold,
        alarm_threshold
    ):
        self.warning_threshold = warning_threshold
        self.alarm_threshold = alarm_threshold

    def evaluate(
        self,
        residual_vibr1,
        residual_vibr2
    ):

        score = max(
            residual_vibr1,
            residual_vibr2
        )

        if score >= self.alarm_threshold:
            return score, "ALARM"

        if score >= self.warning_threshold:
            return score, "WARNING"

        return score, "HEALTHY"