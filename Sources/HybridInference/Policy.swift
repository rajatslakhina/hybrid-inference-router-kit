import Foundation

/// Why the router decided the student's answer was not good enough to ship.
public enum EscalationReason: Equatable, Sendable {
    case emptyInput
    case lowConfidence(Double)
    case narrowMargin(Double)
    case outOfDistribution(knownTokenFraction: Double)
}

/// The rule that decides "answer on device" vs "ask the teacher".
///
/// Three independent signals, because each one catches a failure the others miss:
/// - **confidence**: the student's top probability is below the calibrated threshold;
/// - **margin**: the top two labels are too close (a confident-looking coin flip);
/// - **known-token share**: most of the input is vocabulary the student never trained on, which is
///   how an out-of-distribution input looks to a bag-of-words model even when it sounds sure.
public struct EscalationPolicy: Equatable, Sendable {
    public let minConfidence: Double
    public let minMargin: Double
    public let minKnownTokenFraction: Double

    /// Values are clamped to `0...1`. `NaN` becomes `1`, the most conservative setting, so a bad
    /// config escalates everything rather than silently trusting the student.
    public init(minConfidence: Double, minMargin: Double = 0, minKnownTokenFraction: Double = 0) {
        self.minConfidence = Saturating.unit(minConfidence, nanValue: 1)
        self.minMargin = Saturating.unit(minMargin, nanValue: 1)
        self.minKnownTokenFraction = Saturating.unit(minKnownTokenFraction, nanValue: 1)
    }

    /// Escalate everything. The safe policy for a model that has not been calibrated yet.
    public static let escalateAll = EscalationPolicy(minConfidence: 1, minMargin: 1, minKnownTokenFraction: 1)

    /// `nil` means "answer locally".
    public func escalationReason(for prediction: StudentPrediction) -> EscalationReason? {
        guard prediction.tokenCount > 0, let top = prediction.distribution.top else { return .emptyInput }
        if prediction.knownTokenFraction < minKnownTokenFraction {
            return .outOfDistribution(knownTokenFraction: prediction.knownTokenFraction)
        }
        if top.probability < minConfidence { return .lowConfidence(top.probability) }
        if prediction.distribution.margin < minMargin { return .narrowMargin(prediction.distribution.margin) }
        // A threshold of exactly 1 means "never trust": a probability of 1.0 is not proof.
        if minConfidence >= 1 { return .lowConfidence(top.probability) }
        return nil
    }
}

/// The confidence threshold chosen for one model version, and what it buys on the eval set.
public struct CalibrationResult: Equatable, Sendable {
    /// `nil` when no threshold reaches the target precision: escalate everything.
    public let threshold: Double?
    /// Share of eval examples the student would answer locally at this threshold.
    public let coverage: Double
    /// Accuracy on those locally answered examples. `nil` when coverage is zero.
    public let localPrecision: Double?

    /// The policy to install alongside the model. Margin and out-of-distribution floors are kept.
    public func policy(minMargin: Double = 0, minKnownTokenFraction: Double = 0) -> EscalationPolicy {
        guard let threshold else { return .escalateAll }
        return EscalationPolicy(
            minConfidence: threshold,
            minMargin: minMargin,
            minKnownTokenFraction: minKnownTokenFraction
        )
    }
}

/// Picks the lowest confidence threshold whose locally answered slice of the eval set still meets
/// a precision target. Lower threshold = more traffic on device = less spend.
///
/// Thresholds never carry over between model versions: a retrained model's probabilities mean
/// something different, so each candidate is calibrated on its own before the promotion gate sees it.
public enum ThresholdCalibrator {
    public static func calibrate(
        model: any StudentModel,
        evalSet: EvalSet,
        targetPrecision: Double
    ) -> CalibrationResult {
        let target = Saturating.unit(targetPrecision, nanValue: 1)
        let scored: [(confidence: Double, correct: Bool)] = evalSet.examples.map { example in
            let top = model.predict(example.text).distribution.top
            return (top?.probability ?? 0, top?.label == example.label)
        }
        .sorted { $0.confidence > $1.confidence }

        var correct = 0
        var best: (count: Int, threshold: Double, correct: Int)?
        for index in scored.indices {
            if scored[index].correct { correct += 1 }
            // Only cut between distinct confidences: the policy uses ">=", so every example tied with
            // the threshold is answered locally and has to be counted in the precision.
            let isLast = index + 1 == scored.count
            guard isLast || scored[index + 1].confidence < scored[index].confidence else { continue }
            let count = index + 1
            if Saturating.ratio(correct, count) >= target {
                best = (count, scored[index].confidence, correct)
            }
        }

        // A threshold of 1.0 would be read as "never trust" by the policy, so it cannot be used.
        guard let best, best.threshold < 1 else {
            return CalibrationResult(threshold: nil, coverage: 0, localPrecision: nil)
        }
        return CalibrationResult(
            threshold: best.threshold,
            coverage: Saturating.ratio(best.count, scored.count),
            localPrecision: Saturating.ratio(best.correct, best.count)
        )
    }
}
