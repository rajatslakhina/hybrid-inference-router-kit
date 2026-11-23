import Foundation

// MARK: - Eval set

public enum EvalSetError: Error, Equatable, Sendable {
    case empty
    case blankExample
    case conflictingLabels(normalizedText: String)
}

/// A frozen, versioned set of held-out examples: the yardstick every model version is judged on.
///
/// It is a first-class artifact with a content fingerprint, so a promotion decision can prove that
/// the incumbent and the candidate were scored on exactly the same data.
public struct EvalSet: Sendable {
    public let version: String
    /// De-duplicated by normalised text and sorted, so order of construction never matters.
    public let examples: [LabeledExample]
    /// FNV-1a over the canonical content. Changes when any text or label changes.
    public let fingerprint: UInt64
    let normalizedTexts: Set<String>

    /// - Throws: `.empty` with no examples, `.blankExample` when an example has no tokens, and
    ///   `.conflictingLabels` when two examples normalise to the same text with different labels
    ///   (an eval set that disagrees with itself cannot score anything).
    public init(version: String, examples: [LabeledExample]) throws {
        guard !examples.isEmpty else { throw EvalSetError.empty }
        var byText: [String: LabeledExample] = [:]
        for example in examples {
            let key = Tokenizer.normalized(example.text)
            guard !key.isEmpty else { throw EvalSetError.blankExample }
            if let existing = byText[key] {
                guard existing.label == example.label else {
                    throw EvalSetError.conflictingLabels(normalizedText: key)
                }
                continue
            }
            byText[key] = example
        }
        let keys = byText.keys.sorted()
        self.version = version
        self.examples = keys.compactMap { byText[$0] }
        self.normalizedTexts = Set(keys)
        self.fingerprint = StableHash.fnv1a64(
            keys.compactMap { key in byText[key].map { "\(key)\t\($0.label.rawValue)" } }.joined(separator: "\n")
        )
    }

    /// True when `text` normalises to an eval example. Used by the leak guard.
    public func contains(text: String) -> Bool {
        normalizedTexts.contains(Tokenizer.normalized(text))
    }
}

// MARK: - Evaluation

public struct EvalReport: Equatable, Sendable {
    public let modelVersion: String
    public let evalSetVersion: String
    public let evalFingerprint: UInt64
    public let exampleCount: Int
    /// Top-1 accuracy over the whole eval set, ignoring the policy.
    public let accuracy: Double
    /// Recall for every label that appears in the eval set.
    public let perLabelRecall: [ClassLabel: Double]
    /// Share answered on device under the model's own calibrated policy.
    public let localCoverage: Double
    /// Accuracy on the locally answered share. `nil` when nothing is answered locally.
    public let localPrecision: Double?
}

public enum Evaluator {
    public static func evaluate(
        _ model: any StudentModel,
        on evalSet: EvalSet,
        policy: EscalationPolicy
    ) -> EvalReport {
        var correct = 0
        var local = 0
        var localCorrect = 0
        var totalsByLabel: [ClassLabel: Int] = [:]
        var correctByLabel: [ClassLabel: Int] = [:]

        for example in evalSet.examples {
            let prediction = model.predict(example.text)
            let isCorrect = prediction.distribution.top?.label == example.label
            totalsByLabel[example.label] = Saturating.add(totalsByLabel[example.label, default: 0], 1)
            if isCorrect {
                correct = Saturating.add(correct, 1)
                correctByLabel[example.label] = Saturating.add(correctByLabel[example.label, default: 0], 1)
            }
            if policy.escalationReason(for: prediction) == nil {
                local = Saturating.add(local, 1)
                if isCorrect { localCorrect = Saturating.add(localCorrect, 1) }
            }
        }

        let count = evalSet.examples.count
        var recall: [ClassLabel: Double] = [:]
        for (label, total) in totalsByLabel {
            recall[label] = Saturating.ratio(correctByLabel[label, default: 0], total)
        }
        return EvalReport(
            modelVersion: model.version,
            evalSetVersion: evalSet.version,
            evalFingerprint: evalSet.fingerprint,
            exampleCount: count,
            accuracy: Saturating.ratio(correct, count),
            perLabelRecall: recall,
            localCoverage: Saturating.ratio(local, count),
            localPrecision: local > 0 ? Saturating.ratio(localCorrect, local) : nil
        )
    }
}

// MARK: - Promotion gate

public enum GateFailure: Equatable, Sendable {
    case evalSetMismatch(incumbent: UInt64, candidate: UInt64)
    case accuracyRegressed(from: Double, to: Double)
    case recallRegressed(label: ClassLabel, from: Double, to: Double)
    case localPrecisionBelowFloor(value: Double?, floor: Double)
    case noImprovement
}

public enum PromotionDecision: Equatable, Sendable {
    case promote
    case reject([GateFailure])

    public var isPromotion: Bool { self == .promote }
}

/// Decides whether a retrained student may replace the one in production.
///
/// A retrain is only worth shipping if it is no worse on any label, still precise where it answers
/// locally, and better at something (accuracy or on-device coverage). A teacher that drifted, a
/// poisoned harvest or a bug in the training code all show up here as a rejection instead of as a
/// silent quality drop in production.
public struct PromotionGate: Equatable, Sendable {
    public let maxAccuracyDrop: Double
    public let maxRecallDrop: Double
    public let minLocalPrecision: Double

    /// Tolerance for floating-point noise in ratio comparisons.
    static let epsilon = 1e-9

    public init(maxAccuracyDrop: Double = 0, maxRecallDrop: Double = 0.05, minLocalPrecision: Double = 0.9) {
        self.maxAccuracyDrop = Saturating.unit(maxAccuracyDrop, nanValue: 0)
        self.maxRecallDrop = Saturating.unit(maxRecallDrop, nanValue: 0)
        self.minLocalPrecision = Saturating.unit(minLocalPrecision, nanValue: 1)
    }

    /// - Parameter incumbent: `nil` for the very first model, which only has to clear the precision floor.
    public func decide(candidate: EvalReport, incumbent: EvalReport?) -> PromotionDecision {
        var failures: [GateFailure] = []

        if let precision = candidate.localPrecision, precision + Self.epsilon >= minLocalPrecision {
            // Precision floor met.
        } else {
            failures.append(.localPrecisionBelowFloor(value: candidate.localPrecision, floor: minLocalPrecision))
        }

        guard let incumbent else {
            return failures.isEmpty ? .promote : .reject(failures)
        }

        guard incumbent.evalFingerprint == candidate.evalFingerprint else {
            // Nothing else is comparable across different eval sets.
            return .reject([.evalSetMismatch(incumbent: incumbent.evalFingerprint, candidate: candidate.evalFingerprint)])
        }

        if incumbent.accuracy - candidate.accuracy > maxAccuracyDrop + Self.epsilon {
            failures.append(.accuracyRegressed(from: incumbent.accuracy, to: candidate.accuracy))
        }
        for label in incumbent.perLabelRecall.keys.sorted() {
            let before = incumbent.perLabelRecall[label, default: 0]
            let after = candidate.perLabelRecall[label, default: 0]
            if before - after > maxRecallDrop + Self.epsilon {
                failures.append(.recallRegressed(label: label, from: before, to: after))
            }
        }
        let improvesAccuracy = candidate.accuracy > incumbent.accuracy + Self.epsilon
        let improvesCoverage = candidate.localCoverage > incumbent.localCoverage + Self.epsilon
        if !improvesAccuracy && !improvesCoverage { failures.append(.noImprovement) }

        return failures.isEmpty ? .promote : .reject(failures)
    }
}

// MARK: - Training set assembly

public struct TrainingSet: Equatable, Sendable {
    public let examples: [LabeledExample]
    /// Examples dropped because they normalise to an eval-set example.
    public let excludedForLeak: Int
    /// Seed examples replaced by a harvested (teacher) label for the same text.
    public let seedOverridden: Int
}

/// Merges the hand-written seed set with teacher-labelled examples harvested from production.
public enum TrainingSetBuilder {
    /// Harvested labels win over seed labels for the same normalised text, and anything that appears
    /// in the eval set is dropped: training on the yardstick would make the gate meaningless.
    public static func build(
        seed: [LabeledExample],
        harvested: [LabeledExample],
        excluding evalSet: EvalSet
    ) -> TrainingSet {
        var order: [String] = []
        var byText: [String: LabeledExample] = [:]
        var fromSeed = Set<String>()
        var overridden = 0
        var leaked = 0

        func add(_ example: LabeledExample, isSeed: Bool) {
            let key = Tokenizer.normalized(example.text)
            guard !key.isEmpty else { return }
            guard !evalSet.normalizedTexts.contains(key) else {
                leaked = Saturating.add(leaked, 1)
                return
            }
            if byText[key] == nil {
                order.append(key)
            } else if !isSeed, fromSeed.contains(key) {
                overridden = Saturating.add(overridden, 1)
                fromSeed.remove(key)
            }
            if isSeed { fromSeed.insert(key) }
            byText[key] = example
        }

        for example in seed { add(example, isSeed: true) }
        for example in harvested { add(example, isSeed: false) }

        return TrainingSet(
            examples: order.compactMap { byText[$0] },
            excludedForLeak: leaked,
            seedOverridden: overridden
        )
    }
}
