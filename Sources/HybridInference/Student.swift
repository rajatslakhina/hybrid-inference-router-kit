import Foundation

// MARK: - Core value types

/// A class the feature can output, e.g. `billing` or `bug`.
public struct ClassLabel: RawRepresentable, Hashable, Comparable, Sendable, Codable,
    ExpressibleByStringLiteral, CustomStringConvertible
{
    public let rawValue: String
    public init(rawValue: String) { self.rawValue = rawValue }
    public init(_ rawValue: String) { self.rawValue = rawValue }
    public init(stringLiteral value: String) { self.rawValue = value }
    public var description: String { rawValue }
    public static func < (lhs: ClassLabel, rhs: ClassLabel) -> Bool { lhs.rawValue < rhs.rawValue }
}

/// One piece of supervised data: an input and the label a trusted source gave it.
public struct LabeledExample: Hashable, Sendable, Codable {
    public let text: String
    public let label: ClassLabel
    public init(text: String, label: ClassLabel) {
        self.text = text
        self.label = label
    }
}

public struct ScoredLabel: Equatable, Sendable {
    public let label: ClassLabel
    public let probability: Double
}

/// A probability distribution over labels, sorted most-likely first (ties broken by label name,
/// so the order is deterministic).
public struct Distribution: Equatable, Sendable {
    public let ranked: [ScoredLabel]

    public init(probabilities: [ClassLabel: Double]) {
        ranked = probabilities
            .map { ScoredLabel(label: $0.key, probability: $0.value) }
            .sorted { lhs, rhs in
                lhs.probability != rhs.probability ? lhs.probability > rhs.probability : lhs.label < rhs.label
            }
    }

    public var top: ScoredLabel? { ranked.first }

    /// Gap between the two most likely labels. With a single label it is that label's probability;
    /// with none it is `0`.
    public var margin: Double {
        guard let first = ranked.first else { return 0 }
        guard ranked.count > 1 else { return first.probability }
        return first.probability - ranked[1].probability
    }
}

/// What the on-device student says about one input, plus the signals the router needs to decide
/// whether to trust it.
public struct StudentPrediction: Equatable, Sendable {
    public let modelVersion: String
    public let distribution: Distribution
    /// Tokens in the input.
    public let tokenCount: Int
    /// Tokens the model saw during training. A low share means the input is out of distribution.
    public let knownTokenCount: Int

    public init(modelVersion: String, distribution: Distribution, tokenCount: Int, knownTokenCount: Int) {
        self.modelVersion = modelVersion
        self.distribution = distribution
        self.tokenCount = max(tokenCount, 0)
        self.knownTokenCount = min(max(knownTokenCount, 0), max(tokenCount, 0))
    }

    public var knownTokenFraction: Double { Saturating.ratio(knownTokenCount, tokenCount) }
}

// MARK: - Student protocol

/// The small, app-specific model that serves most traffic on device.
///
/// In production this is a Core ML / Create ML text classifier. The router only depends on this
/// protocol, so swapping the bundled Naive Bayes for a Core ML model touches one adapter.
public protocol StudentModel: Sendable {
    var version: String { get }
    /// Every label the model can output, sorted.
    var labels: [ClassLabel] { get }
    func predict(_ text: String) -> StudentPrediction
}

public enum TrainingError: Error, Equatable, Sendable {
    case invalidSmoothing(Double)
    case noUsableExamples
    case needsAtLeastTwoLabels(found: Int)
}

// MARK: - Bundled student: multinomial Naive Bayes

/// A multinomial Naive Bayes text classifier with Laplace smoothing, trained in milliseconds on
/// device from teacher-labelled examples.
///
/// It is deliberately simple: the point of this package is the system around the model (routing,
/// budgets, the feedback loop and the promotion gate), not the model. Naive Bayes is known to be
/// over-confident, which is exactly why the router never uses a raw probability as its threshold:
/// `ThresholdCalibrator` picks the threshold per model version from a held-out eval set.
public struct NaiveBayesStudent: StudentModel {
    struct LabelStats: Sendable {
        let logPrior: Double
        let tokenCounts: [String: Int]
        let totalTokens: Int
    }

    public let version: String
    public let labels: [ClassLabel]
    public let trainingExampleCount: Int
    let stats: [ClassLabel: LabelStats]
    let vocabulary: Set<String>
    let alpha: Double

    /// Upper bound on `alpha`. Far beyond any useful smoothing, and low enough that `alpha × vocabulary`
    /// can never overflow to infinity and turn every probability into NaN.
    public static let maxAlpha = 1e6

    /// - Throws: `TrainingError` when smoothing is not in `(0, maxAlpha]`, when no example has
    ///   any tokens, or when fewer than two distinct labels remain (a one-class router is pointless).
    public init(training examples: [LabeledExample], version: String, alpha: Double = 1.0) throws {
        guard alpha.isFinite, alpha > 0, alpha <= Self.maxAlpha else { throw TrainingError.invalidSmoothing(alpha) }

        var documents: [ClassLabel: Int] = [:]
        var counts: [ClassLabel: [String: Int]] = [:]
        var totals: [ClassLabel: Int] = [:]
        var vocabulary = Set<String>()
        var usable = 0

        for example in examples {
            let tokens = Tokenizer.tokens(example.text)
            guard !tokens.isEmpty else { continue }
            usable = Saturating.add(usable, 1)
            documents[example.label] = Saturating.add(documents[example.label, default: 0], 1)
            var labelCounts = counts[example.label, default: [:]]
            for token in tokens {
                labelCounts[token] = Saturating.add(labelCounts[token, default: 0], 1)
                vocabulary.insert(token)
            }
            counts[example.label] = labelCounts
            totals[example.label] = Saturating.add(totals[example.label, default: 0], tokens.count)
        }

        guard usable > 0 else { throw TrainingError.noUsableExamples }
        guard documents.count >= 2 else { throw TrainingError.needsAtLeastTwoLabels(found: documents.count) }

        var stats: [ClassLabel: LabelStats] = [:]
        for (label, documentCount) in documents {
            // documentCount >= 1 and usable >= documentCount, so the log argument is in (0, 1].
            stats[label] = LabelStats(
                logPrior: log(Double(documentCount) / Double(usable)),
                tokenCounts: counts[label, default: [:]],
                totalTokens: totals[label, default: 0]
            )
        }

        self.version = version
        self.labels = documents.keys.sorted()
        self.trainingExampleCount = usable
        self.stats = stats
        self.vocabulary = vocabulary
        self.alpha = alpha
    }

    public func predict(_ text: String) -> StudentPrediction {
        let tokens = Tokenizer.tokens(text)
        let known = tokens.filter { vocabulary.contains($0) }
        // Unknown tokens carry no evidence in multinomial NB; they only lower `knownTokenFraction`.
        let vocabularySize = Double(vocabulary.count)

        var logScores: [ClassLabel: Double] = [:]
        for (label, labelStats) in stats {
            // alpha > 0 and vocabularySize >= 1, so the denominator is strictly positive. The log of the
            // numerator and denominator are taken separately: for a tiny alpha (down to the smallest
            // subnormal) the ratio would underflow to 0 and log(0) = -inf would turn the softmax into NaN.
            let logDenominator = log(Double(labelStats.totalTokens) + alpha * vocabularySize)
            var score = labelStats.logPrior
            for token in known {
                score += log(Double(labelStats.tokenCounts[token, default: 0]) + alpha) - logDenominator
            }
            logScores[label] = score
        }

        // Numerically stable softmax: subtract the max so the largest term is exp(0) = 1.
        let maxScore = logScores.values.max() ?? 0
        var probabilities: [ClassLabel: Double] = [:]
        var sum = 0.0
        for (label, score) in logScores {
            let value = exp(score - maxScore)
            probabilities[label] = value
            sum += value
        }
        if sum > 0 {
            for (label, value) in probabilities { probabilities[label] = value / sum }
        }

        return StudentPrediction(
            modelVersion: version,
            distribution: Distribution(probabilities: probabilities),
            tokenCount: tokens.count,
            knownTokenCount: known.count
        )
    }
}
