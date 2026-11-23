import XCTest
@testable import HybridInference

final class FoundationsTests: XCTestCase {
    func testSaturatingAddClampsInsteadOfTrapping() {
        XCTAssertEqual(Saturating.add(Int.max, 1), Int.max)
        XCTAssertEqual(Saturating.add(Int.min, -1), Int.min)
        XCTAssertEqual(Saturating.add(2, 3), 5)
        XCTAssertEqual(Saturating.subtract(Int.min, 1), Int.min)
        XCTAssertEqual(Saturating.subtract(Int.max, -1), Int.max)
    }

    func testRatioIsZeroForNonPositiveDenominator() {
        XCTAssertEqual(Saturating.ratio(3, 0), 0)
        XCTAssertEqual(Saturating.ratio(3, -2), 0)
        XCTAssertEqual(Saturating.ratio(1, 4), 0.25)
    }

    func testUnitClampsAndMapsNaNToTheCallersSafeDefault() {
        XCTAssertEqual(Saturating.unit(.nan, nanValue: 1), 1)
        XCTAssertEqual(Saturating.unit(-.infinity, nanValue: 1), 0)
        XCTAssertEqual(Saturating.unit(2, nanValue: 0), 1)
    }

    /// Golden vectors from the FNV reference. A process-seeded hash (Swift's `Hasher`) would fail these.
    func testStableHashMatchesPublishedFNV1aVectors() {
        XCTAssertEqual(StableHash.fnv1a64(""), 0xcbf2_9ce4_8422_2325)
        XCTAssertEqual(StableHash.fnv1a64("a"), 0xaf63_dc4c_8601_ec8c)
        XCTAssertEqual(StableHash.fnv1a64("foobar"), 0x8594_4171_f739_67e8)
    }

    func testTokenizerLowercasesAndSplitsOnPunctuation() {
        XCTAssertEqual(Tokenizer.tokens("Refund?? NOW, please!"), ["refund", "now", "please"])
        XCTAssertEqual(Tokenizer.normalized("  ...  "), "")
        XCTAssertEqual(Tokenizer.normalized("Café 2FA"), "café 2fa")
    }
}

final class NaiveBayesStudentTests: XCTestCase {
    func testTrainingRejectsInvalidSmoothing() {
        XCTAssertThrowsError(try NaiveBayesStudent(training: Tickets.seed, version: "v", alpha: 0)) {
            XCTAssertEqual($0 as? TrainingError, .invalidSmoothing(0))
        }
        XCTAssertThrowsError(try NaiveBayesStudent(training: Tickets.seed, version: "v", alpha: .nan))
        // Huge but finite alpha would overflow alpha × vocabulary to infinity and make every probability NaN.
        XCTAssertThrowsError(try NaiveBayesStudent(training: Tickets.seed, version: "v", alpha: 1e308))
        XCTAssertNoThrow(try NaiveBayesStudent(training: Tickets.seed, version: "v", alpha: NaiveBayesStudent.maxAlpha))
    }

    func testTrainingRejectsEmptyAndTokenlessInput() {
        XCTAssertThrowsError(try NaiveBayesStudent(training: [], version: "v")) {
            XCTAssertEqual($0 as? TrainingError, .noUsableExamples)
        }
        let blank = [LabeledExample(text: "!!!", label: "a"), LabeledExample(text: " ", label: "b")]
        XCTAssertThrowsError(try NaiveBayesStudent(training: blank, version: "v")) {
            XCTAssertEqual($0 as? TrainingError, .noUsableExamples)
        }
    }

    func testTrainingRejectsASingleLabel() {
        let one = [LabeledExample(text: "a", label: "x"), LabeledExample(text: "b", label: "x")]
        XCTAssertThrowsError(try NaiveBayesStudent(training: one, version: "v")) {
            XCTAssertEqual($0 as? TrainingError, .needsAtLeastTwoLabels(found: 1))
        }
    }

    /// Hand-computed Laplace smoothing: vocab {a, b}; P(a|x) = (1+1)/(1+2) = 2/3, P(a|y) = 1/3,
    /// equal priors, so P(x | "a") = (2/3) / (2/3 + 1/3) = 2/3.
    func testProbabilitiesMatchHandComputedLaplaceSmoothing() throws {
        let model = try NaiveBayesStudent(
            training: [.init(text: "a", label: "x"), .init(text: "b", label: "y")], version: "v"
        )
        let prediction = model.predict("a")
        XCTAssertEqual(prediction.distribution.top?.label, "x")
        XCTAssertEqual(prediction.distribution.top?.probability ?? 0, 2.0 / 3.0, accuracy: 1e-12)
        XCTAssertEqual(prediction.distribution.margin, 1.0 / 3.0, accuracy: 1e-12)
    }

    /// With no known tokens, the posterior is exactly the prior: 3 billing vs 1 bug -> 0.75 / 0.25.
    func testUnknownVocabularyFallsBackToPriorsAndReportsZeroKnownTokens() throws {
        let model = try NaiveBayesStudent(
            training: [
                .init(text: "refund", label: "billing"), .init(text: "invoice", label: "billing"),
                .init(text: "charged", label: "billing"), .init(text: "crash", label: "bug"),
            ],
            version: "v"
        )
        let prediction = model.predict("zebra quantum")
        XCTAssertEqual(prediction.tokenCount, 2)
        XCTAssertEqual(prediction.knownTokenCount, 0)
        XCTAssertEqual(prediction.knownTokenFraction, 0)
        XCTAssertEqual(prediction.distribution.top?.label, "billing")
        XCTAssertEqual(prediction.distribution.top?.probability ?? 0, 0.75, accuracy: 1e-12)
    }

    /// With the smallest subnormal alpha, (0 + alpha) / denominator underflows to 0 and log(0) = -inf
    /// for every label on a mixed input; computing the logs separately keeps everything finite.
    func testTinySmoothingStillGivesFiniteNormalisedProbabilities() throws {
        let model = try NaiveBayesStudent(training: Tickets.seed, version: "v", alpha: 5e-324)
        let prediction = model.predict("refund crashes password")
        for entry in prediction.distribution.ranked { XCTAssertTrue(entry.probability.isFinite) }
        XCTAssertEqual(prediction.distribution.ranked.reduce(0) { $0 + $1.probability }, 1, accuracy: 1e-9)
    }

    func testEmptyInputHasZeroTokensAndProbabilitiesStillSumToOne() throws {
        let model = try NaiveBayesStudent(training: Tickets.seed, version: "v")
        let prediction = model.predict("")
        XCTAssertEqual(prediction.tokenCount, 0)
        XCTAssertEqual(prediction.knownTokenFraction, 0)
        let total = prediction.distribution.ranked.reduce(0) { $0 + $1.probability }
        XCTAssertEqual(total, 1, accuracy: 1e-9)
    }

    func testVeryLongInputStaysFiniteAndNormalised() throws {
        let model = try NaiveBayesStudent(training: Tickets.seed, version: "v")
        let prediction = model.predict(Array(repeating: "refund crashes password", count: 4_000).joined(separator: " "))
        XCTAssertEqual(prediction.distribution.ranked.count, 4)
        for entry in prediction.distribution.ranked { XCTAssertTrue(entry.probability.isFinite) }
        XCTAssertEqual(prediction.distribution.ranked.reduce(0) { $0 + $1.probability }, 1, accuracy: 1e-9)
    }

    func testLabelsAreSortedAndBlankExamplesAreSkipped() throws {
        let model = try NaiveBayesStudent(
            training: Tickets.seed + [.init(text: "???", label: "zzz")], version: "v"
        )
        XCTAssertEqual(model.labels, Tickets.labels)
        XCTAssertEqual(model.trainingExampleCount, Tickets.seed.count)
    }

    func testDistributionMarginHandlesOneAndZeroLabels() {
        XCTAssertEqual(Distribution(probabilities: [:]).margin, 0)
        XCTAssertNil(Distribution(probabilities: [:]).top)
        XCTAssertEqual(Distribution(probabilities: ["a": 0.7]).margin, 0.7)
    }

    func testDistributionBreaksTiesByLabelName() {
        let distribution = Distribution(probabilities: ["b": 0.5, "a": 0.5])
        XCTAssertEqual(distribution.ranked.map(\.label), ["a", "b"])
    }

    func testStudentPredictionClampsInconsistentCounts() {
        let prediction = StudentPrediction(
            modelVersion: "v", distribution: Distribution(probabilities: [:]), tokenCount: 2, knownTokenCount: 9
        )
        XCTAssertEqual(prediction.knownTokenCount, 2)
        XCTAssertEqual(prediction.knownTokenFraction, 1)
    }
}
