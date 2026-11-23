import XCTest
@testable import HybridInference

final class EscalationPolicyTests: XCTestCase {
    private let labels: [ClassLabel] = ["a", "b"]

    private func prediction(_ top: Double, tokens: Int = 3, known: Int = 3) -> StudentPrediction {
        StudentPrediction(
            modelVersion: "v",
            distribution: Distribution(probabilities: ["a": top, "b": 1 - top]),
            tokenCount: tokens,
            knownTokenCount: known
        )
    }

    func testConfidentKnownInputIsAnsweredLocally() {
        let policy = EscalationPolicy(minConfidence: 0.8, minMargin: 0.3, minKnownTokenFraction: 0.5)
        XCTAssertNil(policy.escalationReason(for: prediction(0.9)))
    }

    func testEachSignalEscalatesOnItsOwn() {
        let policy = EscalationPolicy(minConfidence: 0.8, minMargin: 0.7, minKnownTokenFraction: 0.5)
        XCTAssertEqual(policy.escalationReason(for: prediction(0.9, tokens: 0, known: 0)), .emptyInput)
        XCTAssertEqual(policy.escalationReason(for: prediction(0.9, tokens: 4, known: 1)),
                       .outOfDistribution(knownTokenFraction: 0.25))
        XCTAssertEqual(policy.escalationReason(for: prediction(0.6)), .lowConfidence(0.6))
        // 0.82 passes confidence but the margin (0.64) is below 0.7.
        guard case .narrowMargin(let margin)? = policy.escalationReason(for: prediction(0.82)) else {
            return XCTFail("expected narrowMargin")
        }
        XCTAssertEqual(margin, 0.64, accuracy: 1e-12)
    }

    func testNaNConfigurationFailsSafeByEscalating() {
        let policy = EscalationPolicy(minConfidence: .nan)
        XCTAssertEqual(policy.minConfidence, 1)
        XCTAssertNotNil(policy.escalationReason(for: prediction(0.99)))
    }

    func testThresholdOfOneNeverTrustsEvenACertainStudent() {
        XCTAssertEqual(EscalationPolicy.escalateAll.escalationReason(for: prediction(1.0)), .lowConfidence(1.0))
    }
}

final class ThresholdCalibratorTests: XCTestCase {
    private let labels: [ClassLabel] = ["x", "y"]

    /// Builds an eval set of single-word texts, each labelled `x`, and a student whose confidence
    /// and correctness per text are fixed: `(confidence, correct)`.
    private func fixture(_ rows: [(Double, Bool)]) throws -> (FixedStudent, EvalSet) {
        var table: [String: ScoredLabel] = [:]
        var examples: [LabeledExample] = []
        for (index, row) in rows.enumerated() {
            let text = "item\(index)"
            examples.append(.init(text: text, label: "x"))
            table[text] = scored(row.1 ? "x" : "y", row.0)
        }
        let student = FixedStudent(labels: labels, table: table, defaultAnswer: scored("x", 0.5))
        return (student, try EvalSet(version: "e1", examples: examples))
    }

    func testPicksTheLowestThresholdThatMeetsTheTarget() throws {
        let (student, eval) = try fixture([(0.95, true), (0.9, true), (0.8, false), (0.7, true), (0.6, true)])

        let strict = ThresholdCalibrator.calibrate(model: student, evalSet: eval, targetPrecision: 1)
        XCTAssertEqual(strict.threshold, 0.9)
        XCTAssertEqual(strict.coverage, 0.4, accuracy: 1e-12)
        XCTAssertEqual(strict.localPrecision, 1)

        let loose = ThresholdCalibrator.calibrate(model: student, evalSet: eval, targetPrecision: 0.8)
        XCTAssertEqual(loose.threshold, 0.6)
        XCTAssertEqual(loose.coverage, 1, accuracy: 1e-12)
        XCTAssertEqual(loose.localPrecision ?? 0, 0.8, accuracy: 1e-12)

        let middle = ThresholdCalibrator.calibrate(model: student, evalSet: eval, targetPrecision: 0.85)
        XCTAssertEqual(middle.threshold, 0.9)
    }

    /// The policy answers locally at `confidence >= threshold`, so a cut inside a tie would hide the
    /// wrong answer tied at 0.9. Without the tie rule this calibrates to 0.9 with "precision 1".
    func testNeverCutsInsideATie() throws {
        let (student, eval) = try fixture([(0.95, true), (0.9, true), (0.9, false)])
        let result = ThresholdCalibrator.calibrate(model: student, evalSet: eval, targetPrecision: 1)
        XCTAssertEqual(result.threshold, 0.95)
        XCTAssertEqual(result.coverage, 1.0 / 3.0, accuracy: 1e-12)
    }

    func testReturnsEscalateAllWhenNothingMeetsTheTarget() throws {
        let (student, eval) = try fixture([(0.9, false), (0.8, false)])
        let result = ThresholdCalibrator.calibrate(model: student, evalSet: eval, targetPrecision: 0.5)
        XCTAssertNil(result.threshold)
        XCTAssertEqual(result.coverage, 0)
        XCTAssertNil(result.localPrecision)
        XCTAssertEqual(result.policy(), .escalateAll)
    }

    func testAThresholdOfExactlyOneIsNotUsable() throws {
        let (student, eval) = try fixture([(1.0, true), (0.7, false)])
        XCTAssertNil(ThresholdCalibrator.calibrate(model: student, evalSet: eval, targetPrecision: 1).threshold)
    }

    func testCalibratedPolicyKeepsMarginAndDistributionFloors() throws {
        let (student, eval) = try fixture([(0.9, true)])
        let policy = ThresholdCalibrator.calibrate(model: student, evalSet: eval, targetPrecision: 1)
            .policy(minMargin: 0.2, minKnownTokenFraction: 0.5)
        XCTAssertEqual(policy, EscalationPolicy(minConfidence: 0.9, minMargin: 0.2, minKnownTokenFraction: 0.5))
    }
}

final class EvalSetTests: XCTestCase {
    func testRejectsEmptyBlankAndSelfContradictorySets() {
        XCTAssertThrowsError(try EvalSet(version: "e", examples: [])) { XCTAssertEqual($0 as? EvalSetError, .empty) }
        XCTAssertThrowsError(try EvalSet(version: "e", examples: [.init(text: "?!", label: "a")])) {
            XCTAssertEqual($0 as? EvalSetError, .blankExample)
        }
        XCTAssertThrowsError(try EvalSet(version: "e", examples: [
            .init(text: "Refund!", label: "billing"), .init(text: "refund", label: "bug"),
        ])) { XCTAssertEqual($0 as? EvalSetError, .conflictingLabels(normalizedText: "refund")) }
    }

    func testDeduplicatesAndIsOrderIndependent() throws {
        let a = try EvalSet(version: "e", examples: [.init(text: "b", label: "y"), .init(text: "a", label: "x"),
                                                    .init(text: "A!", label: "x")])
        let b = try EvalSet(version: "e", examples: [.init(text: "a", label: "x"), .init(text: "b", label: "y")])
        XCTAssertEqual(a.examples.count, 2)
        XCTAssertEqual(a.fingerprint, b.fingerprint)
        XCTAssertTrue(a.contains(text: "  A ... "))
        XCTAssertFalse(a.contains(text: "c"))
    }

    func testFingerprintChangesWhenALabelChanges() throws {
        let original = try EvalSet(version: "e", examples: [.init(text: "a", label: "x"), .init(text: "b", label: "y")])
        let relabelled = try EvalSet(version: "e", examples: [.init(text: "a", label: "x"), .init(text: "b", label: "x")])
        XCTAssertNotEqual(original.fingerprint, relabelled.fingerprint)
    }
}

final class EvaluatorAndGateTests: XCTestCase {
    private let labels: [ClassLabel] = ["x", "y"]

    private var eval: EvalSet {
        get throws {
            try EvalSet(version: "e1", examples: [
                .init(text: "x1", label: "x"), .init(text: "x2", label: "x"),
                .init(text: "y1", label: "y"), .init(text: "y2", label: "y"),
            ])
        }
    }

    func testEvaluatorComputesAccuracyRecallCoverageAndPrecision() throws {
        // x1 right & confident, x2 wrong & confident, y1 right & unsure, y2 right & confident.
        let student = FixedStudent(labels: labels, table: [
            "x1": scored("x", 0.9), "x2": scored("y", 0.9), "y1": scored("y", 0.6), "y2": scored("y", 0.95),
        ], defaultAnswer: scored("x", 0.5))
        let report = Evaluator.evaluate(student, on: try eval, policy: EscalationPolicy(minConfidence: 0.8))
        XCTAssertEqual(report.accuracy, 0.75)
        XCTAssertEqual(report.perLabelRecall["x"], 0.5)
        XCTAssertEqual(report.perLabelRecall["y"], 1)
        XCTAssertEqual(report.localCoverage, 0.75)
        XCTAssertEqual(report.localPrecision ?? 0, 2.0 / 3.0, accuracy: 1e-12)
        XCTAssertEqual(report.exampleCount, 4)
    }

    private func report(
        version: String = "v", accuracy: Double, recall: [ClassLabel: Double], coverage: Double,
        precision: Double?, fingerprint: UInt64 = 1
    ) -> EvalReport {
        EvalReport(modelVersion: version, evalSetVersion: "e1", evalFingerprint: fingerprint, exampleCount: 4,
                   accuracy: accuracy, perLabelRecall: recall, localCoverage: coverage, localPrecision: precision)
    }

    func testFirstModelOnlyNeedsThePrecisionFloor() {
        let gate = PromotionGate(minLocalPrecision: 0.9)
        XCTAssertEqual(gate.decide(candidate: report(accuracy: 0.5, recall: [:], coverage: 0.2, precision: 0.95),
                                   incumbent: nil), .promote)
        XCTAssertEqual(gate.decide(candidate: report(accuracy: 0.9, recall: [:], coverage: 0, precision: nil),
                                   incumbent: nil),
                       .reject([.localPrecisionBelowFloor(value: nil, floor: 0.9)]))
    }

    func testPromotesWhenCoverageImprovesWithoutRegressions() {
        let incumbent = report(accuracy: 0.75, recall: ["x": 0.5, "y": 1], coverage: 0.25, precision: 1)
        let candidate = report(accuracy: 0.75, recall: ["x": 0.5, "y": 1], coverage: 0.5, precision: 1)
        XCTAssertEqual(PromotionGate().decide(candidate: candidate, incumbent: incumbent), .promote)
    }

    func testRejectsAnIdenticalRetrainAsNoImprovement() {
        let same = report(accuracy: 0.75, recall: ["x": 0.5, "y": 1], coverage: 0.25, precision: 1)
        XCTAssertEqual(PromotionGate().decide(candidate: same, incumbent: same), .reject([.noImprovement]))
    }

    /// Overall accuracy goes up, but one label gets worse. Averages hide this; the gate must not.
    func testRejectsAPerLabelRecallRegressionEvenWhenAccuracyRises() {
        let incumbent = report(accuracy: 0.5, recall: ["x": 1, "y": 0], coverage: 0.5, precision: 1)
        let candidate = report(accuracy: 0.75, recall: ["x": 0.5, "y": 1], coverage: 0.5, precision: 1)
        XCTAssertEqual(PromotionGate(maxRecallDrop: 0.05).decide(candidate: candidate, incumbent: incumbent),
                       .reject([.recallRegressed(label: "x", from: 1, to: 0.5)]))
    }

    func testRejectsAccuracyRegressionAndPrecisionBelowFloorTogether() {
        let incumbent = report(accuracy: 0.9, recall: [:], coverage: 0.5, precision: 1)
        let candidate = report(accuracy: 0.7, recall: [:], coverage: 0.8, precision: 0.8)
        XCTAssertEqual(PromotionGate(minLocalPrecision: 0.9).decide(candidate: candidate, incumbent: incumbent),
                       .reject([.localPrecisionBelowFloor(value: 0.8, floor: 0.9), .accuracyRegressed(from: 0.9, to: 0.7)]))
    }

    func testRejectsReportsFromDifferentEvalSets() {
        let incumbent = report(accuracy: 0.5, recall: [:], coverage: 0.2, precision: 1, fingerprint: 1)
        let candidate = report(accuracy: 0.9, recall: [:], coverage: 0.9, precision: 1, fingerprint: 2)
        XCTAssertEqual(PromotionGate().decide(candidate: candidate, incumbent: incumbent),
                       .reject([.evalSetMismatch(incumbent: 1, candidate: 2)]))
    }
}

final class TrainingSetBuilderTests: XCTestCase {
    func testDropsEvalLeaksLetsHarvestOverrideSeedAndSkipsBlanks() throws {
        let eval = try EvalSet(version: "e", examples: [.init(text: "my card was charged", label: "billing")])
        let result = TrainingSetBuilder.build(
            seed: [.init(text: "app crashes", label: "feature"), .init(text: "add dark mode", label: "feature")],
            harvested: [
                .init(text: "App crashes!", label: "bug"),
                .init(text: "My card was CHARGED", label: "billing"),
                .init(text: "   ", label: "bug"),
            ],
            excluding: eval
        )
        XCTAssertEqual(result.excludedForLeak, 1)
        XCTAssertEqual(result.seedOverridden, 1)
        XCTAssertEqual(result.examples.map(\.label), ["bug", "feature"])
        XCTAssertFalse(result.examples.contains { eval.contains(text: $0.text) })
    }
}
