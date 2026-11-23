import XCTest
@testable import HybridInference

/// The headline property, end to end, with the real Naive Bayes student and the scripted teacher:
/// serve day one → harvest Claude's labels → retrain → calibrate → gate → install → serve unseen day-two tickets.
final class TeacherStudentLoopTests: XCTestCase {
    private let target = 0.9
    private let gate = PromotionGate(maxAccuracyDrop: 0, maxRecallDrop: 0.05, minLocalPrecision: 0.9)

    private func calibrated(_ model: NaiveBayesStudent, _ eval: EvalSet) -> (EscalationPolicy, EvalReport) {
        let policy = ThresholdCalibrator.calibrate(model: model, evalSet: eval, targetPrecision: target).policy()
        return (policy, Evaluator.evaluate(model, on: eval, policy: policy))
    }

    private func serve(_ router: HybridRouter, _ batch: [String]) async -> RouterSnapshot {
        await router.resetCounters()
        for text in batch { _ = await router.route(text) }
        return await router.snapshot()
    }

    func testRetrainingOnTeacherLabelsCutsEscalationsAndPassesTheGate() async throws {
        let eval = try EvalSet(version: "tickets-eval-1", examples: Tickets.eval)
        let v1 = try NaiveBayesStudent(training: Tickets.seed, version: "v1")
        let (policy1, report1) = calibrated(v1, eval)

        let teacher = ScriptedTeacher(modelID: "scripted-claude", rules: Tickets.rules, fallbackLabel: "feature")
        let router = HybridRouter(
            student: v1, teacher: teacher,
            configuration: RouterConfiguration(policy: policy1, budget: BudgetConfiguration(capacityUnits: 1_000),
                                               shadowSampleRate: 0)
        )
        // Day one on v1, then day two (unseen tickets) on v1 as the out-of-sample baseline.
        let dayOne = await serve(router, Tickets.dayOne)
        XCTAssertEqual(dayOne.requests, Tickets.dayOne.count)
        XCTAssertEqual(dayOne.fallbackCount, 0)
        let harvested = await router.harvestedExamples()
        XCTAssertEqual(harvested.count, dayOne.answeredByTeacher)
        let before = await serve(router, Tickets.dayTwo)

        // Retrain on day one's labels only, so day two stays unseen by v2.
        let training = TrainingSetBuilder.build(seed: Tickets.seed, harvested: harvested, excluding: eval)
        XCTAssertEqual(training.excludedForLeak, 0, "traffic is disjoint from the eval set")

        let v2 = try NaiveBayesStudent(training: training.examples, version: "v2")
        let (policy2, report2) = calibrated(v2, eval)
        XCTAssertEqual(gate.decide(candidate: report2, incumbent: report1), .promote)
        XCTAssertGreaterThan(report2.localCoverage, report1.localCoverage)

        await router.install(v2, policy: policy2)
        let after = await serve(router, Tickets.dayTwo)
        XCTAssertLessThan(after.escalationRate, before.escalationRate,
                          "on unseen day-two tickets v1 escalated \(before.escalationRate), v2 \(after.escalationRate)")
        XCTAssertEqual(after.modelVersion, "v2")
    }

    /// The gate's reason to exist: a teacher that drifted (every harvested label rotated to the next
    /// label) produces a candidate the gate must refuse. If the gate only looked at coverage, or only at
    /// average accuracy, this candidate could ship.
    func testGateRejectsAStudentTrainedOnDriftedTeacherLabels() async throws {
        let eval = try EvalSet(version: "tickets-eval-1", examples: Tickets.eval)
        let v1 = try NaiveBayesStudent(training: Tickets.seed, version: "v1")
        let (policy1, report1) = calibrated(v1, eval)
        let teacher = ScriptedTeacher(modelID: "scripted-claude", rules: Tickets.rules, fallbackLabel: "feature")
        let router = HybridRouter(student: v1, teacher: teacher,
                                  configuration: RouterConfiguration(policy: policy1,
                                                                     budget: BudgetConfiguration(capacityUnits: 1_000)))
        _ = await serve(router, Tickets.dayOne)

        let labels = Tickets.labels
        let drifted = await router.harvestedExamples().map { example -> LabeledExample in
            let index = labels.firstIndex(of: example.label) ?? 0
            return LabeledExample(text: example.text, label: labels[(index + 1) % labels.count])
        }
        let training = TrainingSetBuilder.build(seed: Tickets.seed, harvested: drifted, excluding: eval)
        let bad = try NaiveBayesStudent(training: training.examples, version: "v2-drifted")
        let (_, badReport) = calibrated(bad, eval)
        let decision = gate.decide(candidate: badReport, incumbent: report1)
        XCTAssertFalse(decision.isPromotion)
        guard case .reject(let failures) = decision else { return XCTFail("expected rejection") }
        XCTAssertTrue(failures.contains { if case .recallRegressed = $0 { return true } else { return false } },
                      "expected at least one per-label recall regression, got \(failures)")
    }

    /// Leak guard: if eval examples reach training, the guard drops every one of them.
    func testEvalExamplesInTheHarvestNeverReachTraining() throws {
        let eval = try EvalSet(version: "tickets-eval-1", examples: Tickets.eval)
        let poisoned = Tickets.eval.map { LabeledExample(text: $0.text.uppercased() + "!", label: $0.label) }
        let training = TrainingSetBuilder.build(seed: Tickets.seed, harvested: poisoned, excluding: eval)
        XCTAssertEqual(training.excludedForLeak, Tickets.eval.count)
        XCTAssertEqual(training.examples.count, Tickets.seed.count)
    }

    func testScriptedTeacherFaultsSurfaceAsTypedErrors() async {
        let teacher = ScriptedTeacher(modelID: "s", rules: Tickets.rules, fallbackLabel: "feature")
        let expectations: [(ScriptedTeacher.Fault, TeacherError)] = [
            (.outage, .unavailable), (.rateLimited, .rateLimited), (.retired, .modelRetired), (.refuseAll, .refused),
        ]
        for (fault, expected) in expectations {
            await teacher.setFault(fault)
            do {
                _ = try await teacher.label("refund", allowedLabels: Tickets.labels)
                XCTFail("\(fault) should throw")
            } catch {
                XCTAssertEqual(error as? TeacherError, expected)
            }
        }
        await teacher.setFault(.offListLabels)
        let offList = try? await teacher.label("refund", allowedLabels: Tickets.labels)
        XCTAssertEqual(offList, "not-a-real-label")
        await teacher.setFault(.healthy)
        let healthy = try? await teacher.label("The app crashes during a refund", allowedLabels: Tickets.labels)
        XCTAssertEqual(healthy, "bug", "rules are checked in order")
        let fallback = try? await teacher.label("hello there", allowedLabels: Tickets.labels)
        XCTAssertEqual(fallback, "feature")
    }
}
