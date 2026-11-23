import XCTest
@testable import HybridInference
@testable import HybridInferenceUI

extension Tickets {
    /// The same scenario the demo app ships (Demo/DemoApp.swift in hybrid-inference-router-demo-app), so the
    /// numbers in its screenshots are asserted in this package's CI.
    static func demoScenario() throws -> LabScenario {
        LabScenario(
            title: "Ticket triage",
            seed: seed,
            evalSet: try EvalSet(version: "tickets-eval-1", examples: eval),
            trafficBatches: [dayOne, dayTwo],
            teacherModelID: "scripted-claude",
            teacherRules: rules,
            teacherFallback: "feature",
            configuration: RouterConfiguration(
                policy: EscalationPolicy(minConfidence: 1, minMargin: 0, minKnownTokenFraction: 0.25),
                budget: BudgetConfiguration(capacityUnits: 60, windowSeconds: 3600),
                pricePerCallUnits: 1,
                breaker: BreakerConfiguration(failureThreshold: 3, cooldownSeconds: 30),
                teacherTimeout: .seconds(4),
                shadowSampleRate: 0.2,
                shadowSalt: "triage-shadow-3",
                shadowQueueCapacity: 32,
                shadowHeadroomUnits: 10,
                harvestCapacity: 500
            ),
            targetPrecision: 0.9,
            gate: PromotionGate(maxAccuracyDrop: 0, maxRecallDrop: 0.05, minLocalPrecision: 0.9),
            shadowCallsPerRun: 8
        )
    }
}

/// Drives `HybridInferenceLab` through every scripted stage the demo app launches into, and the two
/// interactive paths (retired teacher, price rise) that have no launch stage.
@MainActor
final class LabStageTests: XCTestCase {
    private func makeLab() throws -> HybridInferenceLab {
        HybridInferenceLab(scenario: try Tickets.demoScenario())
    }

    private func snapshot(_ lab: HybridInferenceLab) throws -> RouterSnapshot {
        try XCTUnwrap(lab.snapshot)
    }

    func testBaselineSendsAllOfDayOneToTheTeacher() async throws {
        let lab = try makeLab()
        await lab.run(to: .baseline)
        let state = try snapshot(lab)
        XCTAssertNil(lab.errorMessage)
        XCTAssertEqual(state.modelVersion, "student-v1")
        XCTAssertNil(lab.activeThreshold, "8 seed examples cannot reach 90% precision on the eval set")
        XCTAssertEqual(state.requests, 40)
        XCTAssertEqual(state.answeredByStudent, 0)
        XCTAssertEqual(state.answeredByTeacher, 40)
        XCTAssertEqual(state.spentUnits, 40)
        XCTAssertEqual(state.harvested, 40)
        XCTAssertEqual(lab.nextBatchNumber, 2)
    }

    func testRetrainedPromotesV2WhichServesMostUnseenTicketsOnDevice() async throws {
        let lab = try makeLab()
        await lab.run(to: .retrained)
        let gate = try XCTUnwrap(lab.lastGate)
        XCTAssertEqual(gate.decision, .promote)
        XCTAssertEqual(gate.trainingCount, 48)
        XCTAssertEqual(gate.leakExcluded, 0)
        XCTAssertEqual(gate.incumbent?.accuracy ?? 0, 5.0 / 12.0, accuracy: 1e-12)
        XCTAssertEqual(gate.candidate.accuracy, 1)
        XCTAssertEqual(gate.candidate.localCoverage, 1)
        XCTAssertEqual(lab.activeThreshold ?? 0, 0.594, accuracy: 0.001)

        let state = try snapshot(lab)
        XCTAssertEqual(state.modelVersion, "student-v2")
        XCTAssertEqual(state.requests, 20)
        XCTAssertEqual(state.answeredByStudent, 16)
        XCTAssertEqual(state.answeredByTeacher, 4)
        XCTAssertEqual(state.fallbackCount, 0)
        // Five of the sixteen local answers fall in the shadow sample; all five fit above the headroom.
        XCTAssertEqual(state.shadowAgreements + state.shadowDisagreements, 5)
        XCTAssertEqual(state.spentUnits, 40 + 4 + 5)
    }

    func testDriftedRetrainIsRejectedAndV1StaysInstalled() async throws {
        let lab = try makeLab()
        await lab.run(to: .drifted)
        let gate = try XCTUnwrap(lab.lastGate)
        XCTAssertTrue(gate.drifted)
        guard case .reject(let failures) = gate.decision else { return XCTFail("drifted retrain must be rejected") }
        XCTAssertTrue(failures.contains(.recallRegressed(label: "account", from: 1, to: 0)))
        XCTAssertTrue(failures.contains(.accuracyRegressed(from: 5.0 / 12.0, to: 0)))
        XCTAssertEqual(try snapshot(lab).modelVersion, "student-v1")
    }

    func testOutageOpensTheBreakerWhileTheStudentServes() async throws {
        let lab = try makeLab()
        await lab.run(to: .outage)
        let state = try snapshot(lab)
        XCTAssertEqual(lab.teacherMode, .outage)
        XCTAssertEqual(state.breaker, .open)
        XCTAssertEqual(state.answeredByStudent, 16)
        XCTAssertEqual(state.fallbacks[.teacherFailed], 3)
        XCTAssertEqual(state.fallbacks[.circuitOpen], 1)
        XCTAssertEqual(state.shadowAgreements + state.shadowDisagreements, 0, "no shadow calls while the breaker is open")
        XCTAssertEqual(state.spentUnits, 40, "failed calls are refunded")
    }

    func testRetiredTeacherFallsBackThenMigratesToASuccessor() async throws {
        let lab = try makeLab()
        await lab.run(to: .baseline)
        await lab.setTeacherMode(.retired)
        await lab.runTraffic()                                   // day two on v1: every escalation fails
        var state = try snapshot(lab)
        XCTAssertEqual(state.breaker, .retired)
        XCTAssertEqual(state.fallbacks[.teacherRetired], 20)

        await lab.setTeacherMode(.healthy)                      // migrate to a successor model
        state = try snapshot(lab)
        XCTAssertEqual(state.teacherModelID, "scripted-claude-successor-2")
        XCTAssertEqual(state.breaker, .closed(consecutiveFailures: 0))
        await lab.runTraffic()
        XCTAssertGreaterThan(try snapshot(lab).answeredByTeacher, 0)
    }

    func testAFourTimesPriceRiseExhaustsTheSameBudgetSooner() async throws {
        let lab = try makeLab()
        await lab.prepare()
        await lab.setPriceRaised(true)
        await lab.runTraffic()
        let state = try snapshot(lab)
        XCTAssertEqual(state.pricePerCallUnits, 4)
        XCTAssertEqual(state.answeredByTeacher, 15)
        XCTAssertEqual(state.fallbacks[.budgetExhausted], 25)
        XCTAssertEqual(state.spentUnits, 60)
    }

    func testPrepareResetsTeacherModeAndPrice() async throws {
        let lab = try makeLab()
        await lab.prepare()
        await lab.setPriceRaised(true)
        await lab.setTeacherMode(.outage)
        await lab.prepare()
        XCTAssertFalse(lab.priceRaised)
        XCTAssertEqual(lab.teacherMode, .healthy)
        XCTAssertEqual(try snapshot(lab).pricePerCallUnits, 1)
        XCTAssertEqual(try snapshot(lab).modelVersion, "student-v1")
    }
}
