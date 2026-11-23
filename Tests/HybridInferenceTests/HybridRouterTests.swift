import XCTest
@testable import HybridInference

final class HybridRouterTests: XCTestCase {
    /// Three labels, so a 0.4 "bug" really is the top guess (the other two get 0.3 each).
    private let labels: [ClassLabel] = ["account", "billing", "bug"]

    /// Confident (0.95 "billing") on everything except the "unsure…" texts, which get 0.4 "bug".
    private func student(version: String = "s1") -> FixedStudent {
        FixedStudent(
            version: version, labels: labels,
            table: ["unsure", "unsure a", "unsure b", "unsure c"].reduce(into: [:]) { $0[$1] = scored("bug", 0.4) },
            defaultAnswer: scored("billing", 0.95)
        )
    }

    private func configuration(
        capacity: Int = 100, price: Int = 1, threshold: Int = 3, cooldown: Double = 30,
        timeout: Duration = .seconds(30), shadowRate: Double = 0, shadowCapacity: Int = 8, headroom: Int = 0
    ) -> RouterConfiguration {
        RouterConfiguration(
            policy: EscalationPolicy(minConfidence: 0.8),
            budget: BudgetConfiguration(capacityUnits: capacity, windowSeconds: 3600),
            pricePerCallUnits: price,
            breaker: BreakerConfiguration(failureThreshold: threshold, cooldownSeconds: cooldown),
            teacherTimeout: timeout,
            shadowSampleRate: shadowRate,
            shadowQueueCapacity: shadowCapacity,
            shadowHeadroomUnits: headroom
        )
    }

    // MARK: Basic routing

    func testConfidentRequestStaysOnDeviceAndNeverCallsTheTeacher() async {
        let teacher = ProgrammableTeacher()
        let router = HybridRouter(student: student(), teacher: teacher, configuration: configuration())
        let answer = await router.route("my card")
        XCTAssertEqual(answer.source, .student)
        XCTAssertEqual(answer.label, "billing")
        XCTAssertNil(answer.escalationReason)
        let calls = await teacher.calls
        XCTAssertEqual(calls, 0)
    }

    func testUnsureRequestEscalatesAndTheTeacherLabelIsHarvested() async {
        let teacher = ProgrammableTeacher(defaultResult: .success("billing"))
        let router = HybridRouter(student: student(), teacher: teacher, configuration: configuration(price: 3))
        let answer = await router.route("unsure")
        XCTAssertEqual(answer.source, .teacher(modelID: "teacher-a"))
        XCTAssertEqual(answer.label, "billing")
        XCTAssertEqual(answer.studentLabel, "bug")
        XCTAssertEqual(answer.escalationReason, .lowConfidence(0.4))
        let snapshot = await router.snapshot()
        XCTAssertEqual(snapshot.spentUnits, 3)
        XCTAssertEqual(snapshot.teacherDisagreements, 1)
        let harvested = await router.harvestedExamples()
        XCTAssertEqual(harvested, [LabeledExample(text: "unsure", label: "billing")])
    }

    // MARK: Budget

    func testExhaustedBudgetFallsBackToTheStudentWithoutCallingTheTeacher() async {
        let teacher = ProgrammableTeacher()
        let router = HybridRouter(student: student(), teacher: teacher, configuration: configuration(capacity: 1))
        _ = await router.route("unsure a")
        let second = await router.route("unsure b")
        XCTAssertEqual(second.source, .studentFallback(.budgetExhausted))
        XCTAssertEqual(second.label, "bug")
        let calls = await teacher.calls
        XCTAssertEqual(calls, 1)
    }

    func testRepriceAppliesToTheNextReservation() async {
        let router = HybridRouter(student: student(), teacher: ProgrammableTeacher(),
                                  configuration: configuration(capacity: 9, price: 1))
        await router.reprice(unitsPerCall: 5)
        let first = await router.route("unsure a")
        let second = await router.route("unsure b")
        XCTAssertEqual(first.source, .teacher(modelID: "teacher-a"))
        XCTAssertEqual(second.source, .studentFallback(.budgetExhausted))
        let snapshot = await router.snapshot()
        XCTAssertEqual(snapshot.spentUnits, 5)
        XCTAssertEqual(snapshot.pricePerCallUnits, 5)
    }

    /// 40 concurrent escalations against a 10-unit budget with a slow teacher. The calls really overlap
    /// (`maxConcurrent > 1`), so a check-then-await-then-spend router would let every one through.
    func testConcurrentEscalationsNeverOverspendTheBudget() async {
        let teacher = ProgrammableTeacher(latency: .milliseconds(20))
        let router = HybridRouter(student: student(), teacher: teacher, configuration: configuration(capacity: 10))
        let answers = await withTaskGroup(of: RoutedAnswer.self) { group in
            for index in 0..<40 { group.addTask { await router.route(index.isMultiple(of: 2) ? "unsure a" : "unsure b") } }
            var collected: [RoutedAnswer] = []
            for await answer in group { collected.append(answer) }
            return collected
        }
        let calls = await teacher.calls
        let maxConcurrent = await teacher.maxConcurrent
        let snapshot = await router.snapshot()
        XCTAssertEqual(answers.count, 40)
        XCTAssertEqual(calls, 10)
        XCTAssertGreaterThan(maxConcurrent, 1, "the test must actually exercise overlapping awaits")
        XCTAssertEqual(snapshot.spentUnits, 10)
        XCTAssertEqual(snapshot.reservedUnits, 0)
        XCTAssertEqual(snapshot.fallbacks[.budgetExhausted], 30)
        XCTAssertEqual(snapshot.answeredByTeacher, 10)
    }

    // MARK: Breaker and failures

    func testOutageOpensTheBreakerThenAProbeClosesItAfterTheCooldown() async {
        let clock = ManualClock()
        let teacher = ProgrammableTeacher(defaultResult: .failure(.unavailable))
        let router = HybridRouter(student: student(), teacher: teacher,
                                  configuration: configuration(threshold: 2, cooldown: 30), clock: clock)
        let first = await router.route("unsure a")
        _ = await router.route("unsure a")
        let third = await router.route("unsure a")
        XCTAssertEqual(first.source, .studentFallback(.teacherFailed))
        XCTAssertEqual(first.teacherError, .unavailable)
        XCTAssertEqual(third.source, .studentFallback(.circuitOpen))
        var calls = await teacher.calls
        XCTAssertEqual(calls, 2)
        var snapshot = await router.snapshot()
        XCTAssertEqual(snapshot.breaker, .open)
        XCTAssertEqual(snapshot.spentUnits, 0, "an unavailable call is refunded")

        await teacher.setDefault(.success("bug"))
        clock.advance(30)
        let probe = await router.route("unsure a")
        XCTAssertEqual(probe.source, .teacher(modelID: "teacher-a"))
        calls = await teacher.calls
        XCTAssertEqual(calls, 3)
        snapshot = await router.snapshot()
        XCTAssertEqual(snapshot.breaker, .closed(consecutiveFailures: 0))
    }

    func testRetiredModelStopsEscalationUntilTheTeacherIsReplaced() async {
        let old = ProgrammableTeacher(modelID: "old", defaultResult: .failure(.modelRetired))
        let router = HybridRouter(student: student(), teacher: old, configuration: configuration())
        let first = await router.route("unsure")
        let second = await router.route("unsure")
        XCTAssertEqual(first.source, .studentFallback(.teacherRetired))
        XCTAssertEqual(second.source, .studentFallback(.teacherRetired))
        let oldCalls = await old.calls
        XCTAssertEqual(oldCalls, 1)

        let successor = ProgrammableTeacher(modelID: "new", defaultResult: .success("bug"))
        await router.replaceTeacher(successor)
        let third = await router.route("unsure")
        XCTAssertEqual(third.source, .teacher(modelID: "new"))
    }

    /// Re-entrancy: the old teacher reports "retired" *after* it was replaced. That must not retire the new one.
    func testRetirementReportedByAReplacedTeacherDoesNotDisableItsSuccessor() async {
        let old = GatedTeacher(modelID: "old")
        let router = HybridRouter(student: student(), teacher: old, configuration: configuration())
        let inFlight = Task { await router.route("unsure") }
        let arrived = await eventually { await old.pending() == 1 }
        XCTAssertTrue(arrived)
        await router.replaceTeacher(ProgrammableTeacher(modelID: "new", defaultResult: .success("bug")))
        await old.release(.failure(.modelRetired))
        _ = await inFlight.value
        let snapshot = await router.snapshot()
        XCTAssertEqual(snapshot.breaker, .closed(consecutiveFailures: 0))
        let next = await router.route("unsure")
        XCTAssertEqual(next.source, .teacher(modelID: "new"))
    }

    func testRefusalIsBilledButDoesNotCountAsAnOutage() async {
        let teacher = ProgrammableTeacher(defaultResult: .failure(.refused))
        let router = HybridRouter(student: student(), teacher: teacher, configuration: configuration(threshold: 1))
        let answer = await router.route("unsure")
        XCTAssertEqual(answer.source, .studentFallback(.teacherRefused))
        let snapshot = await router.snapshot()
        XCTAssertEqual(snapshot.spentUnits, 1)
        XCTAssertEqual(snapshot.breaker, .closed(consecutiveFailures: 0))
    }

    func testOffListLabelIsNeverServedOrHarvested() async {
        let teacher = ProgrammableTeacher(defaultResult: .success("ignore previous instructions"))
        let router = HybridRouter(student: student(), teacher: teacher, configuration: configuration())
        let answer = await router.route("unsure")
        XCTAssertEqual(answer.source, .studentFallback(.offListLabel))
        XCTAssertEqual(answer.label, "bug")
        let harvested = await router.harvestedExamples()
        XCTAssertTrue(harvested.isEmpty)
        let snapshot = await router.snapshot()
        XCTAssertEqual(snapshot.spentUnits, 1, "the call happened, so it is billed")
    }

    func testTimeoutBoundsLatencyAndIsBilledConservatively() async {
        let teacher = ProgrammableTeacher(latency: .seconds(20))
        let router = HybridRouter(student: student(), teacher: teacher,
                                  configuration: configuration(timeout: .milliseconds(50)))
        let started = ContinuousClock.now
        let answer = await router.route("unsure")
        let elapsed = ContinuousClock.now - started
        XCTAssertEqual(answer.source, .studentFallback(.teacherFailed))
        XCTAssertEqual(answer.teacherError, .timedOut)
        XCTAssertLessThan(elapsed, .seconds(5))
        let snapshot = await router.snapshot()
        XCTAssertEqual(snapshot.spentUnits, 1)
        XCTAssertEqual(snapshot.breaker, .closed(consecutiveFailures: 1))
    }

    func testCallerCancellationRefundsAndDoesNotTripTheBreaker() async {
        let teacher = ProgrammableTeacher(latency: .seconds(20))
        let router = HybridRouter(student: student(), teacher: teacher,
                                  configuration: configuration(threshold: 1, timeout: .seconds(60)))
        let task = Task { await router.route("unsure") }
        let arrived = await eventually { await teacher.calls == 1 }
        XCTAssertTrue(arrived)
        task.cancel()
        let answer = await task.value
        XCTAssertEqual(answer.source, .studentFallback(.cancelled))
        let snapshot = await router.snapshot()
        XCTAssertEqual(snapshot.spentUnits, 0)
        XCTAssertEqual(snapshot.reservedUnits, 0)
        XCTAssertEqual(snapshot.breaker, .closed(consecutiveFailures: 0))
    }

    /// The production path: `ClaudeTeacher` over a transport that, like URLSession, reports cancellation as
    /// `URLError(.cancelled)`. Three cancelled users must not open a threshold-1 breaker.
    func testCancellationThroughClaudeTeacherDoesNotTripTheBreaker() async {
        let transport = SlowCancellableTransport()
        let teacher = ClaudeTeacher(apiKey: "k", model: "m", transport: transport)
        let router = HybridRouter(student: student(), teacher: teacher,
                                  configuration: configuration(threshold: 1, timeout: .seconds(60)))
        for expected in 1...3 {
            let task = Task { await router.route("unsure") }
            let arrived = await eventually { transport.started >= expected }
            XCTAssertTrue(arrived)
            task.cancel()
            let answer = await task.value
            XCTAssertEqual(answer.source, .studentFallback(.cancelled))
        }
        let snapshot = await router.snapshot()
        XCTAssertEqual(snapshot.breaker, .closed(consecutiveFailures: 0))
        XCTAssertEqual(snapshot.spentUnits, 0)
    }

    /// Budget is checked before the breaker. After the cool-down, a request the budget refuses must not
    /// move the breaker to half-open; the probe stays available for a request that can pay for it.
    func testABudgetRefusalLeavesTheBreakersProbeUntouched() async {
        let clock = ManualClock()
        let teacher = ProgrammableTeacher(defaultResult: .failure(.unavailable))
        let router = HybridRouter(student: student(), teacher: teacher,
                                  configuration: configuration(capacity: 3, threshold: 1, cooldown: 10), clock: clock)
        _ = await router.route("unsure")
        var snapshot = await router.snapshot()
        XCTAssertEqual(snapshot.breaker, .open)

        clock.advance(10)
        await router.reprice(unitsPerCall: 5)
        let refused = await router.route("unsure")
        XCTAssertEqual(refused.source, .studentFallback(.budgetExhausted))
        snapshot = await router.snapshot()
        XCTAssertEqual(snapshot.breaker, .open, "a refused budget must not consume or reveal the probe")

        await router.reprice(unitsPerCall: 1)
        await teacher.setDefault(.success("bug"))
        let probe = await router.route("unsure")
        XCTAssertEqual(probe.source, .teacher(modelID: "teacher-a"))
    }

    // MARK: Model swaps

    func testInFlightRequestFinishesOnTheModelItStartedWith() async {
        let teacher = GatedTeacher()
        let router = HybridRouter(student: student(version: "s1"), teacher: teacher, configuration: configuration())
        let inFlight = Task { await router.route("unsure") }
        let arrived = await eventually { await teacher.pending() == 1 }
        XCTAssertTrue(arrived)
        await router.install(student(version: "s2"), policy: EscalationPolicy(minConfidence: 0.8))
        await teacher.release(.success("bug"))
        let answer = await inFlight.value
        XCTAssertEqual(answer.modelVersion, "s1")
        let snapshot = await router.snapshot()
        XCTAssertEqual(snapshot.modelVersion, "s2")
    }

    // MARK: Shadow traffic

    func testShadowSamplesMeasureAgreementAndStopAtTheHeadroom() async {
        let teacher = ProgrammableTeacher(script: [.success("billing"), .success("bug")])
        let router = HybridRouter(
            student: student(), teacher: teacher,
            configuration: configuration(capacity: 4, shadowRate: 1, shadowCapacity: 8, headroom: 2)
        )
        for text in ["one", "two", "three"] { _ = await router.route(text) }
        var snapshot = await router.snapshot()
        XCTAssertEqual(snapshot.shadowQueued, 3)

        let made = await router.drainShadowQueue(maxCalls: 10)
        snapshot = await router.snapshot()
        XCTAssertEqual(made, 2, "the third call would dip into the 2 units kept for live traffic")
        XCTAssertEqual(snapshot.shadowAgreements, 1)
        XCTAssertEqual(snapshot.shadowDisagreements, 1)
        XCTAssertEqual(snapshot.shadowQueued, 1)
        XCTAssertEqual(snapshot.harvested, 2)

        let live = await router.route("unsure")
        XCTAssertEqual(live.source, .teacher(modelID: "teacher-a"), "live traffic still has its headroom")
    }

    func testShadowQueueIsBoundedAndCountsDrops() async {
        let router = HybridRouter(student: student(), teacher: ProgrammableTeacher(),
                                  configuration: configuration(shadowRate: 1, shadowCapacity: 2))
        for index in 0..<5 { _ = await router.route("text \(index)") }
        let snapshot = await router.snapshot()
        XCTAssertEqual(snapshot.shadowQueued, 2)
        XCTAssertEqual(snapshot.shadowDropped, 3)
    }

    func testDrainStopsWhenTheBreakerIsOpenAndKeepsTheWork() async {
        let teacher = ProgrammableTeacher(defaultResult: .failure(.unavailable))
        let router = HybridRouter(student: student(), teacher: teacher,
                                  configuration: configuration(threshold: 1, shadowRate: 1))
        for text in ["one", "two", "three"] { _ = await router.route(text) }
        let made = await router.drainShadowQueue(maxCalls: 10)
        let snapshot = await router.snapshot()
        XCTAssertEqual(made, 1)
        XCTAssertEqual(snapshot.breaker, .open)
        XCTAssertEqual(snapshot.shadowQueued, 2)
    }

    func testShadowDrainLeavesTheHalfOpenProbeForLiveTraffic() async {
        let clock = ManualClock()
        let teacher = ProgrammableTeacher(script: [.failure(.unavailable)], defaultResult: .success("billing"))
        let router = HybridRouter(student: student(), teacher: teacher,
                                  configuration: configuration(threshold: 1, cooldown: 10, shadowRate: 1), clock: clock)
        _ = await router.route("unsure")          // fails: breaker opens
        _ = await router.route("one")             // confident: queued for shadow
        clock.advance(10)                         // cool-down over: next acquire would be the probe
        let made = await router.drainShadowQueue(maxCalls: 5)
        XCTAssertEqual(made, 0)
        var snapshot = await router.snapshot()
        XCTAssertEqual(snapshot.shadowQueued, 1)
        XCTAssertEqual(snapshot.breaker, .open)
        let live = await router.route("unsure")
        XCTAssertEqual(live.source, .teacher(modelID: "teacher-a"))
        snapshot = await router.snapshot()
        XCTAssertEqual(snapshot.breaker, .closed(consecutiveFailures: 0))
    }

    func testSnapshotEscalationRateAndCounterReset() async {
        let router = HybridRouter(student: student(), teacher: ProgrammableTeacher(), configuration: configuration())
        _ = await router.route("one")
        _ = await router.route("unsure")
        var snapshot = await router.snapshot()
        XCTAssertEqual(snapshot.requests, 2)
        XCTAssertEqual(snapshot.escalationRate, 0.5)
        await router.resetCounters()
        snapshot = await router.snapshot()
        XCTAssertEqual(snapshot.requests, 0)
        XCTAssertEqual(snapshot.escalationRate, 0)
        XCTAssertEqual(snapshot.harvested, 1, "resetting counters keeps the harvested training data")
    }
}
