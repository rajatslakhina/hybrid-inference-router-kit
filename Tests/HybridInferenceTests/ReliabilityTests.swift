import XCTest
@testable import HybridInference

final class CircuitBreakerTests: XCTestCase {
    private func breaker(threshold: Int = 2, cooldown: Double = 10) -> CircuitBreaker {
        CircuitBreaker(configuration: BreakerConfiguration(failureThreshold: threshold, cooldownSeconds: cooldown))
    }

    func testOpensAfterConsecutiveFailuresAndSuccessResetsTheCount() throws {
        var subject = breaker(threshold: 2)
        let first = try XCTUnwrap(subject.acquire(now: 0))
        subject.recordFailure(first, now: 0)
        XCTAssertEqual(subject.status, .closed(consecutiveFailures: 1))
        let second = try XCTUnwrap(subject.acquire(now: 0))
        subject.recordSuccess(second)
        XCTAssertEqual(subject.status, .closed(consecutiveFailures: 0))
        for _ in 0..<2 {
            let permit = try XCTUnwrap(subject.acquire(now: 0))
            subject.recordFailure(permit, now: 0)
        }
        XCTAssertEqual(subject.status, .open)
        XCTAssertNil(subject.acquire(now: 9.9))
    }

    func testHalfOpenAllowsExactlyOneProbe() throws {
        var subject = breaker(threshold: 1, cooldown: 10)
        subject.recordFailure(try XCTUnwrap(subject.acquire(now: 0)), now: 0)
        let probe = try XCTUnwrap(subject.acquire(now: 10))
        XCTAssertTrue(probe.isProbe)
        XCTAssertEqual(subject.status, .halfOpen)
        XCTAssertNil(subject.acquire(now: 10), "a second caller must not get a probe while one is in flight")
        subject.recordSuccess(probe)
        XCTAssertEqual(subject.status, .closed(consecutiveFailures: 0))
    }

    func testFailedProbeReopensForAnotherCooldown() throws {
        var subject = breaker(threshold: 1, cooldown: 10)
        subject.recordFailure(try XCTUnwrap(subject.acquire(now: 0)), now: 0)
        let probe = try XCTUnwrap(subject.acquire(now: 10))
        subject.recordFailure(probe, now: 10)
        XCTAssertEqual(subject.status, .open)
        XCTAssertNil(subject.acquire(now: 19))
        XCTAssertNotNil(subject.acquire(now: 20))
    }

    func testReleasedProbeLetsTheNextCallerProbe() throws {
        var subject = breaker(threshold: 1, cooldown: 0)
        subject.recordFailure(try XCTUnwrap(subject.acquire(now: 0)), now: 0)
        let probe = try XCTUnwrap(subject.acquire(now: 0))
        subject.release(probe)
        XCTAssertNotNil(subject.acquire(now: 0))
    }

    /// The re-entrancy guard. A slow call starts while closed; the breaker then opens, cools down and
    /// sends a probe. When the slow pre-outage success finally lands it must not close the breaker
    /// over the probe's head: only the probe decides.
    func testStaleResultsFromAnOlderGenerationAreIgnored() throws {
        var subject = breaker(threshold: 1, cooldown: 10)
        let slow = try XCTUnwrap(subject.acquire(now: 0))
        let fast = try XCTUnwrap(subject.acquire(now: 0))
        subject.recordFailure(fast, now: 0)
        XCTAssertEqual(subject.status, .open)
        let probe = try XCTUnwrap(subject.acquire(now: 10))
        XCTAssertTrue(probe.isProbe)

        subject.recordSuccess(slow)
        XCTAssertEqual(subject.status, .halfOpen, "a pre-outage success must not close the breaker")
        XCTAssertNil(subject.acquire(now: 10), "the probe is still the only call allowed through")
        subject.recordFailure(slow, now: 10)
        XCTAssertEqual(subject.status, .halfOpen, "a stale failure must not re-open it either")

        subject.recordFailure(probe, now: 10)
        XCTAssertEqual(subject.status, .open)
    }

    func testRetiredIsPermanentUntilReset() throws {
        var subject = breaker()
        subject.retire()
        XCTAssertNil(subject.acquire(now: 1_000_000))
        XCTAssertEqual(subject.status, .retired)
        subject.reset()
        XCTAssertNotNil(subject.acquire(now: 0))
    }

    func testConfigurationSanitisesBadValues() {
        let config = BreakerConfiguration(failureThreshold: -3, cooldownSeconds: .infinity)
        XCTAssertEqual(config.failureThreshold, 1)
        XCTAssertEqual(config.cooldownSeconds, 30)
    }
}

final class EscalationBudgetTests: XCTestCase {
    private func budget(_ capacity: Int, window: Double = 60) -> EscalationBudget {
        EscalationBudget(configuration: BudgetConfiguration(capacityUnits: capacity, windowSeconds: window), now: 0)
    }

    func testReservationsCountAgainstCapacityBeforeTheyAreCommitted() throws {
        var subject = budget(3)
        let a = try XCTUnwrap(subject.reserve(units: 2, now: 0))
        XCTAssertNil(subject.reserve(units: 2, now: 0), "in-flight reservations must block overspend")
        subject.commit(a, now: 1)
        XCTAssertEqual(subject.spent, 2)
        XCTAssertEqual(subject.reserved, 0)
        XCTAssertNotNil(subject.reserve(units: 1, now: 1))
    }

    func testRefundReturnsUnitsAndCommitIsIdempotent() throws {
        var subject = budget(2)
        let a = try XCTUnwrap(subject.reserve(units: 2, now: 0))
        subject.refund(a)
        let b = try XCTUnwrap(subject.reserve(units: 2, now: 0))
        subject.commit(b, now: 0)
        subject.commit(b, now: 0)
        subject.refund(b)
        XCTAssertEqual(subject.spent, 2)
        XCTAssertEqual(subject.outstandingCount, 0)
    }

    func testWindowRollResetsSpendButInFlightReservationsStillCount() throws {
        var subject = budget(2, window: 60)
        let early = try XCTUnwrap(subject.reserve(units: 1, now: 0))
        subject.commit(try XCTUnwrap(subject.reserve(units: 1, now: 0)), now: 0)
        XCTAssertNil(subject.reserve(units: 1, now: 59))
        // New window: spend resets, but `early` is still in flight and still holds 1 unit.
        XCTAssertNotNil(subject.reserve(units: 1, now: 61))
        XCTAssertNil(subject.reserve(units: 1, now: 61))
        // `early` belonged to the old window: settling it now does not charge the new one.
        subject.commit(early, now: 62)
        XCTAssertEqual(subject.spent, 0)
    }

    func testHeadroomKeepsUnitsFreeForLiveTraffic() {
        var subject = budget(3)
        XCTAssertNotNil(subject.reserve(units: 1, now: 0, headroom: 2))
        XCTAssertNil(subject.reserve(units: 1, now: 0, headroom: 2))
        XCTAssertNotNil(subject.reserve(units: 2, now: 0))
    }

    func testClockGoingBackwardsStartsAFreshWindow() throws {
        var subject = budget(1)
        subject.commit(try XCTUnwrap(subject.reserve(units: 1, now: 100)), now: 100)
        XCTAssertNil(subject.reserve(units: 1, now: 100))
        XCTAssertNotNil(subject.reserve(units: 1, now: 50))
    }

    func testHugeValuesSaturateInsteadOfTrapping() throws {
        var subject = budget(10)
        XCTAssertNil(subject.reserve(units: Int.max, now: 0, headroom: Int.max))
        var unlimited = budget(Int.max)
        let a = try XCTUnwrap(unlimited.reserve(units: Int.max, now: 0))
        let b = try XCTUnwrap(unlimited.reserve(units: Int.max, now: 0))
        unlimited.commit(a, now: 0)
        unlimited.commit(b, now: 0)
        XCTAssertEqual(unlimited.spent, Int.max)
        XCTAssertEqual(subject.reserved, 0)
    }

    func testNegativeAndZeroPricesAreFreeAndZeroCapacityAllowsOnlyFreeCalls() {
        var subject = budget(0)
        XCTAssertNotNil(subject.reserve(units: -5, now: 0))
        XCTAssertNil(subject.reserve(units: 1, now: 0))
    }

    func testConfigurationSanitisesBadValues() {
        let config = BudgetConfiguration(capacityUnits: -1, windowSeconds: .nan)
        XCTAssertEqual(config.capacityUnits, 0)
        XCTAssertEqual(config.windowSeconds, 3600)
    }
}

final class BoundedCollectionTests: XCTestCase {
    func testTrainingBufferEvictsOldestAndRelabelMovesToNewest() {
        var buffer = TrainingBuffer(capacity: 2)
        XCTAssertTrue(buffer.insert(.init(text: "a", label: "x")))
        XCTAssertTrue(buffer.insert(.init(text: "b", label: "x")))
        XCTAssertTrue(buffer.insert(.init(text: "A!", label: "y")))   // relabels "a", now newest
        XCTAssertEqual(buffer.examples.map(\.label), ["x", "y"])
        XCTAssertTrue(buffer.insert(.init(text: "c", label: "z")))    // evicts "b"
        XCTAssertEqual(buffer.examples.map(\.text), ["A!", "c"])
        XCTAssertEqual(buffer.evicted, 1)
    }

    func testTrainingBufferRejectsBlankTextAndZeroCapacity() {
        var buffer = TrainingBuffer(capacity: 2)
        XCTAssertFalse(buffer.insert(.init(text: "...", label: "x")))
        var none = TrainingBuffer(capacity: -4)
        XCTAssertFalse(none.insert(.init(text: "a", label: "x")))
        XCTAssertEqual(none.count, 0)
    }

    func testBoundedQueueDropsOldestAndRespectsCapOnPushFront() {
        var queue = BoundedQueue<Int>(capacity: 2)
        queue.append(1); queue.append(2); queue.append(3)
        XCTAssertEqual(queue.items, [2, 3])
        XCTAssertEqual(queue.dropped, 1)
        queue.pushFront(9)
        XCTAssertEqual(queue.items, [2, 3])
        XCTAssertEqual(queue.dropped, 2)
        XCTAssertEqual(queue.popFirst(), 2)
        queue.pushFront(7)
        XCTAssertEqual(queue.items, [7, 3])
        var empty = BoundedQueue<Int>(capacity: 0)
        empty.append(1)
        XCTAssertNil(empty.popFirst())
        XCTAssertEqual(empty.dropped, 1)
    }

    func testShadowSamplerHonoursRateBoundsAndIsDeterministicAcrossInstances() {
        let texts = (0..<1_000).map { "ticket number \($0)" }
        XCTAssertFalse(texts.contains { ShadowSampler(rate: 0, salt: "s").shouldSample($0) })
        XCTAssertFalse(texts.contains { ShadowSampler(rate: .nan, salt: "s").shouldSample($0) })
        XCTAssertTrue(texts.allSatisfy { ShadowSampler(rate: 1, salt: "s").shouldSample($0) })

        let quarter = ShadowSampler(rate: 0.25, salt: "s")
        let picked = texts.filter { quarter.shouldSample($0) }
        XCTAssertGreaterThan(picked.count, 150)
        XCTAssertLessThan(picked.count, 350)
        // Same salt, new instance: identical decisions. Different salt: a different sample.
        XCTAssertEqual(texts.filter { ShadowSampler(rate: 0.25, salt: "s").shouldSample($0) }, picked)
        XCTAssertNotEqual(texts.filter { ShadowSampler(rate: 0.25, salt: "t").shouldSample($0) }, picked)
        // Golden value, computed independently from the FNV-1a definition. A process-seeded hash (`Hasher`)
        // would agree with itself within this test run but fail this.
        XCTAssertEqual(ShadowSampler(rate: 0.25, salt: "s").bucket(for: "ticket number 7"), 4_261)
        // Normalised text decides, so re-asking with different punctuation cannot farm extra samples.
        XCTAssertEqual(quarter.shouldSample("Ticket NUMBER 7!!"), quarter.shouldSample("ticket number 7"))
    }
}
