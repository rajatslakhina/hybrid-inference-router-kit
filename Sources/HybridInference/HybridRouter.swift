import Foundation

// MARK: - Configuration and results

public struct RouterConfiguration: Sendable {
    /// Starting policy. Replace it with the calibrated one via `install(_:policy:)`.
    public var policy: EscalationPolicy
    public var budget: BudgetConfiguration
    /// What one teacher call costs, in the budget's units. Change it at runtime with `reprice`.
    public var pricePerCallUnits: Int
    public var breaker: BreakerConfiguration
    /// Latency budget for one teacher call, in `(0, maxTeacherTimeout]`; a non-positive value falls back to
    /// 8 s. Enforced by cancellation (see `withTimeout`).
    public var teacherTimeout: Duration
    /// Share of confidently-answered requests also sent to the teacher in the background, `0...1`.
    public var shadowSampleRate: Double
    /// Changing the salt re-draws which inputs are in the shadow sample.
    public var shadowSalt: String
    public var shadowQueueCapacity: Int
    /// Units shadow traffic must always leave free for live escalations.
    public var shadowHeadroomUnits: Int
    public var harvestCapacity: Int

    /// Upper bound for `teacherTimeout` (10 minutes).
    public static let maxTeacherTimeout: Duration = .seconds(600)

    /// A zero or negative timeout would time out (and bill, and count as an outage) every escalation, so it
    /// falls back to 8 s. Above `maxTeacherTimeout` it is clamped: no escalation should wait that long, and a
    /// huge value such as `.seconds(Int64.max)` would overflow the sleep deadline. Applied both here and in
    /// `HybridRouter.init`, because the property is publicly settable after initialisation.
    static func sanitizedTimeout(_ timeout: Duration) -> Duration {
        timeout <= .zero ? .seconds(8) : min(timeout, maxTeacherTimeout)
    }

    public init(
        policy: EscalationPolicy = .escalateAll,
        budget: BudgetConfiguration = BudgetConfiguration(capacityUnits: 100),
        pricePerCallUnits: Int = 1,
        breaker: BreakerConfiguration = BreakerConfiguration(),
        teacherTimeout: Duration = .seconds(8),
        shadowSampleRate: Double = 0.05,
        shadowSalt: String = "shadow-v1",
        shadowQueueCapacity: Int = 64,
        shadowHeadroomUnits: Int = 0,
        harvestCapacity: Int = 2_000
    ) {
        self.policy = policy
        self.budget = budget
        self.pricePerCallUnits = max(pricePerCallUnits, 0)
        self.breaker = breaker
        self.teacherTimeout = Self.sanitizedTimeout(teacherTimeout)
        self.shadowSampleRate = shadowSampleRate
        self.shadowSalt = shadowSalt
        self.shadowQueueCapacity = max(shadowQueueCapacity, 0)
        self.shadowHeadroomUnits = max(shadowHeadroomUnits, 0)
        self.harvestCapacity = max(harvestCapacity, 0)
    }
}

/// Why an escalation ended with the student's answer instead of the teacher's.
public enum FallbackReason: String, CaseIterable, Sendable {
    case budgetExhausted
    case circuitOpen
    case teacherRetired
    case teacherFailed
    case teacherRefused
    case offListLabel
    case cancelled
}

public enum AnswerSource: Equatable, Sendable {
    case student
    case teacher(modelID: String)
    case studentFallback(FallbackReason)
}

public struct RoutedAnswer: Equatable, Sendable, Identifiable {
    public let id: Int
    public let text: String
    /// `nil` only when the student has no labels at all and the teacher did not answer.
    public let label: ClassLabel?
    public let source: AnswerSource
    /// Why the student was not trusted; `nil` when it answered on its own.
    public let escalationReason: EscalationReason?
    public let teacherError: TeacherError?
    public let studentLabel: ClassLabel?
    public let studentConfidence: Double
    public let modelVersion: String
}

public struct RouterSnapshot: Equatable, Sendable {
    public let modelVersion: String
    public let teacherModelID: String
    public let requests: Int
    public let answeredByStudent: Int
    public let answeredByTeacher: Int
    public let fallbacks: [FallbackReason: Int]
    /// Every teacher call started, live or shadow, whatever its outcome.
    public let teacherCalls: Int
    /// Live escalations where the teacher's label differed from the student's top guess.
    public let teacherDisagreements: Int
    public let spentUnits: Int
    public let reservedUnits: Int
    public let capacityUnits: Int
    public let pricePerCallUnits: Int
    public let breaker: BreakerStatus
    public let shadowQueued: Int
    public let shadowDropped: Int
    public let shadowAgreements: Int
    public let shadowDisagreements: Int
    public let harvested: Int
    public let harvestEvicted: Int

    public var fallbackCount: Int { fallbacks.values.reduce(0, Saturating.add) }
    /// Share of requests the student was not trusted with.
    public var escalationRate: Double {
        Saturating.ratio(Saturating.subtract(requests, answeredByStudent), requests)
    }
}

// MARK: - Router

/// Routes each request to the on-device student or to the teacher (Claude), within a spend budget,
/// behind a circuit breaker, and turns every teacher answer into training data for the next student.
///
/// Concurrency model: an actor, so all state is mutated serially, but `route` suspends while the
/// teacher answers and other requests run in between. Everything that must hold across that
/// suspension is settled *before* the `await` (budget reservation, breaker permit, the model and
/// policy snapshot) and re-validated *after* it (permit generation, allowed labels).
public actor HybridRouter {
    struct ShadowItem: Sendable {
        let text: String
        let studentLabel: ClassLabel
    }

    struct Counters: Sendable {
        var requests = 0
        var student = 0
        var teacher = 0
        var fallbacks: [FallbackReason: Int] = [:]
        var teacherCalls = 0
        var disagreements = 0
        var shadowAgreements = 0
        var shadowDisagreements = 0
    }

    private var student: any StudentModel
    private var policy: EscalationPolicy
    private var teacher: any TeacherProvider
    private var budget: EscalationBudget
    private var breaker: CircuitBreaker
    private var harvest: TrainingBuffer
    private var shadowQueue: BoundedQueue<ShadowItem>
    private var price: Int
    private let timeout: Duration
    private let sampler: ShadowSampler
    private let shadowHeadroom: Int
    private let clock: any TimeSource
    private var nextID = 0
    private var counters = Counters()

    public init(
        student: any StudentModel,
        teacher: any TeacherProvider,
        configuration: RouterConfiguration = RouterConfiguration(),
        clock: any TimeSource = MonotonicTimeSource()
    ) {
        self.student = student
        self.policy = configuration.policy
        self.teacher = teacher
        self.budget = EscalationBudget(configuration: configuration.budget, now: clock.now())
        self.breaker = CircuitBreaker(configuration: configuration.breaker)
        self.harvest = TrainingBuffer(capacity: configuration.harvestCapacity)
        self.shadowQueue = BoundedQueue(capacity: configuration.shadowQueueCapacity)
        // Re-sanitised here: `RouterConfiguration`'s properties are `var`, so a caller can set them after init.
        self.price = max(configuration.pricePerCallUnits, 0)
        self.timeout = RouterConfiguration.sanitizedTimeout(configuration.teacherTimeout)
        self.sampler = ShadowSampler(rate: configuration.shadowSampleRate, salt: configuration.shadowSalt)
        self.shadowHeadroom = configuration.shadowHeadroomUnits
        self.clock = clock
    }

    // MARK: Serving

    public func route(_ text: String) async -> RoutedAnswer {
        let id = nextID
        nextID = Saturating.add(nextID, 1)
        counters.requests = Saturating.add(counters.requests, 1)

        // Snapshot before any suspension: a concurrent `install` must not change this request's model.
        let model = student
        let prediction = model.predict(text)
        let studentTop = prediction.distribution.top

        func answer(
            _ label: ClassLabel?, _ source: AnswerSource, _ reason: EscalationReason?, _ error: TeacherError? = nil
        ) -> RoutedAnswer {
            RoutedAnswer(
                id: id, text: text, label: label, source: source, escalationReason: reason, teacherError: error,
                studentLabel: studentTop?.label, studentConfidence: studentTop?.probability ?? 0,
                modelVersion: model.version
            )
        }

        guard let reason = policy.escalationReason(for: prediction) else {
            counters.student = Saturating.add(counters.student, 1)
            if let top = studentTop, sampler.shouldSample(text) {
                shadowQueue.append(ShadowItem(text: text, studentLabel: top.label))
            }
            return answer(studentTop?.label, .student, nil)
        }

        func fallback(_ why: FallbackReason, _ error: TeacherError? = nil) -> RoutedAnswer {
            counters.fallbacks[why] = Saturating.add(counters.fallbacks[why, default: 0], 1)
            return answer(studentTop?.label, .studentFallback(why), reason, error)
        }

        let now = clock.now()
        // Budget first, then breaker: a refused budget must not burn the breaker's single half-open probe.
        guard let reservation = budget.reserve(units: price, now: now) else { return fallback(.budgetExhausted) }
        guard let permit = breaker.acquire(now: now) else {
            budget.refund(reservation)
            return fallback(breaker.status == .retired ? .teacherRetired : .circuitOpen)
        }

        counters.teacherCalls = Saturating.add(counters.teacherCalls, 1)
        let activeTeacher = teacher
        let allowed = model.labels
        let outcome = await callTeacher(activeTeacher, text: text, allowed: allowed)

        // ---- Suspension point above: anything may have changed. Re-validate, never assume. ----
        switch outcome {
        case .success(let label):
            budget.commit(reservation, now: clock.now())
            breaker.recordSuccess(permit)
            // The teacher is untrusted output: an off-list label is never served and never trained on.
            guard allowed.contains(label) else { return fallback(.offListLabel) }
            harvest.insert(LabeledExample(text: text, label: label))
            counters.teacher = Saturating.add(counters.teacher, 1)
            if studentTop?.label != label {
                counters.disagreements = Saturating.add(counters.disagreements, 1)
            }
            return answer(label, .teacher(modelID: activeTeacher.modelID), reason)
        case .cancelled:
            budget.refund(reservation)
            breaker.release(permit)
            return fallback(.cancelled)
        case .failure(let error):
            settle(error, reservation: reservation, permit: permit)
            switch error {
            case .modelRetired: return fallback(.teacherRetired, error)
            case .refused: return fallback(.teacherRefused, error)
            default: return fallback(.teacherFailed, error)
            }
        }
    }

    /// Sends up to `maxCalls` queued shadow samples to the teacher and records agreement.
    ///
    /// Run it off the critical path (e.g. after a screen settles, or from a background task). It stops
    /// early, leaving work queued, when the budget would dip into `shadowHeadroomUnits` or the breaker
    /// is not closed. A half-open breaker's single probe is left for live traffic. Returns the number of
    /// teacher calls made.
    @discardableResult
    public func drainShadowQueue(maxCalls: Int) async -> Int {
        var made = 0
        while made < maxCalls, let item = shadowQueue.popFirst() {
            guard case .closed = breaker.state else {
                shadowQueue.pushFront(item)
                break
            }
            let now = clock.now()
            guard let reservation = budget.reserve(units: price, now: now, headroom: shadowHeadroom) else {
                shadowQueue.pushFront(item)
                break
            }
            guard let permit = breaker.acquire(now: now) else {
                budget.refund(reservation)
                shadowQueue.pushFront(item)
                break
            }
            made += 1
            counters.teacherCalls = Saturating.add(counters.teacherCalls, 1)
            let allowed = student.labels
            let outcome = await callTeacher(teacher, text: item.text, allowed: allowed)
            switch outcome {
            case .success(let label):
                budget.commit(reservation, now: clock.now())
                breaker.recordSuccess(permit)
                guard allowed.contains(label) else { continue }
                harvest.insert(LabeledExample(text: item.text, label: label))
                if label == item.studentLabel {
                    counters.shadowAgreements = Saturating.add(counters.shadowAgreements, 1)
                } else {
                    counters.shadowDisagreements = Saturating.add(counters.shadowDisagreements, 1)
                }
            case .cancelled:
                budget.refund(reservation)
                breaker.release(permit)
                shadowQueue.pushFront(item)
                return made
            case .failure(let error):
                settle(error, reservation: reservation, permit: permit)
            }
        }
        return made
    }

    // MARK: Lifecycle

    /// Swaps in a new student and the policy calibrated for it. Requests already in flight finish on
    /// the model they started with.
    public func install(_ model: any StudentModel, policy: EscalationPolicy) {
        student = model
        self.policy = policy
    }

    /// Moves to a new teacher (e.g. after the provider retires a model) and resets the breaker.
    /// Results from calls to the old teacher that are still in flight no longer move the breaker.
    public func replaceTeacher(_ newTeacher: any TeacherProvider) {
        teacher = newTeacher
        breaker.reset()
    }

    /// The provider changed its price. Applies to reservations made from now on.
    public func reprice(unitsPerCall: Int) {
        price = max(unitsPerCall, 0)
    }

    public func harvestedExamples() -> [LabeledExample] { harvest.examples }

    public func resetCounters() { counters = Counters() }

    public func snapshot() -> RouterSnapshot {
        budget.roll(now: clock.now())
        return RouterSnapshot(
            modelVersion: student.version,
            teacherModelID: teacher.modelID,
            requests: counters.requests,
            answeredByStudent: counters.student,
            answeredByTeacher: counters.teacher,
            fallbacks: counters.fallbacks,
            teacherCalls: counters.teacherCalls,
            teacherDisagreements: counters.disagreements,
            spentUnits: budget.spent,
            reservedUnits: budget.reserved,
            capacityUnits: budget.configuration.capacityUnits,
            pricePerCallUnits: price,
            breaker: breaker.status,
            shadowQueued: shadowQueue.count,
            shadowDropped: shadowQueue.dropped,
            shadowAgreements: counters.shadowAgreements,
            shadowDisagreements: counters.shadowDisagreements,
            harvested: harvest.count,
            harvestEvicted: harvest.evicted
        )
    }

    // MARK: Internals

    enum Outcome: Sendable {
        case success(ClassLabel)
        case failure(TeacherError)
        case cancelled
    }

    private nonisolated func callTeacher(
        _ teacher: any TeacherProvider, text: String, allowed: [ClassLabel]
    ) async -> Outcome {
        do {
            let label = try await withTimeout(timeout) {
                try await teacher.label(text, allowedLabels: allowed)
            }
            return .success(label)
        } catch is CancellationError {
            return .cancelled
        } catch {
            // Whatever the teacher threw, if *our* caller was cancelled this is a cancellation, not an outage.
            if Task.isCancelled { return .cancelled }
            return .failure(error as? TeacherError ?? .unavailable)
        }
    }

    private func settle(_ error: TeacherError, reservation: EscalationBudget.Reservation, permit: CircuitBreaker.Permit) {
        if error.isBillable {
            budget.commit(reservation, now: clock.now())
        } else {
            budget.refund(reservation)
        }
        if error == .modelRetired {
            // Only if the breaker is still in the generation this call started in: a retirement reported
            // by a teacher that has since been replaced must not disable the new one.
            if permit.generation == breaker.generation { breaker.retire() }
        } else if error.countsAsOutage {
            breaker.recordFailure(permit, now: clock.now())
        } else {
            breaker.recordSuccess(permit)
        }
    }
}

private enum TimeoutRace<Value: Sendable>: Sendable {
    case value(Value)
    case timerFired
    case timerCancelled
}

/// Runs `operation`, or throws `TeacherError.timedOut` once `timeout` elapses.
///
/// Implemented with a task group and cancellation. If the *caller* is cancelled, the timer child stands
/// down quietly and the operation's own outcome is what surfaces, so the router always sees what the
/// teacher actually did. Limit: a task group always waits for its children, so the timeout only bounds
/// latency for operations that honour cancellation (`URLSession` and `Task.sleep` do). An operation that
/// ignores cancellation delays the timeout until it returns.
func withTimeout<T: Sendable>(
    _ timeout: Duration,
    operation: @escaping @Sendable () async throws -> T
) async throws -> T {
    try await withThrowingTaskGroup(of: TimeoutRace<T>.self) { group in
        group.addTask { .value(try await operation()) }
        group.addTask {
            do {
                try await Task.sleep(for: timeout)
                return .timerFired
            } catch {
                return .timerCancelled
            }
        }
        defer { group.cancelAll() }
        while let next = try await group.next() {
            switch next {
            case .value(let value): return value
            case .timerFired: throw TeacherError.timedOut
            case .timerCancelled: continue
            }
        }
        throw TeacherError.unavailable
    }
}
