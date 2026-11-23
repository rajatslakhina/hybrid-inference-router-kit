import Foundation

// MARK: - Circuit breaker

public struct BreakerConfiguration: Equatable, Sendable {
    /// Consecutive transport failures that open the breaker. At least 1.
    public let failureThreshold: Int
    /// How long the breaker stays open before letting one probe through.
    public let cooldownSeconds: Double

    public init(failureThreshold: Int = 3, cooldownSeconds: Double = 30) {
        self.failureThreshold = max(failureThreshold, 1)
        self.cooldownSeconds = cooldownSeconds.isFinite && cooldownSeconds >= 0 ? cooldownSeconds : 30
    }
}

/// Public, read-only view of the breaker for dashboards.
public enum BreakerStatus: Equatable, Sendable {
    case closed(consecutiveFailures: Int)
    case open
    case halfOpen
    /// The teacher's model was retired by the provider. Only `replaceTeacher` re-enables escalation.
    case retired
}

/// Closed → open after N consecutive failures → half-open after a cool-down (exactly one probe) →
/// closed on probe success, open again on probe failure.
///
/// Every state change bumps `generation`, and each permit remembers the generation it was issued in.
/// The router awaits the teacher between `acquire` and `record…`, so other requests can change the
/// state in the meantime (actor re-entrancy). A result from an older generation is ignored, so a slow
/// success that started before an outage cannot close a breaker that has since opened.
struct CircuitBreaker: Sendable {
    enum State: Equatable, Sendable {
        case closed(consecutiveFailures: Int)
        case open(until: Double)
        case halfOpen(probeInFlight: Bool)
        case retired
    }

    struct Permit: Equatable, Sendable {
        let generation: UInt64
        let isProbe: Bool
    }

    let configuration: BreakerConfiguration
    private(set) var state: State = .closed(consecutiveFailures: 0)
    private(set) var generation: UInt64 = 0

    init(configuration: BreakerConfiguration) {
        self.configuration = configuration
    }

    var status: BreakerStatus {
        switch state {
        case .closed(let failures): return .closed(consecutiveFailures: failures)
        case .open: return .open
        case .halfOpen: return .halfOpen
        case .retired: return .retired
        }
    }

    mutating func acquire(now: Double) -> Permit? {
        switch state {
        case .closed:
            return Permit(generation: generation, isProbe: false)
        case .open(let until):
            guard now >= until else { return nil }
            transition(to: .halfOpen(probeInFlight: true))
            return Permit(generation: generation, isProbe: true)
        case .halfOpen(let probeInFlight):
            guard !probeInFlight else { return nil }
            state = .halfOpen(probeInFlight: true)
            return Permit(generation: generation, isProbe: true)
        case .retired:
            return nil
        }
    }

    /// Gives back a permit that was never used (e.g. the budget said no after the breaker said yes).
    mutating func release(_ permit: Permit) {
        guard permit.isProbe, permit.generation == generation, case .halfOpen = state else { return }
        state = .halfOpen(probeInFlight: false)
    }

    mutating func recordSuccess(_ permit: Permit) {
        guard permit.generation == generation else { return }
        switch state {
        case .closed: state = .closed(consecutiveFailures: 0)
        case .halfOpen: transition(to: .closed(consecutiveFailures: 0))
        case .open, .retired: break
        }
    }

    mutating func recordFailure(_ permit: Permit, now: Double) {
        guard permit.generation == generation else { return }
        switch state {
        case .closed(let failures):
            let next = Saturating.add(failures, 1)
            if next >= configuration.failureThreshold {
                transition(to: .open(until: now + configuration.cooldownSeconds))
            } else {
                state = .closed(consecutiveFailures: next)
            }
        case .halfOpen:
            transition(to: .open(until: now + configuration.cooldownSeconds))
        case .open, .retired:
            break
        }
    }

    mutating func retire() { transition(to: .retired) }
    mutating func reset() { transition(to: .closed(consecutiveFailures: 0)) }

    private mutating func transition(to newState: State) {
        state = newState
        generation &+= 1
    }
}

// MARK: - Escalation budget

public struct BudgetConfiguration: Equatable, Sendable {
    /// Spend allowed per window, in your own cost units (e.g. tenths of a cent). `Int.max` = unlimited.
    public let capacityUnits: Int
    /// Window length in seconds. A non-positive or non-finite value falls back to one hour.
    public let windowSeconds: Double

    public init(capacityUnits: Int, windowSeconds: Double = 3600) {
        self.capacityUnits = max(capacityUnits, 0)
        self.windowSeconds = windowSeconds.isFinite && windowSeconds > 0 ? windowSeconds : 3600
    }
}

/// A fixed-window spend cap with reserve → commit/refund accounting.
///
/// The router reserves before it awaits the teacher, so concurrent requests cannot all pass a
/// "spent < cap" check and then all spend (the classic check-then-act race across an `await`).
/// Reservations that are still in flight when a window rolls over keep counting against the new
/// window until they settle, and their spend is booked to the window they were made in. That errs
/// towards under-spending a window, never over-spending it.
struct EscalationBudget: Sendable {
    struct Reservation: Equatable, Sendable {
        let id: UInt64
        let units: Int
        let windowStart: Double
    }

    let configuration: BudgetConfiguration
    private(set) var windowStart: Double
    private(set) var spent = 0
    private var outstanding: [UInt64: Int] = [:]
    private var nextID: UInt64 = 0

    init(configuration: BudgetConfiguration, now: Double) {
        self.configuration = configuration
        self.windowStart = now
    }

    var reserved: Int { outstanding.values.reduce(0, Saturating.add) }
    var outstandingCount: Int { outstanding.count }

    mutating func roll(now: Double) {
        // `now < windowStart` only happens with an injected clock that moves backwards; start over.
        if now < windowStart || now >= windowStart + configuration.windowSeconds {
            windowStart = now
            spent = 0
        }
    }

    /// - Parameter headroom: units that must stay free after this reservation (shadow traffic uses it
    ///   so it can never consume the last of the budget that live escalations need).
    mutating func reserve(units: Int, now: Double, headroom: Int = 0) -> Reservation? {
        roll(now: now)
        let cost = max(units, 0)
        let needed = Saturating.add(Saturating.add(Saturating.add(spent, reserved), cost), max(headroom, 0))
        guard needed <= configuration.capacityUnits else { return nil }
        let reservation = Reservation(id: nextID, units: cost, windowStart: windowStart)
        nextID &+= 1
        outstanding[reservation.id] = cost
        return reservation
    }

    /// Books the spend. Idempotent: committing or refunding twice does nothing the second time.
    mutating func commit(_ reservation: Reservation, now: Double) {
        guard outstanding.removeValue(forKey: reservation.id) != nil else { return }
        roll(now: now)
        if reservation.windowStart == windowStart {
            spent = Saturating.add(spent, reservation.units)
        }
    }

    mutating func refund(_ reservation: Reservation) {
        outstanding.removeValue(forKey: reservation.id)
    }
}

// MARK: - Bounded collections

/// Teacher-labelled examples harvested from production, capped and de-duplicated.
///
/// Keyed by normalised text: a newer label for the same text replaces the older one and moves it to
/// the back. When full, the oldest example is evicted. Memory is bounded no matter how long the app runs.
struct TrainingBuffer: Sendable {
    let capacity: Int
    private var order: [String] = []
    private var byKey: [String: LabeledExample] = [:]
    private(set) var evicted = 0

    init(capacity: Int) {
        self.capacity = max(capacity, 0)
    }

    var count: Int { order.count }
    var examples: [LabeledExample] { order.compactMap { byKey[$0] } }

    @discardableResult
    mutating func insert(_ example: LabeledExample) -> Bool {
        let key = Tokenizer.normalized(example.text)
        guard !key.isEmpty, capacity > 0 else { return false }
        if byKey[key] != nil {
            order.removeAll { $0 == key }
        } else if order.count >= capacity, !order.isEmpty {
            let oldest = order.removeFirst()
            byKey[oldest] = nil
            evicted = Saturating.add(evicted, 1)
        }
        order.append(key)
        byKey[key] = example
        return true
    }
}

/// FIFO queue with a hard cap; when full, the oldest item is dropped and counted.
struct BoundedQueue<Element: Sendable>: Sendable {
    let capacity: Int
    private(set) var items: [Element] = []
    private(set) var dropped = 0

    init(capacity: Int) {
        self.capacity = max(capacity, 0)
    }

    var count: Int { items.count }

    mutating func append(_ element: Element) {
        guard capacity > 0 else {
            dropped = Saturating.add(dropped, 1)
            return
        }
        if items.count >= capacity, !items.isEmpty {
            items.removeFirst()
            dropped = Saturating.add(dropped, 1)
        }
        items.append(element)
    }

    mutating func popFirst() -> Element? {
        guard !items.isEmpty else { return nil }
        return items.removeFirst()
    }

    /// Puts an item back at the head (used when a drain stops early). Respects the cap.
    mutating func pushFront(_ element: Element) {
        guard items.count < capacity else {
            dropped = Saturating.add(dropped, 1)
            return
        }
        items.insert(element, at: 0)
    }
}

// MARK: - Shadow sampling

/// Decides which confidently-answered requests are also sent to the teacher, in the background,
/// to measure how often the student is confidently wrong.
///
/// Deterministic per normalised text (stable hash, not `Hasher`), so the same input is always in or
/// always out of the sample: re-asking a question cannot be used to farm teacher calls.
struct ShadowSampler: Sendable {
    let rate: Double
    let salt: String

    init(rate: Double, salt: String) {
        self.rate = Saturating.unit(rate, nanValue: 0)
        self.salt = salt
    }

    func shouldSample(_ text: String) -> Bool {
        guard rate > 0 else { return false }
        guard rate < 1 else { return true }
        return Double(bucket(for: text)) < rate * 10_000
    }

    /// The input's fixed position in `0..<10_000`. Stable across processes and devices.
    func bucket(for text: String) -> UInt64 {
        StableHash.fnv1a64(salt + "\u{1F}" + Tokenizer.normalized(text)) % 10_000
    }
}
