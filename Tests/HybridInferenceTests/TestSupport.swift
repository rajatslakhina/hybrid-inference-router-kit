import Foundation
@testable import HybridInference

/// A clock the test moves by hand.
final class ManualClock: TimeSource, @unchecked Sendable {
    private let lock = NSLock()
    private var value: Double

    init(_ start: Double = 1_000) { value = start }

    func now() -> Double {
        lock.lock()
        defer { lock.unlock() }
        return value
    }

    func advance(_ seconds: Double) {
        lock.lock()
        value += seconds
        lock.unlock()
    }

    func set(_ newValue: Double) {
        lock.lock()
        value = newValue
        lock.unlock()
    }
}

/// A student whose answer for each input is fixed by the test.
struct FixedStudent: StudentModel {
    let version: String
    let labels: [ClassLabel]
    let table: [String: ScoredLabel]
    let defaultAnswer: ScoredLabel

    init(version: String = "fixed-v1", labels: [ClassLabel], table: [String: ScoredLabel] = [:], defaultAnswer: ScoredLabel) {
        self.version = version
        self.labels = labels.sorted()
        self.table = Dictionary(uniqueKeysWithValues: table.map { (Tokenizer.normalized($0.key), $0.value) })
        self.defaultAnswer = defaultAnswer
    }

    func predict(_ text: String) -> StudentPrediction {
        let top = table[Tokenizer.normalized(text)] ?? defaultAnswer
        var probabilities: [ClassLabel: Double] = [top.label: top.probability]
        let rest = labels.filter { $0 != top.label }
        if !rest.isEmpty {
            for label in rest { probabilities[label] = (1 - top.probability) / Double(rest.count) }
        }
        let tokens = Tokenizer.tokens(text).count
        return StudentPrediction(
            modelVersion: version,
            distribution: Distribution(probabilities: probabilities),
            tokenCount: tokens,
            knownTokenCount: tokens
        )
    }
}

func scored(_ label: ClassLabel, _ probability: Double) -> ScoredLabel {
    ScoredLabel(label: label, probability: probability)
}

/// A teacher that replays scripted results, with optional latency, and records concurrency.
actor ProgrammableTeacher: TeacherProvider {
    nonisolated let modelID: String
    private var script: [Result<ClassLabel, TeacherError>]
    private var defaultResult: Result<ClassLabel, TeacherError>
    private let latency: Duration
    private(set) var calls = 0
    private(set) var maxConcurrent = 0
    private var inFlight = 0

    init(
        modelID: String = "teacher-a",
        script: [Result<ClassLabel, TeacherError>] = [],
        defaultResult: Result<ClassLabel, TeacherError> = .success("billing"),
        latency: Duration = .zero
    ) {
        self.modelID = modelID
        self.script = script
        self.defaultResult = defaultResult
        self.latency = latency
    }

    func setDefault(_ result: Result<ClassLabel, TeacherError>) { defaultResult = result }

    func label(_ text: String, allowedLabels: [ClassLabel]) async throws -> ClassLabel {
        calls += 1
        inFlight += 1
        maxConcurrent = max(maxConcurrent, inFlight)
        defer { inFlight -= 1 }
        if latency > .zero { try await Task.sleep(for: latency) }
        let result = script.isEmpty ? defaultResult : script.removeFirst()
        return try result.get()
    }
}

/// A teacher that holds every call until the test releases it, so interleavings are exact.
actor GatedTeacher: TeacherProvider {
    nonisolated let modelID: String
    private var waiting: [CheckedContinuation<ClassLabel, Error>] = []
    private(set) var calls = 0

    init(modelID: String = "gated") { self.modelID = modelID }

    func label(_ text: String, allowedLabels: [ClassLabel]) async throws -> ClassLabel {
        calls += 1
        return try await withCheckedThrowingContinuation { continuation in
            waiting.append(continuation)
        }
    }

    func pending() -> Int { waiting.count }

    @discardableResult
    func release(_ result: Result<ClassLabel, TeacherError>) -> Bool {
        guard !waiting.isEmpty else { return false }
        waiting.removeFirst().resume(with: result.mapError { $0 as Error })
        return true
    }
}

/// Behaves like URLSession on cancellation: waits until cancelled, then throws URLError(.cancelled).
final class SlowCancellableTransport: HTTPTransport, @unchecked Sendable {
    private let lock = NSLock()
    private var count = 0

    var started: Int { lock.withLock { count } }

    func send(_ request: URLRequest) async throws -> (Data, HTTPURLResponse) {
        lock.withLock { count += 1 }
        do {
            try await Task.sleep(for: .seconds(60))
        } catch {
            throw URLError(.cancelled)
        }
        throw URLError(.timedOut)
    }
}

/// Waits until cancelled, then throws an ordinary `TeacherError.unavailable` instead of `CancellationError`.
actor UnavailableOnCancelTeacher: TeacherProvider {
    nonisolated let modelID = "unavailable-on-cancel"
    private(set) var calls = 0

    func label(_ text: String, allowedLabels: [ClassLabel]) async throws -> ClassLabel {
        calls += 1
        do {
            try await Task.sleep(for: .seconds(60))
        } catch {
            throw TeacherError.unavailable
        }
        return "billing"
    }
}

/// Polls `condition` for up to ~2 s. Used only to wait for a gated call to *arrive*, never to sleep
/// past a race.
func eventually(_ condition: @Sendable () async -> Bool) async -> Bool {
    for _ in 0..<2_000 {
        if await condition() { return true }
        try? await Task.sleep(for: .milliseconds(1))
    }
    return false
}

// MARK: - Support-ticket dataset shared by the end-to-end tests

enum Tickets {
    static let labels: [ClassLabel] = ["account", "billing", "bug", "feature"]

    static let rules: [ScriptedTeacher.Rule] = [
        .init(keywords: ["crash", "crashes", "freezes", "error", "broken", "bug", "blank", "stuck", "fails"], label: "bug"),
        .init(keywords: ["refund", "charged", "invoice", "payment", "subscription", "price", "receipt", "billing", "card"], label: "billing"),
        .init(keywords: ["password", "login", "email", "locked", "username", "account", "accounts", "sign"], label: "account"),
        .init(keywords: ["add", "wish", "would", "request", "feature", "export"], label: "feature"),
    ]

    static let seed: [LabeledExample] = [
        .init(text: "I was charged twice for my subscription", label: "billing"),
        .init(text: "Please send the invoice for last month", label: "billing"),
        .init(text: "The app crashes when I open settings", label: "bug"),
        .init(text: "Screen freezes after I log in", label: "bug"),
        .init(text: "I forgot my password", label: "account"),
        .init(text: "Cannot change the email on my account", label: "account"),
        .init(text: "Please add dark mode", label: "feature"),
        .init(text: "I wish I could export to PDF", label: "feature"),
    ]

    static let eval: [LabeledExample] = [
        .init(text: "Why was my card charged again", label: "billing"),
        .init(text: "Refund the annual plan please", label: "billing"),
        .init(text: "Receipt shows the wrong price", label: "billing"),
        .init(text: "Photos fail with an error on upload", label: "bug"),
        .init(text: "The widget is blank on my home screen", label: "bug"),
        .init(text: "Sync is stuck at fifty percent", label: "bug"),
        .init(text: "Reset link for my password never arrives", label: "account"),
        .init(text: "My account is locked after three tries", label: "account"),
        .init(text: "How do I change my username", label: "account"),
        .init(text: "Would love a landscape mode on iPad", label: "feature"),
        .init(text: "Please add a shortcut for quick notes", label: "feature"),
        .init(text: "Wish the calendar had a week view", label: "feature"),
    ]

    /// Day-one traffic: what student v1 serves and Claude labels.
    static let dayOne: [String] = [
        "I was charged for a plan I cancelled",
        "Need a refund for the duplicate payment",
        "Where can I download my invoice",
        "The price went up without notice",
        "My card was charged but nothing unlocked",
        "Send me a receipt for the annual subscription",
        "Refund please, I bought it by mistake",
        "Payment failed but the money left my bank",
        "Billing shows two charges this month",
        "Can I get a refund on the family plan",
        "The app crashes on launch since the update",
        "Upload fails with an unknown error",
        "Home screen widget stays blank",
        "Video playback freezes after ten seconds",
        "Search is broken for names with accents",
        "Sync gets stuck and never finishes",
        "Crash when I rotate the phone",
        "Error message when saving a photo",
        "Notifications are broken after the update",
        "Settings screen is blank on iPad",
        "I cannot reset my password",
        "My account got locked after the update",
        "Change the email address on my account",
        "Login code never arrives by email",
        "Can I pick a new username",
        "Two factor login keeps failing",
        "Password reset link expired",
        "Someone else signed into my account",
        "Locked out after too many login attempts",
        "Merge my two accounts into one",
        "Please add a dark mode",
        "Would love an Apple Watch app",
        "Wish there was a landscape layout",
        "Feature request: export to CSV",
        "Please add widgets for the lock screen",
        "Would be great to share lists with family",
        "Request: keyboard shortcuts on iPad",
        "I wish the timer had custom sounds",
        "Add an option to hide completed tasks",
        "Would love Siri support for quick add",
    ]

    /// Day-two traffic: new tickets neither student has trained on, so the before/after comparison is
    /// out of sample.
    static let dayTwo: [String] = [
        "Charged after my free trial ended",
        "Refund for an accidental purchase",
        "Invoice is missing my company name",
        "Why is the price higher in my country",
        "My card was charged three times",
        "App crashes when I tap share",
        "Export fails with an error code",
        "Calendar view is blank after sync",
        "Keyboard freezes in the notes editor",
        "Dark mode is broken on the settings page",
        "I cannot log into my account",
        "Password change keeps failing",
        "Email verification never arrives",
        "My username is already taken",
        "Account locked after password reset",
        "Would love a widget for the home screen",
        "Please add folders for notes",
        "Wish I could sync with Google Calendar",
        "Feature request: pin favourite lists",
        "Add support for Apple Pencil",
    ]
}
