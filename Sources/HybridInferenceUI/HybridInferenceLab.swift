import Foundation
import HybridInference
import Observation

/// Everything the lab needs to replay the teacher–student loop. The app owns it, so the dataset,
/// prices and thresholds are app decisions rather than library defaults.
public struct LabScenario: Sendable {
    public let title: String
    public let seed: [LabeledExample]
    public let evalSet: EvalSet
    /// Successive days of traffic, served in order and then cycled. Keep them disjoint so "after
    /// retraining" is measured on tickets the new student never trained on.
    public let trafficBatches: [[String]]
    public let teacherModelID: String
    public let teacherRules: [ScriptedTeacher.Rule]
    public let teacherFallback: ClassLabel
    public let configuration: RouterConfiguration
    public let targetPrecision: Double
    public let gate: PromotionGate
    public let shadowCallsPerRun: Int

    public init(
        title: String,
        seed: [LabeledExample],
        evalSet: EvalSet,
        trafficBatches: [[String]],
        teacherModelID: String,
        teacherRules: [ScriptedTeacher.Rule],
        teacherFallback: ClassLabel,
        configuration: RouterConfiguration,
        targetPrecision: Double,
        gate: PromotionGate,
        shadowCallsPerRun: Int = 8
    ) {
        self.title = title
        self.seed = seed
        self.evalSet = evalSet
        self.trafficBatches = trafficBatches.filter { !$0.isEmpty }
        self.teacherModelID = teacherModelID
        self.teacherRules = teacherRules
        self.teacherFallback = teacherFallback
        self.configuration = configuration
        self.targetPrecision = targetPrecision
        self.gate = gate
        self.shadowCallsPerRun = max(shadowCallsPerRun, 0)
    }
}

/// Scripted starting points, so the demo can be launched straight into each state (used for screenshots).
public enum LabStage: String, CaseIterable, Sendable {
    /// Student v1 trained on the seed set; no traffic yet.
    case fresh
    /// v1 has served day one; Claude labelled what v1 was unsure about.
    case baseline
    /// v2 was trained on Claude's day-one labels, passed the gate and served day two (unseen tickets).
    case retrained
    /// A retrain on drifted labels was attempted and the gate rejected it.
    case drifted
    /// v2 is serving day two while the teacher is down: the breaker opens and the student carries the load.
    case outage
}

public enum TeacherMode: String, CaseIterable, Sendable, Identifiable {
    case healthy, outage, retired
    public var id: String { rawValue }

    var fault: ScriptedTeacher.Fault {
        switch self {
        case .healthy: return .healthy
        case .outage: return .outage
        case .retired: return .retired
        }
    }
}

public struct GateRecord: Equatable, Sendable {
    public let candidate: EvalReport
    public let incumbent: EvalReport?
    public let decision: PromotionDecision
    public let candidateThreshold: Double?
    public let trainingCount: Int
    public let leakExcluded: Int
    public let drifted: Bool
}

/// Drives a `HybridRouter` through the full loop: serve → harvest → retrain → calibrate → gate → install.
@MainActor
@Observable
public final class HybridInferenceLab {
    public let scenario: LabScenario
    public private(set) var snapshot: RouterSnapshot?
    public private(set) var recent: [RoutedAnswer] = []
    public private(set) var incumbent: EvalReport?
    public private(set) var activeThreshold: Double?
    public private(set) var lastGate: GateRecord?
    public private(set) var isBusy = false
    public private(set) var errorMessage: String?
    public private(set) var teacherMode: TeacherMode = .healthy
    public private(set) var priceRaised = false
    public private(set) var runs = 0

    /// Rows kept for display; older rows are dropped so memory stays bounded.
    public static let recentLimit = 12

    private var router: HybridRouter?
    private var teacher: ScriptedTeacher
    private var teacherGeneration = 1
    private var studentGeneration = 0

    public init(scenario: LabScenario) {
        self.scenario = scenario
        self.teacher = ScriptedTeacher(
            modelID: scenario.teacherModelID,
            rules: scenario.teacherRules,
            fallbackLabel: scenario.teacherFallback
        )
    }

    // MARK: Actions

    /// Trains student v1 on the seed set, calibrates it, and starts the router.
    public func prepare() async {
        await perform { [self] in
            let model = try NaiveBayesStudent(training: scenario.seed, version: nextStudentVersion())
            let calibration = ThresholdCalibrator.calibrate(
                model: model, evalSet: scenario.evalSet, targetPrecision: scenario.targetPrecision
            )
            let policy = calibration.policy(
                minMargin: scenario.configuration.policy.minMargin,
                minKnownTokenFraction: scenario.configuration.policy.minKnownTokenFraction
            )
            let router = HybridRouter(student: model, teacher: teacher, configuration: scenario.configuration)
            await router.install(model, policy: policy)
            self.router = router
            incumbent = Evaluator.evaluate(model, on: scenario.evalSet, policy: policy)
            activeThreshold = calibration.threshold
            studentGeneration += 1
            recent = []
            lastGate = nil
            runs = 0
            snapshot = await router.snapshot()
        }
    }

    /// The batch the next `runTraffic()` will serve.
    public var nextBatch: [String] {
        guard !scenario.trafficBatches.isEmpty else { return [] }
        return scenario.trafficBatches[runs % scenario.trafficBatches.count]
    }

    /// Serves the next day of traffic, then spends a little budget on shadow samples.
    public func runTraffic() async {
        await perform { [self] in
            guard let router else { return }
            await router.resetCounters()
            for text in nextBatch {
                let answer = await router.route(text)
                recent.insert(answer, at: 0)
                if recent.count > Self.recentLimit { recent.removeLast(recent.count - Self.recentLimit) }
            }
            await router.drainShadowQueue(maxCalls: scenario.shadowCallsPerRun)
            runs = runs < Int.max ? runs + 1 : 0   // never wraps negative, so `runs % count` stays a valid index
            snapshot = await router.snapshot()
        }
    }

    /// Retrains on seed + Claude's harvested labels and lets the promotion gate decide.
    ///
    /// - Parameter drifted: rotate every harvested label to the next label first, simulating a teacher
    ///   whose behaviour changed underneath you (a silent model update, a prompt regression).
    public func retrain(drifted: Bool) async {
        await perform { [self] in
            guard let router else { return }
            var harvested = await router.harvestedExamples()
            if drifted { harvested = Self.rotateLabels(harvested) }
            let training = TrainingSetBuilder.build(
                seed: scenario.seed, harvested: harvested, excluding: scenario.evalSet
            )
            let candidate = try NaiveBayesStudent(training: training.examples, version: nextStudentVersion())
            let calibration = ThresholdCalibrator.calibrate(
                model: candidate, evalSet: scenario.evalSet, targetPrecision: scenario.targetPrecision
            )
            let policy = calibration.policy(
                minMargin: scenario.configuration.policy.minMargin,
                minKnownTokenFraction: scenario.configuration.policy.minKnownTokenFraction
            )
            let report = Evaluator.evaluate(candidate, on: scenario.evalSet, policy: policy)
            let decision = scenario.gate.decide(candidate: report, incumbent: incumbent)
            lastGate = GateRecord(
                candidate: report, incumbent: incumbent, decision: decision,
                candidateThreshold: calibration.threshold, trainingCount: training.examples.count,
                leakExcluded: training.excludedForLeak, drifted: drifted
            )
            studentGeneration += 1
            if decision.isPromotion {
                await router.install(candidate, policy: policy)
                incumbent = report
                activeThreshold = calibration.threshold
            }
            snapshot = await router.snapshot()
        }
    }

    public func setTeacherMode(_ mode: TeacherMode) async {
        await perform { [self] in
            guard let router, mode != teacherMode else { return }
            if teacherMode == .retired {
                // A retired model never comes back: migrate to a successor model, which resets the breaker.
                teacherGeneration += 1
                teacher = ScriptedTeacher(
                    modelID: "\(scenario.teacherModelID)-successor-\(teacherGeneration)",
                    rules: scenario.teacherRules,
                    fallbackLabel: scenario.teacherFallback
                )
                await router.replaceTeacher(teacher)
            }
            await teacher.setFault(mode.fault)
            teacherMode = mode
            snapshot = await router.snapshot()
        }
    }

    /// Simulates the provider quadrupling its price. Budget capacity stays the same.
    public func setPriceRaised(_ raised: Bool) async {
        await perform { [self] in
            guard let router else { return }
            let base = scenario.configuration.pricePerCallUnits
            let (raisedPrice, overflow) = base.multipliedReportingOverflow(by: 4)
            await router.reprice(unitsPerCall: raised ? (overflow ? Int.max : raisedPrice) : base)
            priceRaised = raised
            snapshot = await router.snapshot()
        }
    }

    /// Replays the scripted steps that lead to `stage`.
    public func run(to stage: LabStage) async {
        await prepare()
        guard stage != .fresh else { return }
        await runTraffic()
        switch stage {
        case .fresh, .baseline:
            return
        case .retrained:
            await retrain(drifted: false)
            await runTraffic()
        case .drifted:
            await retrain(drifted: true)
        case .outage:
            await retrain(drifted: false)
            await setTeacherMode(.outage)
            await runTraffic()
        }
    }

    // MARK: Helpers

    private func nextStudentVersion() -> String { "student-v\(studentGeneration + 1)" }

    private func perform(_ work: @MainActor () async throws -> Void) async {
        guard !isBusy else { return }
        isBusy = true
        defer { isBusy = false }
        do {
            errorMessage = nil
            try await work()
        } catch {
            errorMessage = String(describing: error)
        }
    }

    static func rotateLabels(_ examples: [LabeledExample]) -> [LabeledExample] {
        let labels = Array(Set(examples.map(\.label))).sorted()
        guard labels.count > 1 else { return examples }
        var next: [ClassLabel: ClassLabel] = [:]
        for (index, label) in labels.enumerated() {
            next[label] = labels[(index + 1) % labels.count]
        }
        return examples.map { LabeledExample(text: $0.text, label: next[$0.label] ?? $0.label) }
    }
}
