import HybridInference
import SwiftUI

/// The demo surface: what the student answers on device, what Claude answers, what it costs, and
/// whether the latest retrain was allowed to ship.
public struct HybridInferenceLabView: View {
    @State private var lab: HybridInferenceLab
    private let initialStage: LabStage

    public init(scenario: LabScenario, initialStage: LabStage = .baseline) {
        _lab = State(initialValue: HybridInferenceLab(scenario: scenario))
        self.initialStage = initialStage
    }

    public var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    RoutingSummaryCard(lab: lab)
                    ControlsCard(lab: lab)
                    if let record = lab.lastGate { GateCard(record: record) }
                    RecentRoutesCard(answers: lab.recent)
                    if let message = lab.errorMessage {
                        Text(message).font(.footnote).foregroundStyle(.red)
                    }
                }
                .padding(16)
            }
            .background(Color.secondary.opacity(0.08))
            .navigationTitle(lab.scenario.title)
        }
        .task { await lab.run(to: initialStage) }
    }
}

// MARK: - Summary

private struct RoutingSummaryCard: View {
    let lab: HybridInferenceLab

    var body: some View {
        Card {
            HStack(alignment: .firstTextBaseline) {
                Label(lab.snapshot?.modelVersion ?? "training…", systemImage: "cpu")
                    .font(.headline)
                Spacer()
                BreakerChip(status: lab.snapshot?.breaker)
            }
            if let snapshot = lab.snapshot, snapshot.requests > 0 {
                let onDevice = Self.share(snapshot.answeredByStudent, snapshot.requests)
                Text("\(onDevice, format: .percent.precision(.fractionLength(0))) answered on device")
                    .font(.title2.weight(.bold))
                HStack(spacing: 8) {
                    Tile(title: "On device", value: snapshot.answeredByStudent, tint: .green)
                    Tile(title: "Claude", value: snapshot.answeredByTeacher, tint: .indigo)
                    Tile(title: "Fallback", value: snapshot.fallbackCount, tint: .orange)
                }
                SpendBar(spent: snapshot.spentUnits, capacity: snapshot.capacityUnits, price: snapshot.pricePerCallUnits)
                Text(Self.footer(snapshot: snapshot, threshold: lab.activeThreshold))
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            } else {
                Text("Run traffic to see how requests are routed.")
                    .foregroundStyle(.secondary)
            }
        }
    }

    static func share(_ part: Int, _ whole: Int) -> Double {
        whole > 0 ? Double(part) / Double(whole) : 0
    }

    static func footer(snapshot: RouterSnapshot, threshold: Double?) -> String {
        let thresholdText = threshold.map { String(format: "%.2f", $0) } ?? "none (escalate all)"
        var parts = [
            "\(snapshot.requests) requests",
            "threshold \(thresholdText)",
            "\(snapshot.harvested) Claude labels harvested",
        ]
        if snapshot.shadowAgreements + snapshot.shadowDisagreements > 0 {
            parts.append("shadow \(snapshot.shadowAgreements) agree / \(snapshot.shadowDisagreements) disagree")
        }
        let reasons = FallbackReason.allCases.compactMap { reason -> String? in
            guard let count = snapshot.fallbacks[reason], count > 0 else { return nil }
            return "\(reason.rawValue) ×\(count)"
        }
        if !reasons.isEmpty { parts.append("fallbacks: " + reasons.joined(separator: ", ")) }
        parts.append("teacher \(snapshot.teacherModelID)")
        return parts.joined(separator: " · ")
    }
}

private struct Tile: View {
    let title: String
    let value: Int
    let tint: Color

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text("\(value)").font(.title3.weight(.semibold).monospacedDigit())
            Text(title).font(.caption).foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(10)
        .background(tint.opacity(0.14), in: RoundedRectangle(cornerRadius: 10))
    }
}

private struct SpendBar: View {
    let spent: Int
    let capacity: Int
    let price: Int

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            ProgressView(value: fraction)
                .tint(fraction > 0.85 ? .red : .indigo)
            Text("Claude spend \(spent) / \(capacity) units this window · \(price) per call")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    private var fraction: Double {
        guard capacity > 0 else { return 1 }
        return min(max(Double(spent) / Double(capacity), 0), 1)
    }
}

private struct BreakerChip: View {
    let status: BreakerStatus?

    var body: some View {
        Text(text)
            .font(.caption.weight(.semibold))
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(color.opacity(0.18), in: Capsule())
            .foregroundStyle(color)
    }

    private var text: String {
        switch status {
        case .none: return "starting"
        case .closed(let failures)?: return failures == 0 ? "Claude: healthy" : "Claude: \(failures) failures"
        case .open?: return "Claude: breaker open"
        case .halfOpen?: return "Claude: probing"
        case .retired?: return "Claude: model retired"
        }
    }

    private var color: Color {
        switch status {
        case .closed(let failures)?: return failures == 0 ? .green : .orange
        case .open?, .retired?: return .red
        case .halfOpen?: return .orange
        case .none: return .secondary
        }
    }
}

// MARK: - Controls

private struct ControlsCard: View {
    let lab: HybridInferenceLab

    var body: some View {
        Card {
            HStack(spacing: 8) {
                Button("Serve batch \(lab.nextBatchNumber)/\(lab.scenario.trafficBatches.count) (\(lab.nextBatch.count))") { Task { await lab.runTraffic() } }
                    .buttonStyle(.borderedProminent)
                    .accessibilityIdentifier("runTraffic")
                Button("Retrain + gate") { Task { await lab.retrain(drifted: false) } }
                    .buttonStyle(.bordered)
                    .accessibilityIdentifier("retrain")
            }
            Button("Retrain on drifted labels") { Task { await lab.retrain(drifted: true) } }
                .buttonStyle(.bordered)
                .tint(.orange)
                .accessibilityIdentifier("retrainDrifted")
            Picker(
                "Teacher",
                selection: Binding(
                    get: { lab.teacherMode },
                    set: { mode in Task { await lab.setTeacherMode(mode) } }
                )
            ) {
                ForEach(TeacherMode.allCases) { mode in Text(mode.rawValue.capitalized).tag(mode) }
            }
            .pickerStyle(.segmented)
            Toggle(
                "Provider raises price 4×",
                isOn: Binding(get: { lab.priceRaised }, set: { raised in Task { await lab.setPriceRaised(raised) } })
            )
            .font(.subheadline)
        }
        .disabled(lab.isBusy)
    }
}

// MARK: - Gate

private struct GateCard: View {
    let record: GateRecord

    var body: some View {
        Card {
            HStack {
                Label(
                    record.decision.isPromotion ? "Promoted \(record.candidate.modelVersion)"
                        : "Rejected \(record.candidate.modelVersion)",
                    systemImage: record.decision.isPromotion ? "checkmark.seal.fill" : "xmark.octagon.fill"
                )
                .font(.headline)
                .foregroundStyle(record.decision.isPromotion ? .green : .red)
                Spacer()
                if record.drifted {
                    Text("drifted labels").font(.caption).foregroundStyle(.orange)
                }
            }
            Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 4) {
                GridRow {
                    Text("Eval set").foregroundStyle(.secondary)
                    Text(record.incumbent?.modelVersion ?? "—").foregroundStyle(.secondary)
                    Text(record.candidate.modelVersion).foregroundStyle(.secondary)
                }
                row("Accuracy", record.incumbent?.accuracy, record.candidate.accuracy)
                row("On-device share", record.incumbent?.localCoverage, record.candidate.localCoverage)
                row("Local precision", record.incumbent?.localPrecision, record.candidate.localPrecision)
            }
            .font(.subheadline.monospacedDigit())
            Text(
                "Trained on \(record.trainingCount) examples · \(record.leakExcluded) dropped as eval-set leaks · "
                    + "eval set \(record.candidate.evalSetVersion), \(record.candidate.exampleCount) examples"
            )
            .font(.caption)
            .foregroundStyle(.secondary)
            if case .reject(let failures) = record.decision {
                ForEach(Array(failures.enumerated()), id: \.offset) { _, failure in
                    Label(Self.describe(failure), systemImage: "exclamationmark.triangle")
                        .font(.caption)
                        .foregroundStyle(.red)
                }
            }
        }
    }

    private func row(_ title: String, _ before: Double?, _ after: Double?) -> some View {
        GridRow {
            Text(title)
            Text(Self.percent(before))
            Text(Self.percent(after)).fontWeight(.semibold)
        }
    }

    static func percent(_ value: Double?) -> String {
        guard let value else { return "—" }
        return value.formatted(.percent.precision(.fractionLength(0)))
    }

    static func describe(_ failure: GateFailure) -> String {
        switch failure {
        case .evalSetMismatch: return "Scored on a different eval set"
        case .accuracyRegressed(let from, let to): return "Accuracy fell \(percent(from)) → \(percent(to))"
        case .recallRegressed(let label, let from, let to):
            return "Recall for \(label) fell \(percent(from)) → \(percent(to))"
        case .localPrecisionBelowFloor(let value?, let floor):
            return "Local precision \(percent(value)) is below the \(percent(floor)) floor"
        case .localPrecisionBelowFloor(nil, _):
            return "Answers nothing on device, so there is no local precision to trust"
        case .noImprovement: return "No gain in accuracy or on-device share"
        }
    }
}

// MARK: - Recent routes

private struct RecentRoutesCard: View {
    let answers: [RoutedAnswer]

    var body: some View {
        Card {
            Text("Latest requests").font(.headline)
            if answers.isEmpty {
                Text("No requests yet.").foregroundStyle(.secondary)
            }
            ForEach(answers) { answer in
                HStack(alignment: .top, spacing: 10) {
                    Image(systemName: icon(for: answer.source))
                        .foregroundStyle(tint(for: answer.source))
                        .frame(width: 20)
                    VStack(alignment: .leading, spacing: 2) {
                        Text(answer.text).font(.subheadline).lineLimit(1)
                        Text(detail(for: answer)).font(.caption).foregroundStyle(.secondary)
                    }
                    Spacer(minLength: 4)
                    Text(answer.label?.rawValue ?? "—")
                        .font(.caption.weight(.semibold))
                        .padding(.horizontal, 8)
                        .padding(.vertical, 3)
                        .background(tint(for: answer.source).opacity(0.15), in: Capsule())
                }
            }
        }
    }

    private func icon(for source: AnswerSource) -> String {
        switch source {
        case .student: return "iphone"
        case .teacher: return "sparkles"
        case .studentFallback: return "exclamationmark.shield"
        }
    }

    private func tint(for source: AnswerSource) -> Color {
        switch source {
        case .student: return .green
        case .teacher: return .indigo
        case .studentFallback: return .orange
        }
    }

    private func detail(for answer: RoutedAnswer) -> String {
        let confidence = String(format: "%.2f", answer.studentConfidence)
        switch answer.source {
        case .student: return "on device · confidence \(confidence)"
        case .teacher(let model): return "Claude (\(model)) · student was \(confidence)"
        case .studentFallback(let reason): return "fallback: \(reason.rawValue) · confidence \(confidence)"
        }
    }
}

// MARK: - Card

private struct Card<Content: View>: View {
    @ViewBuilder let content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 10) { content }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(14)
            .background(.background, in: RoundedRectangle(cornerRadius: 14))
    }
}
