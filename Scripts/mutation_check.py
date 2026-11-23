#!/usr/bin/env python3
"""Mutation check: break each load-bearing guard on purpose and prove a named test catches it.

Each mutant is a one-line edit to the real source. The script applies it, confirms the package still
compiles (a mutant that does not compile proves nothing), runs only the test that is supposed to
catch it, and restores the file. Any mutant whose test still passes is a "survivor": that test was
not actually guarding the property, and the script exits non-zero.

Run from the package root:  python3 Scripts/mutation_check.py
"""
import pathlib
import re
import subprocess
import sys

SRC = pathlib.Path("Sources/HybridInference")

MUTANTS = [
    (
        "budget-ignores-in-flight-reservations",
        SRC / "Reliability.swift",
        "let needed = Saturating.add(Saturating.add(Saturating.add(spent, reserved), cost), max(headroom, 0))",
        "let needed = Saturating.add(Saturating.add(spent, cost), max(headroom, 0))",
        "HybridRouterTests/testConcurrentEscalationsNeverOverspendTheBudget",
    ),
    (
        "calibrator-cuts-inside-ties",
        SRC / "Policy.swift",
        "guard isLast || scored[index + 1].confidence < scored[index].confidence else { continue }",
        "_ = isLast",
        "ThresholdCalibratorTests/testNeverCutsInsideATie",
    ),
    (
        "breaker-accepts-stale-success",
        SRC / "Reliability.swift",
        "mutating func recordSuccess(_ permit: Permit) {\n        guard permit.generation == generation else { return }",
        "mutating func recordSuccess(_ permit: Permit) {",
        "CircuitBreakerTests/testStaleResultsFromAnOlderGenerationAreIgnored",
    ),
    (
        "breaker-accepts-stale-failure",
        SRC / "Reliability.swift",
        "mutating func recordFailure(_ permit: Permit, now: Double) {\n        guard permit.generation == generation else { return }",
        "mutating func recordFailure(_ permit: Permit, now: Double) {",
        "CircuitBreakerTests/testStaleResultsFromAnOlderGenerationAreIgnored",
    ),
    (
        "retirement-ignores-generation",
        SRC / "HybridRouter.swift",
        "if permit.generation == breaker.generation { breaker.retire() }",
        "breaker.retire()",
        "HybridRouterTests/testRetirementReportedByAReplacedTeacherDoesNotDisableItsSuccessor",
    ),
    (
        "router-serves-off-list-labels",
        SRC / "HybridRouter.swift",
        "guard allowed.contains(label) else { return fallback(.offListLabel) }",
        "guard allowed.contains(label) || true else { return fallback(.offListLabel) }",
        "HybridRouterTests/testOffListLabelIsNeverServedOrHarvested",
    ),
    (
        "cancellation-counts-as-outage",
        SRC / "HybridRouter.swift",
        "} catch is CancellationError {\n            return .cancelled",
        "} catch is CancellationError {\n            return .failure(.unavailable)",
        "HybridRouterTests/testCallerCancellationRefundsAndDoesNotTripTheBreaker",
    ),
    (
        "training-set-leaks-eval-examples",
        SRC / "Evaluation.swift",
        "guard !evalSet.normalizedTexts.contains(key) else {",
        "guard !evalSet.normalizedTexts.contains(key) || true else {",
        "TeacherStudentLoopTests/testEvalExamplesInTheHarvestNeverReachTraining",
    ),
    (
        "gate-ignores-per-label-recall",
        SRC / "Evaluation.swift",
        "if before - after > maxRecallDrop + Self.epsilon {",
        "if false && before - after > maxRecallDrop + Self.epsilon {",
        "TeacherStudentLoopTests/testGateRejectsAStudentTrainedOnDriftedTeacherLabels",
    ),
    (
        "breaker-checked-before-budget",
        SRC / "HybridRouter.swift",
        """        guard let reservation = budget.reserve(units: price, now: now) else { return fallback(.budgetExhausted) }
        guard let permit = breaker.acquire(now: now) else {
            budget.refund(reservation)
            return fallback(breaker.status == .retired ? .teacherRetired : .circuitOpen)
        }""",
        """        guard let permit = breaker.acquire(now: now) else {
            return fallback(breaker.status == .retired ? .teacherRetired : .circuitOpen)
        }
        guard let reservation = budget.reserve(units: price, now: now) else {
            breaker.release(permit)
            return fallback(.budgetExhausted)
        }""",
        "HybridRouterTests/testABudgetRefusalLeavesTheBreakersProbeUntouched",
    ),
    (
        "shadow-drain-takes-the-half-open-probe",
        SRC / "HybridRouter.swift",
        """            guard case .closed = breaker.state else {
                shadowQueue.pushFront(item)
                break
            }
""",
        "",
        "HybridRouterTests/testShadowDrainLeavesTheHalfOpenProbeForLiveTraffic",
    ),
    (
        "policy-trusts-non-finite-confidence",
        SRC / "Policy.swift",
        """        guard top.probability.isFinite, prediction.distribution.margin.isFinite else {
            return .lowConfidence(top.probability)
        }
""",
        "",
        "EscalationPolicyTests/testNonFinitePredictionIsNeverTrusted",
    ),
    (
        "router-ignores-its-callers-cancellation",
        SRC / "HybridRouter.swift",
        "            if Task.isCancelled { return .cancelled }\n",
        "",
        "HybridRouterTests/testRouterTreatsAnyErrorAfterItsCallerWasCancelledAsCancellation",
    ),
]


def run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True)


def main() -> int:
    killed, survivors, broken = [], [], []
    for name, path, old, new, test in MUTANTS:
        original = path.read_text()
        if original.count(old) != 1:
            broken.append(f"{name}: target text found {original.count(old)} times in {path}")
            continue
        try:
            path.write_text(original.replace(old, new))
            build = run("swift", "build", "--build-tests")
            if build.returncode != 0:
                broken.append(f"{name}: mutant does not compile\n{build.stdout[-2000:]}{build.stderr[-2000:]}")
                continue
            result = run("swift", "test", "--skip-build", "--filter", test)
            # A kill needs evidence: the filter matched at least one test and at least one test failed.
            # A typo in the filter, or a crash before any test ran, is reported as invalid, not as a kill.
            counts = re.findall(r"Executed (\d+) tests?, with (\d+) failures?", result.stdout + result.stderr)
            if not counts or int(counts[-1][0]) == 0:
                broken.append(f"{name}: filter {test!r} ran no tests")
                print("INVALID  " + name, flush=True)
                continue
            caught = int(counts[-1][1]) > 0
            (killed if caught else survivors).append(f"{name} -> {test}")
            print(("KILLED   " if caught else "SURVIVED ") + name, flush=True)
        finally:
            path.write_text(original)

    total = len(MUTANTS)
    print(f"\nMutation check: {len(killed)}/{total} killed, {len(survivors)} survived, {len(broken)} invalid")
    for line in survivors:
        print(f"  SURVIVOR: {line}")
    for line in broken:
        print(f"  INVALID:  {line}")
    return 0 if len(killed) == total else 1


if __name__ == "__main__":
    sys.exit(main())
