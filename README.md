# HybridInference

**Train the small model, keep Claude as the escalation path.**

A Swift package for the routing, budget and feedback-loop layer of an iOS AI feature. A small on-device
*student* model answers most requests. When it is unsure, a large *teacher* model (Claude) answers instead,
within a spend budget and behind a circuit breaker. Every answer the teacher gives becomes training data. The
next student is retrained on that data, calibrated, and has to pass a promotion gate before it ships.

[![CI](https://github.com/rajatslakhina/hybrid-inference-router-kit/actions/workflows/ci.yml/badge.svg)](https://github.com/rajatslakhina/hybrid-inference-router-kit/actions/workflows/ci.yml)
![Swift 6](https://img.shields.io/badge/Swift-6-orange) ![Platforms](https://img.shields.io/badge/platforms-iOS%2017%20%7C%20macOS%2014-blue) ![License: MIT](https://img.shields.io/badge/license-MIT-green)

Demo app: (added after the companion repo is pushed — see below)

---

## Why this matters

If an AI feature calls a frontier model on every request, every request pays for it four ways: latency,
money, availability (the provider's outage becomes yours), and vendor risk. Vendor risk covers price changes,
changes to usage limits or guardrails, and model retirement. Natasha Murashev's
[*Why You Should Train Your Own App-specific Models*](https://www.natashatherobot.com/p/train-app-specific-models)
makes the product case for a small app-specific model. This package is the engineering around that case, and
it is the part teams usually get wrong:

- **Routing you can defend.** The confidence threshold is calibrated per model version on a held-out eval set
  against a precision target. It is never a magic `0.8` copied from a blog post.
- **A budget that holds under concurrency.** The spend cap is reserved *before* the `await`, so 40 parallel
  escalations against a 10-call budget make exactly 10 calls. A test checks this.
- **Failure that degrades, not breaks.** Outage, rate limiting, a timeout, a retired model or a price hike
  each leave the user with the student's answer and a typed reason. None of them is an error screen.
- **A closed data loop with a gate.** Teacher labels are harvested, de-duplicated and kept away from the eval
  set (a leak guard), then retrained, calibrated and gated per label. A teacher that drifted is caught before
  its labels ship.

## The system

```
                    ┌──────────────── on device ────────────────┐
request ──▶ Student.predict ──▶ EscalationPolicy ──── local ──▶ answer (+ maybe a shadow sample)
                                      │ escalate (low confidence · narrow margin · out of distribution)
                                      ▼
                     EscalationBudget.reserve ── no ──▶ student answer, reason: budgetExhausted
                                      ▼
                     CircuitBreaker.acquire   ── no ──▶ student answer, reason: circuitOpen / teacherRetired
                                      ▼
                     Teacher (Claude) under a timeout     ── fails ──▶ student answer, typed TeacherError
                                      ▼
                     allowed-label check ── off-list ──▶ student answer, never trained on
                                      ▼
                     answer + harvest ─▶ TrainingBuffer (bounded, de-duplicated)

  offline loop:  TrainingSetBuilder (seed + harvest − eval leaks)
                 ─▶ NaiveBayesStudent(training:)  ─▶ ThresholdCalibrator  ─▶ Evaluator
                 ─▶ PromotionGate.decide(candidate, incumbent)  ─▶ router.install(model, policy)
```

Shadow traffic: a deterministic sample of *confident* answers is queued and later sent to the teacher with
`drainShadowQueue(maxCalls:)`, off the critical path. This measures how often the student is confidently wrong,
which escalations alone can never show. Shadow calls always leave `shadowHeadroomUnits` of budget free for live
traffic.

## The architecture

Two library products and no app target. The demo app lives in its own repository and depends on this one as a
version-pinned remote package.

| Type | Role |
| --- | --- |
| `StudentModel` (protocol), `NaiveBayesStudent` | On-device model seam. The bundled multinomial Naive Bayes trains in milliseconds; a Core ML classifier drops in behind the same protocol. |
| `TeacherProvider` (protocol), `ClaudeTeacher`, `ScriptedTeacher` | Teacher seam. `ClaudeTeacher` calls the Anthropic Messages API through an injectable `HTTPTransport`; `ScriptedTeacher` is an offline stand-in with switchable faults. |
| `EscalationPolicy`, `ThresholdCalibrator`, `CalibrationResult` | When to trust the student, and how that threshold is chosen per model version. |
| `HybridRouter` (actor) | Serving path: policy → budget → breaker → teacher → validation → harvest, plus shadow draining, model install, teacher replacement and repricing. |
| `EscalationBudget`, `CircuitBreaker` (internal) | Reserve/commit/refund spend accounting; a breaker with a generation counter for re-entrancy safety. |
| `EvalSet`, `Evaluator`, `EvalReport`, `PromotionGate` | Versioned, fingerprinted yardstick and the ship/no-ship decision. |
| `TrainingSetBuilder`, `TrainingBuffer` | Harvest merge with a leak guard; a bounded, de-duplicating buffer. |
| `HybridInferenceUI`: `HybridInferenceLab`, `HybridInferenceLabView`, `LabScenario` | `@Observable` view model and SwiftUI screen that run the whole loop. The app supplies the scenario (data, prices, thresholds). |

## Design decisions and trade-offs

1. **Reserve before `await`, settle after.** `HybridRouter` is an actor, but `route` suspends while the teacher
   answers and other requests run in between. A `spent < cap` check followed by an `await` and then a spend is
   the classic re-entrancy bug. The router reserves first, then commits (billable) or refunds (not billable)
   afterwards. *Rejected:* a per-request token bucket. It counts calls, not cost, so it cannot follow a price
   change.
2. **The breaker ignores stale results.** Every state change bumps a generation, and each permit carries the
   generation it was issued in. A slow success that started before an outage cannot close a breaker that is now
   half-open; the probe alone decides. A "model retired" error from a teacher that has since been replaced
   cannot retire its successor. *Rejected:* a timestamp check, which is ambiguous at equal times and with an
   injected clock.
3. **Thresholds are calibrated, never configured.** `ThresholdCalibrator` picks the lowest confidence whose
   locally answered slice of the eval set meets a precision target, under the same margin and
   out-of-distribution floors the policy will ship with, so the coverage it reports is the coverage you get.
   A non-finite probability (NaN or infinity) is never trusted. It never cuts inside a tie, because the
   policy answers locally at `>=` and a tied wrong answer would be hidden. Each retrained model gets its own
   threshold, because a new model's probabilities mean something different.
4. **The gate checks every label, not the average.** A candidate is rejected if any label's recall falls by
   more than `maxRecallDrop`, even when overall accuracy rises. It is also rejected if accuracy falls, if local
   precision is under the floor, if it was scored on a different eval set (fingerprint mismatch), or if it is
   no better than the incumbent.
5. **Teacher output is untrusted input.** The router accepts only labels the current student knows. An off-list
   label (a garbled answer or a prompt-injection echo) is never served and never trained on. It is still billed,
   because the call happened.
6. **Failure accounting is explicit.** `TeacherError.isBillable` and `.countsAsOutage` decide, per error,
   whether the budget is charged and whether the breaker counts a failure. A refusal costs money but proves the
   service is up. A cancelled caller is refunded and does not count as an outage, so users leaving a screen
   cannot trip the breaker.
7. **Shadow sampling is deterministic.** Sampling uses a stable FNV-1a hash of the normalised text, not
   `Hasher`, which is seeded per process. The same input is always in or always out of the sample, so asking
   the same question again cannot be used to farm teacher calls.
8. **Naive Bayes ships, Core ML plugs in.** The package is about the system, so the bundled student is
   deliberately simple and dependency-free. *Rejected:* bundling Create ML training on device, which is not
   available on iOS and would hide the routing design behind a framework.

## Failure modes

| What happens | What the user gets | Budget | Breaker |
| --- | --- | --- | --- |
| Provider outage / 5xx / network error | student answer, `teacherFailed` | refunded | counts a failure |
| HTTP 429 | student answer, `teacherFailed` | refunded | counts a failure |
| HTTP 401 / 403 / 407 (credentials) | student answer, `teacherFailed` (`.rejectedRequest`) | refunded | counts a failure |
| Other 4xx (400, 413, 422: usually one bad input) | student answer, `teacherFailed` (`.rejectedRequest`) | refunded | healthy |
| Teacher exceeds `teacherTimeout` | student answer, `teacherFailed` (`.timedOut`) | charged (conservative) | counts a failure |
| Model retired (HTTP 404) | student answer, `teacherRetired`, until `replaceTeacher` | refunded | retired |
| Refusal | student answer, `teacherRefused` | charged | healthy |
| Off-list label | student answer, `offListLabel`; not harvested | charged | healthy |
| Budget spent for the window | student answer, `budgetExhausted`; teacher not called | — | untouched |
| Breaker open | student answer, `circuitOpen`; teacher not called | refunded | — |
| Caller cancelled (the calling task was cancelled; URLSession's `URLError(.cancelled)` counts only then) | student answer, `cancelled` | refunded | untouched |
| Blank or punctuation-only input | escalates as `.emptyInput` and is billed, but never harvested | charged | as for the call |
| Price raised (`reprice`) | fewer escalations fit the same cap | new price applies to new reservations | — |
| Teacher labels drift | a retrain on them is rejected by `PromotionGate` | — | — |

## Usage

```swift
import HybridInference

let eval = try EvalSet(version: "tickets-eval-1", examples: heldOut)
let v1 = try NaiveBayesStudent(training: seed, version: "student-v1")
let policy = ThresholdCalibrator.calibrate(model: v1, evalSet: eval, targetPrecision: 0.9).policy()

let teacher = ClaudeTeacher(apiKey: proxyCredential, model: "<claude model id>", endpoint: yourProxyURL)
let router = HybridRouter(
    student: v1, teacher: teacher,
    configuration: RouterConfiguration(policy: policy,
                                       budget: BudgetConfiguration(capacityUnits: 500, windowSeconds: 3600),
                                       pricePerCallUnits: 1, teacherTimeout: .seconds(4))
)

let answer = await router.route("I was charged twice")   // .student, .teacher(...) or .studentFallback(reason)

// Later, off the critical path:
let training = TrainingSetBuilder.build(seed: seed, harvested: await router.harvestedExamples(), excluding: eval)
let v2 = try NaiveBayesStudent(training: training.examples, version: "student-v2")
let calibration = ThresholdCalibrator.calibrate(model: v2, evalSet: eval, targetPrecision: 0.9)
let candidate = Evaluator.evaluate(v2, on: eval, policy: calibration.policy())
let incumbent = Evaluator.evaluate(v1, on: eval, policy: policy)
if PromotionGate().decide(candidate: candidate, incumbent: incumbent).isPromotion {
    await router.install(v2, policy: calibration.policy())
}
```

Add it with Swift Package Manager:

```swift
.package(url: "https://github.com/rajatslakhina/hybrid-inference-router-kit.git", from: "1.2.0")
```

## Verification

- `swift build -Xswiftc -warnings-as-errors` from a clean `.build`: 0 warnings, 0 errors.
- `swift test`: 103 XCTest tests, 0 failures. They include hand-computed Laplace probabilities, a golden
  FNV-1a vector and a golden shadow-sample bucket, calibration tie handling and floors, every breaker
  transition, budget window roll-over and saturation, 40 overlapping escalations against a 10-call budget (the
  test asserts the calls really overlapped), cancellation through `ClaudeTeacher` with a URLSession-style
  transport, `URLError(.cancelled)` without a cancelled task treated as a failure, API-key redaction, input
  truncation, timeout, teacher replacement during an in-flight call, a stubbed-transport suite for
  `ClaudeTeacher`, and the end-to-end loop measured on day-two tickets that neither student trained on.
  `LabStageTests` drives `HybridInferenceLab` with the demo app's exact scenario and asserts the numbers shown
  in its four screenshots, plus the retired-teacher migration and the 4× price rise. The demo's CI checks that
  its scenario still matches this copy.
- `python3 Scripts/mutation_check.py`: 13 mutants. Each one removes or reorders one guard: in-flight
  reservations, tie handling, stale breaker results ×2, retirement generation, the off-list label check,
  cancellation accounting, the router recognising its caller's cancellation, the eval leak guard, per-label
  recall in the gate, budget-before-breaker order, the shadow drain taking the half-open probe, and the
  non-finite confidence check. A mutant counts as
  killed only if its named test ran and failed.
- CI ([Actions](https://github.com/rajatslakhina/hybrid-inference-router-kit/actions)) runs three `macos-15`
  jobs on pushes to `main`, on `v*` tags and on pull requests: build and test with warnings as errors, an iOS
  Simulator build of both library targets, and the mutation check.
- The demo app was run on an iOS Simulator; see the demo repository for what was run and the screenshots.

## Honest limits

- **Toy data.** The tests and the demo use a 4-label support-ticket set: 8 seed examples, a 12-example eval set,
  and two disjoint 40- and 20-ticket days. With 3 eval examples per label, one example moves a label's recall by
  33 points, so the gate's numbers here show the mechanism, not production statistics.
- **The demo's eval result is partly circular.** The scripted teacher's keyword rules agree with every eval
  label, and v2 is trained on that teacher's labels, so v2's 100% eval score is partly built in. With a real
  teacher, the eval set must be labelled independently of it.
- **Naive Bayes is a stand-in.** It is over-confident and blind to word order. Production should use a Core ML
  classifier behind `StudentModel`, trained off device.
- **`ClaudeTeacher` is tested against a stubbed transport only.** No live API call is made by the tests or the
  demo. The 404 → retired and 5xx → unavailable mappings are reasonable defaults, not a contract with any provider.
  Any 404 is treated as a permanent retirement, including a wrong proxy path, until `replaceTeacher` is called.
- **The timeout relies on cancellation.** A task group waits for its children, so a teacher that ignores
  cancellation delays the fallback until it returns. `URLSession` and `Task.sleep` honour cancellation.
- **The budget is a fixed window.** Up to twice the capacity can be spent across a window boundary (the end of
  one window plus the start of the next). It lives in memory, so a relaunch starts a new window. Enforce the
  real spend limit on your server as well.
- **Nothing is persisted.** The harvest, the breaker and the budget live in memory, so the data loop closes
  only within one process. Production needs a durable, consented store for harvested examples.
- **No response cache, and blank input is billed.** Asking the same uncertain question twice escalates twice.
  Blank or punctuation-only input escalates (as `.emptyInput`) and is charged, although there is nothing to
  label. Short-circuit both before `route` if they matter to you.
- **Billing assumptions are conservative estimates.** They are not provider billing rules. Real cost should come
  from the usage the provider reports.
- **Budget units are per call, not per token.** `ClaudeTeacher` truncates input to `maxInputCharacters`
  (4,000 by default) and caps output at `maxTokens`, so the cost of one call is bounded, but a long ticket
  still costs more than a short one for the same unit.
- **Prompt injection is reduced, not removed.** The input is fenced (every `<` and `>` in it is escaped) and the
  label list is enforced, but a crafted ticket can still steer Claude towards a different *allowed* label.
  The promotion gate catches it only if the eval set covers the case.
- **Privacy is out of scope.** Escalation sends user text to a third party, and so does shadow sampling (by
  default 5% of the inputs the student answered on device, once the app calls `drainShadowQueue`). Harvesting
  stores that text for training.
  Shipping this needs consent, redaction and a retention policy, none of which this package implements.
- **The teacher is treated as ground truth.** Teacher mistakes become training data. The gate limits the damage
  only as far as the eval set reaches.

## License

MIT. See [LICENSE](LICENSE).
