import Foundation

// MARK: - Overflow-safe arithmetic

/// Integer arithmetic that clamps instead of trapping.
///
/// Every counter and budget sum in this package goes through these helpers, so a
/// hostile or very long-lived input can saturate a number but can never crash the app.
enum Saturating {
    static func add(_ a: Int, _ b: Int) -> Int {
        let (result, overflow) = a.addingReportingOverflow(b)
        guard overflow else { return result }
        return b > 0 ? Int.max : Int.min
    }

    static func subtract(_ a: Int, _ b: Int) -> Int {
        let (result, overflow) = a.subtractingReportingOverflow(b)
        guard overflow else { return result }
        return b < 0 ? Int.max : Int.min
    }

    /// `numerator / denominator` as a `Double`, or `0` when the denominator is not positive.
    static func ratio(_ numerator: Int, _ denominator: Int) -> Double {
        guard denominator > 0 else { return 0 }
        return Double(numerator) / Double(denominator)
    }

    /// Clamps to `0...1`. `NaN` becomes `nanValue`, so callers choose the safe default.
    static func unit(_ value: Double, nanValue: Double) -> Double {
        guard !value.isNaN else { return nanValue }
        return min(max(value, 0), 1)
    }
}

// MARK: - Stable hashing

/// A hash that is identical across processes, launches and devices.
///
/// Swift's `Hasher` is randomly seeded per process, so it cannot back anything that
/// must agree tomorrow (shadow sampling, eval-set fingerprints). This is FNV-1a 64-bit.
/// It is not cryptographic and is never used for security.
public enum StableHash {
    public static func fnv1a64(_ string: String) -> UInt64 {
        var hash: UInt64 = 0xcbf2_9ce4_8422_2325
        for byte in string.utf8 {
            hash ^= UInt64(byte)
            hash = hash &* 0x0000_0100_0000_01b3
        }
        return hash
    }
}

// MARK: - Text normalisation

/// Lower-cases and splits on anything that is not a letter or a number.
///
/// The same normal form is used for model features, de-duplication and the
/// eval-set leak guard, so "Refund??" and "refund" are the same example everywhere.
public enum Tokenizer {
    public static func tokens(_ text: String) -> [String] {
        text.lowercased()
            .split(whereSeparator: { !($0.isLetter || $0.isNumber) })
            .map(String.init)
    }

    public static func normalized(_ text: String) -> String {
        tokens(text).joined(separator: " ")
    }
}

// MARK: - Time

/// The router's clock. Injected so budget windows and breaker cool-downs are testable.
public protocol TimeSource: Sendable {
    /// Seconds on a monotonic timeline. Only differences are meaningful.
    func now() -> Double
}

/// Uptime-based monotonic clock: unaffected by the user changing the wall-clock time.
public struct MonotonicTimeSource: TimeSource {
    public init() {}
    public func now() -> Double { ProcessInfo.processInfo.systemUptime }
}
