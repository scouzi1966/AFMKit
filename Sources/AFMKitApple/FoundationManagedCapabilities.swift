#if canImport(FoundationModels)
import Foundation
import Security

@available(macOS 27.0, *)
/// Reads Apple managed capabilities from the signed host process.
public enum AFMFoundationManagedCapabilities {
    public static let privateCloudComputeEntitlement =
        AFMFoundationNativeProviderCapabilities.privateCloudComputeEntitlement

    /// Returns whether the current executable's code signature contains the PCC entitlement.
    public nonisolated static func currentProcessHasPrivateCloudComputeEntitlement() -> Bool {
        guard let task = SecTaskCreateFromSelf(nil),
              let value = SecTaskCopyValueForEntitlement(
                task,
                privateCloudComputeEntitlement as CFString,
                nil
              ) else {
            return false
        }
        return entitlementValueIsEnabled(value)
    }

    static func entitlementValueIsEnabled(_ value: Any?) -> Bool {
        (value as? Bool) == true
    }
}
#endif
