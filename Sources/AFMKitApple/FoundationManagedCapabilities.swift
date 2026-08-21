#if canImport(FoundationModels)
import Foundation
import Security

@available(macOS 27.0, *)
/// Reads Apple managed capabilities from the signed host process.
public enum AFMFoundationManagedCapabilities {
    private static let adHocSignatureFlag: UInt32 = 0x0002

    public static let privateCloudComputeEntitlement =
        AFMFoundationNativeProviderCapabilities.privateCloudComputeEntitlement

    /// Returns whether the current executable's code signature contains the PCC entitlement.
    public nonisolated static func currentProcessHasPrivateCloudComputeEntitlement() -> Bool {
        var code: SecCode?
        guard SecCodeCopySelf(SecCSFlags(), &code) == errSecSuccess,
              let code,
              SecCodeCheckValidity(
                code,
                SecCSFlags(
                    rawValue: kSecCSStrictValidate | kSecCSCheckAllArchitectures
                ),
                nil
              ) == errSecSuccess else {
            return false
        }

        var staticCode: SecStaticCode?
        var signingInformation: CFDictionary?
        guard SecCodeCopyStaticCode(code, SecCSFlags(), &staticCode) == errSecSuccess,
              let staticCode,
              SecCodeCopySigningInformation(
                staticCode,
                SecCSFlags(rawValue: kSecCSSigningInformation),
                &signingInformation
              ) == errSecSuccess,
              let signingInformation = signingInformation as? [CFString: Any],
              let flags = (signingInformation[kSecCodeInfoFlags] as? NSNumber)?.uint32Value,
              let task = SecTaskCreateFromSelf(nil),
              let value = SecTaskCopyValueForEntitlement(
                task,
                privateCloudComputeEntitlement as CFString,
                nil
              ) else {
            return false
        }
        return signedHostEntitlementIsEnabled(
            signatureIsValid: true,
            signatureIsAdHoc: flags & adHocSignatureFlag != 0,
            entitlementValue: value
        )
    }

    static func entitlementValueIsEnabled(_ value: Any?) -> Bool {
        (value as? Bool) == true
    }

    static func signedHostEntitlementIsEnabled(
        signatureIsValid: Bool,
        signatureIsAdHoc: Bool,
        entitlementValue: Any?
    ) -> Bool {
        signatureIsValid && !signatureIsAdHoc && entitlementValueIsEnabled(entitlementValue)
    }
}
#endif
