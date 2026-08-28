import Foundation

/// Shared destructive-path validation for checkpoint converters.
///
/// This lives below the public dispatcher so direct provider use receives the
/// same protection as application-level callers.
enum CheckpointConversionPathSafety {
    struct ValidatedPaths {
        let source: URL
        let output: URL
    }

    static func validate(source: URL, output: URL) throws -> ValidatedPaths {
        let sourceURL = source.standardizedFileURL
        let outputURL = output.standardizedFileURL
        guard sourceURL.isFileURL, outputURL.isFileURL else {
            throw PathError.nonLocal
        }

        let resolvedSource = sourceURL.resolvingSymlinksInPath()
        let resolvedOutput = outputURL.resolvingSymlinksInPath()
        guard !isFilesystemOrVolumeRoot(resolvedOutput),
              !contains(resolvedSource, resolvedOutput),
              !contains(resolvedOutput, resolvedSource)
        else {
            throw PathError.unsafe
        }
        return ValidatedPaths(source: sourceURL, output: outputURL)
    }

    static func isFilesystemOrVolumeRoot(
        _ url: URL,
        mountedVolumes: [URL]? = nil
    ) -> Bool {
        let resolvedPath = url.standardizedFileURL.resolvingSymlinksInPath().path
        if resolvedPath == "/" { return true }
        let volumes = mountedVolumes ?? FileManager.default.mountedVolumeURLs(
            includingResourceValuesForKeys: nil,
            options: []) ?? []
        return volumes.contains {
            $0.standardizedFileURL.resolvingSymlinksInPath().path == resolvedPath
        }
    }

    private static func contains(_ directory: URL, _ candidate: URL) -> Bool {
        let parent = directory.standardizedFileURL.pathComponents
        let child = candidate.standardizedFileURL.pathComponents
        guard parent.count <= child.count else { return false }
        return Array(child.prefix(parent.count)) == parent
    }

    enum PathError: Error {
        case nonLocal
        case unsafe
    }
}
