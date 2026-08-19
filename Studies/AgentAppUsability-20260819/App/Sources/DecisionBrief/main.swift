import DecisionBriefCore
import SwiftUI
import UniformTypeIdentifiers

private let supportedSourceTypes = [
    UTType.plainText,
    UTType(filenameExtension: "md"),
    UTType(filenameExtension: "markdown"),
].compactMap { $0 }

@main
struct DecisionBriefApp: App {
    var body: some Scene {
        WindowGroup("DecisionBrief") {
            DecisionBriefView()
        }
    }
}

struct DecisionBriefView: View {
    @StateObject private var model: DecisionBriefViewModel
    @State private var showingImporter = false

    init() {
        let service: any DecisionModelService
        if let configured = try? AFMKitDecisionModelService() {
            service = configured
        } else {
            service = UnavailableModelService()
        }
        _model = StateObject(wrappedValue: DecisionBriefViewModel(service: service))
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Text("DecisionBrief")
                    .font(.title.bold())
                Spacer()
                Text("Local analysis | \(decisionBriefModelID)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .accessibilityIdentifier("modelStatus")
            }

            HStack {
                Text("Sources")
                    .font(.headline)
                Spacer()
                Button {
                    showingImporter = true
                } label: {
                    Label("Add notes", systemImage: "plus")
                }
                .accessibilityIdentifier("addSources")
            }

            List {
                ForEach(model.sources) { source in
                    HStack {
                        Text(source.label)
                        Spacer()
                        Button {
                            model.remove(source)
                        } label: {
                            Image(systemName: "minus.circle")
                        }
                        .buttonStyle(.borderless)
                        .help("Remove source")
                        .accessibilityLabel("Remove \(source.label)")
                    }
                }
            }
            .frame(minHeight: 100)
            .accessibilityIdentifier("sourceList")

            Text("Decision objective")
                .font(.headline)
            TextField(
                "What decision should this pre-read support?",
                text: $model.objective,
                axis: .vertical
            )
            .textFieldStyle(.roundedBorder)
            .accessibilityIdentifier("objectiveField")

            HStack {
                statusView
                Spacer()
                if model.state.isRunning {
                    Button("Cancel") {
                        model.cancel()
                    }
                    .accessibilityIdentifier("cancelButton")
                }
                Button("Generate") {
                    model.generate()
                }
                .disabled(!model.canGenerate)
                .keyboardShortcut(.return)
                .accessibilityIdentifier("generateButton")
            }

            if let error = model.errorMessage {
                Text(error)
                    .foregroundStyle(.red)
                    .accessibilityIdentifier("errorMessage")
            }
            if let summary = model.completionSummary {
                Text(summary)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .accessibilityIdentifier("completionSummary")
            }
            if !model.brief.isEmpty {
                ScrollView {
                    Text(model.brief)
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding()
                }
                .background(.quaternary.opacity(0.35))
                .accessibilityIdentifier("briefOutput")
            }
            Spacer()
        }
        .padding(24)
        .frame(minWidth: 700, minHeight: 560)
        .fileImporter(
            isPresented: $showingImporter,
            allowedContentTypes: supportedSourceTypes,
            allowsMultipleSelection: true
        ) { result in
            switch result {
            case .success(let urls):
                model.add(urls: urls)
            case .failure(let error):
                let cocoaError = error as NSError
                if cocoaError.code != NSUserCancelledError {
                    model.report(error: error)
                }
            }
        }
        .onDisappear {
            model.shutdown()
        }
    }

    @ViewBuilder
    private var statusView: some View {
        switch model.state {
        case .empty:
            Text("Select notes and enter an objective")
                .foregroundStyle(.secondary)
        case .ready:
            Text("Ready")
        case .loading(let progress):
            ProgressView(value: progress) {
                Text("Loading model...")
            }
            .frame(width: 220)
        case .generating:
            ProgressView("Generating...")
        case .completed:
            Label("Completed", systemImage: "checkmark.circle")
        case .cancelled:
            Label("Cancelled | ready to retry", systemImage: "pause.circle")
        case .failed:
            Label("Failed | retry available", systemImage: "exclamationmark.triangle")
        }
    }
}

private final class UnavailableModelService: DecisionModelService, @unchecked Sendable {
    func load(progress: @escaping @Sendable (Double) -> Void) async throws {
        throw NSError(
            domain: "DecisionBrief",
            code: 1,
            userInfo: [NSLocalizedDescriptionKey: "The local model could not be configured."]
        )
    }

    func stream(prompt: String) -> AsyncThrowingStream<DecisionModelEvent, Error> {
        AsyncThrowingStream { $0.finish() }
    }

    func unload() async {}
}
