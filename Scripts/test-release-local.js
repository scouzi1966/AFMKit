#!/usr/bin/env node
"use strict";

const assert = require("node:assert/strict");
const publicationDefaults = require("./release-publication");
const {
    cleanValidationEnvironment,
    createGitHubRESTClient,
    parseArguments,
    runLocalRelease,
} = require("./release-local");

const ROOT = "/fixture/AFMKit";
const SHA = "1".repeat(40);
const REPOSITORY = {
    owner: "owner",
    repo: "AFMKit",
    nameWithOwner: "owner/AFMKit",
    defaultBranch: "main",
};

function commandName(command, args) {
    return `${command} ${args.join(" ")}`;
}

function fakeCommandRunner(options = {}) {
    const calls = [];
    let headReads = 0;
    const runCommand = (command, args, commandOptions = {}) => {
        const name = commandName(command, args);
        calls.push({ name, options: commandOptions });
        if (command === "git" && args.includes("rev-parse")) {
            headReads += 1;
            return `${options.headAfterValidation && headReads > 1 ? "2".repeat(40) : SHA}\n`;
        }
        if (command === "git" && args.includes("status") && !args.includes("submodule")) {
            return options.dirty ? " M Package.swift\n" : "";
        }
        if (command === "git" && args.includes("submodule")) {
            if (args.includes("foreach") && options.dirtySubmodule) {
                throw new Error("dirty submodule");
            }
            return options.uninitializedSubmodule ? `-${SHA} vendor/ds4\n` : ` ${SHA} vendor/ds4\n`;
        }
        if (command.endsWith("Scripts/validate-release.sh")) {
            if (options.validationFailure) {
                throw new Error("qualification failed");
            }
            return "";
        }
        if (command.endsWith("Scripts/validate-release-tag.sh")) {
            return "";
        }
        if (command === "gh" && args[0] === "release") {
            return JSON.stringify({
                tagName: "v1.2.3",
                isDraft: false,
                isPrerelease: false,
                url: "https://github.com/owner/AFMKit/releases/tag/v1.2.3",
            });
        }
        throw new Error(`Unexpected command: ${name}`);
    };
    return { calls, runCommand };
}

function fakePublication(options = {}) {
    const calls = [];
    const publication = {
        releaseSemVer: publicationDefaults.releaseSemVer,
        async assertReleaseCandidate() {
            calls.push("candidate");
            if (options.stale) {
                throw new Error("Default branch main moved; refusing stale candidate.");
            }
            return options.recovery ? { recoverablePublication: true } : {};
        },
        async ensurePublicationIntent() {
            calls.push("intent");
            return options.recovery ? { created: false } : { created: true };
        },
        async publishRelease() {
            calls.push("publish");
            return { tag: { created: !options.recovery }, release: { created: !options.recovery } };
        },
        async validatePublishedRelease() {
            calls.push("verify");
            return {
                tag_name: "v1.2.3",
                draft: false,
                prerelease: false,
                html_url: "https://github.com/owner/AFMKit/releases/tag/v1.2.3",
            };
        },
    };
    return { calls, publication };
}

async function runFixture({ commandOptions, publicationOptions } = {}) {
    const command = fakeCommandRunner(commandOptions);
    const remote = fakePublication(publicationOptions);
    const result = await runLocalRelease({
        root: ROOT,
        tagName: "v1.2.3",
        yes: true,
        environment: { GH_TOKEN: "secret", PATH: "/usr/bin" },
        repositoryInfo: REPOSITORY,
        github: {},
        runCommand: command.runCommand,
        publication: remote.publication,
        log() {},
    });
    return { command, remote, result };
}

async function run() {
    let fixture = await runFixture();
    assert.deepEqual(fixture.remote.calls, ["candidate", "intent", "publish", "verify"]);
    const validationIndex = fixture.command.calls.findIndex(({ name }) =>
        name.startsWith(`${ROOT}/Scripts/validate-release.sh `)
    );
    const releaseViewIndex = fixture.command.calls.findIndex(({ name }) =>
        name.startsWith("gh release view")
    );
    assert.ok(validationIndex > 0);
    assert.ok(releaseViewIndex > validationIndex);
    const validationEnvironment = fixture.command.calls[validationIndex].options.env;
    assert.equal(validationEnvironment.GH_TOKEN, undefined);
    assert.equal(validationEnvironment.GITHUB_TOKEN, undefined);
    assert.equal(validationEnvironment.AFMKIT_DOWNSTREAM_SHA, SHA);
    assert.equal(validationEnvironment.AFMKIT_DOWNSTREAM_TAG, "v1.2.3");

    const command = fakeCommandRunner({ validationFailure: true });
    const remote = fakePublication();
    await assert.rejects(
        () => runLocalRelease({
            root: ROOT,
            tagName: "v1.2.3",
            yes: true,
            repositoryInfo: REPOSITORY,
            github: {},
            runCommand: command.runCommand,
            publication: remote.publication,
            log() {},
        }),
        /qualification failed/
    );
    assert.deepEqual(remote.calls, ["candidate"]);

    for (const dirtyState of [
        { dirty: true, expected: /clean worktree/ },
        { dirtySubmodule: true, expected: /clean submodule/ },
        { uninitializedSubmodule: true, expected: /initialized submodules/ },
    ]) {
        const dirtyCommand = fakeCommandRunner(dirtyState);
        const untouchedRemote = fakePublication();
        await assert.rejects(
            () => runLocalRelease({
                root: ROOT,
                tagName: "v1.2.3",
                yes: true,
                repositoryInfo: REPOSITORY,
                github: {},
                runCommand: dirtyCommand.runCommand,
                publication: untouchedRemote.publication,
                log() {},
            }),
            dirtyState.expected
        );
        assert.deepEqual(untouchedRemote.calls, []);
    }

    const staleCommand = fakeCommandRunner();
    const staleRemote = fakePublication({ stale: true });
    await assert.rejects(
        () => runLocalRelease({
            root: ROOT,
            tagName: "v1.2.3",
            yes: true,
            repositoryInfo: REPOSITORY,
            github: {},
            runCommand: staleCommand.runCommand,
            publication: staleRemote.publication,
            log() {},
        }),
        /stale candidate/
    );
    assert.deepEqual(staleRemote.calls, ["candidate"]);
    assert.equal(staleCommand.calls.some(({ name }) => name.includes("validate-release.sh")), false);

    fixture = await runFixture({ publicationOptions: { recovery: true } });
    assert.deepEqual(fixture.remote.calls, ["candidate", "intent", "publish", "verify"]);
    assert.equal(fixture.result.intent.created, false);

    const movedCommand = fakeCommandRunner({ headAfterValidation: true });
    const movedRemote = fakePublication();
    await assert.rejects(
        () => runLocalRelease({
            root: ROOT,
            tagName: "v1.2.3",
            yes: true,
            repositoryInfo: REPOSITORY,
            github: {},
            runCommand: movedCommand.runCommand,
            publication: movedRemote.publication,
            log() {},
        }),
        /Local HEAD moved/
    );
    assert.deepEqual(movedRemote.calls, ["candidate"]);

    assert.deepEqual(parseArguments(["--yes", "--repo", "owner/AFMKit", "v1.2.3"]), {
        tagName: "v1.2.3",
        yes: true,
        repository: "owner/AFMKit",
    });
    assert.throws(() => parseArguments(["--repo", "invalid", "v1.2.3"]), /Invalid/);

    const cleanEnvironment = cleanValidationEnvironment({
        GH_TOKEN: "one",
        GITHUB_TOKEN: "two",
        PATH: "/usr/bin",
    });
    assert.deepEqual(cleanEnvironment, { PATH: "/usr/bin" });

    const token = "ghp_fixture_secret";
    let authorization = null;
    const github = createGitHubRESTClient({
        token,
        fetchImpl: async (_url, options) => {
            authorization = options.headers.Authorization;
            return {
                ok: false,
                status: 500,
                async text() {
                    return JSON.stringify({ message: `fixture failure ${token}` });
                },
            };
        },
    });
    let redactedError;
    try {
        await github.rest.git.getRef({ owner: "owner", repo: "AFMKit", ref: "heads/main" });
    } catch (error) {
        redactedError = error;
    }
    assert.equal(authorization, `Bearer ${token}`);
    assert.match(redactedError.message, /fixture failure/);
    assert.equal(redactedError.message.includes(token), false);

    let transportAttempts = 0;
    let retryHeaders = null;
    const retryingGitHub = createGitHubRESTClient({
        token,
        fetchImpl: async (_url, options) => {
            transportAttempts += 1;
            retryHeaders = options.headers;
            if (transportAttempts === 1) {
                throw new Error(`stale connection ${token}`);
            }
            return {
                ok: true,
                status: 200,
                async text() {
                    return JSON.stringify({ object: { type: "commit", sha: SHA } });
                },
            };
        },
    });
    const retriedRef = await retryingGitHub.rest.git.getRef({
        owner: "owner",
        repo: "AFMKit",
        ref: "heads/main",
    });
    assert.equal(transportAttempts, 2);
    assert.equal(retriedRef.data.object.sha, SHA);
    assert.equal(retryHeaders.Connection, "close");

    for (const invoke of [
        (client) => client.rest.git.createRef({
            owner: "owner",
            repo: "AFMKit",
            ref: "refs/tags/fixture",
            sha: SHA,
        }),
        (client) => client.rest.git.createTag({
            owner: "owner",
            repo: "AFMKit",
            tag: "v1.2.3",
            message: "AFMKit v1.2.3",
            object: SHA,
            type: "commit",
        }),
        (client) => client.rest.repos.createRelease({
            owner: "owner",
            repo: "AFMKit",
            tag_name: "v1.2.3",
            draft: false,
            prerelease: false,
        }),
    ]) {
        const requests = [];
        const retryingWriteClient = createGitHubRESTClient({
            token,
            fetchImpl: async (url, options) => {
                requests.push({ url, options });
                if (requests.length === 1) {
                    throw new Error("response lost after write may have completed");
                }
                return {
                    ok: true,
                    status: 201,
                    async text() {
                        return JSON.stringify({ sha: SHA, id: 1 });
                    },
                };
            },
        });
        await invoke(retryingWriteClient);
        assert.equal(requests.length, 2);
        assert.equal(requests[0].options.method, "POST");
        assert.equal(requests[1].options.method, "POST");
        assert.equal(requests[0].options.body, requests[1].options.body);
        assert.equal(requests[0].options.headers.Connection, "close");
    }

    let failedTransportAttempts = 0;
    const failingGitHub = createGitHubRESTClient({
        token,
        fetchImpl: async () => {
            failedTransportAttempts += 1;
            throw new Error(`transport failure ${token}`);
        },
    });
    await assert.rejects(
        () => failingGitHub.rest.git.createRef({
            owner: "owner",
            repo: "AFMKit",
            ref: "refs/tags/fixture",
            sha: SHA,
        }),
        (error) => {
            assert.match(error.message, /failed after retry/);
            assert.equal(error.message.includes(token), false);
            return true;
        }
    );
    assert.equal(failedTransportAttempts, 2);

    process.stdout.write("Local release ordering, recovery, and credential-safety checks passed.\n");
}

run().catch((error) => {
    process.stderr.write(`${error.stack || error.message}\n`);
    process.exitCode = 1;
});
