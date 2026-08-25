#!/usr/bin/env node
"use strict";

const { spawnSync } = require("node:child_process");
const path = require("node:path");
const readline = require("node:readline/promises");
const process = require("node:process");
const publicationDefaults = require("./release-publication");

function commandError(command, result) {
    const detail = String(result.stderr || "").trim();
    const suffix = detail ? `: ${detail}` : "";
    return new Error(`${command} failed with exit code ${result.status}${suffix}`);
}

function runProcess(command, args, options = {}) {
    const passthrough = Boolean(options.passthrough);
    const result = spawnSync(command, args, {
        cwd: options.cwd,
        env: options.env,
        encoding: "utf8",
        stdio: passthrough ? "inherit" : ["ignore", "pipe", "pipe"],
    });
    if (result.error) {
        throw new Error(`${command} could not start: ${result.error.message}`);
    }
    if (result.status !== 0) {
        throw commandError(command, result);
    }
    return passthrough ? "" : String(result.stdout || "");
}

function apiError(status, message) {
    const error = new Error(`GitHub API ${status}: ${message || "request failed"}`);
    error.status = status;
    return error;
}

function redacted(value, secrets) {
    let output = String(value || "");
    for (const secret of secrets) {
        if (secret) {
            output = output.split(secret).join("[REDACTED]");
        }
    }
    return output;
}

function createGitHubRESTClient({ token, fetchImpl = globalThis.fetch, apiBase = "https://api.github.com" }) {
    if (!token || typeof token !== "string") {
        throw new Error("GitHub authentication did not return a token.");
    }
    if (typeof fetchImpl !== "function") {
        throw new Error("This release command requires a Node runtime with fetch support.");
    }

    async function request(method, endpoint, body) {
        let response;
        try {
            response = await fetchImpl(`${apiBase}${endpoint}`, {
                method,
                headers: {
                    Accept: "application/vnd.github+json",
                    Authorization: `Bearer ${token}`,
                    "User-Agent": "AFMKit-local-release",
                    "X-GitHub-Api-Version": "2022-11-28",
                    ...(body === undefined ? {} : { "Content-Type": "application/json" }),
                },
                body: body === undefined ? undefined : JSON.stringify(body),
            });
        } catch (error) {
            throw new Error(`GitHub API request failed: ${redacted(error.message, [token])}`);
        }

        const text = await response.text();
        let data = {};
        if (text) {
            try {
                data = JSON.parse(text);
            } catch {
                data = {};
            }
        }
        if (!response.ok) {
            throw apiError(response.status, redacted(data.message, [token]));
        }
        return { data };
    }

    function repositoryPath(owner, repo, suffix) {
        return `/repos/${encodeURIComponent(owner)}/${encodeURIComponent(repo)}${suffix}`;
    }

    return {
        rest: {
            git: {
                getRef({ owner, repo, ref }) {
                    return request(
                        "GET",
                        repositoryPath(owner, repo, `/git/ref/${encodeURIComponent(ref)}`)
                    );
                },
                getTag({ owner, repo, tag_sha: tagSha }) {
                    return request(
                        "GET",
                        repositoryPath(owner, repo, `/git/tags/${encodeURIComponent(tagSha)}`)
                    );
                },
                createRef({ owner, repo, ref, sha }) {
                    return request("POST", repositoryPath(owner, repo, "/git/refs"), {
                        ref,
                        sha,
                    });
                },
                createTag({ owner, repo, ...tag }) {
                    return request("POST", repositoryPath(owner, repo, "/git/tags"), tag);
                },
            },
            repos: {
                getReleaseByTag({ owner, repo, tag }) {
                    return request(
                        "GET",
                        repositoryPath(owner, repo, `/releases/tags/${encodeURIComponent(tag)}`)
                    );
                },
                createRelease({ owner, repo, ...release }) {
                    return request("POST", repositoryPath(owner, repo, "/releases"), release);
                },
                getLatestRelease({ owner, repo }) {
                    return request("GET", repositoryPath(owner, repo, "/releases/latest"));
                },
            },
        },
    };
}

function parseRepository(value) {
    const match = /^([^/\s]+)\/([^/\s]+)$/.exec(value || "");
    if (!match) {
        throw new Error(`Invalid GitHub repository: ${value}`);
    }
    return { owner: match[1], repo: match[2] };
}

function resolveRepository({ root, repository, runCommand }) {
    const args = ["repo", "view"];
    if (repository) {
        args.push(repository);
    }
    args.push("--json", "nameWithOwner,defaultBranchRef");
    const metadata = JSON.parse(runCommand("gh", args, { cwd: root }));
    const parsed = parseRepository(metadata.nameWithOwner);
    const defaultBranch = metadata.defaultBranchRef?.name;
    if (!defaultBranch) {
        throw new Error("GitHub repository metadata has no default branch.");
    }
    return { ...parsed, nameWithOwner: metadata.nameWithOwner, defaultBranch };
}

function cleanValidationEnvironment(environment) {
    const clean = { ...environment };
    delete clean.GH_TOKEN;
    delete clean.GITHUB_TOKEN;
    return clean;
}

function assertCleanCheckout({ root, expectedSha, runCommand }) {
    const qualifiedSha = runCommand(
        "git",
        ["-C", root, "rev-parse", "HEAD"]
    ).trim();
    if (!/^[0-9a-f]{40}$/.test(qualifiedSha)) {
        throw new Error(`Local HEAD is not a full commit SHA: ${qualifiedSha}`);
    }
    if (expectedSha && qualifiedSha !== expectedSha) {
        throw new Error(`Local HEAD moved from ${expectedSha} to ${qualifiedSha}.`);
    }

    const status = runCommand(
        "git",
        ["-C", root, "status", "--porcelain=v1", "--untracked-files=all"]
    ).trim();
    if (status) {
        throw new Error("Local release requires a clean worktree.");
    }

    const submoduleStatus = runCommand(
        "git",
        ["-C", root, "submodule", "status", "--recursive"]
    );
    const invalidSubmodule = submoduleStatus.split("\n").find((line) => /^[+-U]/.test(line));
    if (invalidSubmodule) {
        throw new Error("Local release requires initialized submodules at committed revisions.");
    }
    try {
        runCommand("git", [
            "-C",
            root,
            "submodule",
            "foreach",
            "--recursive",
            "--quiet",
            'test -z "$(git status --porcelain=v1 --untracked-files=all)"',
        ]);
    } catch {
        throw new Error("Local release requires clean submodule worktrees.");
    }
    return qualifiedSha;
}

async function defaultConfirmation({ repository, tagName, qualifiedSha }) {
    if (!process.stdin.isTTY || !process.stdout.isTTY) {
        throw new Error("Interactive confirmation requires a terminal; pass --yes explicitly.");
    }
    const interface_ = readline.createInterface({ input: process.stdin, output: process.stdout });
    try {
        const expected = `publish ${tagName}`;
        const answer = await interface_.question(
            `Publish ${repository}@${qualifiedSha}? Type '${expected}': `
        );
        if (answer !== expected) {
            throw new Error("Release confirmation did not match; nothing was published.");
        }
    } finally {
        interface_.close();
    }
}

function verifyGhReleaseRecord(record, tagName, prerelease) {
    if (record.tagName !== tagName || record.isDraft !== false || record.isPrerelease !== prerelease) {
        throw new Error(`gh release verification failed for ${tagName}.`);
    }
    if (!record.url) {
        throw new Error(`gh release verification returned no URL for ${tagName}.`);
    }
}

async function runLocalRelease(options) {
    const root = options.root;
    const tagName = options.tagName;
    const runCommand = options.runCommand || runProcess;
    const publication = options.publication || publicationDefaults;
    const log = options.log || ((message) => process.stdout.write(`${message}\n`));

    publication.releaseSemVer(tagName);
    runCommand(path.join(root, "Scripts/validate-release-tag.sh"), [tagName], { cwd: root });
    const qualifiedSha = assertCleanCheckout({ root, runCommand });
    const repository = options.repositoryInfo || resolveRepository({
        root,
        repository: options.repository,
        runCommand,
    });

    let github = options.github;
    if (!github) {
        const token = runCommand(
            "gh",
            ["auth", "token", "--hostname", "github.com"],
            { cwd: root }
        ).trim();
        github = createGitHubRESTClient({ token, fetchImpl: options.fetchImpl });
    }

    const publicationArguments = {
        github,
        owner: repository.owner,
        repo: repository.repo,
        tagName,
        qualifiedSha,
        defaultBranch: repository.defaultBranch,
    };

    // Phase 1: all checks before this point and the full qualification below are read-only.
    await publication.assertReleaseCandidate(publicationArguments);
    log(`Release candidate: ${repository.nameWithOwner} ${tagName} ${qualifiedSha}`);
    if (!options.yes) {
        await (options.confirm || defaultConfirmation)({
            repository: repository.nameWithOwner,
            tagName,
            qualifiedSha,
        });
    }

    // Phase 2: no remote mutation is allowed before this exact validator succeeds.
    const validationEnvironment = cleanValidationEnvironment(options.environment || process.env);
    validationEnvironment.AFMKIT_DOWNSTREAM_SHA = qualifiedSha;
    validationEnvironment.AFMKIT_DOWNSTREAM_TAG = tagName;
    runCommand(path.join(root, "Scripts/validate-release.sh"), [], {
        cwd: root,
        env: validationEnvironment,
        passthrough: true,
    });
    assertCleanCheckout({ root, expectedSha: qualifiedSha, runCommand });

    // Phase 3: the immutable intent is the first remote mutation and closes the branch race.
    const intent = await publication.ensurePublicationIntent(publicationArguments);
    const result = await publication.publishRelease(publicationArguments);

    // Phase 4: independently re-read the tag and release after all recovery/create paths.
    const release = await publication.validatePublishedRelease(publicationArguments);
    const releaseView = JSON.parse(runCommand("gh", [
        "release",
        "view",
        tagName,
        "--repo",
        repository.nameWithOwner,
        "--json",
        "tagName,isDraft,isPrerelease,url",
    ], { cwd: root }));
    verifyGhReleaseRecord(
        releaseView,
        tagName,
        publication.releaseSemVer(tagName).prerelease
    );
    log(`Published ${tagName} at ${qualifiedSha}: ${release.html_url || releaseView.url}`);
    return { intent, publication: result, release, qualifiedSha, repository };
}

function parseArguments(arguments_) {
    let yes = false;
    let repository = null;
    let tagName = null;
    for (let index = 0; index < arguments_.length; index += 1) {
        const argument = arguments_[index];
        if (argument === "--yes") {
            yes = true;
        } else if (argument === "--repo") {
            index += 1;
            repository = arguments_[index];
            if (!repository) {
                throw new Error("--repo requires OWNER/REPO.");
            }
        } else if (!tagName) {
            tagName = argument;
        } else {
            throw new Error(`Unexpected argument: ${argument}`);
        }
    }
    if (!tagName) {
        throw new Error("Usage: release-local.js [--yes] [--repo OWNER/REPO] vMAJOR.MINOR.PATCH[-PRERELEASE]");
    }
    if (repository) {
        parseRepository(repository);
    }
    return { tagName, yes, repository };
}

async function main() {
    const arguments_ = parseArguments(process.argv.slice(2));
    const root = path.resolve(__dirname, "..");
    await runLocalRelease({ root, environment: process.env, ...arguments_ });
}

if (require.main === module) {
    main().catch((error) => {
        process.stderr.write(`${error.message}\n`);
        process.exitCode = 1;
    });
}

module.exports = {
    assertCleanCheckout,
    cleanValidationEnvironment,
    createGitHubRESTClient,
    parseArguments,
    redacted,
    runLocalRelease,
    verifyGhReleaseRecord,
};
