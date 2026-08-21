#!/usr/bin/env node
"use strict";

const assert = require("node:assert/strict");
const {
    assertReleaseCandidate,
    ensureReleaseTag,
    publishRelease,
    releaseSemVer,
} = require("./release-publication");

const SHA = "1".repeat(40);
const OTHER_SHA = "2".repeat(40);

function apiError(status) {
    return Object.assign(new Error(`HTTP ${status}`), { status });
}

function fakeGitHub(options = {}) {
    const state = {
        branchSha: options.branchSha || SHA,
        tagRef: options.tagRef || null,
        tagObjects: new Map(options.tagObjects || []),
        release: options.release || null,
        latestRelease: options.latestRelease || options.release || null,
        createRefRace: Boolean(options.createRefRace),
        createReleaseRace: Boolean(options.createReleaseRace),
        calls: {
            createTag: 0,
            createRef: 0,
            createRelease: 0,
            createReleaseArguments: null,
        },
    };

    const github = {
        rest: {
            git: {
                async getRef({ ref }) {
                    if (ref === "heads/main") {
                        return { data: { object: { type: "commit", sha: state.branchSha } } };
                    }
                    if (ref === "tags/v1.2.3") {
                        if (!state.tagRef) {
                            throw apiError(404);
                        }
                        return { data: { object: state.tagRef } };
                    }
                    throw apiError(404);
                },
                async getTag({ tag_sha: tagSha }) {
                    const object = state.tagObjects.get(tagSha);
                    if (!object) {
                        throw apiError(404);
                    }
                    return { data: { object } };
                },
                async createTag({ object }) {
                    state.calls.createTag += 1;
                    state.tagObjects.set("annotated-tag", { type: "commit", sha: object });
                    return { data: { sha: "annotated-tag" } };
                },
                async createRef() {
                    state.calls.createRef += 1;
                    state.tagRef = { type: "tag", sha: "annotated-tag" };
                    if (state.createRefRace) {
                        state.createRefRace = false;
                        throw apiError(422);
                    }
                    return { data: { ref: "refs/tags/v1.2.3" } };
                },
            },
            repos: {
                async getReleaseByTag() {
                    if (!state.release) {
                        throw apiError(404);
                    }
                    return { data: state.release };
                },
                async createRelease(arguments_) {
                    state.calls.createRelease += 1;
                    state.calls.createReleaseArguments = arguments_;
                    state.release = {
                        id: 7,
                        tag_name: arguments_.tag_name,
                        draft: arguments_.draft,
                        prerelease: arguments_.prerelease,
                    };
                    state.latestRelease = arguments_.prerelease ? null : state.release;
                    if (state.createReleaseRace) {
                        state.createReleaseRace = false;
                        throw apiError(422);
                    }
                    return { data: state.release };
                },
                async getLatestRelease() {
                    if (!state.latestRelease) {
                        throw apiError(404);
                    }
                    return { data: state.latestRelease };
                },
            },
        },
    };
    return { github, state };
}

async function run() {
    let fixture = fakeGitHub();
    let result = await publishRelease({
        github: fixture.github,
        owner: "owner",
        repo: "repo",
        tagName: "v1.2.3",
        qualifiedSha: SHA,
        defaultBranch: "main",
    });
    assert.equal(result.tag.created, true);
    assert.equal(result.release.created, true);
    assert.equal(fixture.state.calls.createTag, 1);
    assert.equal(fixture.state.calls.createRef, 1);
    assert.equal(fixture.state.calls.createRelease, 1);
    assert.equal(fixture.state.calls.createReleaseArguments.prerelease, false);
    assert.equal(fixture.state.calls.createReleaseArguments.make_latest, "true");

    result = await publishRelease({
        github: fixture.github,
        owner: "owner",
        repo: "repo",
        tagName: "v1.2.3",
        qualifiedSha: SHA,
        defaultBranch: "main",
    });
    assert.equal(result.tag.created, false);
    assert.equal(result.release.created, false);
    assert.equal(fixture.state.calls.createTag, 1);
    assert.equal(fixture.state.calls.createRef, 1);
    assert.equal(fixture.state.calls.createRelease, 1);

    fixture = fakeGitHub({ tagRef: { type: "commit", sha: SHA } });
    result = await publishRelease({
        github: fixture.github,
        owner: "owner",
        repo: "repo",
        tagName: "v1.2.3",
        qualifiedSha: SHA,
        defaultBranch: "main",
    });
    assert.equal(result.tag.created, false);
    assert.equal(result.release.created, true);

    fixture = fakeGitHub({
        branchSha: OTHER_SHA,
        tagRef: { type: "commit", sha: SHA },
        release: { id: 8, tag_name: "v1.2.3", draft: false, prerelease: false },
    });
    result = await publishRelease({
        github: fixture.github,
        owner: "owner",
        repo: "repo",
        tagName: "v1.2.3",
        qualifiedSha: SHA,
        defaultBranch: "main",
    });
    assert.equal(result.tag.created, false);
    assert.equal(result.release.created, false);

    fixture = fakeGitHub({ tagRef: { type: "commit", sha: OTHER_SHA } });
    await assert.rejects(
        () => ensureReleaseTag({
            github: fixture.github,
            owner: "owner",
            repo: "repo",
            tagName: "v1.2.3",
            qualifiedSha: SHA,
            defaultBranch: "main",
        }),
        /already points/
    );

    fixture = fakeGitHub({ branchSha: OTHER_SHA });
    await assert.rejects(
        () => assertReleaseCandidate({
            github: fixture.github,
            owner: "owner",
            repo: "repo",
            tagName: "v1.2.3",
            qualifiedSha: SHA,
            defaultBranch: "main",
        }),
        /refusing to tag stale candidate/
    );

    fixture = fakeGitHub({ createRefRace: true });
    result = await ensureReleaseTag({
        github: fixture.github,
        owner: "owner",
        repo: "repo",
        tagName: "v1.2.3",
        qualifiedSha: SHA,
        defaultBranch: "main",
    });
    assert.equal(result.recoveredRace, true);

    fixture = fakeGitHub({ createReleaseRace: true });
    result = await publishRelease({
        github: fixture.github,
        owner: "owner",
        repo: "repo",
        tagName: "v1.2.3",
        qualifiedSha: SHA,
        defaultBranch: "main",
    });
    assert.equal(result.release.recoveredRace, true);

    fixture = fakeGitHub();
    result = await publishRelease({
        github: fixture.github,
        owner: "owner",
        repo: "repo",
        tagName: "v1.2.3-rc.1",
        qualifiedSha: SHA,
        defaultBranch: "main",
    });
    assert.equal(result.release.created, true);
    assert.equal(fixture.state.calls.createReleaseArguments.prerelease, true);
    assert.equal(fixture.state.calls.createReleaseArguments.make_latest, "false");

    fixture = fakeGitHub({
        tagRef: { type: "commit", sha: SHA },
        release: { id: 9, tag_name: "v1.2.3", draft: true, prerelease: false },
    });
    await assert.rejects(
        () => publishRelease({
            github: fixture.github,
            owner: "owner",
            repo: "repo",
            tagName: "v1.2.3",
            qualifiedSha: SHA,
            defaultBranch: "main",
        }),
        /must not be a draft/
    );

    fixture = fakeGitHub({
        tagRef: { type: "commit", sha: SHA },
        release: { id: 10, tag_name: "v1.2.3-rc.1", draft: false, prerelease: false },
    });
    await assert.rejects(
        () => publishRelease({
            github: fixture.github,
            owner: "owner",
            repo: "repo",
            tagName: "v1.2.3-rc.1",
            qualifiedSha: SHA,
            defaultBranch: "main",
        }),
        /prerelease state/
    );

    fixture = fakeGitHub({
        tagRef: { type: "commit", sha: SHA },
        release: { id: 11, tag_name: "v1.2.3-rc.1", draft: false, prerelease: true },
        latestRelease: { id: 11, tag_name: "v1.2.3-rc.1" },
    });
    await assert.rejects(
        () => publishRelease({
            github: fixture.github,
            owner: "owner",
            repo: "repo",
            tagName: "v1.2.3-rc.1",
            qualifiedSha: SHA,
            defaultBranch: "main",
        }),
        /must not be the latest release/
    );

    assert.deepEqual(releaseSemVer("v1.2.3"), { prerelease: false });
    assert.deepEqual(releaseSemVer("v1.2.3-rc.1"), { prerelease: true });
    assert.throws(() => releaseSemVer("v1.2.3+build.1"), /not SwiftPM-compatible/);

    console.log("13 release publication regression scenarios passed.");
}

run().catch((error) => {
    console.error(error);
    process.exit(1);
});
