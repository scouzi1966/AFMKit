"use strict";

function statusIs(error, status) {
    return error && Number(error.status) === status;
}

function validateQualifiedSha(qualifiedSha) {
    if (!/^[0-9a-f]{40}$/.test(qualifiedSha)) {
        throw new Error(`Invalid qualified commit SHA: ${qualifiedSha}`);
    }
}

function releaseSemVer(tagName) {
    const numeric = "(?:0|[1-9][0-9]*)";
    const alphanumeric = "(?:[0-9]*[A-Za-z-][0-9A-Za-z-]*)";
    const prereleaseIdentifier = `(?:${numeric}|${alphanumeric})`;
    const pattern = new RegExp(
        `^v${numeric}\\.${numeric}\\.${numeric}` +
        `(?:-(${prereleaseIdentifier}(?:\\.${prereleaseIdentifier})*))?$`
    );
    const match = pattern.exec(tagName);
    if (!match) {
        throw new Error(
            `Release tag ${tagName} is not SwiftPM-compatible SemVer without build metadata.`
        );
    }
    return { prerelease: Boolean(match[1]) };
}

function publicationIntentRef(tagName) {
    releaseSemVer(tagName);
    return `tags/afmkit-publication-${tagName}`;
}

async function getPublicationIntentCommit(github, owner, repo, tagName) {
    const refName = publicationIntentRef(tagName);
    try {
        const ref = await github.rest.git.getRef({ owner, repo, ref: refName });
        if (ref.data.object.type !== "commit") {
            throw new Error(
                `Publication intent ${refName} must point directly to a commit.`
            );
        }
        return ref.data.object.sha;
    } catch (error) {
        if (statusIs(error, 404)) {
            return null;
        }
        throw error;
    }
}

async function getTagCommit(github, owner, repo, tagName) {
    let object;
    try {
        const ref = await github.rest.git.getRef({
            owner,
            repo,
            ref: `tags/${tagName}`,
        });
        object = ref.data.object;
    } catch (error) {
        if (statusIs(error, 404)) {
            return null;
        }
        throw error;
    }

    for (let depth = 0; depth < 8; depth += 1) {
        if (object.type === "commit") {
            return object.sha;
        }
        if (object.type !== "tag") {
            throw new Error(`Tag ${tagName} points to unsupported Git object type ${object.type}.`);
        }
        const tag = await github.rest.git.getTag({
            owner,
            repo,
            tag_sha: object.sha,
        });
        object = tag.data.object;
    }
    throw new Error(`Tag ${tagName} contains too many nested annotated tags.`);
}

async function assertReleaseCandidate({
    github,
    owner,
    repo,
    tagName,
    qualifiedSha,
    defaultBranch,
}) {
    validateQualifiedSha(qualifiedSha);
    releaseSemVer(tagName);
    const taggedSha = await getTagCommit(github, owner, repo, tagName);
    if (taggedSha !== null) {
        if (taggedSha !== qualifiedSha) {
            throw new Error(`Tag ${tagName} already points to ${taggedSha}, not ${qualifiedSha}.`);
        }
        return { recoverableTag: true };
    }

    const intentSha = await getPublicationIntentCommit(
        github,
        owner,
        repo,
        tagName
    );
    if (intentSha !== null) {
        if (intentSha !== qualifiedSha) {
            throw new Error(
                `Publication intent for ${tagName} points to ${intentSha}, ` +
                `not ${qualifiedSha}.`
            );
        }
        return { recoverableTag: false, recoverablePublication: true };
    }

    const branch = await github.rest.git.getRef({
        owner,
        repo,
        ref: `heads/${defaultBranch}`,
    });
    if (branch.data.object.sha !== qualifiedSha) {
        throw new Error(
            `Default branch ${defaultBranch} moved to ${branch.data.object.sha}; ` +
            `refusing to tag stale candidate ${qualifiedSha}.`
        );
    }
    return { recoverableTag: false, recoverablePublication: false };
}

async function ensurePublicationIntent({
    github,
    owner,
    repo,
    tagName,
    qualifiedSha,
    defaultBranch,
}) {
    const candidate = await assertReleaseCandidate({
        github,
        owner,
        repo,
        tagName,
        qualifiedSha,
        defaultBranch,
    });
    if (candidate.recoverableTag) {
        return { created: false, finalTagExists: true };
    }
    if (candidate.recoverablePublication) {
        return { created: false };
    }

    const refName = publicationIntentRef(tagName);
    try {
        await github.rest.git.createRef({
            owner,
            repo,
            ref: `refs/${refName}`,
            sha: qualifiedSha,
        });
        return { created: true };
    } catch (error) {
        if (!statusIs(error, 422)) {
            throw error;
        }
        const racedSha = await getPublicationIntentCommit(
            github,
            owner,
            repo,
            tagName
        );
        if (racedSha !== qualifiedSha) {
            throw new Error(
                `Publication intent for ${tagName} was created concurrently for ` +
                `${racedSha}, not ${qualifiedSha}.`
            );
        }
        return { created: false, recoveredRace: true };
    }
}

async function ensureReleaseTag({
    github,
    owner,
    repo,
    tagName,
    qualifiedSha,
    defaultBranch,
}) {
    const candidate = await assertReleaseCandidate({
        github,
        owner,
        repo,
        tagName,
        qualifiedSha,
        defaultBranch,
    });
    if (candidate.recoverableTag) {
        return { created: false };
    }

    const tag = await github.rest.git.createTag({
        owner,
        repo,
        tag: tagName,
        message: `AFMKit ${tagName}`,
        object: qualifiedSha,
        type: "commit",
    });
    try {
        await github.rest.git.createRef({
            owner,
            repo,
            ref: `refs/tags/${tagName}`,
            sha: tag.data.sha,
        });
        return { created: true };
    } catch (error) {
        if (!statusIs(error, 422)) {
            throw error;
        }
        const racedSha = await getTagCommit(github, owner, repo, tagName);
        if (racedSha !== qualifiedSha) {
            throw new Error(
                `Tag ${tagName} was created concurrently for ${racedSha}, not ${qualifiedSha}.`
            );
        }
        return { created: false, recoveredRace: true };
    }
}

function validateReleaseRecord(release, tagName, prerelease) {
    if (release.tag_name !== tagName) {
        throw new Error(
            `Existing release tag is ${release.tag_name}, expected ${tagName}.`
        );
    }
    if (release.draft !== false) {
        throw new Error(`Release ${tagName} must not be a draft.`);
    }
    if (release.prerelease !== prerelease) {
        throw new Error(
            `Release ${tagName} prerelease state is ${release.prerelease}, ` +
            `expected ${prerelease}.`
        );
    }
}

async function validateLatestState({ github, owner, repo, release, prerelease }) {
    let latest = null;
    try {
        latest = (await github.rest.repos.getLatestRelease({ owner, repo })).data;
    } catch (error) {
        if (!statusIs(error, 404)) {
            throw error;
        }
    }

    if (prerelease) {
        if (latest && latest.id === release.id) {
            throw new Error(`Prerelease ${release.tag_name} must not be the latest release.`);
        }
        return;
    }
    if (!latest || latest.id !== release.id) {
        throw new Error(`Stable release ${release.tag_name} must be the latest release.`);
    }
}

async function validateExistingRelease({ github, owner, repo, release, tagName, prerelease }) {
    validateReleaseRecord(release, tagName, prerelease);
    await validateLatestState({ github, owner, repo, release, prerelease });
}

async function ensureGitHubRelease({ github, owner, repo, tagName, qualifiedSha }) {
    const { prerelease } = releaseSemVer(tagName);
    try {
        const release = await github.rest.repos.getReleaseByTag({
            owner,
            repo,
            tag: tagName,
        });
        await validateExistingRelease({
            github,
            owner,
            repo,
            release: release.data,
            tagName,
            prerelease,
        });
        return { created: false, release: release.data };
    } catch (error) {
        if (!statusIs(error, 404)) {
            throw error;
        }
    }

    try {
        const release = await github.rest.repos.createRelease({
            owner,
            repo,
            tag_name: tagName,
            target_commitish: qualifiedSha,
            name: `AFMKit ${tagName}`,
            generate_release_notes: true,
            draft: false,
            prerelease,
            make_latest: prerelease ? "false" : "true",
        });
        validateReleaseRecord(release.data, tagName, prerelease);
        return { created: true, release: release.data };
    } catch (error) {
        if (!statusIs(error, 422)) {
            throw error;
        }
        const release = await github.rest.repos.getReleaseByTag({
            owner,
            repo,
            tag: tagName,
        });
        await validateExistingRelease({
            github,
            owner,
            repo,
            release: release.data,
            tagName,
            prerelease,
        });
        return { created: false, recoveredRace: true, release: release.data };
    }
}

async function publishRelease({
    github,
    owner,
    repo,
    tagName,
    qualifiedSha,
    defaultBranch,
}) {
    const tag = await ensureReleaseTag({
        github,
        owner,
        repo,
        tagName,
        qualifiedSha,
        defaultBranch,
    });
    const release = await ensureGitHubRelease({
        github,
        owner,
        repo,
        tagName,
        qualifiedSha,
    });
    return { tag, release };
}

module.exports = {
    assertReleaseCandidate,
    ensureGitHubRelease,
    ensurePublicationIntent,
    ensureReleaseTag,
    getPublicationIntentCommit,
    getTagCommit,
    publicationIntentRef,
    publishRelease,
    releaseSemVer,
    validateReleaseRecord,
};
