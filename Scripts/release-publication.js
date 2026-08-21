"use strict";

function statusIs(error, status) {
    return error && Number(error.status) === status;
}

function validateQualifiedSha(qualifiedSha) {
    if (!/^[0-9a-f]{40}$/.test(qualifiedSha)) {
        throw new Error(`Invalid qualified commit SHA: ${qualifiedSha}`);
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
    const taggedSha = await getTagCommit(github, owner, repo, tagName);
    if (taggedSha !== null) {
        if (taggedSha !== qualifiedSha) {
            throw new Error(`Tag ${tagName} already points to ${taggedSha}, not ${qualifiedSha}.`);
        }
        return { recoverableTag: true };
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
    return { recoverableTag: false };
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

async function ensureGitHubRelease({ github, owner, repo, tagName, qualifiedSha }) {
    try {
        const release = await github.rest.repos.getReleaseByTag({
            owner,
            repo,
            tag: tagName,
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
        });
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
    ensureReleaseTag,
    getTagCommit,
    publishRelease,
};
