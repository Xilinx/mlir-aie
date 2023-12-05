import argparse
import json
import os
import sys

import requests


def get_pull_request_comments(
    github_api_url, github_token, github_api_timeout, repo, pull_number
):
    for page in range(1, 100):
        result = requests.get(
            f"{github_api_url}/repos/{repo}/pulls/{pull_number:d}/comments?page={page:d}",
            headers={
                "Accept": "application/vnd.github.v3+json",
                "Authorization": f"token {github_token}",
            },
            timeout=github_api_timeout,
        )

        assert result.status_code == requests.codes.ok  # pylint: disable=no-member

        chunk = json.loads(result.text)

        if len(chunk) == 0:
            break

        for item in chunk:
            yield item


def delete_pull_request_comment(
    github_api_url, github_token, github_api_timeout, repo, comment_id
):
    result = requests.get(
        f"{github_api_url}/repos/{repo}/pulls/comments/{comment_id}",
        headers={
            "Accept": "application/vnd.github.v3+json",
            "Authorization": f"token {github_token}",
        },
        timeout=github_api_timeout,
    )

    assert result.status_code == requests.codes.ok  # pylint: disable=no-member


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repository",
        type=str,
        required=True,
        help="Name of the repository containing the code",
    )
    parser.add_argument(
        "--pull-number",
        type=int,
        required=True,
    )
    args = parser.parse_args()

    # The GitHub API token is sensitive information, pass it through the environment
    github_token = os.environ.get("GITHUB_TOKEN")
    github_api_url = os.environ.get("GITHUB_API_URL")
    github_api_timeout = 10

    for comment in get_pull_request_comments(
        github_api_url,
        github_token,
        github_api_timeout,
        args.repository,
        args.pull_number,
    ):
        if comment["user"]["login"] == "github-actions[bot]":
            delete_pull_request_comment(
                github_api_url,
                github_token,
                github_api_timeout,
                args.repository,
                comment["id"],
            )


if __name__ == "__main__":
    sys.exit(main())
