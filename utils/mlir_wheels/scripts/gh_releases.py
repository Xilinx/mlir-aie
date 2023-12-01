import os
import time

from github import Github
import datetime

# Authentication is defined via github.Auth
from github import Auth

# using an access token
auth = Auth.Token(os.environ["GITHUB_API_TOKEN"])

twomonthsago = datetime.date.today() - datetime.timedelta(days=30)

# First create a Github instance:

# Public Web Github
g = Github(auth=auth)

n_deleted = 0
for _ in range(100):
    n_deleted = 0
    repo = g.get_repo("makslevental/wheels")
    release = repo.get_release(113028511)
    assets = release.get_assets()
    for ass in assets:
        if "35ca6498" in ass.name:
            continue
        if ass.created_at.date() < twomonthsago:
            print(ass.name)
            assert ass.delete_asset()
            n_deleted += 1

    repo = g.get_repo("makslevental/mlir-wheels")
    release = repo.get_release(111725799)
    assets = release.get_assets()
    for ass in assets:
        if "35ca6498" in ass.name:
            continue
        if ass.created_at.date() < twomonthsago:
            print(ass.name)
            assert ass.delete_asset()
            n_deleted += 1

    if n_deleted == 0:
        break
    time.sleep(5)

if n_deleted > 0:
    raise Exception("missed some")
