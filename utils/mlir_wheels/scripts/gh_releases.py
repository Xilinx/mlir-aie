# Copyright (C) 2023 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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
    for a in assets:
        if "35ca6498" in a.name:
            continue
        if a.created_at.date() < twomonthsago:
            print(a.name)
            assert a.delete_asset()
            n_deleted += 1

    repo = g.get_repo("makslevental/mlir-wheels")
    release = repo.get_release(111725799)
    assets = release.get_assets()
    for a in assets:
        if "35ca6498" in a.name:
            continue
        if a.created_at.date() < twomonthsago:
            print(a.name)
            assert a.delete_asset()
            n_deleted += 1

    if n_deleted == 0:
        break
    time.sleep(5)

if n_deleted > 0:
    raise Exception("missed some")
