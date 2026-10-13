# Copyright 2026 Micael Oliveira
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
octopus-profiling package.
"""

from contextlib import suppress
from importlib.metadata import PackageNotFoundError, version

__version__ = "unknown"
with suppress(PackageNotFoundError):
    __version__ = version("octopus-profiling")

from octopus_profiling.application import OctopusApplication, OctopusGroupSizeConstraint
from octopus_profiling.manager import OctopusManager
from octopus_profiling.profiling_parser import (
    OctopusProfilingParser,
    OctopusRankProfilingParser,
    OctopusTreeProfilingParser,
)

__all__ = [
    "OctopusApplication",
    "OctopusGroupSizeConstraint",
    "OctopusManager",
    "OctopusProfilingParser",
    "OctopusRankProfilingParser",
    "OctopusTreeProfilingParser",
]
