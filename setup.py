# Copyright 2026 Micael Oliveira
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# This file is only needed for the automatic versioning in the conda recipe
import setuptools_scm
from setuptools import setup

setup(version=setuptools_scm.get_version())
