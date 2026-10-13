# Copyright 2026 Micael Oliveira
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from conftest import INPUT_FILE

from octopus_profiling.inp import read_variables, set_variables


def test_read_variables():
    """Tests that top-level variables are read, ignoring blocks and comments."""
    variables = read_variables(INPUT_FILE)

    assert variables == {
        "calculationmode": "gs",
        "unitsoutput": "eV_Angstrom",
        "radius": "4.5*angstrom",
        "parstates": "0",
        "profilingmode": "prof_memory",
    }


def test_set_variables_replaces_in_place():
    """Tests that variables already set are changed where they are, matching names case-insensitively."""
    text = set_variables(INPUT_FILE, {"ParStates": 2, "ProfilingMode": "prof_time"})

    lines = text.splitlines()
    assert lines[8] == "ParStates = 2"
    assert lines[9] == "ProfilingMode = prof_time"
    assert len(lines) == len(INPUT_FILE.splitlines())


def test_set_variables_appends_missing():
    """Tests that variables not set yet are appended, and that booleans are written as yes or no."""
    text = set_variables(INPUT_FILE, {"ParDomains": 4, "ProfilingAllNodes": True})

    assert text.endswith("ProfilingMode = prof_memory\n\nParDomains = 4\nProfilingAllNodes = yes\n")
    assert read_variables(text)["pardomains"] == "4"


def test_set_variables_leaves_blocks_alone():
    """Tests that a block line looking like an assignment is not changed."""
    text = "Radius = 4\n%Species\n  Radius = 3\n%\n"

    assert set_variables(text, {"Radius": 5}) == "Radius = 5\n%Species\n  Radius = 3\n%\n"
