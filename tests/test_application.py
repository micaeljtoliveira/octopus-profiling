# Copyright 2026 Micael Oliveira
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import pytest
from access.config.parallel_allocation_strategies import RootAllocation
from access.config.parallel_component import ComponentLayout
from access.config.parallel_domain import Domain, DomainDecompositionSpec
from access.config.parallel_layouts import iter_layouts
from access.config.parallel_mpi_grid import MPICartesianGrid
from conftest import PARALLEL_STDOUT, SERIAL_STDOUT, write_experiment

from octopus_profiling.application import OctopusApplication, OctopusGroupSizeConstraint


@pytest.fixture
def app():
    return OctopusApplication(name="ch4", n_points=1000, n_states=8, n_kpoints=2, min_points_per_domain=300)


def octopus_layout(grid, threads=1, domain=(1000, 8, 2)):
    n_ranks = grid[0] * grid[1] * grid[2]
    return ComponentLayout(
        name="octopus",
        n_cores=n_ranks * threads,
        n_ranks=n_ranks,
        threads_per_rank=threads,
        decomposition=DomainDecompositionSpec(Domain(domain), MPICartesianGrid(grid)),
    )


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"name": "x", "n_points": 0, "n_states": 1}, {"name": "x", "n_points": 1, "n_states": 1, "n_kpoints": 0}],
)
def test_invalid_application(kwargs):
    """Tests that an application without a name or with a non-positive size is rejected."""
    with pytest.raises(ValueError):
        OctopusApplication(**kwargs)


def test_layouts(app):
    """Tests that the layout search finds every way of dividing the processes among the strategies."""
    layouts = list(iter_layouts(app.parallel_component, 4, allocations=RootAllocation(thread_range=(1, 2))))

    found = {(layout.decomposition.grid.shape, layout.threads_per_rank) for layout in layouts}
    # 4 domains would hold fewer than 300 points each, and there are only 2 k-points
    assert found == {
        ((1, 2, 2), 1),
        ((1, 4, 1), 1),
        ((2, 1, 2), 1),
        ((2, 2, 1), 1),
        ((1, 1, 2), 2),
        ((1, 2, 1), 2),
        ((2, 1, 1), 2),
    }


@pytest.mark.parametrize(
    ("grid", "satisfied"),
    [
        ((3, 1, 1), True),
        ((4, 1, 1), False),
        ((1, 8, 1), True),
        ((1, 16, 1), False),
        ((1, 1, 2), True),
        ((1, 1, 4), False),
    ],
)
def test_group_size_constraint(grid, satisfied):
    """Tests the limits on the number of groups of each strategy."""
    assert OctopusGroupSizeConstraint(300).is_satisfied(octopus_layout(grid)) == satisfied


def test_group_size_constraint_needs_decomposition():
    layout = ComponentLayout(name="octopus", n_cores=1, n_ranks=1, threads_per_rank=1, decomposition=None)
    with pytest.raises(ValueError):
        OctopusGroupSizeConstraint().is_satisfied(layout)


def test_experiment_name(app):
    assert app.experiment_name(octopus_layout((2, 4, 1), threads=2)) == "octopus-layout_ch4_dom2_st4_kp1_omp2"


def test_experiment_name_rejects_other_layouts(app):
    layout = ComponentLayout(name="ocean", n_cores=1, n_ranks=1, threads_per_rank=1, decomposition=None)
    with pytest.raises(ValueError):
        app.experiment_name(layout)


def test_config_changes(app):
    assert app.config_changes(octopus_layout((2, 4, 1))) == {
        "inp": {
            "ParDomains": 2,
            "ParStates": 4,
            "ParKPoints": 1,
            "ProfilingMode": "prof_time",
            "ProfilingAllNodes": "yes",
        }
    }


def test_config_changes_leaves_profiling_alone():
    app = OctopusApplication(name="ch4", n_points=1000, n_states=8, profiling_mode=None, profile_all_ranks=False)
    assert app.config_changes(octopus_layout((2, 1, 1), domain=(1000, 8, 1))) == {
        "inp": {"ParDomains": 2, "ParStates": 1, "ParKPoints": 1}
    }


def test_logs(app, tmp_path):
    """Tests that the per-rank log is only found when every rank wrote its profile."""
    write_experiment(tmp_path / "one", ranks=1)
    write_experiment(tmp_path / "two", ranks=2)

    assert set(app.component_logs(tmp_path / "one")) == {"time", "time_tree"}
    assert set(app.component_logs(tmp_path / "two")) == {"time", "time_tree", "time_ranks"}
    assert app.component_for_log("time_ranks") == "octopus"


def test_parse_layout_parallel(app, tmp_path):
    write_experiment(tmp_path, stdout=PARALLEL_STDOUT)

    assert app.parse_layout(tmp_path) == octopus_layout((2, 2, 1), threads=2)


def test_parse_layout_serial(app, tmp_path):
    write_experiment(tmp_path, stdout=SERIAL_STDOUT)

    assert app.parse_layout(tmp_path) == octopus_layout((1, 1, 1))


@pytest.mark.parametrize(
    "stdout",
    [
        None,
        "Nothing about the parallelization here\n",
        PARALLEL_STDOUT.replace("ParStates  ", "ParOther   "),
        PARALLEL_STDOUT.replace("Number of processes           :       4", "Number of processes : 5"),
    ],
    ids=["no stdout", "no parallelization", "other strategy", "slaves"],
)
def test_parse_layout_unknown(app, tmp_path, stdout):
    write_experiment(tmp_path, stdout=stdout)

    assert app.parse_layout(tmp_path) is None
