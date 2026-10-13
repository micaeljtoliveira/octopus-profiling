# Copyright 2026 Micael Oliveira
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Octopus, as an application to be profiled.

Octopus divides its MPI processes among several parallelization strategies at once: the real-space mesh is divided
into domains, the electronic states among groups of processes, and so on for the k-points. Unlike a coupled model,
whose components run on disjoint sets of cores, these are levels of a single decomposition: the number of processes
is the product of the number of groups of each strategy.

This is expressed as a single component whose domain has one axis per strategy, (mesh points, states, k-points),
so that each process grid found by the layout search is one way of setting ParDomains, ParStates and ParKPoints.
OpenMP threads come from the thread range of the allocation strategy passed to the layout search.
"""

import re
from dataclasses import dataclass
from pathlib import Path

from access.config.parallel_component import ComponentLayout, LocalConstraint, ParallelComponent
from access.config.parallel_domain import Domain, DomainDecompositionSpec
from access.config.parallel_mpi_grid import MPICartesianGrid
from access.profiling.application import Application, LogLocator, LogSpec, log_at

from octopus_profiling.profiling_parser import (
    OctopusProfilingParser,
    OctopusRankProfilingParser,
    OctopusTreeProfilingParser,
)

# Name of the single component of the layout
OCTOPUS_COMPONENT = "octopus"

# Parallelization strategies, in the order of the axes of the domain
PARALLEL_STRATEGIES = ("ParDomains", "ParStates", "ParKPoints")


@dataclass(frozen=True)
class OctopusGroupSizeConstraint(LocalConstraint):
    """Restricts the number of groups of each Octopus parallelization strategy.

    There cannot be more groups of states or k-points than there are states or k-points, and each domain should hold
    a minimum number of mesh points for the parallelization in domains to be worth its communication cost.

    Args:
        min_points_per_domain (int): Minimum number of mesh points in each domain.
    """

    min_points_per_domain: int = 1

    def is_satisfied(self, layout: ComponentLayout) -> bool:
        if layout.decomposition is None:
            raise ValueError(f"{type(self).__name__} requires a layout with a domain decomposition.")
        (n_points, n_states, n_kpoints) = layout.decomposition.domain.shape
        (n_domains, n_state_groups, n_kpoint_groups) = layout.decomposition.grid.shape
        return (
            n_points // n_domains >= self.min_points_per_domain
            and n_state_groups <= n_states
            and n_kpoint_groups <= n_kpoints
        )


def _rank_files_locator() -> LogLocator:
    """Returns a locator for the flat profile of rank 0, but only if the other ranks wrote theirs too."""

    def locate(output_dir: Path) -> Path | None:
        if not (output_dir / "profiling" / "time.000001").is_file():
            return None
        return output_dir / "profiling" / "time.000000"

    return locate


@dataclass(frozen=True)
class OctopusApplication(Application):
    """One Octopus calculation, set up one particular way.

    The sizes of the system are what bound the parallelization: they can be read from the output of a previous run
    of the same input. The number of mesh points is printed next to the size of the ParDomains group, and the number
    of states in the "States" section.

    Args:
        name (str): Short name for this calculation.
        n_points (int): Number of points of the real-space mesh.
        n_states (int): Number of electronic states.
        n_kpoints (int): Number of k-points. Defaults to 1.
        min_points_per_domain (int): Minimum number of mesh points each domain must hold. Defaults to 1.
        experiment_prefix (str): Prefix for the names of the experiments generated for this calculation.
        stdout (str): Name of the file the standard output of Octopus is written to, relative to the experiment
            directory. Defaults to "out".
        profiling_mode (str | None): Value of the ProfilingMode variable to set in the generated experiments. None
            leaves whatever the control sets. Defaults to "prof_time".
        profile_all_ranks (bool): Whether the generated experiments should write the profile of all MPI ranks, which
            is needed to study the load balance. Defaults to True.

    Raises:
        ValueError: If no name is given or if a size is not positive.
    """

    # name is defaulted because every field after it is, not because it is optional: __post_init__ refuses an empty
    # one.
    name: str = ""
    n_points: int = 0
    n_states: int = 0
    n_kpoints: int = 1
    min_points_per_domain: int = 1
    experiment_prefix: str = "octopus-layout"
    stdout: str = "out"
    profiling_mode: str | None = "prof_time"
    profile_all_ranks: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("OctopusApplication.name must be non-empty.")
        for field in ("n_points", "n_states", "n_kpoints", "min_points_per_domain"):
            if getattr(self, field) <= 0:
                raise ValueError(f"OctopusApplication.{field} must be positive, got {getattr(self, field)}.")

    @property
    def domain(self) -> Domain:
        """Domain with one axis per parallelization strategy: (mesh points, states, k-points)."""
        return Domain((self.n_points, self.n_states, self.n_kpoints))

    @property
    def parallel_component(self) -> ParallelComponent:
        return ParallelComponent(
            OCTOPUS_COMPONENT,
            domain=self.domain,
            local_constraints=(OctopusGroupSizeConstraint(self.min_points_per_domain),),
        )

    @property
    def logs(self) -> tuple[LogSpec, ...]:
        return (
            LogSpec("time", log_at("profiling/time.000000"), OctopusProfilingParser(), component=OCTOPUS_COMPONENT),
            LogSpec(
                "time_tree",
                log_at("profiling/time.000000.tree"),
                OctopusTreeProfilingParser(),
                component=OCTOPUS_COMPONENT,
                optional=True,
            ),
            LogSpec(
                "time_ranks",
                _rank_files_locator(),
                OctopusRankProfilingParser(),
                component=OCTOPUS_COMPONENT,
                optional=True,
            ),
        )

    @staticmethod
    def _group_sizes(layout: ComponentLayout) -> tuple[int, int, int]:
        """Returns the number of groups of each parallelization strategy in a layout.

        Raises:
            ValueError: If the layout is not a layout of an Octopus calculation.
        """
        if (
            layout.name != OCTOPUS_COMPONENT
            or not layout.is_leaf
            or layout.decomposition is None
            or len(layout.decomposition.grid.shape) != len(PARALLEL_STRATEGIES)
        ):
            raise ValueError(f"Layout '{layout.name}' is not a layout of an Octopus calculation.")
        n_domains, n_state_groups, n_kpoint_groups = layout.decomposition.grid.shape
        return n_domains, n_state_groups, n_kpoint_groups

    def experiment_name(self, layout: ComponentLayout) -> str:
        n_domains, n_state_groups, n_kpoint_groups = self._group_sizes(layout)
        return (
            f"{self.experiment_prefix}_{self.name}_dom{n_domains}_st{n_state_groups}_kp{n_kpoint_groups}"
            f"_omp{layout.threads_per_rank}"
        )

    def config_changes(self, layout: ComponentLayout) -> dict:
        variables: dict[str, object] = dict(zip(PARALLEL_STRATEGIES, self._group_sizes(layout), strict=True))
        if self.profiling_mode is not None:
            variables["ProfilingMode"] = self.profiling_mode
        if self.profile_all_ranks:
            variables["ProfilingAllNodes"] = True
        return {"inp": variables}

    def parse_layout(self, output_dir: Path) -> ComponentLayout | None:
        """Returns the layout an experiment ran, as Octopus reported it in its standard output.

        The standard output is read rather than the input file, since the input may leave the parallelization to
        Octopus, and only the output says what it chose.

        Args:
            output_dir (Path): The experiment directory.

        Returns:
            ComponentLayout | None: The layout, or None if the standard output does not say, or if the run used a
                parallelization strategy other than domains, states and k-points.
        """
        stdout = output_dir / self.stdout
        if not stdout.is_file():
            return None
        text = stdout.read_text(errors="replace")

        if re.search(r"Octopus will run in \*serial\*", text):
            n_ranks, n_threads = 1, 1
            groups = dict.fromkeys(PARALLEL_STRATEGIES, 1)
        else:
            processes = re.search(r"Number of processes\s*:\s*(\d+)", text)
            threads = re.search(r"Number of threads per process\s*:\s*(\d+)", text)
            if processes is None or threads is None:
                return None
            n_ranks, n_threads = int(processes.group(1)), int(threads.group(1))
            groups = {
                strategy: int(size) for strategy, size in re.findall(r"Number of nodes in (\w+)\s+group:\s*(\d+)", text)
            }
            if any(strategy not in PARALLEL_STRATEGIES for strategy in groups):
                return None
            groups = {strategy: groups.get(strategy, 1) for strategy in PARALLEL_STRATEGIES}

        grid = tuple(groups[strategy] for strategy in PARALLEL_STRATEGIES)
        if grid[0] * grid[1] * grid[2] != n_ranks:
            # Some processes were used for something else, e.g. as Poisson solver slaves
            return None
        return ComponentLayout(
            name=OCTOPUS_COMPONENT,
            n_cores=n_ranks * n_threads,
            n_ranks=n_ranks,
            threads_per_rank=n_threads,
            decomposition=DomainDecompositionSpec(self.domain, MPICartesianGrid(grid)),
        )
