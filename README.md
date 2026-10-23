# octopus-profiling

Tools to profile the [Octopus](https://octopus-code.org) code, built on top of
[access-profiling](https://github.com/ACCESS-NRI/access-profiling).

## Features

- Parsers for the time profiling files Octopus writes when `ProfilingMode` is set:
  - `OctopusProfilingParser`: flat profile (`profiling/time.NNNNNN`), with call counts, cumulative and self
    times, time fractions, throughput and bandwidth.
  - `OctopusTreeProfilingParser`: call tree (`profiling/time.NNNNNN.tree`), with the fraction of the total and of
    the parent time spent in each region.
  - `OctopusRankProfilingParser`: flat profiles of all MPI ranks (written when `ProfilingAllNodes = yes`), to
    study the load balance with `access.profiling.parser.aggregate_pe_data`.
- `OctopusApplication`: describes a calculation, so that its valid parallel layouts (number of domains, groups of
  states and groups of k-points, and OpenMP threads) can be searched for.
- `OctopusManager`: generates, runs, parses, plots and archives scaling experiments, running Octopus locally with
  `mpirun`.

## Installation

This package needs the development versions of access-profiling and access-config-utils. Install them from
source first, then this package, inside a virtual environment:

```shell
git clone https://github.com/ACCESS-NRI/access-config-utils.git
git clone https://github.com/ACCESS-NRI/access-profiling.git
pip install -e ./access-config-utils -e ./access-profiling
pip install -e "./octopus-profiling[devel,test]"
```

## Usage

### Analysing existing runs

```python
from pathlib import Path

from access.profiling import RegionGroup
from access.profiling.metrics import tavg
from octopus_profiling import OctopusApplication, OctopusManager

# The numbers of states and k-points bound the parallelization in states and k-points. Octopus writes them to
# restart/gs/states (nst and nik). The number of domains is not bounded, as Octopus partitions the mesh itself.
app = OctopusApplication(name="ch4", n_states=8)
manager = OctopusManager(Path("experiments"), Path("archive"), app)

manager.add_experiment_from_directory("1core", Path("1core"))
manager.add_experiment_from_directory("2cores", Path("2cores"))
manager.parse_profiling_data()

manager.plot_scaling_data([RegionGroup("time", ["COMPLETE_RUN", "DVLPSI"])], tavg)
```

Each experiment directory is expected to hold the standard output of Octopus in a file called `out` (see the
`stdout` argument of `OctopusApplication`), and the `exec` and `profiling` directories Octopus writes. The parsed
data of each experiment is keyed by log name:

| Log          | File                           | Present when                         |
|--------------|--------------------------------|--------------------------------------|
| `time`       | `profiling/time.000000`        | always                               |
| `time_tree`  | `profiling/time.000000.tree`   | Octopus wrote the call tree          |
| `time_ranks` | `profiling/time.??????`        | all ranks wrote their profile        |

### Generating and running scaling experiments

```python
from access.config.parallel_allocation_strategies import RootAllocation
from access.profiling.control import ExistingDirectoryControlSource

app = OctopusApplication(name="ch4", n_states=8)
manager = OctopusManager(
    Path("experiments"),
    Path("archive"),
    app,
    ExistingDirectoryControlSource(Path("/path/to/control")),
    mpirun_args=["--bind-to", "core"],
    preamble="module load Octopus",
)

# One experiment per valid layout on 1 and 2 nodes of 4 cores, with 1 or 2 OpenMP threads per process
manager.generate_scaling_experiments(
    [1, 2], cores_per_node=4, walltime=0.5, allocations=RootAllocation(thread_range=(1, 2))
)
manager.run_experiments()
manager.parse_profiling_data()
manager.plot_scaling_data(
    [RegionGroup("time", ["COMPLETE_RUN"])],
    tavg,
    experiments=manager.select_best_experiments("time", "COMPLETE_RUN", tavg),
)
```

Each experiment is a copy of the control directory, without the outputs of any previous run of it, with
`ParDomains`, `ParStates`, `ParKPoints`, `ProfilingMode` and `ProfilingAllNodes` set in its `inp` file. It also
holds a `run.sh` script running Octopus with the right number of processes and threads, which can be run by hand
as well. Experiments are run one after the other, and `run_experiments` blocks until all of them are done.

## Development

```shell
pytest
ruff check src tests
ruff format --check src tests
```

## License

This project is licensed under the Mozilla Public License 2.0. See [LICENSE](LICENSE).
