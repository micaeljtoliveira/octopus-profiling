# Copyright 2026 Micael Oliveira
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import matplotlib
import pytest

# Plots are made in tests, but never shown
matplotlib.use("Agg")

# Excerpt of a flat profile written by Octopus (profiling/time.000000)
FLAT_PROFILE = """\
                                                                                   CUMULATIVE TIME                                 |                         SELF TIME
                                          --------------------------------------------------------------------------|-------------------------------------------------------------
TAG                                          NUM_CALLS      TOTAL_TIME   TIME_PER_CALL        MIN_TIME    MFLOPS  MBYTES/S   %TIME |       TOTAL_TIME   TIME_PER_CALL    MFLOPS  MBYTES/S   %TIME
====================================================================================================================|=============================================================
DNL_OPERATOR_BATCH                                1133        6.714707        0.005926        0.002229   18780.8       0.0    24.6 |         6.714707        0.005926   18780.8       0.0    20.6
MESH_INIT_STAGE_2                                    1        5.939396        5.939396        5.939396       0.0       0.0    21.7 |         5.939396        5.939396       0.0       0.0    19.7
DVLPSI                                            1127        1.623787        0.001441        0.000233    1556.2    7564.8     7.1 |         1.623787        0.001441    1556.2    7564.8     7.1
COMPLETE_RUN                                         1       22.840441       22.840441       22.840441    6077.4     617.3   100.0 |         0.889086        0.889086       0.0       0.0     3.9
"""  # noqa: E501

# Excerpt of a call tree written by Octopus (profiling/time.000000.tree)
TREE_PROFILE = """\
Tree level, region                      % of total % of parent   Full time
| COMPLETE_RUN                            100.00%    100.00%       22.8404
-| ELECTRONS_CONSTRUCTOR                   20.84%     20.84%        4.7603
--| IONS_CONSTRUCTOR                        0.02%      0.11%        0.0052
---| SPECIES_FACTORY_INIT                   0.00%      1.98%        0.0001
--| MESH_INIT_STAGE_2                      20.81%     99.86%        4.7537
-| EPOT_GENERATE                            0.00%      0.00%        0.0010
--| SUBMESH_INIT                            0.00%     72.67%        0.0008
-| ION_ELEC_LOC_INTER                       0.16%      0.16%        0.0360
--| SUBMESH_INIT                            0.02%      9.59%        0.0035
// modeline for vim to enable folding (put in ~/.vimrc: set modeline modelineexpr)
// vim: fdm=expr fde=getline(v\\:lnum)=~'.*\\|.*'?len(split(getline(v\\:lnum))[0])-1\\:0
"""

# Excerpts of the standard output of Octopus
SERIAL_STDOUT = """\
Number of states         =        8
************************** Parallelization ***************************
Info: Octopus will run in *serial*
**********************************************************************
"""

PARALLEL_STDOUT = """\
Number of states         =        8
************************** Parallelization ***************************
Info: Octopus will run in *parallel*

      Number of processes           :       4
      Number of threads per process :       2

Info: Number of nodes in ParStates   group:     2 (           8)
Info: Number of nodes in ParDomains  group:     2 (      612525)
Info: Octopus will waste at least  0.00% of computer time.
**********************************************************************
"""

# Excerpt of an Octopus input file
INPUT_FILE = """\
CalculationMode = gs
UnitsOutput = eV_Angstrom

Radius = 4.5*angstrom
%Coordinates
  "C" | 0 | 0 | 0
%

parstates = 0  # set by hand
ProfilingMode = prof_memory
"""


def write_experiment(path, stdout=None, status=None, profile=True, ranks=1, exit_code=None):
    """Creates the files an Octopus run leaves in an experiment directory."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "inp").write_text(INPUT_FILE)
    if stdout is not None:
        (path / "out").write_text(stdout)
    if status is not None:
        (path / "exec").mkdir(exist_ok=True)
        (path / "exec" / f"oct-status-{status}").touch()
    if profile:
        (path / "profiling").mkdir(exist_ok=True)
        for rank in range(ranks):
            (path / "profiling" / f"time.{rank:06d}").write_text(FLAT_PROFILE)
        (path / "profiling" / "time.000000.tree").write_text(TREE_PROFILE)
    if exit_code is not None:
        (path / "exit_code").write_text(f"{exit_code}\n")
    return path


@pytest.fixture
def flat_profile_file(tmp_path):
    path = tmp_path / "time.000000"
    path.write_text(FLAT_PROFILE)
    return path


@pytest.fixture
def tree_profile_file(tmp_path):
    path = tmp_path / "time.000000.tree"
    path.write_text(TREE_PROFILE)
    return path
