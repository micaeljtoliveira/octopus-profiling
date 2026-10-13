# Copyright 2026 Micael Oliveira
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import math

import pytest
from access.profiling.experiment import ProfilingLog
from access.profiling.metrics import count, tavg, tfrac, tmin
from access.profiling.parser import aggregate_pe_data
from conftest import FLAT_PROFILE

from octopus_profiling.metrics import bandwidth, pfrac, sbandwidth, sthroughput, sttot, throughput, ttot
from octopus_profiling.profiling_parser import (
    OctopusProfilingParser,
    OctopusRankProfilingParser,
    OctopusTreeProfilingParser,
    stavg,
    stfrac,
)


@pytest.fixture(scope="module")
def octopus_profiling():
    """Expected parsed content of the flat profile."""
    return {
        "region": ["DNL_OPERATOR_BATCH", "MESH_INIT_STAGE_2", "DVLPSI", "COMPLETE_RUN"],
        count: [1133, 1, 1127, 1],
        ttot: [6.714707, 5.939396, 1.623787, 22.840441],
        tavg: [0.005926, 5.939396, 0.001441, 22.840441],
        tmin: [0.002229, 5.939396, 0.000233, 22.840441],
        throughput: [18780.8, 0.0, 1556.2, 6077.4],
        bandwidth: [0.0, 0.0, 7564.8, 617.3],
        tfrac: [24.6, 21.7, 7.1, 100.0],
        sttot: [6.714707, 5.939396, 1.623787, 0.889086],
        stavg: [0.005926, 5.939396, 0.001441, 0.889086],
        sthroughput: [18780.8, 0.0, 1556.2, 0.0],
        sbandwidth: [0.0, 0.0, 7564.8, 0.0],
        stfrac: [20.6, 19.7, 7.1, 3.9],
    }


def test_flat_parser(flat_profile_file, octopus_profiling):
    """Tests the Octopus flat profile parser."""
    results = OctopusProfilingParser().parse(flat_profile_file)

    assert results == octopus_profiling
    assert set(OctopusProfilingParser().metrics) == set(octopus_profiling) - {"region"}


def test_flat_parser_overflowing_columns(tmp_path):
    """Tests that values running into each other or overflowing their column are read from the column widths."""
    header = "".join(FLAT_PROFILE.splitlines(keepends=True)[:4])
    data = (
        "COMPLETE_RUN                                         1       19.315002       19.315002"
        "       19.31500272108326.5     771.2   100.0 |         0.817474        0.817474       0.0       0.0     4.2\n"
        "DSCALAPACK_HERK                                     29        0.665958        0.022964"
        "        0.020880**********       0.0     3.4 |         0.665958        0.022964**********       0.0     3.4\n"
    )
    path = tmp_path / "time.000000"
    path.write_text(header + data)

    results = OctopusProfilingParser().parse(path)

    assert results["region"] == ["COMPLETE_RUN", "DSCALAPACK_HERK"]
    assert results[count] == [1, 29]
    assert results[tmin] == [19.315002, 0.020880]
    assert results[throughput][0] == 72108326.5
    assert math.isnan(results[throughput][1])
    assert math.isnan(results[sthroughput][1])
    assert results[stfrac] == [4.2, 3.4]


def test_flat_parser_skips_other_lines(tmp_path):
    """Tests that lines after the header that are not data lines are skipped."""
    lines = FLAT_PROFILE.splitlines(keepends=True)
    data_line = lines[-1]
    path = tmp_path / "time.000000"
    path.write_text(
        "".join(lines[:4])
        + "too short | 1\n"
        + data_line.replace("COMPLETE_RUN", "TWO WORDS   ")
        + data_line.replace("6077.4", "  NaNx")
        + data_line
    )

    assert OctopusProfilingParser().parse(path)["region"] == ["COMPLETE_RUN"]


def test_flat_parser_no_data(tmp_path):
    """Tests that a file without profiling data is rejected."""
    path = tmp_path / "time.000000"
    path.write_text(FLAT_PROFILE.split("=====")[0])

    with pytest.raises(ValueError, match="No Octopus profiling data found"):
        OctopusProfilingParser().parse(path)


def test_flat_parser_missing_file(tmp_path):
    """Tests that a missing file is reported."""
    with pytest.raises(FileNotFoundError):
        OctopusProfilingParser().parse(tmp_path / "time.000000")


def test_flat_profiling_log(flat_profile_file):
    """Tests that the flat profile can be turned into a dataset."""
    ds = ProfilingLog(flat_profile_file, OctopusProfilingParser()).parse()

    assert ds.sizes == {"region": 4}
    assert ds[tfrac].sel(region="COMPLETE_RUN").pint.dequantify().item() == 100.0
    assert str(ds[throughput].pint.units) == "megacount / second"


def test_tree_parser(tree_profile_file):
    """Tests the Octopus call tree parser."""
    results = OctopusTreeProfilingParser().parse(tree_profile_file)

    assert list(results) == ["COMPLETE_RUN"]
    root = results["COMPLETE_RUN"]
    assert root[ttot] == 22.8404
    assert root[tfrac] == 100.0
    assert [key for key in root if isinstance(key, str)] == [
        "ELECTRONS_CONSTRUCTOR",
        "EPOT_GENERATE",
        "ION_ELEC_LOC_INTER",
    ]
    ions = root["ELECTRONS_CONSTRUCTOR"]["IONS_CONSTRUCTOR"]
    assert ions[pfrac] == 0.11
    assert ions["SPECIES_FACTORY_INIT"] == {ttot: 0.0001, tfrac: 0.0, pfrac: 1.98}
    assert root["ION_ELEC_LOC_INTER"]["SUBMESH_INIT"][pfrac] == 9.59


def test_tree_profiling_log(tree_profile_file):
    """Tests that the call tree is flattened, giving unique names to regions called from several places."""
    ds = ProfilingLog(tree_profile_file, OctopusTreeProfilingParser()).parse()

    assert ds.region.values.tolist() == [
        "COMPLETE_RUN",
        "ELECTRONS_CONSTRUCTOR",
        "IONS_CONSTRUCTOR",
        "SPECIES_FACTORY_INIT",
        "MESH_INIT_STAGE_2",
        "EPOT_GENERATE",
        "SUBMESH_INIT",
        "ION_ELEC_LOC_INTER",
        "SUBMESH_INIT_2",
    ]
    assert ds[pfrac].sel(region="SUBMESH_INIT_2").pint.dequantify().item() == 9.59


def test_tree_parser_no_data(tmp_path):
    """Tests that a file without a call tree is rejected."""
    path = tmp_path / "time.000000.tree"
    path.write_text("Tree level, region                      % of total % of parent   Full time\n")

    with pytest.raises(ValueError, match="No Octopus call tree found"):
        OctopusTreeProfilingParser().parse(path)


def test_tree_parser_missing_parent(tmp_path):
    """Tests that a region deeper than the one before it by more than one level is rejected."""
    path = tmp_path / "time.000000.tree"
    path.write_text("| COMPLETE_RUN   100.00%    100.00%       22.8404\n--| ORPHAN   1.00%    1.00%       0.1\n")

    with pytest.raises(ValueError, match="no parent"):
        OctopusTreeProfilingParser().parse(path)


@pytest.fixture
def rank_profiles(tmp_path):
    """Flat profiles of two ranks, listing their regions in different orders, one of them missing a region."""
    lines = FLAT_PROFILE.splitlines()
    header, data = lines[:4], lines[4:]
    (tmp_path / "time.000000").write_text("\n".join(header + data) + "\n")
    rank1 = [line.replace("1133", "1000") for line in reversed(data) if not line.startswith("DVLPSI")]
    (tmp_path / "time.000001").write_text("\n".join(header + rank1) + "\n")
    (tmp_path / "time.000000.tree").write_text("not a flat profile\n")
    return tmp_path / "time.000000"


def test_rank_parser(rank_profiles):
    """Tests the parser of the flat profiles of all ranks."""
    results = OctopusRankProfilingParser().parse(rank_profiles)

    assert results["pe"] == [0, 1]
    assert results["region"] == ["DNL_OPERATOR_BATCH", "MESH_INIT_STAGE_2", "DVLPSI", "COMPLETE_RUN"]
    assert results[count][0] == [1133.0, 1000.0]
    assert results[tavg][3] == [22.840441, 22.840441]
    assert results[tavg][2][0] == 0.001441
    assert math.isnan(results[tavg][2][1])


def test_rank_profiling_log(rank_profiles):
    """Tests that the profiles of all ranks can be turned into a dataset and aggregated over ranks."""
    ds = ProfilingLog(rank_profiles, OctopusRankProfilingParser()).parse()
    assert ds.sizes == {"region": 4, "pe": 2}

    aggregated = aggregate_pe_data(ds)
    assert aggregated["count_max_pe"].sel(region="DNL_OPERATOR_BATCH").pint.dequantify().item() == 1133.0
    assert aggregated["count_min_pe"].sel(region="DNL_OPERATOR_BATCH").pint.dequantify().item() == 1000.0
