# Copyright 2026 Micael Oliveira
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Parsers for the time profiling files written by Octopus.

When ProfilingMode is set, Octopus writes the following files to the "profiling" directory:

    time.NNNNNN       flat profile of rank NNNNNN, one line per region, sorted by self time
    time.NNNNNN.tree  call tree of rank NNNNNN

Only rank 0 writes them, unless ProfilingAllNodes is set to yes.
"""

import math
import os
import re
from pathlib import Path

from access.profiling.metrics import count, tavg, tfrac, tmin
from access.profiling.parser import ProfilingData, ProfilingParser, _read_text_file, _test_file

from octopus_profiling.metrics import (
    bandwidth,
    pfrac,
    sbandwidth,
    stavg,
    stfrac,
    sthroughput,
    sttot,
    throughput,
    ttot,
)

__all__ = [
    "OctopusProfilingParser",
    "OctopusRankProfilingParser",
    "OctopusTreeProfilingParser",
    "stavg",
    "stfrac",
]

# Columns of the flat profile, in order. The cumulative ones come before the "|" separator, the self ones after it.
_CUMULATIVE_COLUMNS = (count, ttot, tavg, tmin, throughput, bandwidth, tfrac)
_SELF_COLUMNS = (sttot, stavg, sthroughput, sbandwidth, stfrac)

# Widths of the columns, as Octopus writes them with a fixed Fortran format. Columns are not always parted by spaces: a
# value as wide as its column runs into the one before it (e.g. "19.31500272108326.5" for MIN_TIME and MFLOPS), and a
# value too wide for its column is written as asterisks. The region name, before the cumulative columns, is the only
# one of variable width, so the columns are counted from the "|" separator.
_CUMULATIVE_WIDTHS = (14, 16, 16, 16, 10, 10, 8)
_SELF_WIDTHS = (16, 16, 10, 10, 8)


def _split_columns(text: str, widths: tuple[int, ...]) -> list[str]:
    """Splits right-aligned text into columns of the given widths, counted from its end."""
    fields = []
    end = len(text)
    for width in reversed(widths):
        fields.append(text[max(0, end - width) : end].strip())
        end -= width
    return fields[::-1]


def _fortran_number(field: str) -> float:
    """Converts a field written by Fortran to a number, NaN if it overflowed its column."""
    if field and set(field) == {"*"}:
        return math.nan
    return float(field)


class OctopusProfilingParser(ProfilingParser):
    """Parser for the flat profile Octopus writes to profiling/time.NNNNNN."""

    _metrics = [*_CUMULATIVE_COLUMNS, *_SELF_COLUMNS]

    def parse(self, file_path: str | Path | os.PathLike) -> ProfilingData:
        """Parses an Octopus flat profile.

        Args:
            file_path (str | Path | os.PathLike): Path to the time.NNNNNN file.

        Returns:
            ProfilingData: Profiling data in the flat format. Time fractions are percentages.

        Raises:
            TypeError: If file_path is not a valid path.
            FileNotFoundError: If file_path is not a file or doesn't exist.
            ValueError: If file_path is not a text file or does not contain any Octopus profiling data.
        """
        stream = _read_text_file(file_path)

        results: ProfilingData = {"region": []}
        for metric in self.metrics:
            results[metric] = []

        # The data follows the line of "=" closing the header
        in_data = False
        for line in stream.splitlines():
            if not in_data:
                in_data = line.startswith("====")
                continue

            cumulative, sep, self_part = line.partition("|")
            cumulative = cumulative.rstrip()
            self_part = self_part.rstrip()
            n_cumulative = sum(_CUMULATIVE_WIDTHS)
            if not sep or len(cumulative) <= n_cumulative or len(self_part) < sum(_SELF_WIDTHS):
                continue

            region = cumulative[:-n_cumulative].strip()
            fields = _split_columns(cumulative, _CUMULATIVE_WIDTHS) + _split_columns(self_part, _SELF_WIDTHS)
            if not region or len(region.split()) != 1:
                continue
            try:
                values = [_fortran_number(field) for field in fields]
            except ValueError:
                continue
            # The number of calls is an integer, unless it overflowed its column
            if not math.isnan(values[0]):
                values[0] = int(values[0])

            results["region"].append(region)
            for metric, value in zip(self.metrics, values, strict=True):
                results[metric].append(value)

        if not results["region"]:
            raise ValueError("No Octopus profiling data found")

        return results


# A line of the call tree: one "-" per level below the root, a "|", the region name, the percentage of the total time,
# the percentage of the parent time, and the time.
_TREE_LINE = re.compile(
    r"^(?P<depth>-*)\|\s*(?P<region>\S+)\s+(?P<tfrac>[\d.]+)%\s+(?P<pfrac>[\d.]+)%\s+(?P<ttot>[\d.eE+-]+)\s*$"
)


class OctopusTreeProfilingParser(ProfilingParser):
    """Parser for the call tree Octopus writes to profiling/time.NNNNNN.tree.

    The data is returned in the hierarchical format, with each region nested inside the region that called it.
    """

    _metrics = [ttot, tfrac, pfrac]

    def parse(self, file_path: str | Path | os.PathLike) -> ProfilingData:
        """Parses an Octopus call tree.

        Args:
            file_path (str | Path | os.PathLike): Path to the time.NNNNNN.tree file.

        Returns:
            ProfilingData: Profiling data in the hierarchical format. Time fractions are percentages.

        Raises:
            TypeError: If file_path is not a valid path.
            FileNotFoundError: If file_path is not a file or doesn't exist.
            ValueError: If file_path is not a text file, does not contain any call tree, or if the tree skips a
                level.
        """
        stream = _read_text_file(file_path)

        result: ProfilingData = {}
        # Stack of the regions enclosing the current line, with their depth
        stack: list[tuple[dict, int]] = [(result, -1)]
        for line in stream.splitlines():
            match = _TREE_LINE.match(line)
            if not match:
                continue

            depth = len(match.group("depth"))
            while stack[-1][1] >= depth:
                stack.pop()
            if stack[-1][1] != depth - 1:
                raise ValueError(f"Region '{match.group('region')}' in {file_path} has no parent region.")

            node = stack[-1][0].setdefault(match.group("region"), {})
            node.update(
                {
                    ttot: float(match.group("ttot")),
                    tfrac: float(match.group("tfrac")),
                    pfrac: float(match.group("pfrac")),
                }
            )
            stack.append((node, depth))

        if not result:
            raise ValueError("No Octopus call tree found")

        return result


_RANK_FILE = re.compile(r"^time\.(\d{6})$")


class OctopusRankProfilingParser(ProfilingParser):
    """Parser for the flat profiles of all MPI ranks.

    Octopus writes a flat profile per rank when ProfilingAllNodes is set. This parser is given any of them, reads all
    the time.NNNNNN files in the same directory, and returns the data in the per-PE format, which can be reduced with
    access.profiling.parser.aggregate_pe_data to study the load balance.

    Regions are not necessarily the same in all ranks. The data of a rank that does not have a given region is NaN.
    """

    _metrics = OctopusProfilingParser._metrics

    def parse(self, file_path: str | Path | os.PathLike) -> ProfilingData:
        """Parses the flat profiles of all MPI ranks.

        Args:
            file_path (str | Path | os.PathLike): Path to one of the time.NNNNNN files.

        Returns:
            ProfilingData: Profiling data in the per-PE format, with the PE being the MPI rank.

        Raises:
            TypeError: If file_path is not a valid path.
            FileNotFoundError: If file_path is not a file or doesn't exist.
            ValueError: If one of the files does not contain any Octopus profiling data.
        """
        path = _test_file(file_path)

        rank_files = {}
        for candidate in path.parent.iterdir():
            match = _RANK_FILE.match(candidate.name)
            if match and candidate.is_file():
                rank_files[int(match.group(1))] = candidate

        flat_parser = OctopusProfilingParser()
        ranks = sorted(rank_files)
        rank_data = [flat_parser.parse(rank_files[rank]) for rank in ranks]

        # Keep the regions in the order they first appear, starting with the lowest rank
        regions = list(dict.fromkeys(region for data in rank_data for region in data["region"]))

        results: ProfilingData = {"region": regions, "pe": ranks}
        for metric in self.metrics:
            results[metric] = [[] for _ in regions]

        for data in rank_data:
            index = {region: i for i, region in enumerate(data["region"])}
            for metric in self.metrics:
                for row, region in zip(results[metric], regions, strict=True):
                    row.append(float(data[metric][index[region]]) if region in index else math.nan)

        return results
