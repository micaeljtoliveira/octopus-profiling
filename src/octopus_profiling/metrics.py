# Copyright 2026 Micael Oliveira
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Profiling metrics specific to Octopus.

These complement the generic metrics defined in access.profiling.metrics (count, tavg, tmin, tfrac, ...). Octopus
reports every region twice: once including the time spent in its subregions (cumulative) and once excluding it
(self). The metrics prefixed with "s" are the self counterparts of the cumulative ones.

Octopus reports throughputs in MFLOPS, but pint has no unit for floating point operations, so these are expressed
as millions of counts per second.
"""

from access.profiling.metrics import ProfilingMetric
from pint import Unit

# Cumulative metrics, i.e., including the time spent in subregions
ttot = ProfilingMetric("total time", Unit("second"), "Total time spent in region, summed over all calls")
throughput = ProfilingMetric("throughput", Unit("megacount/second"), "Floating point operations per second in region")
bandwidth = ProfilingMetric("bandwidth", Unit("megabyte/second"), "Memory bandwidth in region")

# Self metrics, i.e., excluding the time spent in subregions
sttot = ProfilingMetric(
    "total self-time", Unit("second"), "Total time spent in region, summed over all calls, excluding subregions"
)
stavg = ProfilingMetric(
    "average self-time", Unit("second"), "Average time spent in region, excluding calls to subregions"
)
sthroughput = ProfilingMetric(
    "self throughput", Unit("megacount/second"), "Floating point operations per second in region, excluding subregions"
)
sbandwidth = ProfilingMetric(
    "self bandwidth", Unit("megabyte/second"), "Memory bandwidth in region, excluding subregions"
)
stfrac = ProfilingMetric(
    "self-time fraction", Unit("%"), "Fraction of total time spent in region, excluding calls to subregions"
)

# Call tree metrics
pfrac = ProfilingMetric("parent time fraction", Unit("%"), "Fraction of the parent region time spent in region")
