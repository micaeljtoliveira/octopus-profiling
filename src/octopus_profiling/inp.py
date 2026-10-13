# Copyright 2026 Micael Oliveira
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Minimal reading and editing of Octopus input files.

Octopus input files are made of variable assignments ("Name = value") and blocks delimited by lines starting with
"%". Variable names are case-insensitive and "#" starts a comment. Values can be arbitrary expressions, so they are
kept as strings.

This is not a full parser: only top-level variable assignments are read or changed, and everything else, blocks
included, is left untouched.
"""

import re
from collections.abc import Mapping

_ASSIGNMENT = re.compile(r"^\s*(?P<name>[A-Za-z_]\w*)\s*=\s*(?P<value>.*?)\s*$")


def _strip_comment(line: str) -> str:
    """Removes a trailing comment from a line, ignoring "#" inside double-quoted strings."""
    in_string = False
    for i, char in enumerate(line):
        if char == '"':
            in_string = not in_string
        elif char == "#" and not in_string:
            return line[:i]
    return line


def _assignments(lines: list[str]):
    """Yields the index, name and value of every top-level variable assignment in the lines."""
    in_block = False
    for i, line in enumerate(lines):
        code = _strip_comment(line)
        if code.strip().startswith("%"):
            in_block = not in_block
            continue
        if in_block:
            continue
        match = _ASSIGNMENT.match(code)
        if match:
            yield i, match.group("name"), match.group("value")


def read_variables(text: str) -> dict[str, str]:
    """Returns the top-level variables set in an Octopus input file.

    Args:
        text (str): Contents of the input file.

    Returns:
        dict[str, str]: Values of the variables, keyed by their lower-case names. If a variable is set more than
            once, the last value is returned.
    """
    return {name.lower(): value for _, name, value in _assignments(text.splitlines())}


def _format_value(value: object) -> str:
    """Formats a value as Octopus expects it."""
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def set_variables(text: str, values: Mapping[str, object]) -> str:
    """Sets top-level variables in an Octopus input file.

    Variables already set in the file are changed in place, matching their names case-insensitively, while the others
    are appended at the end of the file.

    Args:
        text (str): Contents of the input file.
        values (Mapping[str, object]): New values of the variables, keyed by name. Booleans are written as "yes" or
            "no", everything else as str() writes it.

    Returns:
        str: The new contents of the input file.
    """
    lines = text.splitlines()
    pending = {name.lower(): (name, value) for name, value in values.items()}

    for i, name, _ in list(_assignments(lines)):
        if name.lower() in pending:
            requested_name, value = pending[name.lower()]
            lines[i] = f"{requested_name} = {_format_value(value)}"

    # Variables set more than once were changed everywhere above, so only the ones not found are left to append
    found = {name.lower() for _, name, _ in _assignments(lines)}
    missing = [(name, value) for key, (name, value) in pending.items() if key not in found]
    if missing:
        if lines and lines[-1].strip():
            lines.append("")
        lines.extend(f"{name} = {_format_value(value)}" for name, value in missing)

    return "\n".join(lines) + "\n"
