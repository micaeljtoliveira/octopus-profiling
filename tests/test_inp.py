# Copyright 2026 Micael Oliveira
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import pytest
from access.config import Config, UnsupportedEntryError
from conftest import INPUT_FILE
from lark.exceptions import UnexpectedCharacters

from octopus_profiling.inp import (
    OctopusInputParser,
    full_name,
    get_variable,
    namespace_candidates,
    set_variables,
)

# An input file of a multisystem calculation, with everything the format can hold
MULTISYSTEM = """\
# Two systems
CalculationMode = td
include extra.inp

%Systems
  "Earth" | classical_particle
  "Moon"  | classical_particle
%

Spacing = 0.5
TDTimeStep = 0.05/27.2   # in Hartree
Name = "run # 1"
Earth.Mass = 5.97e24
Earth.ParStates = 2
%Earth.Coordinates
  0 | 0 | 0
  1 | 0 | 0
%
Moon.Spacing = 0.1*angstrom
%Moon.Species
  "O" | species_pseudo | file | "O#1.psf"
%
Solar.Earth.Radius = 6371
%Empty
%
"""


@pytest.fixture(scope="module")
def parser():
    return OctopusInputParser()


def roundtrip(parser, config):
    """Checks that the text of a configuration reads back as the same configuration."""
    assert dict(parser.parse(str(config))) == dict(config)


def test_roundtrip(parser):
    """Test that files are written back exactly as they were read."""
    for text in (INPUT_FILE, MULTISYSTEM):
        assert str(parser.parse(text)) == text


def test_content(parser):
    """Test that each kind of value is read as the type it is."""
    config = parser.parse(MULTISYSTEM)

    assert config == {
        "calculationmode": "td",
        "systems": ['"Earth" | classical_particle', '"Moon"  | classical_particle'],
        "spacing": 0.5,
        "tdtimestep": "0.05/27.2",
        "name": '"run # 1"',
        "earth": {"mass": 5.97e24, "parstates": 2, "coordinates": ["0 | 0 | 0", "1 | 0 | 0"]},
        "moon": {"spacing": "0.1*angstrom", "species": '"O" | species_pseudo | file | "O#1.psf"'},
        "solar": {"earth": {"radius": 6371}},
        "empty": None,
    }
    assert isinstance(config["earth"], Config)


def test_case_insensitive(parser):
    """Test that names are stored in lower case and found in any case, as Octopus does."""
    config = parser.parse(MULTISYSTEM)

    assert config["CALCULATIONMODE"] == "td"
    assert config["Earth"]["ParStates"] == 2


def test_edit(parser):
    """Test that edits change only the values they change."""
    config = parser.parse(MULTISYSTEM)
    config["spacing"] = 0.4
    config["CalculationMode"] = "gs"
    config["earth"]["parstates"] = 4
    config["moon"]["spacing"] = "0.2*angstrom"
    config["earth"]["coordinates"][1] = "2 | 0 | 0"

    assert str(config) == (
        MULTISYSTEM.replace("Spacing = 0.5", "Spacing = 0.4")
        .replace("= td", "= gs")
        .replace("ParStates = 2", "ParStates = 4")
        .replace("0.1*angstrom", "0.2*angstrom")
        .replace("  1 | 0 | 0", "  2 | 0 | 0")
    )
    roundtrip(parser, config)


def test_add(parser):
    """Test that new variables and blocks are written with their whole dotted names."""
    config = parser.parse("Radius = 4\nA.X = 1\n")
    config["ParDomains"] = 2
    config["Spacing"] = "0.1*angstrom"
    config["ProfilingMode"] = "prof_time"
    config["File"] = '"x.xyz"'
    config["a"]["Y"] = "yes"
    config["a"]["B"] = {"Z": 2}
    config["Earth"] = {"Mass": 1, "Coordinates": ["0 | 0", "1 | 0"]}
    config["a"]["b"]["Empty"] = None

    assert str(config) == (
        "Radius = 4\n"
        "A.X = 1\n"
        "A.Y = yes\n"
        "A.B.Z = 2\n"
        "%A.B.Empty\n"
        "%\n"
        "ParDomains = 2\n"
        "Spacing = 0.1*angstrom\n"
        "ProfilingMode = prof_time\n"
        'File = "x.xyz"\n'
        "Earth.Mass = 1\n"
        "%Earth.Coordinates\n"
        "  0 | 0\n"
        "  1 | 0\n"
        "%\n"
    )
    roundtrip(parser, config)


def test_resize(parser):
    """Test that a block grows and shrinks, the rows that stay written as they were."""
    config = parser.parse(MULTISYSTEM)
    config["earth"]["coordinates"].append("2 | 0 | 0")
    assert "  1 | 0 | 0\n  2 | 0 | 0\n%\n" in str(config)
    roundtrip(parser, config)

    config["systems"] = ['"Earth" | classical_particle', '"Moon"  | classical_particle', '"Sun" | classical_particle']
    del config["earth"]["coordinates"][0]
    assert "%Earth.Coordinates\n  1 | 0 | 0\n  2 | 0 | 0\n%\n" in str(config)
    assert '"Moon"  | classical_particle\n  "Sun" | classical_particle\n%' in str(config)
    roundtrip(parser, config)


def test_delete(parser):
    config = parser.parse(MULTISYSTEM)
    del config["earth"]["mass"]

    assert str(config) == MULTISYSTEM.replace("Earth.Mass = 5.97e24\n", "")


def test_malformed(parser):
    with pytest.raises(UnexpectedCharacters):
        parser.parse("Radius = 4\n~\n")


def test_variable_named_like_namespace(parser):
    """Test that a variable sharing its name with a namespace is refused, as it cannot be represented."""
    with pytest.raises(ValueError, match="namespace"):
        parser.parse("a = 6.0*angstrom\n%A.Spacing\n  a | a | a\n%\n")


def test_type_change(parser):
    config = parser.parse(MULTISYSTEM)
    with pytest.raises(TypeError):
        config["spacing"] = "0.5*angstrom"


def test_no_logicals(parser):
    """Test that a logical is refused, as Octopus writes it as a name."""
    config = parser.parse(MULTISYSTEM)
    with pytest.raises(UnsupportedEntryError):
        config["ProfilingAllNodes"] = True


def test_quoted_comment_character(parser):
    """Test that a row holding a "#" inside quotes cannot be rewritten."""
    config = parser.parse('%Species\n  "O" | file | "O#1.psf"\n  "H" | 1\n%\n')
    with pytest.raises(TypeError):
        config["species"][1] = '"H" | 2'


@pytest.mark.parametrize(
    ("namespace", "expected"),
    [
        ("", []),
        ("A", [("A",)]),
        ("A.B.C", [("A", "B", "C"), ("B", "C"), ("C",), ("A", "B"), ("B",), ("A",)]),
    ],
)
def test_namespace_candidates(namespace, expected):
    """Test the order in which Octopus looks for a variable, from its parse_get_full_name."""
    assert list(namespace_candidates(namespace)) == expected


@pytest.fixture(scope="module")
def nested(parser):
    return parser.parse("X = 0\nA.X = 1\nA.B.X = 2\nC.X = 3\nB.C.X = 4\nY = 5\nZ.Y.W = 6\n")


@pytest.mark.parametrize(
    ("name", "namespace", "expected_name", "expected_value"),
    [
        ("X", "", "X", 0),
        ("X", "A", "A.X", 1),
        ("X", "A.B", "A.B.X", 2),
        # "B.C.X" is tried before "C.X", and both before the ancestors "A.B" and "A"
        ("X", "A.B.C", "B.C.X", 4),
        ("X", "D.C", "C.X", 3),
        ("X", "D.B", "X", 0),
        ("Y", "A.B", "Y", 5),
        # "Z.Y" is a namespace, not the variable "Y"
        ("Y", "Z", "Y", 5),
        ("W", "A", "W", None),
    ],
)
def test_resolution(nested, name, namespace, expected_name, expected_value):
    """Test that a variable read in a namespace resolves as Octopus resolves it."""
    assert full_name(nested, name, namespace) == expected_name
    assert get_variable(nested, name, namespace) == expected_value


def test_get_variable_default(nested):
    assert get_variable(nested, "W", "A", default="none") == "none"


class TestSetVariables:
    def test_replace(self):
        """Test that a variable is changed in place, whatever the case it is written in."""
        text = set_variables(INPUT_FILE, {"ParStates": 2, "ProfilingMode": "prof_time"})

        assert text == INPUT_FILE.replace("parstates = 0", "parstates = 2").replace("prof_memory", "prof_time")

    def test_append(self):
        text = set_variables(INPUT_FILE, {"ParDomains": 4, "ProfilingAllNodes": "yes"})

        assert text == INPUT_FILE + "ParDomains = 4\nProfilingAllNodes = yes\n"

    def test_type_change(self):
        """Test that a value of another type replaces the old one at the end of the file."""
        text = set_variables("ParDomains = auto\nRadius = 4\n", {"ParDomains": 2})

        assert text == "Radius = 4\nParDomains = 2\n"

    def test_namespaces(self):
        """Test that a variable added to a namespace is written after the last line of that namespace."""
        text = set_variables(
            MULTISYSTEM, {"Earth.ParStates": 1, "Moon.ParStates": 3, "Solar.Earth.Mass": 1, "Mars.Mass": 2}
        )

        assert text == (
            MULTISYSTEM.replace("Earth.ParStates = 2", "Earth.ParStates = 1")
            .replace('"O#1.psf"\n%\n', '"O#1.psf"\n%\nMoon.ParStates = 3\n')
            .replace("Radius = 6371\n", "Radius = 6371\nSolar.Earth.Mass = 1\n")
            + "Mars.Mass = 2\n"
        )

    def test_namespace_as_variable(self):
        with pytest.raises(ValueError, match="'Earth' is a namespace"):
            set_variables(MULTISYSTEM, {"Earth": 1})

    def test_variable_as_namespace(self):
        with pytest.raises(ValueError, match="'Spacing' in 'Spacing.X' is a variable"):
            set_variables(MULTISYSTEM, {"Spacing.X": 1})
