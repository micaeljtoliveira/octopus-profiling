# Copyright 2026 Micael Oliveira
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Parser for Octopus input files.

The format is described in the Octopus manual:

https://www.octopus-code.org/documentation/main/manual/basics/input_file/

An input file assigns values to variables (``Name = value``) and fills blocks, which are tables
written between a line ``%Name`` and a line ``%``, one row per line and the columns of a row parted
by ``|``. Values are expressions, which Octopus evaluates, so ``Radius = 4.5*angstrom`` is as valid
as ``Radius = 8.5``. Names are case-insensitive and ``#`` starts a comment.

The parser reads a file as follows:

- a variable whose value is an integer or a real is read as an ``int`` or a ``float``;
- one whose value is a bare name (``gs``, ``yes``, ``eV_Angstrom``) is read as a ``str``;
- any other value is an expression, read as a ``str`` holding its text: ``"4.5*angstrom"``. A
  quoted string is an expression too and keeps its quotes, so a new string is assigned as
  ``'"file.xyz"'``. Logicals are names, ``"yes"`` and ``"no"``;
- a block is read as the list of its rows, each a ``str``, as written: ``['"C" | 0 | 0 | 0', ...]``.
  A block with a single row is read as that row, and an empty one as ``None``.

Variables and blocks may be scoped to a namespace, a feature of the Octopus multisystem framework:
``Earth.Mass = 1`` and ``%Earth.Coordinates`` belong to the system ``Earth``, and namespaces nest,
as in ``SolarSystem.Earth.Mass``. The input parser itself just stores the dotted names, and the
meaning is given by the code reading them: ``parse_get_full_name`` in ``src/basic/parser.F90``. A
namespaced variable is read here as a variable of a nested ``Config``, ``config["earth"]["mass"]``,
and ``full_name`` and ``get_variable`` resolve a name in a namespace the way Octopus does.

Note: expressions are not evaluated, and included files (``include file``) are kept but not read. A
variable named like a namespace (``a = 6.0*angstrom`` next to ``%A.Spacing``) cannot be represented,
since it would have to be a value and a nested ``Config`` at once, so such a file is refused. A row
or an expression holding a ``#`` inside quotes can be read but not rewritten. A new block needs at
least two rows, as a single string is written as a variable.
"""

from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from access.config import Config, ConfigParser
from access.config.grammar_contract import BLOCK_RULE, KEY_LIST, KEY_NULL, KEY_VALUE, START_RULE

# A quoted string, kept whole inside an expression so that a "#" in it is not taken for a comment
_QUOTED = r'"[^"\r\n]*"'


class OctopusInputParser(ConfigParser):
    """Octopus input file parser.

    Names are stored in lower case, as Octopus reads them case-insensitively.
    """

    @property
    def key_case(self) -> str:
        """str: ``"lower"``, as Octopus ignores the case of names."""
        return "lower"

    @property
    def block_rules(self) -> Sequence[str]:
        """Sequence[str]: Rules holding the contents of a namespace.

        Octopus has no delimited block of variables: a namespace is only ever written as a prefix of
        the names of its variables. The generic block rule, which this grammar does not define, is
        listed first to say just that, so that a variable added to a namespace is written as a line
        of its own carrying the whole dotted name.
        """
        return (BLOCK_RULE, "ns_body", "ns_table_body")

    @property
    def entry_templates(self) -> Mapping[tuple[str, str], str]:
        """Mapping[tuple[str, str], str]: The text of a new variable and of a new block.

        Written with the whole dotted name, so that one template serves any depth of namespace.
        """
        return {
            (START_RULE, KEY_VALUE): "{keys:.} = {value}\n",
            (START_RULE, KEY_LIST): "%{keys:.}\n  {value}\n  {value}\n%\n",
            (START_RULE, KEY_NULL): "%{keys:.}\n%\n",
        }

    @property
    def grammar(self) -> str:
        return r"""
// "start" must be a non-transparent rule: the root node is the container that new entries
// are added to, and a transparent rule collapses away for a single-entry file.
start: (empty_line | include | variable | table)*

?variable: key_value | ns_variable
key_value: ws* key eq ws* value line_end

// "A.X = 1" is the namespace "A" holding "X = 1", as a Fortran "a%b = 1" is a derived type
ns_variable: ws* key dot ns_body -> key_block
ns_body: key_value | ns_variable

// The number of rows of a block decides its category: none is a valueless entry, one a
// scalar, more a list.
?table: table_null | table_value | table_list | ns_table
table_null: ws* open tail_null
table_value: ws* open tail_value
table_list: ws* open tail_list

// The "%" comes before the namespaces in "%A.B.X", so the chain of a block is a rule of its
// own, holding what follows the "%".
ns_table: ws* open key dot ns_table_body -> key_block
ns_table_body: tail_null | tail_value | tail_list | ns_tail
ns_tail: key dot ns_table_body -> key_block

tail_null: key line_end empty_line* close -> key_null
tail_value: key line_end empty_line* ws* text_line line_end empty_line* close -> key_value
tail_list: key line_end empty_line* ws* text_line (row_sep ws* text_line)+ line_end empty_line* close -> key_list

include: ws* INCLUDE ws+ INCLUDED_FILE line_end
INCLUDE: "include"
INCLUDED_FILE: /[^\s#]+/

// Punctuation is kept in rules of its own, so that the reconstructor can tell a block from a
// variable, and the whitespace before "=" in "eq" so that it stays on its side of the "=".
open: "%"
dot: "."
row_sep: line_end empty_line*
close: ws* "%" line_end
eq: ws* "="

// Most specific first: anything that is not a number nor a name is an expression.
?value: integer | float | identifier | text_line

// An expression or a row of a block: its text without the whitespace around it, never
// starting with the "%" that closes a block, and with quoted strings kept whole.
text_line: TEXT
TEXT: /(?!%)(?:QUOTED|[^\s#"])(?:(?:QUOTED|[^\r\n#"])*(?:QUOTED|[^\s#"]))?/

key: KEY
KEY: /[A-Za-z_]\w*/

empty_line: line_end
line_end: (comment | ws*) NEWLINE

%import config.integer
%import config.float
%import config.identifier
%import config.comment
%import config.ws
%import config.NEWLINE
""".replace("QUOTED", _QUOTED)

    def parse(self, stream: str) -> Config:
        """Parse the given text.

        Args:
            stream (str): The text to parse.

        Returns:
            Config: The variables and blocks of the input file.

        Raises:
            ValueError: If a variable is named like a namespace.
            lark.exceptions.UnexpectedInput: If the text is not a valid input file.
        """
        try:
            return super().parse(stream)
        except ValueError as error:
            raise ValueError(
                f"{error}. Octopus allows a variable and a namespace to share a name, but this parser reads a "
                "namespace as a nested Config, so it cannot hold both."
            ) from error


def namespace_candidates(namespace: str) -> Iterator[tuple[str, ...]]:
    """Yields the namespaces a variable is looked for in, in the order Octopus looks.

    Octopus goes from the right-most ancestor of the namespace to the left-most, and for each
    ancestor from its most complete path to its least complete one. So for "A.B.C" the order is
    "A.B.C", "B.C", "C", "A.B", "B" and "A" (see ``parse_get_full_name`` in Octopus).

    Args:
        namespace (str): The namespace, its components parted by ".". Empty for the global one.

    Yields:
        tuple[str, ...]: The components of each candidate namespace.
    """
    parts = tuple(namespace.split(".")) if namespace else ()
    for end in range(len(parts), 0, -1):
        for start in range(end):
            yield parts[start:end]


def _lookup(config: Mapping, path: Sequence[str]) -> tuple[bool, Any]:
    """Returns whether a variable is defined at a path of names, and its value if it is."""
    container: Any = config
    for name in path:
        if not isinstance(container, Mapping) or name not in container:
            return False, None
        container = container[name]
    # A namespace is not a variable
    if isinstance(container, Mapping):
        return False, None
    return True, container


def full_name(config: Mapping, name: str, namespace: str = "") -> str:
    """Returns the name Octopus reads a variable under when it is read in a namespace.

    This is the variable in the first candidate namespace (see ``namespace_candidates``) that
    defines it, or the global variable if none does.

    Args:
        config (Mapping): The parsed input file.
        name (str): Name of the variable.
        namespace (str): The namespace the variable is read in, its components parted by ".".
            Empty, the default, for the global one.

    Returns:
        str: The dotted name of the variable, e.g. "Earth.Mass", or ``name`` itself.
    """
    for candidate in namespace_candidates(namespace):
        if _lookup(config, [*candidate, name])[0]:
            return ".".join([*candidate, name])
    return name


def get_variable(config: Mapping, name: str, namespace: str = "", default: Any = None) -> Any:
    """Returns the value of a variable, as Octopus reads it in a namespace.

    Args:
        config (Mapping): The parsed input file.
        name (str): Name of the variable.
        namespace (str): The namespace the variable is read in, its components parted by ".".
            Empty, the default, for the global one.
        default (Any): What to return if the variable is not defined. None by default.

    Returns:
        Any: The value of the variable, or ``default``.
    """
    defined, value = _lookup(config, full_name(config, name, namespace).split("."))
    return value if defined else default


def _assign(container: Config, path: Sequence[str], value: Any, full: str) -> None:
    """Assigns a value to the variable at a path of names below a container."""
    head, *rest = path
    if not rest:
        if isinstance(container.get(head), Config):
            raise ValueError(f"'{full}' is a namespace, not a variable.")
        try:
            container[head] = value
        except TypeError:
            # The value is of another type than the one it replaces, which cannot be written in place
            del container[head]
            container[head] = value
    elif head not in container:
        nested: Any = value
        for name in reversed(rest):
            nested = {name: nested}
        container[head] = nested
    elif isinstance(container[head], Config):
        _assign(container[head], rest, value, full)
    else:
        raise ValueError(f"'{head}' in '{full}' is a variable, not a namespace.")


def set_variables(text: str, values: Mapping[str, Any]) -> str:
    """Sets variables in an Octopus input file.

    Variables already set are changed where they are, unless the new value is of another type, in
    which case they are rewritten at the end of the file, like the variables not set yet.

    Args:
        text (str): Contents of the input file.
        values (Mapping[str, Any]): New values of the variables, keyed by name. A name may be
            namespaced, e.g. "Earth.ParStates".

    Returns:
        str: The new contents of the input file.

    Raises:
        ValueError: If a name is used both as a namespace and as a variable.
    """
    config = OctopusInputParser().parse(text)
    for name, value in values.items():
        _assign(config, name.split("."), value, name)
    return str(config)
