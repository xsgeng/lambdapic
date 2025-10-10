from __future__ import annotations

import importlib
import inspect
import logging
import pkgutil
import textwrap
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)


@dataclass
class DocRecord:
    """Container describing a documented symbol."""

    name: str
    qualified_name: str
    summary: str = ""
    kind: str = "symbol"
    stage: Optional[str] = None


@dataclass
class DocumentationIndex:
    """Discover and cache LambdaPIC documentation metadata."""

    simulations: List[DocRecord] = field(default_factory=list)
    callbacks: List[DocRecord] = field(default_factory=list)
    docs: Dict[str, str] = field(default_factory=dict)
    sources: Dict[str, str] = field(default_factory=dict)
    import_errors: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.refresh()

    def refresh(self) -> None:
        """Re-scan LambdaPIC for simulations and callbacks."""
        self.docs.clear()
        self.sources.clear()
        self.simulations = self._discover_simulations()
        self.callbacks = self._discover_callbacks()

    def _discover_simulations(self) -> List[DocRecord]:
        records: List[DocRecord] = []
        try:
            module = importlib.import_module("lambdapic.simulation")
        except Exception as exc:  # pragma: no cover - catastrophic failure
            logger.exception("Failed to import lambdapic.simulation: %s", exc)
            self.import_errors["lambdapic.simulation"] = str(exc)
            return records

        for attr_name, attr in inspect.getmembers(module):
            if attr_name.startswith("_"):
                continue
            if not inspect.isclass(attr):
                continue
            if attr.__module__ != module.__name__:
                continue

            qualified_name = f"{module.__name__}.{attr_name}"
            doc = inspect.getdoc(attr) or ""
            summary = doc.splitlines()[0] if doc else ""
            records.append(
                DocRecord(
                    name=attr_name,
                    qualified_name=qualified_name,
                    summary=summary,
                    kind="class",
                )
            )
            self.docs[qualified_name] = doc

        records.sort(key=lambda record: record.name.lower())
        return records

    def _discover_callbacks(self) -> List[DocRecord]:
        records: List[DocRecord] = []
        try:
            package = importlib.import_module("lambdapic.callback")
            callback_module = importlib.import_module("lambdapic.callback.callback")
            callback_base = getattr(callback_module, "Callback")
        except Exception as exc:  # pragma: no cover - catastrophic failure
            logger.exception("Failed to import callback package: %s", exc)
            self.import_errors["lambdapic.callback"] = str(exc)
            return records

        for module_info in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
            try:
                module = importlib.import_module(module_info.name)
            except Exception as exc:
                logger.warning("Skipping callback module %s: %s", module_info.name, exc)
                self.import_errors[module_info.name] = str(exc)
                continue

            for attr_name in dir(module):
                if attr_name.startswith("_"):
                    continue
                qualified_name = f"{module_info.name}.{attr_name}"
                try:
                    attr = getattr(module, attr_name)
                except AttributeError:
                    continue

                summary = ""
                stage: Optional[str] = None
                doc = ""

                if inspect.isclass(attr) and issubclass(attr, callback_base) and attr is not callback_base:
                    doc = inspect.getdoc(attr) or ""
                    summary = doc.splitlines()[0] if doc else ""
                    stage = getattr(attr, "stage", None)
                    records.append(
                        DocRecord(
                            name=attr_name,
                            qualified_name=qualified_name,
                            summary=summary,
                            kind="class",
                            stage=stage,
                        )
                    )
                    self.docs[qualified_name] = doc
                elif inspect.isfunction(attr) and getattr(attr, "stage", None) is not None:
                    doc = inspect.getdoc(attr) or ""
                    summary = doc.splitlines()[0] if doc else ""
                    stage = getattr(attr, "stage", None)
                    records.append(
                        DocRecord(
                            name=attr_name,
                            qualified_name=qualified_name,
                            summary=summary,
                            kind="function",
                            stage=stage,
                        )
                    )
                    self.docs[qualified_name] = doc

        records.sort(key=lambda record: (record.name.lower(), record.qualified_name))
        return records

    def get_doc(self, symbol: str) -> Optional[str]:
        """Return the docstring for a fully-qualified symbol name."""
        if symbol in self.docs:
            return self.docs[symbol]

        module_name, _, attr_name = symbol.rpartition(".")
        if not module_name:
            return None

        try:
            module = importlib.import_module(module_name)
            attr = getattr(module, attr_name)
        except Exception:
            return None

        doc = inspect.getdoc(attr) or ""
        self.docs[symbol] = doc
        return doc

    def get_source(self, symbol: str) -> Optional[str]:
        """Return the source code for a fully-qualified symbol name."""
        if symbol in self.sources:
            return self.sources[symbol]

        module_name, _, attr_name = symbol.rpartition(".")
        if not module_name:
            return None

        try:
            module = importlib.import_module(module_name)
            attr = getattr(module, attr_name)
        except Exception:
            return None

        try:
            source = inspect.getsource(attr)
        except (OSError, TypeError):
            return None

        self.sources[symbol] = source
        return source


def _format_import_notes(import_errors: Dict[str, str]) -> List[str]:
    lines: List[str] = []
    if import_errors:
        lines.append("")
        lines.append("Notes:")
        for module, error in sorted(import_errors.items()):
            lines.append(f"- {module}: failed to import ({error})")
    return lines


def _render_simulation_index(index: DocumentationIndex) -> str:
    lines = ["Available simulation classes:"]
    if not index.simulations:
        lines.append("- No simulations discovered.")
    else:
        for record in index.simulations:
            summary = f": {record.summary}" if record.summary else ""
            lines.append(f"- {record.qualified_name}{summary}")
    lines.extend(_format_import_notes(index.import_errors))
    return "\n".join(lines)


def _render_callback_index(index: DocumentationIndex) -> str:
    lines = ["Available callbacks:"]
    if not index.callbacks:
        lines.append("- No callbacks discovered.")
    else:
        for record in index.callbacks:
            parts = [record.qualified_name]
            if record.stage:
                parts.append(f"stage={record.stage}")
            descriptor = " [" + ", ".join(parts[1:]) + "]" if len(parts) > 1 else ""
            summary = f": {record.summary}" if record.summary else ""
            lines.append(f"- {record.qualified_name}{descriptor}{summary}")
    lines.extend(_format_import_notes(index.import_errors))
    return "\n".join(lines)


doc_index = DocumentationIndex()
mcp_app = FastMCP(
    name="LambdaPIC Manual",
    instructions=(
        "LambdaPIC MCP documentation server. "
        "Tools: list_simulations, list_callbacks, get_doc, get_code, read_manual. "
        "Use read_manual() for a quickstart guide."
    ),
)


@mcp_app.tool()
def list_simulations() -> str:
    """
    List available simulation classes and their summaries.
    """
    return _render_simulation_index(doc_index)


@mcp_app.tool()
def list_callbacks() -> str:
    """
    List available callbacks with their stages and summaries.
    """
    return _render_callback_index(doc_index)


@mcp_app.tool()
def get_doc(symbol: str) -> str:
    """
    Retrieve the docstring for the requested LambdaPIC symbol.

    Parameters:
        symbol (str): Fully qualified Python name, for example ``lambdapic.simulation.Simulation``.
    """
    if not symbol:
        raise ValueError("symbol must be a non-empty string")
    doc = doc_index.get_doc(symbol)
    if not doc:
        return f"No documentation found for '{symbol}'."
    return f"{symbol}\n\n{doc}"


@mcp_app.tool()
def get_code(symbol: str) -> str:
    """
    Retrieve the source code for the requested LambdaPIC symbol.
    Use with caution. Only when `get_doc` is insufficient.

    Parameters:
        symbol (str): Fully qualified Python name, for example ``lambdapic.simulation.Simulation``.
    """
    if not symbol:
        raise ValueError("symbol must be a non-empty string")
    source = doc_index.get_source(symbol)
    if not source:
        return f"No source code found for '{symbol}'."
    return f"{symbol}\n\n{source}"


@mcp_app.tool()
def read_manual() -> str:
    """
    Provide a comprehensive manual for AI agents to understand LambdaPIC.
    Contains detailed examples, best practices, and troubleshooting guides.
    READ THIS FIRST.
    """
    from pathlib import Path
    manual_path = Path(__file__).parent / "mcp_manual.md"
    with open(manual_path, 'r', encoding='utf-8') as f:
        return f.read()


def run() -> None:
    """Launch the FastMCP server over stdio."""
    logging.basicConfig(level=logging.INFO)
    mcp_app.run("stdio")


__all__ = [
    "DocumentationIndex",
    "DocRecord",
    "doc_index",
    "mcp_app",
    "list_simulations",
    "list_callbacks",
    "get_doc",
    "get_code",
    "read_manual",
    "run",
]
