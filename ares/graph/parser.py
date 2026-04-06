from __future__ import annotations

import ast
import logging
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import networkx as nx

try:
    from tree_sitter import Language, Parser
except ImportError:  # pragma: no cover - optional dependency
    Language = None
    Parser = None

try:  # pragma: no cover - optional dependency
    import tree_sitter_javascript as tsjavascript
except ImportError:  # pragma: no cover - optional dependency
    tsjavascript = None

try:  # pragma: no cover - optional dependency
    import tree_sitter_python as tspython
except ImportError:  # pragma: no cover - optional dependency
    tspython = None

try:  # pragma: no cover - optional dependency
    import tree_sitter_typescript as tstypescript
except ImportError:  # pragma: no cover - optional dependency
    tstypescript = None


SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "vendor",
    "venv",
}
SUPPORTED_EXTENSIONS = {".py", ".js", ".jsx", ".ts", ".tsx"}
TEST_PATH_PARTS = ("test", "tests", "__tests__", "spec")
NOISE_FILES = {"package-lock.json", "yarn.lock", "go.sum", "Cargo.lock"}


class RepoParser:
    def __init__(self, repo_path: str):
        self.repo_path = os.path.abspath(repo_path)
        self.G = nx.DiGraph()
        self._name_index: dict[str, set[str]] = defaultdict(set)
        self._methods_by_class: dict[tuple[str, str], set[str]] = defaultdict(set)
        self._file_imports: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._file_nodes: set[str] = set()
        self.parsers = self._build_parsers()

    def _build_parsers(self) -> dict[str, Any]:
        parsers: dict[str, Any] = {}
        if Parser is None or Language is None:
            return parsers
        if tspython is not None:
            parsers[".py"] = self._get_parser(tspython.language())
        if tsjavascript is not None:
            js_parser = self._get_parser(tsjavascript.language())
            parsers[".js"] = js_parser
            parsers[".jsx"] = js_parser
        if tstypescript is not None:
            ts_parser = self._get_parser(tstypescript.language_typescript())
            parsers[".ts"] = ts_parser
            parsers[".tsx"] = ts_parser
        return parsers

    def _get_parser(self, language: Any) -> Parser:
        parser = Parser()
        parser.language = Language(language)
        return parser

    def parse_repo(self) -> nx.DiGraph:
        # Collect files first so we can show progress
        to_parse: list[tuple[str, str]] = []
        for root, dirs, files in os.walk(self.repo_path):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for filename in files:
                ext = os.path.splitext(filename)[1]
                if ext not in SUPPORTED_EXTENSIONS and filename not in NOISE_FILES:
                    continue
                to_parse.append((os.path.join(root, filename), ext))

        total = len(to_parse)
        log_interval = max(1, total // 10)
        for i, (filepath, ext) in enumerate(to_parse, 1):
            if i == 1 or i % log_interval == 0 or i == total:
                print(
                    f"[parse_repo] Parsing files: {i}/{total}",
                    file=sys.stderr,
                    flush=True,
                )
            self._parse_file(filepath, ext)
        self._extract_call_edges()
        return self.G

    def _parse_file(self, filepath: str, ext: str) -> None:
        rel_path = os.path.relpath(filepath, self.repo_path)
        source_bytes = Path(filepath).read_bytes()
        source_text = source_bytes.decode("utf-8", errors="ignore")
        self.G.add_node(
            rel_path,
            type="file",
            ext=ext,
            file=rel_path,
            source=source_text,
            signature=source_text.splitlines()[0] if source_text.splitlines() else "",
        )
        self._file_nodes.add(rel_path)
        if ext == ".py":
            self._parse_python(filepath, rel_path, source_text, source_bytes)
            return
        parser = self.parsers.get(ext)
        if parser is None:
            return
        tree = parser.parse(source_bytes)
        language = "typescript" if ext in {".ts", ".tsx"} else "javascript"
        self._walk_tree_sitter(
            tree.root_node,
            rel_path,
            source_bytes,
            source_text,
            language=language,
            parent_class=None,
        )

    def _parse_python(
        self, filepath: str, rel_path: str, source_text: str, source_bytes: bytes
    ) -> None:
        parser = self.parsers.get(".py")
        if parser is not None:
            tree = parser.parse(source_bytes)
            self._walk_tree_sitter(
                tree.root_node,
                rel_path,
                source_bytes,
                source_text,
                language="python",
                parent_class=None,
            )
            return
        tree = ast.parse(source_text)
        self._walk_python_ast(tree, rel_path, source_text, parent_class=None)

    def _walk_python_ast(
        self, node: ast.AST, filepath: str, source_text: str, parent_class: str | None
    ) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                class_id = f"{filepath}::{child.name}"
                class_source = ast.get_source_segment(source_text, child) or ""
                self.G.add_node(
                    class_id,
                    type="class",
                    name=child.name,
                    file=filepath,
                    line_start=child.lineno,
                    line_end=getattr(child, "end_lineno", child.lineno),
                    params=[],
                    source=class_source,
                    signature=(class_source.splitlines() or [f"class {child.name}:"])[0].strip(),
                    language="python",
                )
                self.G.add_edge(filepath, class_id, type="contains")
                for base in child.bases:
                    base_name = getattr(base, "id", None)
                    if base_name:
                        self.G.nodes[class_id].setdefault("bases", []).append(base_name)
                self._walk_python_ast(child, filepath, source_text, parent_class=child.name)
                continue
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                node_id = (
                    f"{filepath}::{parent_class}.{child.name}"
                    if parent_class
                    else f"{filepath}::{child.name}"
                )
                function_source = ast.get_source_segment(source_text, child) or ""
                params = [arg.arg for arg in child.args.args]
                calls = self._collect_python_calls(child)
                self.G.add_node(
                    node_id,
                    type="function",
                    name=child.name,
                    file=filepath,
                    line_start=child.lineno,
                    line_end=getattr(child, "end_lineno", child.lineno),
                    params=params,
                    source=function_source,
                    signature=(function_source.splitlines() or [f"def {child.name}(...):"])[0].strip(),
                    calls=calls,
                    language="python",
                    parent_class=parent_class,
                )
                self.G.add_edge(filepath, node_id, type="contains")
                self._name_index[child.name].add(node_id)
                if parent_class:
                    self._methods_by_class[(filepath, parent_class)].add(node_id)
                self._walk_python_ast(child, filepath, source_text, parent_class=parent_class)
                continue
            if isinstance(child, ast.Import):
                for alias in child.names:
                    self._file_imports[filepath].append(
                        {
                            "module": alias.name,
                            "name": alias.asname or alias.name.split(".")[-1],
                            "resolved": self._resolve_python_module(filepath, alias.name, 0),
                        }
                    )
            if isinstance(child, ast.ImportFrom):
                for alias in child.names:
                    self._file_imports[filepath].append(
                        {
                            "module": child.module or "",
                            "name": alias.asname or alias.name,
                            "resolved": self._resolve_python_module(
                                filepath, child.module or alias.name, child.level
                            ),
                        }
                    )
            self._walk_python_ast(child, filepath, source_text, parent_class=parent_class)

    def _collect_python_calls(self, function_node: ast.AST) -> list[dict[str, str]]:
        calls: list[dict[str, str]] = []
        for subnode in ast.walk(function_node):
            if not isinstance(subnode, ast.Call):
                continue
            func_node = subnode.func
            if isinstance(func_node, ast.Name):
                calls.append({"raw": func_node.id, "name": func_node.id, "receiver": ""})
            elif isinstance(func_node, ast.Attribute):
                receiver = self._attr_receiver(func_node.value)
                calls.append(
                    {
                        "raw": f"{receiver}.{func_node.attr}" if receiver else func_node.attr,
                        "name": func_node.attr,
                        "receiver": receiver or "",
                    }
                )
        return calls

    def _attr_receiver(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            prefix = self._attr_receiver(node.value)
            return f"{prefix}.{node.attr}" if prefix else node.attr
        return ""

    def _walk_tree_sitter(
        self,
        node: Any,
        filepath: str,
        source_bytes: bytes,
        source_text: str,
        language: str,
        parent_class: str | None,
    ) -> None:
        node_type = node.type
        if node_type in {"function_definition", "async_function_definition"} and language == "python":
            self._add_tree_sitter_function(
                node, filepath, source_bytes, language, parent_class=parent_class
            )
            return
        if node_type == "class_definition" and language == "python":
            class_name_node = node.child_by_field_name("name")
            class_name = self._node_text(class_name_node, source_bytes)
            class_id = f"{filepath}::{class_name}"
            class_source = self._node_text(node, source_bytes)
            self.G.add_node(
                class_id,
                type="class",
                name=class_name,
                file=filepath,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                params=[],
                source=class_source,
                signature=(class_source.splitlines() or [f"class {class_name}:"])[0].strip(),
                language=language,
            )
            self.G.add_edge(filepath, class_id, type="contains")
            for child in node.children:
                self._walk_tree_sitter(
                    child,
                    filepath,
                    source_bytes,
                    source_text,
                    language=language,
                    parent_class=class_name,
                )
            return
        if node_type in {"function_declaration", "method_definition"} and language in {"javascript", "typescript"}:
            self._add_tree_sitter_function(
                node, filepath, source_bytes, language, parent_class=parent_class
            )
            return
        if node_type in {"lexical_declaration", "variable_declaration"} and language in {
            "javascript",
            "typescript",
        }:
            self._add_tree_sitter_variable_function(
                node, filepath, source_bytes, language, parent_class
            )
        if node_type == "class_declaration" and language in {"javascript", "typescript"}:
            name_node = node.child_by_field_name("name")
            class_name = self._node_text(name_node, source_bytes)
            class_id = f"{filepath}::{class_name}"
            class_source = self._node_text(node, source_bytes)
            self.G.add_node(
                class_id,
                type="class",
                name=class_name,
                file=filepath,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                params=[],
                source=class_source,
                signature=(class_source.splitlines() or [f"class {class_name} {{"])[0].strip(),
                language=language,
            )
            self.G.add_edge(filepath, class_id, type="contains")
            for child in node.children:
                self._walk_tree_sitter(
                    child,
                    filepath,
                    source_bytes,
                    source_text,
                    language=language,
                    parent_class=class_name,
                )
            return
        if node_type in {"import_statement", "import_from_statement"} and language == "python":
            self._record_tree_sitter_python_import(node, filepath, source_bytes)
        if node_type == "import_statement" and language in {"javascript", "typescript"}:
            self._record_tree_sitter_js_import(node, filepath, source_bytes)
        for child in node.children:
            self._walk_tree_sitter(
                child,
                filepath,
                source_bytes,
                source_text,
                language=language,
                parent_class=parent_class,
            )

    def _record_tree_sitter_python_import(self, node: Any, filepath: str, source_bytes: bytes) -> None:
        text = self._node_text(node, source_bytes)
        if text.startswith("from "):
            match = re.match(r"from\s+([.\w]+)\s+import\s+(.+)", text)
            if not match:
                return
            module, names = match.groups()
            level = len(module) - len(module.lstrip("."))
            module_name = module.lstrip(".")
            for imported in [part.strip() for part in names.split(",")]:
                name = imported.split(" as ")[0].strip()
                self._file_imports[filepath].append(
                    {
                        "module": module_name,
                        "name": name,
                        "resolved": self._resolve_python_module(filepath, module_name or name, level),
                    }
                )
        elif text.startswith("import "):
            names = text[len("import ") :].split(",")
            for imported in names:
                name = imported.split(" as ")[0].strip()
                self._file_imports[filepath].append(
                    {
                        "module": name,
                        "name": name.split(".")[-1],
                        "resolved": self._resolve_python_module(filepath, name, 0),
                    }
                )

    def _record_tree_sitter_js_import(self, node: Any, filepath: str, source_bytes: bytes) -> None:
        text = self._node_text(node, source_bytes)
        match = re.search(r"from\s+['\"]([^'\"]+)['\"]", text)
        if not match:
            return
        module = match.group(1)
        names = re.findall(r"[{,]\s*([A-Za-z_$][\w$]*)", text)
        if "import " in text and "{" not in text:
            head = text.replace("import", "", 1).split("from", 1)[0].strip()
            if head:
                names.append(head)
        resolved = self._resolve_js_module(filepath, module)
        for name in names or [module.split("/")[-1]]:
            self._file_imports[filepath].append(
                {"module": module, "name": name, "resolved": resolved}
            )

    def _add_tree_sitter_variable_function(
        self, node: Any, filepath: str, source_bytes: bytes, language: str, parent_class: str | None
    ) -> None:
        for child in node.children:
            if child.type != "variable_declarator":
                continue
            name_node = child.child_by_field_name("name")
            value_node = child.child_by_field_name("value")
            if value_node is None or value_node.type not in {"arrow_function", "function", "function_expression"}:
                continue
            function_name = self._node_text(name_node, source_bytes)
            self._add_tree_sitter_function(
                value_node,
                filepath,
                source_bytes,
                language,
                parent_class=parent_class,
                override_name=function_name,
            )

    def _add_tree_sitter_function(
        self,
        node: Any,
        filepath: str,
        source_bytes: bytes,
        language: str,
        parent_class: str | None,
        override_name: str | None = None,
    ) -> None:
        name_node = node.child_by_field_name("name")
        function_name = override_name or self._node_text(name_node, source_bytes) or "anonymous"
        node_id = (
            f"{filepath}::{parent_class}.{function_name}"
            if parent_class
            else f"{filepath}::{function_name}"
        )
        params = []
        parameters_node = node.child_by_field_name("parameters")
        if parameters_node is not None:
            for child in parameters_node.children:
                if child.type in {",", "(", ")", ":"}:
                    continue
                params.append(self._node_text(child, source_bytes))
        function_source = self._node_text(node, source_bytes)
        calls = self._collect_tree_sitter_calls(node, source_bytes)
        self.G.add_node(
            node_id,
            type="function",
            name=function_name,
            file=filepath,
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            params=params,
            source=function_source,
            signature=(function_source.splitlines() or [function_name])[0].strip(),
            calls=calls,
            language=language,
            parent_class=parent_class,
        )
        self.G.add_edge(filepath, node_id, type="contains")
        self._name_index[function_name].add(node_id)
        if parent_class:
            self._methods_by_class[(filepath, parent_class)].add(node_id)

    def _collect_tree_sitter_calls(self, node: Any, source_bytes: bytes) -> list[dict[str, str]]:
        calls: list[dict[str, str]] = []
        stack = [node]
        while stack:
            current = stack.pop()
            if current.type == "call":
                function_node = current.child_by_field_name("function")
                if function_node is not None:
                    raw = self._node_text(function_node, source_bytes)
                    receiver = ""
                    name = raw
                    if "." in raw:
                        receiver, name = raw.rsplit(".", 1)
                    calls.append({"raw": raw, "name": name, "receiver": receiver})
            elif current.type == "call_expression":
                function_node = current.child_by_field_name("function")
                if function_node is not None:
                    raw = self._node_text(function_node, source_bytes)
                    receiver = ""
                    name = raw
                    if "." in raw:
                        receiver, name = raw.rsplit(".", 1)
                    calls.append({"raw": raw, "name": name, "receiver": receiver})
            stack.extend(reversed(current.children))
        return calls

    def _extract_call_edges(self) -> None:
        for filepath, imports in self._file_imports.items():
            for imported in imports:
                resolved = imported.get("resolved")
                if resolved and resolved in self.G:
                    self._add_edge_type(filepath, resolved, "imports")
        for node_id, data in list(self.G.nodes(data=True)):
            if data.get("type") != "function":
                continue
            caller_file = data["file"]
            parent_class = data.get("parent_class")
            for call in data.get("calls", []):
                for callee_id in self._resolve_call_candidates(
                    call["name"], call.get("receiver", ""), caller_file, parent_class
                ):
                    self._add_edge_type(node_id, callee_id, "calls")
                    if self._is_test_path(caller_file):
                        self._add_edge_type(node_id, callee_id, "tests")

    def _resolve_call_candidates(
        self, call_name: str, receiver: str, caller_file: str, parent_class: str | None
    ) -> list[str]:
        matches: list[str] = []
        if receiver in {"self", "cls"} and parent_class:
            matches.extend(
                sorted(
                    node_id
                    for node_id in self._methods_by_class.get((caller_file, parent_class), set())
                    if self.G.nodes[node_id].get("name") == call_name
                )
            )
        if not matches:
            same_file = [
                node_id
                for node_id in self._name_index.get(call_name, set())
                if self.G.nodes[node_id].get("file") == caller_file
            ]
            matches.extend(sorted(same_file))
        if not matches:
            imported_matches = []
            for imported in self._file_imports.get(caller_file, []):
                resolved = imported.get("resolved")
                if not resolved:
                    continue
                imported_matches.extend(
                    node_id
                    for node_id, data in self.G.nodes(data=True)
                    if data.get("file") == resolved and data.get("name") == call_name
                )
            matches.extend(sorted(set(imported_matches)))
        if not matches and len(self._name_index.get(call_name, set())) == 1:
            matches.extend(sorted(self._name_index[call_name]))
        return list(dict.fromkeys(matches))

    def _resolve_python_module(self, filepath: str, module: str, level: int) -> str | None:
        file_dir = Path(filepath).parent
        base_dir = file_dir
        for _ in range(max(level - 1, 0)):
            base_dir = base_dir.parent
        module_parts = [part for part in module.split(".") if part]
        candidate = base_dir.joinpath(*module_parts)
        for suffix in (".py", "/__init__.py"):
            resolved = Path(f"{candidate}{suffix}" if suffix.startswith(".") else str(candidate) + suffix)
            rel = str(resolved).replace("\\", "/")
            if rel in self._file_nodes or (Path(self.repo_path) / rel).exists():
                return rel
        return None

    def _resolve_js_module(self, filepath: str, module: str) -> str | None:
        if not module.startswith("."):
            return None
        file_dir = Path(filepath).parent
        candidate = (file_dir / module).resolve()
        repo_root = Path(self.repo_path).resolve()
        try_paths = [candidate]
        for ext in (".js", ".jsx", ".ts", ".tsx"):
            try_paths.append(Path(str(candidate) + ext))
        for path in try_paths:
            try:
                rel = str(path.relative_to(repo_root)).replace("\\", "/")
            except ValueError:
                continue
            if rel in self._file_nodes or path.exists():
                return rel
        return None

    def _node_text(self, node: Any, source_bytes: bytes) -> str:
        if node is None:
            return ""
        return source_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="ignore")

    def _add_edge_type(self, source: str, target: str, edge_type: str, **attrs: Any) -> None:
        if self.G.has_edge(source, target):
            edge = self.G.edges[source, target]
            types = set(edge.get("types", []))
            if edge.get("type"):
                types.add(edge["type"])
            types.add(edge_type)
            edge["types"] = sorted(types)
            edge["type"] = edge_type if len(types) == 1 else edge.get("type", edge_type)
            edge.update(attrs)
            return
        self.G.add_edge(source, target, type=edge_type, types=[edge_type], **attrs)

    @staticmethod
    def _is_test_path(path: str) -> bool:
        lowered = path.lower()
        filename = os.path.basename(lowered)
        return any(part in lowered.split("/") for part in TEST_PATH_PARTS) or filename.startswith(
            "test_"
        ) or filename.endswith(("_test.py", ".test.js", ".spec.ts", ".spec.js", ".test.ts"))
