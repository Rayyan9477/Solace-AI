"""
Import dependency analyzer to detect circular imports and dependency issues.

This tool helps identify and resolve circular dependency problems in the codebase.
"""

import ast
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class ImportAnalyzer:
    """Analyzes Python files to detect import dependencies and circular imports"""

    def __init__(self, root_dir: str):
        """
        Initialize the import analyzer.

        Args:
            root_dir: Root directory of the Python project
        """
        self.root_dir = Path(root_dir)
        self.imports: Dict[str, Set[str]] = defaultdict(set)
        self.circular_deps: List[List[str]] = []

    def analyze_file(self, file_path: Path) -> Set[str]:
        """
        Analyze a single Python file for its imports.

        Args:
            file_path: Path to the Python file

        Returns:
            Set of module names imported by this file
        """
        imports = set()

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read(), filename=str(file_path))

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name.split('.')[0])

                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        # Handle relative imports
                        if node.level > 0:
                            # Relative import - resolve to absolute
                            rel_path = self._resolve_relative_import(
                                file_path,
                                node.module,
                                node.level
                            )
                            if rel_path:
                                imports.add(rel_path)
                        else:
                            # Absolute import
                            imports.add(node.module.split('.')[0])

        except Exception as e:
            logger.warning(f"Failed to parse {file_path}: {e}")

        return imports

    def _resolve_relative_import(
        self,
        file_path: Path,
        module: Optional[str],
        level: int
    ) -> Optional[str]:
        """
        Resolve a relative import to an absolute module path.

        Args:
            file_path: Path to the file containing the import
            module: Module name (can be None for '.' imports)
            level: Number of dots in the relative import

        Returns:
            Absolute module path or None if cannot be resolved
        """
        # Get the package path by going up 'level' directories
        current_dir = file_path.parent

        for _ in range(level - 1):
            current_dir = current_dir.parent
            if current_dir == self.root_dir.parent:
                return None

        # Convert to module path
        rel_to_root = current_dir.relative_to(self.root_dir)
        parts = list(rel_to_root.parts)

        if module:
            parts.extend(module.split('.'))

        return '.'.join(parts) if parts else None

    def analyze_directory(self, directory: Optional[Path] = None) -> Dict[str, Set[str]]:
        """
        Analyze all Python files in a directory recursively.

        Args:
            directory: Directory to analyze (defaults to root_dir)

        Returns:
            Dictionary mapping module paths to their dependencies
        """
        if directory is None:
            directory = self.root_dir

        for py_file in directory.rglob('*.py'):
            if '__pycache__' in str(py_file):
                continue

            # Convert file path to module path
            try:
                rel_path = py_file.relative_to(self.root_dir)
                module_path = str(rel_path.with_suffix('')).replace(os.sep, '.')

                # Analyze imports
                imports = self.analyze_file(py_file)
                self.imports[module_path] = imports

            except ValueError:
                # File is not under root_dir
                continue

        return self.imports

    def find_circular_dependencies(self) -> List[List[str]]:
        """
        Find circular dependencies using depth-first search.

        Returns:
            List of circular dependency chains
        """
        circular_deps = []
        visited = set()
        rec_stack = set()

        def dfs(module: str, path: List[str]) -> None:
            """
            Depth-first search to detect cycles.

            Args:
                module: Current module being visited
                path: Current path in the DFS traversal
            """
            visited.add(module)
            rec_stack.add(module)
            path.append(module)

            # Check dependencies
            for dependency in self.imports.get(module, set()):
                if dependency not in visited:
                    dfs(dependency, path.copy())
                elif dependency in rec_stack:
                    # Found a cycle
                    cycle_start = path.index(dependency)
                    cycle = path[cycle_start:] + [dependency]
                    if cycle not in circular_deps:
                        circular_deps.append(cycle)

            rec_stack.remove(module)

        # Run DFS from each module
        for module in self.imports.keys():
            if module not in visited:
                dfs(module, [])

        self.circular_deps = circular_deps
        return circular_deps

    def get_dependency_graph(self) -> Dict[str, List[str]]:
        """
        Get the dependency graph as adjacency list.

        Returns:
            Dictionary mapping modules to list of their dependencies
        """
        return {module: list(deps) for module, deps in self.imports.items()}

    def get_reverse_dependencies(self) -> Dict[str, Set[str]]:
        """
        Get reverse dependencies (what depends on each module).

        Returns:
            Dictionary mapping modules to set of modules that depend on them
        """
        reverse_deps: Dict[str, Set[str]] = defaultdict(set)

        for module, dependencies in self.imports.items():
            for dep in dependencies:
                reverse_deps[dep].add(module)

        return reverse_deps

    def find_highly_coupled_modules(self, threshold: int = 10) -> List[Tuple[str, int]]:
        """
        Find modules with high coupling (many dependencies or dependents).

        Args:
            threshold: Minimum number of connections to be considered highly coupled

        Returns:
            List of (module, connection_count) tuples sorted by count
        """
        reverse_deps = self.get_reverse_dependencies()
        coupling_scores = []

        for module in self.imports.keys():
            # Count both dependencies and reverse dependencies
            dep_count = len(self.imports.get(module, set()))
            rdep_count = len(reverse_deps.get(module, set()))
            total = dep_count + rdep_count

            if total >= threshold:
                coupling_scores.append((module, total))

        return sorted(coupling_scores, key=lambda x: x[1], reverse=True)

    def generate_report(self) -> str:
        """
        Generate a comprehensive report of import analysis.

        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 80)
        report.append("IMPORT DEPENDENCY ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")

        # Summary
        report.append(f"Total modules analyzed: {len(self.imports)}")
        report.append(f"Total import relationships: {sum(len(deps) for deps in self.imports.values())}")
        report.append("")

        # Circular dependencies
        circular = self.find_circular_dependencies()
        report.append(f"Circular dependencies found: {len(circular)}")
        if circular:
            report.append("")
            report.append("CIRCULAR DEPENDENCY CHAINS:")
            report.append("-" * 80)
            for i, cycle in enumerate(circular, 1):
                report.append(f"\n{i}. Cycle:")
                for j, module in enumerate(cycle):
                    if j == len(cycle) - 1:
                        report.append(f"   {module} -> {cycle[0]} (CYCLE)")
                    else:
                        report.append(f"   {module} ->")
        else:
            report.append("No circular dependencies detected!")

        report.append("")
        report.append("-" * 80)

        # Highly coupled modules
        highly_coupled = self.find_highly_coupled_modules(threshold=5)
        if highly_coupled:
            report.append("")
            report.append("HIGHLY COUPLED MODULES (>= 5 connections):")
            report.append("-" * 80)
            for module, count in highly_coupled[:10]:  # Top 10
                report.append(f"  {module}: {count} connections")

        report.append("")
        report.append("=" * 80)

        return "\n".join(report)


def analyze_project_imports(root_dir: str = "src") -> ImportAnalyzer:
    """
    Analyze imports for the entire project.

    Args:
        root_dir: Root directory to analyze

    Returns:
        ImportAnalyzer with analysis results
    """
    analyzer = ImportAnalyzer(root_dir)
    analyzer.analyze_directory()
    return analyzer


if __name__ == "__main__":
    # When run as a script, analyze the src directory
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else "src"
    analyzer = analyze_project_imports(root)

    print(analyzer.generate_report())

    # Save detailed results
    with open("import_analysis.txt", "w") as f:
        f.write(analyzer.generate_report())
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("FULL DEPENDENCY GRAPH:\n")
        f.write("=" * 80 + "\n\n")

        for module, deps in sorted(analyzer.get_dependency_graph().items()):
            f.write(f"{module}:\n")
            for dep in sorted(deps):
                f.write(f"  -> {dep}\n")
            f.write("\n")

    print("\nDetailed analysis saved to import_analysis.txt")
