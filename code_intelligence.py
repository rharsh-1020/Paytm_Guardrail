import os
import re
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Optional imports: keep graceful fallback behavior.
try:
    from tree_sitter_languages import get_parser as ts_get_parser
except Exception:
    ts_get_parser = None

try:
    from neo4j import GraphDatabase
except Exception:
    GraphDatabase = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CODE_DIR = PROJECT_ROOT
CODE_DB_DIR = PROJECT_ROOT / "chroma_code_db"
GRAPH_EXPORT_PATH = PROJECT_ROOT / "semantic_graph.json"
EMBEDDING_MODEL = "flax-sentence-embeddings/st-codesearch-distilroberta-base"
MAX_FILE_SIZE_BYTES = 400_000


LANG_BY_EXT = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".jsx": "javascript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
}


IGNORE_DIR_NAMES = {
    ".git",
    "venv",
    ".venv",
    "node_modules",
    "__pycache__",
    "dist",
    "build",
    "chroma_db",
    "chroma_code_db",
}


def disable_broken_local_proxy():
    proxy_keys = [
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "GIT_HTTP_PROXY",
        "GIT_HTTPS_PROXY",
    ]
    for key in proxy_keys:
        value = os.environ.get(key)
        if value and ("127.0.0.1:9" in value or "localhost:9" in value):
            os.environ.pop(key, None)


@dataclass
class CodeNode:
    text: str
    metadata: Dict


# 🔥 Only scan relevant microservices
TARGET_SERVICES = {
    "payment_service",
    "mandate_service",
    "notification_service",
    "wallet_service",
    "dummy_services",
}

def is_relevant_file(path: Path) -> bool:
    path_str = str(path).replace('\\', '/').lower()
    return any(service in path_str for service in TARGET_SERVICES)

class CodeIntelligence:
    def __init__(self, code_dir: Path):
        self.code_dir = code_dir
        self.parsers: Dict[str, object] = {}

    def _get_parser(self, language: str):
        if ts_get_parser is None:
            return None
        if language in self.parsers:
            return self.parsers[language]
        try:
            parser = ts_get_parser(language)
            self.parsers[language] = parser
            return parser
        except Exception:
            self.parsers[language] = None
            return None

    def iter_source_files(self) -> List[Path]:
        files: List[Path] = []
        all_candidates: List[Path] = []
        for root, dirs, names in os.walk(self.code_dir):
            dirs[:] = [d for d in dirs if d not in IGNORE_DIR_NAMES]
            root_path = Path(root)
            for name in names:
                path = root_path / name
                if path.suffix.lower() not in LANG_BY_EXT:
                    continue
                try:
                    if path.stat().st_size > MAX_FILE_SIZE_BYTES:
                        continue
                except OSError:
                    continue
                all_candidates.append(path)
                if is_relevant_file(path):
                    files.append(path)

        # If strict service filters match nothing, fall back to whole codebase scan.
        if not files and all_candidates:
            print("[AST][WARN] No files matched TARGET_SERVICES; falling back to all source files.")
            files = all_candidates
        return files

    @staticmethod
    def _node_text(src_bytes: bytes, start: int, end: int) -> str:
        return src_bytes[start:end].decode("utf-8", errors="ignore")

    @staticmethod
    def _extract_basic_semantics(text: str) -> Dict[str, List[str]]:
        constants = re.findall(r"\b\d+(?:\.\d+)?\b", text)
        config_refs = re.findall(r"[\w\-/]+\.(?:ya?ml|json|toml|ini|env)", text)
        apis = re.findall(r"\b(?:GET|POST|PUT|PATCH|DELETE)\b|/api/[\w\-/]+", text)
        env_vars = re.findall(r"\b[A-Z_]{3,}\b", text)
        return {
            "constants": sorted(set(constants))[:50],
            "config_refs": sorted(set(config_refs))[:50],
            "apis": sorted(set(apis))[:50],
            "env_vars": sorted(set(env_vars))[:50],
        }

    def _extract_nodes_treesitter(self, path: Path, language: str) -> List[CodeNode]:
        parser = self._get_parser(language)
        if parser is None:
            return []

        source = path.read_bytes()
        tree = parser.parse(source)
        root = tree.root_node

        # Generic + language-specific node targets.
        target_types = {
            "function_definition",
            "class_definition",
            "method_definition",
            "function_declaration",
            "method_declaration",
            "lexical_declaration",
            "assignment",
            "pair",
        }

        nodes: List[CodeNode] = []
        stack = [root]

        while stack:
            node = stack.pop()
            stack.extend(node.children)

            if node.type not in target_types:
                continue

            text = self._node_text(source, node.start_byte, node.end_byte).strip()
            if not text or len(text) < 10:
                continue

            name = "unknown"
            for child in node.children:
                if child.type in {"identifier", "property_identifier", "string", "name"}:
                    name = self._node_text(source, child.start_byte, child.end_byte).strip().strip('"\'')
                    break

            sem = self._extract_basic_semantics(text)
            nodes.append(
                CodeNode(
                    text=text,
                    metadata={
                        "file": str(path.relative_to(self.code_dir)).replace("\\", "/"),
                        "language": language,
                        "name": name,
                        "type": node.type,
                        **sem,
                    },
                )
            )

        return nodes

    def _extract_nodes_python_ast_fallback(self, path: Path) -> List[CodeNode]:
        import ast

        source = path.read_text(encoding="utf-8", errors="ignore")
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return []

        nodes: List[CodeNode] = []
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            segment = ast.get_source_segment(source, node)
            if not segment:
                continue
            sem = self._extract_basic_semantics(segment)
            nodes.append(
                CodeNode(
                    text=segment,
                    metadata={
                        "file": str(path.relative_to(self.code_dir)).replace("\\", "/"),
                        "language": "python",
                        "name": node.name,
                        "type": type(node).__name__,
                        **sem,
                    },
                )
            )

        return nodes

    def extract_code_nodes(self) -> List[CodeNode]:
        files = self.iter_source_files()
        nodes: List[CodeNode] = []

        for path in files:
            language = LANG_BY_EXT.get(path.suffix.lower())
            if not language:
                continue

            extracted = self._extract_nodes_treesitter(path, language)
            if not extracted and language == "python":
                extracted = self._extract_nodes_python_ast_fallback(path)

            nodes.extend(extracted)

        print(f"[AST] Extracted {len(nodes)} semantic code nodes from {len(files)} files")
        return nodes


def build_vector_store(nodes: List[CodeNode]) -> Chroma:
    if not nodes:
        raise RuntimeError("No code nodes extracted. Nothing to embed.")

    docs = []
    for n in nodes:
        safe_meta = {}
        for key, value in n.metadata.items():
            if isinstance(value, list):
                safe_meta[key] = json.dumps(value)
            elif isinstance(value, (str, int, float, bool)) or value is None:
                safe_meta[key] = value if value is not None else ""
            else:
                safe_meta[key] = str(value)
        docs.append(Document(page_content=n.text, metadata=safe_meta))

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=str(CODE_DB_DIR),
    )
    return db


def find_relevant_code(db: Chroma, rule_text: str, k: int = 5):
    results = db.similarity_search_with_score(rule_text, k=k)
    ranked = []
    for doc, score in results:
        ranked.append({"doc": doc, "similarity": 1 / (1 + float(score))})
    return sorted(ranked, key=lambda x: x["similarity"], reverse=True)


def build_semantic_graph(nodes: List[CodeNode]) -> Dict:
    services = {}
    edges = []

    for n in nodes:
        file_path = n.metadata["file"]
        service = file_path.split("/")[0] if "/" in file_path else "root"
        services.setdefault(service, {"files": set(), "components": []})
        services[service]["files"].add(file_path)
        services[service]["components"].append({
            "name": n.metadata.get("name", "unknown"),
            "type": n.metadata.get("type", "unknown"),
            "file": file_path,
            "config_refs": n.metadata.get("config_refs", []),
            "apis": n.metadata.get("apis", []),
        })
        for api in n.metadata.get("apis", []):
            edges.append({
                "from": file_path,
                "to": api,
                "type": "EXPOSES_API",
            })
        for cfg in n.metadata.get("config_refs", []):
            edges.append(
                {
                    "from": file_path,
                    "to": cfg,
                    "type": "REFERENCES_CONFIG",
                }
            )

    graph = {
        "services": {
            name: {
                "files": sorted(list(data["files"])),
                "components": data["components"],
            }
            for name, data in services.items()
        },
        "edges": edges,
    }
    return graph


def export_semantic_graph(graph: Dict, out_path: Path):
    out_path.write_text(json.dumps(graph, indent=2), encoding="utf-8")
    print(f"[GRAPH] Exported semantic graph to {out_path}")


def rule_to_query(rule: Dict) -> str:
    parts = []

    rule_type = rule.get("rule_type", "")
    value = str(rule.get("constraint_value", ""))

    if rule_type == "limit":
        parts.append(f"payment limit threshold {value} transaction cap mandate")

    elif rule_type == "obligation":
        parts.append("send notification pre debit alert mandate afa validation required")

    elif rule_type == "prohibition":
        parts.append("no charges fee not allowed mandate restriction")

    if value:
        parts.append(value)

    return " ".join(parts)

def extract_numeric_value(text: str) -> Optional[int]:
    if not text:
        return None

    # Remove commas and currency symbols
    cleaned = re.sub(r"[\u20B9,]", "", text)
    nums = re.findall(r"\d+", cleaned)

    if not nums:
        return None

    return int(nums[0])


def push_graph_to_neo4j(graph: Dict):
    if GraphDatabase is None:
        print("[GRAPH] neo4j package not installed; skipping Neo4j push")
        return

    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")
    if not (uri and user and password):
        print("[GRAPH] NEO4J_* env vars missing; skipping Neo4j push")
        return

    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session() as session:
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (s:Service) REQUIRE s.name IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (f:File) REQUIRE f.path IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Component) REQUIRE c.id IS UNIQUE")

        for service_name, payload in graph["services"].items():
            session.run("MERGE (s:Service {name:$name})", name=service_name)
            for file_path in payload["files"]:
                session.run(
                    """
                    MERGE (f:File {path:$path})
                    MERGE (s:Service {name:$service})
                    MERGE (s)-[:OWNS]->(f)
                    """,
                    path=file_path,
                    service=service_name,
                )
            for component in payload["components"]:
                comp_id = f"{component['file']}::{component['name']}::{component['type']}"
                session.run(
                    """
                    MERGE (c:Component {id:$id})
                    SET c.name=$name, c.type=$type
                    MERGE (f:File {path:$file})
                    MERGE (f)-[:DECLARES]->(c)
                    """,
                    id=comp_id,
                    name=component["name"],
                    type=component["type"],
                    file=component["file"],
                )

        for edge in graph["edges"]:
            if edge["type"] == "REFERENCES_CONFIG":
                session.run(
                    """
                    MERGE (f:File {path:$from_path})
                    MERGE (cfg:Config {path:$to_path})
                    MERGE (f)-[:REFERENCES_CONFIG]->(cfg)
                    """,
                    from_path=edge["from"],
                    to_path=edge["to"],
                )
            elif edge["type"] == "EXPOSES_API":
                session.run(
                    """
                    MERGE (f:File {path:$from_path})
                    MERGE (api:Api {path:$to_path})
                    MERGE (f)-[:EXPOSES_API]->(api)
                    """,
                    from_path=edge["from"],
                    to_path=edge["to"],
                )

    driver.close()
    print("[GRAPH] Pushed semantic graph to Neo4j")
def is_domain_relevant(text: str) -> bool:
    domain_keywords = [
        "payment",
        "mandate",
        "transaction",
        "wallet",
        "notify",
        "notification",
        "afa",
    ]
    return any(k in text for k in domain_keywords)

def detect_violations(rule: Dict, matched_docs: List[Dict]) -> List[Dict]:
    violations = []
    seen = set()

    rule_value = extract_numeric_value(str(rule.get("constraint_value")))

    def _meta_list(metadata: Dict, key: str) -> List[str]:
        raw = metadata.get(key, [])
        if isinstance(raw, list):
            return [str(x) for x in raw]
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    return [str(x) for x in parsed]
            except Exception:
                pass
            return [x.strip() for x in raw.split("|") if x.strip()]
        return []

    for item in matched_docs:
        doc = item["doc"]
        meta = doc.metadata

        constants = [
            extract_numeric_value(c)
            for c in _meta_list(meta, "constants")
        ]
        constants = [c for c in constants if c is not None]

        code_text = doc.page_content.lower()

        # 🔥 FILTER OUT NON-DOMAIN CODE
        if not is_domain_relevant(code_text):
            continue

        # -------------------------------
        # LIMIT RULE CHECK
        # -------------------------------
        if rule.get("rule_type") == "limit" and rule_value:
            for value in constants:
                if value < rule_value and value > 100:
                    issue = {
                        "file": meta.get("file"),
                        "component": meta.get("name"),
                        "issue": f"Limit {value} < required {rule_value}",
                        "similarity": item["similarity"],
                    }
                    key = (issue["file"], issue["component"], issue["issue"])
                    if key not in seen:
                        seen.add(key)
                        violations.append(issue)

        # -------------------------------
        # TIME RULE CHECK
        # -------------------------------
        if "hour" in str(rule.get("constraint_value", "")).lower() and rule_value:
            for value in constants:
                if 0 < value <= 168 and value < rule_value:
                    issue = {
                        "file": meta.get("file"),
                        "component": meta.get("name"),
                        "issue": f"Lead time {value}h < required {rule_value}h",
                        "similarity": item["similarity"],
                    }
                    key = (issue["file"], issue["component"], issue["issue"])
                    if key not in seen:
                        seen.add(key)
                        violations.append(issue)

        # -------------------------------
        # PROHIBITION CHECK
        # -------------------------------
        if rule.get("rule_type") == "prohibition":
            keywords = ["charge", "fee", "levy"]
            if any(k in code_text for k in keywords):
                issue = {
                    "file": meta.get("file"),
                    "component": meta.get("name"),
                    "issue": "Potential prohibited action detected",
                    "similarity": item["similarity"],
                }
                key = (issue["file"], issue["component"], issue["issue"])
                if key not in seen:
                    seen.add(key)
                    violations.append(issue)

        # -------------------------------
        # OBLIGATION CHECK (missing logic)
        # -------------------------------
        if rule.get("rule_type") == "obligation":

            required_keywords = ["notify", "notification", "alert"]
            has_behavior = any(k in code_text for k in required_keywords)

            # 🔥 Only flag if this looks like relevant service code
            if ("mandate" in code_text or "payment" in code_text) and not has_behavior:
                issue = {
                    "file": meta.get("file"),
                    "component": meta.get("name"),
                    "issue": "Missing required notification/alert behavior",
                    "similarity": item["similarity"],
                }
                key = (issue["file"], issue["component"], issue["issue"])
                if key not in seen:
                    seen.add(key)
                    violations.append(issue)

    return violations



def run_code_intelligence_pipeline(rules: Optional[List[Dict]] = None) -> Dict:
    disable_broken_local_proxy()
    print("[START] Phase 3: Code Intelligence System")

    ci = CodeIntelligence(CODE_DIR)
    nodes = ci.extract_code_nodes()
    db = build_vector_store(nodes)

    graph = build_semantic_graph(nodes)
    export_semantic_graph(graph, GRAPH_EXPORT_PATH)
    push_graph_to_neo4j(graph)

    if rules is None:
        rules = [
            {"rule_type": "limit", "constraint_value": "100000"},
            {"rule_type": "obligation", "constraint_value": "24 hours"},
        ]

    all_violations: List[Dict] = []

    for rule in rules:
        safe_rule = str(rule).encode("ascii", errors="replace").decode("ascii")
        print("\n===================================")
        print(f"[CHECK] Rule: {safe_rule}")
        print("===================================")

        query = rule_to_query(rule)
        matches = find_relevant_code(db, query, k=5)
        violations = detect_violations(rule, matches)

        if not violations:
            print("[OK] No obvious violations detected for this rule in top matches")
            continue

        for v in violations:
            violation = {
                **v,
                "rule_type": rule.get("rule_type", ""),
                "constraint_value": str(rule.get("constraint_value", "")),
            }
            all_violations.append(violation)
            print("[VIOLATION]")
            print(f"File: {violation['file']}")
            print(f"Component: {violation['component']}")
            print(f"Issue: {violation['issue']}")

    return {
        "graph_path": str(GRAPH_EXPORT_PATH),
        "violations": all_violations,
    }


def main():
    run_code_intelligence_pipeline()


if __name__ == "__main__":
    main()
