"""Tests for all tools in the SatyaRepro tool library.

Layer 1 (static): tested deterministically — no LLM needed.
Layer 2 + reports: tested with MockClient to verify prompt routing and response pass-through.
Parsers: tested with in-memory temp files.
"""
import json
import os
import textwrap
import tempfile

import pytest

from satyarepro.client.mock import MockClient
from satyarepro.tools import create_default_registry
from satyarepro.tools.layer1.checkpoint_check import CheckpointCheck
from satyarepro.tools.layer1.dependency_check import DependencyCheck
from satyarepro.tools.layer1.seed_check import SeedCheck
from satyarepro.tools.layer1.split_check import SplitCheck
from satyarepro.tools.layer2.leakage_detector import LeakageDetector
from satyarepro.tools.layer2.provenance_checker import ProvenanceChecker
from satyarepro.tools.layer2.subgroup_reporter import SubgroupReporter
from satyarepro.tools.parsers.notebook_parser import NotebookParser
from satyarepro.tools.parsers.script_parser import ScriptParser
from satyarepro.tools.reports.dmsp_generator import DMSPGenerator
from satyarepro.tools.reports.tripod_ai_generator import TripodAIGenerator
from satyarepro.types import CompletionResponse, Usage


# ── helpers ────────────────────────────────────────────────────────────────────

def _ok_response(text: str) -> CompletionResponse:
    return CompletionResponse(
        content=text,
        raw_content=[{"type": "text", "text": text}],
        usage=Usage(10, 5),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Layer 1 — SeedCheck
# ══════════════════════════════════════════════════════════════════════════════

class TestSeedCheck:
    @pytest.fixture
    def tool(self):
        return SeedCheck()

    async def test_numpy_with_seed(self, tool):
        code = "import numpy as np\nnp.random.seed(42)\nX = np.zeros((10, 3))"
        result = json.loads(await tool.execute(code=code))
        assert result["status"] == "ok"
        assert "numpy" in result["seeded"]

    async def test_numpy_missing_seed(self, tool):
        code = "import numpy as np\nX = np.zeros((10, 3))"
        result = json.loads(await tool.execute(code=code))
        assert result["status"] == "issues_found"
        assert any("numpy" in issue for issue in result["issues"])

    async def test_torch_with_manual_seed(self, tool):
        code = "import torch\ntorch.manual_seed(0)\nmodel = torch.nn.Linear(4, 1)"
        result = json.loads(await tool.execute(code=code))
        assert result["status"] == "ok"
        assert "torch" in result["seeded"]

    async def test_torch_missing_seed(self, tool):
        code = "import torch\nmodel = torch.nn.Linear(4, 1)"
        result = json.loads(await tool.execute(code=code))
        assert result["status"] == "issues_found"
        assert any("torch" in issue for issue in result["issues"])

    async def test_random_module_seeded(self, tool):
        code = "import random\nrandom.seed(7)\nsamples = random.sample(range(100), 10)"
        result = json.loads(await tool.execute(code=code))
        assert result["status"] == "ok"

    async def test_tensorflow_seeded(self, tool):
        code = "import tensorflow as tf\ntf.random.set_seed(1)\nmodel = tf.keras.Sequential()"
        result = json.loads(await tool.execute(code=code))
        assert result["status"] == "ok"
        assert "tensorflow" in result["seeded"]

    async def test_multiple_frameworks_partial(self, tool):
        code = textwrap.dedent("""\
            import numpy as np
            import torch
            np.random.seed(42)
            model = torch.nn.Linear(4, 1)
        """)
        result = json.loads(await tool.execute(code=code))
        assert result["status"] == "issues_found"
        assert "numpy" in result["seeded"]
        assert any("torch" in issue for issue in result["issues"])

    async def test_no_ml_imports(self, tool):
        code = "x = 1 + 2\nprint(x)"
        result = json.loads(await tool.execute(code=code))
        assert result["status"] == "ok"
        assert result["frameworks_detected"] == []

    async def test_syntax_error_returns_error_key(self, tool):
        result = json.loads(await tool.execute(code="def (broken"))
        assert "error" in result

    async def test_alias_import(self, tool):
        code = "import numpy as np\nnp.random.seed(99)"
        result = json.loads(await tool.execute(code=code))
        assert result["status"] == "ok"


# ══════════════════════════════════════════════════════════════════════════════
# Layer 1 — DependencyCheck
# ══════════════════════════════════════════════════════════════════════════════

class TestDependencyCheck:
    @pytest.fixture
    def tool(self):
        return DependencyCheck()

    async def test_all_pinned(self, tool):
        reqs = "numpy==1.26.0\nscikit-learn==1.4.0\npandas==2.2.0\n"
        result = json.loads(await tool.execute(requirements=reqs))
        assert result["status"] == "ok"

    async def test_unpinned_detected(self, tool):
        reqs = "numpy\nscikit-learn==1.4.0\npandas\n"
        result = json.loads(await tool.execute(requirements=reqs))
        assert result["status"] == "issues_found"
        assert "numpy" in result["unpinned_packages"]
        assert "pandas" in result["unpinned_packages"]
        assert "scikit-learn" not in result["unpinned_packages"]

    async def test_comments_and_blank_lines_ignored(self, tool):
        reqs = "# this is a comment\n\nnumpy==1.26.0\n"
        result = json.loads(await tool.execute(requirements=reqs))
        assert result["status"] == "ok"

    async def test_extras_syntax_handled(self, tool):
        reqs = "uvicorn[standard]==0.32.0\nfastapi\n"
        result = json.loads(await tool.execute(requirements=reqs))
        assert result["status"] == "issues_found"
        assert "fastapi" in result["unpinned_packages"]
        assert "uvicorn" not in result["unpinned_packages"]

    async def test_conda_env_pinned(self, tool):
        env = "name: myenv\ndependencies:\n  - numpy=1.26.0\n  - scikit-learn=1.4.0\n"
        result = json.loads(await tool.execute(requirements=env, file_type="environment.yml"))
        assert result["status"] == "ok"

    async def test_conda_env_unpinned(self, tool):
        env = "name: myenv\ndependencies:\n  - numpy\n  - scikit-learn=1.4.0\n"
        result = json.loads(await tool.execute(requirements=env, file_type="environment.yml"))
        assert result["status"] == "issues_found"
        assert "numpy" in result["unpinned_packages"]

    async def test_empty_requirements(self, tool):
        result = json.loads(await tool.execute(requirements=""))
        assert result["status"] == "ok"


# ══════════════════════════════════════════════════════════════════════════════
# Layer 1 — SplitCheck
# ══════════════════════════════════════════════════════════════════════════════

class TestSplitCheck:
    @pytest.fixture
    def tool(self):
        return SplitCheck()

    async def test_fit_on_train_data_ok(self, tool):
        code = textwrap.dedent("""\
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train_s = scaler.transform(X_train)
            X_test_s = scaler.transform(X_test)
        """)
        result = json.loads(await tool.execute(code=code))
        assert result["status"] == "ok"

    async def test_fit_on_full_data_flagged(self, tool):
        code = textwrap.dedent("""\
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            scaler.fit(X)
        """)
        result = json.loads(await tool.execute(code=code))
        assert result["status"] == "issues_found"
        assert any("X" in issue["call"] for issue in result["issues"])

    async def test_fit_transform_on_full_data_flagged(self, tool):
        code = textwrap.dedent("""\
            from sklearn.preprocessing import StandardScaler
            X_scaled = StandardScaler().fit_transform(X_all)
        """)
        result = json.loads(await tool.execute(code=code))
        assert result["status"] == "issues_found"

    async def test_no_fitting_ok(self, tool):
        code = "y_pred = model.predict(X_test)\nprint(y_pred)"
        result = json.loads(await tool.execute(code=code))
        assert result["status"] == "ok"

    async def test_syntax_error_returns_error_key(self, tool):
        result = json.loads(await tool.execute(code="class ("))
        assert "error" in result

    async def test_line_number_reported(self, tool):
        code = textwrap.dedent("""\
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            scaler.fit(X)
        """)
        result = json.loads(await tool.execute(code=code))
        assert result["issues"][0]["line"] == 3


# ══════════════════════════════════════════════════════════════════════════════
# Layer 1 — CheckpointCheck
# ══════════════════════════════════════════════════════════════════════════════

class TestCheckpointCheck:
    @pytest.fixture
    def tool(self):
        return CheckpointCheck()

    async def test_training_with_save_ok(self, tool):
        code = textwrap.dedent("""\
            import torch
            model = torch.nn.Linear(4, 1)
            for epoch in range(10):
                loss.backward()
                optimizer.step()
            torch.save(model.state_dict(), "model.pt")
        """)
        result = json.loads(await tool.execute(code=code))
        assert result["status"] == "ok"

    async def test_training_without_save_flagged(self, tool):
        code = textwrap.dedent("""\
            import torch
            model = torch.nn.Linear(4, 1)
            for epoch in range(10):
                loss.backward()
                optimizer.step()
        """)
        result = json.loads(await tool.execute(code=code))
        assert result["status"] == "issues_found"

    async def test_sklearn_fit_with_joblib_ok(self, tool):
        code = textwrap.dedent("""\
            from sklearn.ensemble import RandomForestClassifier
            import joblib
            model = RandomForestClassifier()
            model.fit(X_train, y_train)
            joblib.dump(model, "rf_model.pkl")
        """)
        result = json.loads(await tool.execute(code=code))
        assert result["status"] == "ok"

    async def test_no_training_not_applicable(self, tool):
        code = "import numpy as np\nX = np.load('data.npy')"
        result = json.loads(await tool.execute(code=code))
        assert result["status"] == "ok"
        assert "not applicable" in result["message"]

    async def test_keras_fit_with_save_ok(self, tool):
        code = textwrap.dedent("""\
            model.fit(X_train, y_train, epochs=10)
            model.save("my_model.keras")
        """)
        result = json.loads(await tool.execute(code=code))
        assert result["status"] == "ok"

    async def test_syntax_error_returns_error_key(self, tool):
        result = json.loads(await tool.execute(code="for ("))
        assert "error" in result


# ══════════════════════════════════════════════════════════════════════════════
# Layer 2 — LLM-powered tools (MockClient)
# ══════════════════════════════════════════════════════════════════════════════

_SAMPLE_CODE = "import pandas as pd\ndf = pd.read_csv('patients.csv')\n"
_SAMPLE_AUDIT = '{"seed_check": "ok", "split_check": "issues_found"}'


class TestLeakageDetector:
    async def test_returns_llm_response(self):
        mock = MockClient()
        mock.enqueue(_ok_response("No leakage detected."))
        tool = LeakageDetector(client=mock)
        result = await tool.execute(code=_SAMPLE_CODE)
        assert result == "No leakage detected."

    async def test_code_is_sent_to_client(self):
        mock = MockClient()
        tool = LeakageDetector(client=mock)
        await tool.execute(code=_SAMPLE_CODE)
        assert mock.calls[0]["type"] == "complete"
        assert _SAMPLE_CODE in mock.calls[0]["messages"][0]["content"]

    async def test_schema_name(self):
        assert LeakageDetector().schema.name == "leakage_detector"


class TestSubgroupReporter:
    async def test_returns_llm_response(self):
        mock = MockClient()
        mock.enqueue(_ok_response("Age: partial. Sex: no. Race: no."))
        tool = SubgroupReporter(client=mock)
        result = await tool.execute(code=_SAMPLE_CODE)
        assert "Age" in result

    async def test_code_forwarded(self):
        mock = MockClient()
        tool = SubgroupReporter(client=mock)
        await tool.execute(code=_SAMPLE_CODE)
        assert _SAMPLE_CODE in mock.calls[0]["messages"][0]["content"]

    async def test_schema_name(self):
        assert SubgroupReporter().schema.name == "subgroup_reporter"


class TestProvenanceChecker:
    async def test_returns_llm_response(self):
        mock = MockClient()
        mock.enqueue(_ok_response("Dataset source: partial."))
        tool = ProvenanceChecker(client=mock)
        result = await tool.execute(code=_SAMPLE_CODE)
        assert "Dataset source" in result

    async def test_code_forwarded(self):
        mock = MockClient()
        tool = ProvenanceChecker(client=mock)
        await tool.execute(code=_SAMPLE_CODE)
        assert _SAMPLE_CODE in mock.calls[0]["messages"][0]["content"]

    async def test_schema_name(self):
        assert ProvenanceChecker().schema.name == "provenance_checker"


# ══════════════════════════════════════════════════════════════════════════════
# Report generators (MockClient)
# ══════════════════════════════════════════════════════════════════════════════

class TestTripodAIGenerator:
    async def test_returns_llm_response(self):
        mock = MockClient()
        mock.enqueue(_ok_response("TRIPOD-AI checklist: Item 1 ✓"))
        tool = TripodAIGenerator(client=mock)
        result = await tool.execute(audit_results=_SAMPLE_AUDIT)
        assert "TRIPOD-AI" in result

    async def test_audit_results_forwarded(self):
        mock = MockClient()
        tool = TripodAIGenerator(client=mock)
        await tool.execute(audit_results=_SAMPLE_AUDIT)
        assert _SAMPLE_AUDIT in mock.calls[0]["messages"][0]["content"]

    async def test_schema_name(self):
        assert TripodAIGenerator().schema.name == "tripod_ai_generator"


class TestDMSPGenerator:
    async def test_returns_llm_response(self):
        mock = MockClient()
        mock.enqueue(_ok_response("DMSP Draft: 1. Data Type ..."))
        tool = DMSPGenerator(client=mock)
        result = await tool.execute(audit_results=_SAMPLE_AUDIT)
        assert "DMSP" in result

    async def test_audit_results_forwarded(self):
        mock = MockClient()
        tool = DMSPGenerator(client=mock)
        await tool.execute(audit_results=_SAMPLE_AUDIT)
        assert _SAMPLE_AUDIT in mock.calls[0]["messages"][0]["content"]

    async def test_schema_name(self):
        assert DMSPGenerator().schema.name == "dmsp_generator"


# ══════════════════════════════════════════════════════════════════════════════
# Parsers
# ══════════════════════════════════════════════════════════════════════════════

class TestNotebookParser:
    @pytest.fixture
    def tool(self):
        return NotebookParser()

    @pytest.fixture
    def sample_notebook(self, tmp_path):
        nb = {
            "nbformat": 4,
            "nbformat_minor": 5,
            "metadata": {},
            "cells": [
                {"cell_type": "markdown", "source": ["# Title"], "metadata": {}},
                {"cell_type": "code", "source": ["import numpy as np"], "metadata": {}, "outputs": []},
                {"cell_type": "code", "source": ["x = np.array([1, 2, 3])"], "metadata": {}, "outputs": []},
                {"cell_type": "code", "source": ["   \n"], "metadata": {}, "outputs": []},  # blank cell
            ],
        }
        path = tmp_path / "test.ipynb"
        path.write_text(json.dumps(nb), encoding="utf-8")
        return str(path)

    async def test_extracts_code_cells(self, tool, sample_notebook):
        result = await tool.execute(path=sample_notebook)
        assert "import numpy as np" in result
        assert "x = np.array([1, 2, 3])" in result

    async def test_skips_markdown_cells(self, tool, sample_notebook):
        result = await tool.execute(path=sample_notebook)
        assert "# Title" not in result

    async def test_skips_blank_code_cells(self, tool, sample_notebook):
        result = await tool.execute(path=sample_notebook)
        assert result.count("# ── cell ──") == 1  # only one separator between two non-blank cells

    async def test_empty_notebook_returns_message(self, tool, tmp_path):
        nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": {}, "cells": []}
        path = tmp_path / "empty.ipynb"
        path.write_text(json.dumps(nb))
        result = await tool.execute(path=str(path))
        assert "No code cells" in result

    async def test_schema_name(self, tool):
        assert tool.schema.name == "notebook_parser"


class TestScriptParser:
    @pytest.fixture
    def tool(self):
        return ScriptParser()

    async def test_reads_file(self, tool, tmp_path):
        script = tmp_path / "model.py"
        script.write_text("import torch\nprint('hello')", encoding="utf-8")
        result = await tool.execute(path=str(script))
        assert "import torch" in result
        assert "print('hello')" in result

    async def test_schema_name(self, tool):
        assert tool.schema.name == "script_parser"


# ══════════════════════════════════════════════════════════════════════════════
# Registry
# ══════════════════════════════════════════════════════════════════════════════

class TestRegistry:
    def test_all_twelve_tools_registered(self):
        registry = create_default_registry()
        names = {s.name for s in registry.schemas()}
        expected = {
            "seed_check", "dependency_check", "split_check", "checkpoint_check",
            "leakage_detector", "subgroup_reporter", "provenance_checker",
            "tripod_ai_generator", "dmsp_generator",
            "notebook_parser", "script_parser", "repo_fetcher",
        }
        assert names == expected

    async def test_dispatch_layer1_tool(self):
        registry = create_default_registry()
        result = json.loads(
            await registry.dispatch("seed_check", {"code": "import numpy as np\nnp.random.seed(0)"})
        )
        assert result["status"] == "ok"

    async def test_dispatch_unknown_tool_raises(self):
        registry = create_default_registry()
        with pytest.raises(ValueError, match="Unknown tool"):
            await registry.dispatch("nonexistent", {})

    def test_mock_client_shared_with_layer2(self):
        mock = MockClient()
        registry = create_default_registry(client=mock)
        names = {s.name for s in registry.schemas()}
        assert "leakage_detector" in names
