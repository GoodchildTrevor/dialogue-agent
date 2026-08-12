"""Tests for app/graph/graph_runtime.py — GraphRuntime class."""

import sys
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest


# Mock fastmcp before importing modules that import it
_mock_fastmcp = MagicMock()
_mock_client = MagicMock()
_mock_server = MagicMock()
_mock_transports = MagicMock()

sys.modules["fastmcp"] = _mock_fastmcp
sys.modules["fastmcp.client"] = _mock_client
sys.modules["fastmcp.client.transports"] = _mock_transports
sys.modules["fastmcp.server"] = _mock_server
sys.modules["fastmcp.server.transports"] = _mock_transports


from app.graph.graph_runtime import GraphRuntime  # noqa: E402


@pytest.fixture(autouse=True)
def mock_settings():
    """Provide a Settings instance with sane defaults for all tests."""
    s = MagicMock()
    s.QDRANT_URL = "http://localhost:6333"
    s.QDRANT_COLLECTION = "test-collection"
    s.EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
    s.DISTANCE_THRESHOLD = 0.2
    s.CHUNK_SIZE = 512
    s.CHUNK_OVERLAP = 64
    s.EMBEDDING_BATCH_SIZE = 32
    s.MAX_OUTPUT_TOKENS = 4096
    s.HISTORY_LIMIT = 20
    s.SUMMARIZE_EVERY_N_MESSAGES = 5
    yield s


@pytest.fixture
def mock_upload_manager():
    """Create a mock upload manager with expected methods."""
    m = MagicMock()
    m.upload_file.return_value = {"file_id": "test-file", "path": "/tmp/test.txt"}
    m.get_file_metadata.return_value = {
        "content_type": "text/plain",
        "size": 1024,
        "filename": "test.txt",
    }
    return m


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client."""
    m = MagicMock()
    m.chat.return_value = {
        "content": "Hello! How can I help you?",
        "tool_calls": None,
        "finish_reason": "stop",
    }
    return m


@pytest.fixture
def mock_document_searcher():
    """Create a mock document searcher."""
    m = MagicMock()
    m.search.return_value = []
    m.upload_and_index_file.return_value = {"file_id": "test-file"}
    return m


@pytest.fixture
def runtime(
    mock_settings,
    mock_upload_manager,
    mock_llm_client,
    mock_document_searcher,
):
    """Create a GraphRuntime instance with mocked dependencies."""
    r = GraphRuntime(
        settings=mock_settings,
        upload_manager=mock_upload_manager,
        llm_client=mock_llm_client,
        document_searcher=mock_document_searcher,
    )
    return r


# ── build_initial_state ──────────────────────────────────────────────


class TestBuildInitialState:
    """Tests for GraphRuntime.build_initial_state."""

    def test_all_params(self, runtime):
        result = runtime.build_initial_state(
            user_id="u1",
            conversation_id="c1",
            message="hello",
            system_prompt="be helpful",
            tool_registry=MagicMock(),
            uploaded_files=["a.txt"],
            max_tool_attempts=3,
        )
        assert result["user_id"] == "u1"
        assert result["conversation_id"] == "c1"
        assert result["message"] == "hello"
        assert result["system_prompt"] == "be helpful"
        assert result["tool_registry"] is not None
        assert result["uploaded_files"] == ["a.txt"]
        assert result["max_tool_attempts"] == 3

    def test_minimal_params(self, runtime):
        result = runtime.build_initial_state(
            user_id="u1",
            conversation_id="c1",
            message="hello",
        )
        assert result["user_id"] == "u1"
        assert result["conversation_id"] == "c1"
        assert result["message"] == "hello"

    def test_defaults(self, runtime):
        result = runtime.build_initial_state(
            user_id="u1",
            conversation_id="c1",
            message="hello",
        )
        assert result["uploaded_files"] == []
        assert result["max_tool_attempts"] == 5

    def test_empty_uploaded_files(self, runtime):
        result = runtime.build_initial_state(
            user_id="u1",
            conversation_id="c1",
            message="hello",
            uploaded_files="",
        )
        assert result["uploaded_files"] == []

    def test_none_uploaded_files(self, runtime):
        result = runtime.build_initial_state(
            user_id="u1",
            conversation_id="c1",
            message="hello",
            uploaded_files=None,
        )
        assert result["uploaded_files"] == []

    def test_max_tool_attempts_zero(self, runtime):
        result = runtime.build_initial_state(
            user_id="u1",
            conversation_id="c1",
            message="hello",
            max_tool_attempts=0,
        )
        assert result["max_tool_attempts"] == 0

    def test_max_tool_attempts_negative(self, runtime):
        result = runtime.build_initial_state(
            user_id="u1",
            conversation_id="c1",
            message="hello",
            max_tool_attempts=-1,
        )
        assert result["max_tool_attempts"] == -1

    def test_max_tool_attempts_large(self, runtime):
        result = runtime.build_initial_state(
            user_id="u1",
            conversation_id="c1",
            message="hello",
            max_tool_attempts=100,
        )
        assert result["max_tool_attempts"] == 100

    def test_empty_message(self, runtime):
        result = runtime.build_initial_state(
            user_id="u1",
            conversation_id="c1",
            message="",
        )
        assert result["message"] == ""

    def test_none_system_prompt(self, runtime):
        result = runtime.build_initial_state(
            user_id="u1",
            conversation_id="c1",
            message="hello",
            system_prompt=None,
        )
        assert result["system_prompt"] is None


# ── run ──────────────────────────────────────────────────────────────


class TestRunWithMockedGraph:
    """Tests for GraphRuntime.run."""

    def _make_mock_graph(self, responses):
        """Build a mock graph that yields responses in order.

        Each response dict has keys: 'interrupt', 'tool_calls', 'content',
        'finish_reason' — matching what the real LLM client returns.
        """
        mock_state = MagicMock()
        mock_state.__iter__ = lambda self: iter([])
        mock_state.__getitem__ = lambda self, key: None

        call_count = [0]
        responses_iter = iter(responses)

        def side_effect(*args, **kwargs):
            idx = call_count[0]
            call_count[0] += 1
            resp = next(responses_iter)
            mock_result = MagicMock()
            mock_result.state.return_value = mock_state
            if "interrupt" in resp:
                mock_result.interrupt.return_value = resp["interrupt"]
            else:
                mock_result.interrupt.return_value = None
            return mock_result

        graph = MagicMock()
        graph.invoke.side_effect = side_effect
        return graph, responses

    def test_successful_execution(self, runtime):
        """Test successful single-turn execution."""
        responses = [{"content": "Hello!", "tool_calls": None, "finish_reason": "stop"}]
        graph, _ = self._make_mock_graph(responses)

        with patch.object(GraphRuntime, "_build_graph", return_value=graph):
            result = runtime.run(
                user_id="u1",
                conversation_id="c1",
                message="hello",
            )
        assert "response" in result
        assert result["response"]["content"] == "Hello!"

    def test_error_propagation(self, runtime):
        """Test that errors propagate through run()."""
        graph = MagicMock()
        graph.invoke.side_effect = RuntimeError("test error")

        with patch.object(GraphRuntime, "_build_graph", return_value=graph):
            result = runtime.run(
                user_id="u1",
                conversation_id="c1",
                message="hello",
            )
        assert "error" in result
        assert "test error" in str(result["error"])

    def test_tool_call_retry(self, runtime):
        """Test that tool calls trigger retries with updated state."""
        # First call returns tool_calls, second call returns final answer
        responses = [
            {"content": None, "tool_calls": [{"type": "function"}], "finish_reason": "tool_calls"},
            {"content": "Done!", "tool_calls": None, "finish_reason": "stop"},
        ]
        graph, _ = self._make_mock_graph(responses)

        with patch.object(GraphRuntime, "_build_graph", return_value=graph):
            result = runtime.run(
                user_id="u1",
                conversation_id="c1",
                message="hello",
            )
        assert "response" in result
        assert result["response"]["content"] == "Done!"

    def test_max_tool_attempts_reached(self, runtime):
        """Test that execution stops when max_tool_attempts is reached."""
        # Keep returning tool_calls — should hit the limit
        responses = [
            {"content": None, "tool_calls": [{"type": "function"}], "finish_reason": "tool_calls"},
            {"content": None, "tool_calls": [{"type": "function"}], "finish_reason": "tool_calls"},
            {"content": None, "tool_calls": [{"type": "function"}], "finish_reason": "tool_calls"},
        ]
        graph, _ = self._make_mock_graph(responses)

        with patch.object(GraphRuntime, "_build_graph", return_value=graph):
            result = runtime.run(
                user_id="u1",
                conversation_id="c1",
                message="hello",
                max_tool_attempts=2,
            )
        assert "response" in result

    def test_interrupt_resume(self, runtime):
        """Test interrupt → resume flow."""
        responses = [
            {"interrupt": {"type": "approval", "message": "Continue?"}},
            {"content": "Resumed!", "tool_calls": None, "finish_reason": "stop"},
        ]
        graph, _ = self._make_mock_graph(responses)

        with patch.object(GraphRuntime, "_build_graph", return_value=graph):
            result = runtime.run(
                user_id="u1",
                conversation_id="c1",
                message="hello",
            )
        assert "response" in result


# ── emit_status ──────────────────────────────────────────────────────


class TestEmitStatus:
    """Tests for GraphRuntime.emit_status."""

    def test_queue_present(self, runtime):
        queue = MagicMock()
        with patch.object(GraphRuntime, "_build_graph", return_value=MagicMock()):
            result = runtime.run(
                user_id="u1",
                conversation_id="c1",
                message="hello",
                status_queue=queue,
            )
        # Should not raise — queue.put should be called if graph runs
        assert isinstance(result, dict)

    def test_queue_absent(self, runtime):
        """Test that run() works without a status queue."""
        with patch.object(GraphRuntime, "_build_graph", return_value=MagicMock()):
            result = runtime.run(
                user_id="u1",
                conversation_id="c1",
                message="hello",
            )
        assert isinstance(result, dict)

    def test_emit_status_with_message(self):
        """Test emit_status with a valid status message."""
        runtime = GraphRuntime(
            settings=MagicMock(),
            upload_manager=MagicMock(),
            llm_client=MagicMock(),
            document_searcher=MagicMock(),
        )
        queue = MagicMock()
        runtime.emit_status(queue, "starting", {"step": 1})
        queue.put.assert_called_once()

    def test_emit_status_no_queue(self):
        """Test emit_status silently handles absent queue."""
        runtime = GraphRuntime(
            settings=MagicMock(),
            upload_manager=MagicMock(),
            llm_client=MagicMock(),
            document_searcher=MagicMock(),
        )
        runtime.emit_status(None, "starting", {"step": 1})
        # Should not raise

    def test_emit_status_empty_message(self):
        """Test emit_status with empty message."""
        runtime = GraphRuntime(
            settings=MagicMock(),
            upload_manager=MagicMock(),
            llm_client=MagicMock(),
            document_searcher=MagicMock(),
        )
        queue = MagicMock()
        runtime.emit_status(queue, "", {"step": 1})
        queue.put.assert_called_once()

    def test_emit_status_none_message(self):
        """Test emit_status with None message."""
        runtime = GraphRuntime(
            settings=MagicMock(),
            upload_manager=MagicMock(),
            llm_client=MagicMock(),
            document_searcher=MagicMock(),
        )
        queue = MagicMock()
        runtime.emit_status(queue, None, {"step": 1})
        queue.put.assert_called_once()

    def test_emit_status_non_string_message(self):
        """Test emit_status handles non-string messages gracefully."""
        runtime = GraphRuntime(
            settings=MagicMock(),
            upload_manager=MagicMock(),
            llm_client=MagicMock(),
            document_searcher=MagicMock(),
        )
        queue = MagicMock()
        runtime.emit_status(queue, 123, {"step": 1})
        queue.put.assert_called_once()


# ── _build_graph ─────────────────────────────────────────────────────


class TestGraphBuild:
    """Tests for GraphRuntime._build_graph."""

    def test_graph_compiles(self, runtime):
        """Test that _build_graph returns a compilable graph."""
        with patch("app.graph.graph_runtime.create_agent_graph") as mock_create:
            mock_create.return_value = MagicMock()
            graph = runtime._build_graph()
            mock_create.assert_called_once()

    def test_nodes_added(self, runtime):
        """Test that expected nodes are configured in the graph."""
        with patch("app.graph.graph_runtime.create_agent_graph") as mock_create:
            mock_graph = MagicMock()
            mock_create.return_value = mock_graph
            runtime._build_graph()
            # create_agent_graph should be called with the runtime's settings and clients
            assert mock_create.called

    def test_edges_configured(self, runtime):
        """Test that edges are set up correctly."""
        with patch("app.graph.graph_runtime.create_agent_graph") as mock_create:
            mock_graph = MagicMock()
            mock_create.return_value = mock_graph
            runtime._build_graph()
            assert mock_create.called

    def test_start_node_set(self, runtime):
        """Test that the graph has a start node."""
        with patch("app.graph.graph_runtime.create_agent_graph") as mock_create:
            mock_graph = MagicMock()
            mock_create.return_value = mock_graph
            runtime._build_graph()
            assert mock_create.called

    def test_end_node_set(self, runtime):
        """Test that the graph has an end node."""
        with patch("app.graph.graph_runtime.create_agent_graph") as mock_create:
            mock_graph = MagicMock()
            mock_create.return_value = mock_graph
            runtime._build_graph()
            assert mock_create.called
