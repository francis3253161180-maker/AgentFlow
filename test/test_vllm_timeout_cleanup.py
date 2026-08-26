import asyncio
import os
import unittest
from types import SimpleNamespace

from agentflow.server import AgentFlowServer, ServerDataStore
from agentflow.verl.async_server import PatchedvLLMServer


class _FakeOutputProcessor:
    def __init__(self, request_ids):
        self.request_states = {request_id: object() for request_id in request_ids}

    def get_num_unfinished_requests(self):
        return len(self.request_states)


class _FakeEngine:
    def __init__(self, request_ids=()):
        self.output_processor = _FakeOutputProcessor(request_ids)
        self.aborted = []
        self.reset_called = False
        self.sleep_called = False

    def get_num_unfinished_requests(self):
        return self.output_processor.get_num_unfinished_requests()

    async def abort(self, request_id):
        self.aborted.append(request_id)
        self.output_processor.request_states.pop(request_id, None)

    async def reset_prefix_cache(self):
        self.reset_called = True

    async def sleep(self):
        self.sleep_called = True

    async def check_health(self):
        return None


def _fake_server(engine, active_chat_requests=0):
    actor_class = PatchedvLLMServer.__ray_actor_class__
    server = object.__new__(actor_class)
    server.engine = engine
    server.config = SimpleNamespace(rollout=SimpleNamespace(free_cache_engine=True))
    server._request_state_lock = asyncio.Lock()
    server._cleanup_lock = asyncio.Lock()
    server._accepting_requests = True
    server._active_chat_requests = active_chat_requests
    server._test_hold_requests_seen = 0
    server._test_hold_requests_used = 0
    return server


class VLLMTimeoutCleanupTest(unittest.IsolatedAsyncioTestCase):
    async def test_abort_drain_precedes_reset_and_sleep(self):
        server = _fake_server(_FakeEngine(["request-a", "request-b"]))
        result = await server.cleanup(reason="unit_timeout")

        self.assertTrue(result["drained"])
        self.assertEqual(server.engine.aborted, ["request-a", "request-b"])
        self.assertTrue(server.engine.reset_called)
        self.assertTrue(server.engine.sleep_called)

    async def test_active_handler_times_out_without_reset(self):
        old_timeout = os.environ.get("AGENTFLOW_VLLM_CLEANUP_DRAIN_TIMEOUT_SECONDS")
        os.environ["AGENTFLOW_VLLM_CLEANUP_DRAIN_TIMEOUT_SECONDS"] = "0.1"
        try:
            server = _fake_server(_FakeEngine(), active_chat_requests=1)
            result = await server.cleanup(reason="unit_active_handler")
        finally:
            if old_timeout is None:
                os.environ.pop("AGENTFLOW_VLLM_CLEANUP_DRAIN_TIMEOUT_SECONDS", None)
            else:
                os.environ["AGENTFLOW_VLLM_CLEANUP_DRAIN_TIMEOUT_SECONDS"] = old_timeout

        self.assertFalse(result["drained"])
        self.assertFalse(result["sleep_started"])
        self.assertFalse(server.engine.reset_called)

    async def test_task_queue_rejects_new_work_after_stop(self):
        server = AgentFlowServer()
        server._store = ServerDataStore()
        await server.queue_task({"question": "before cleanup"})
        await server.stop_accepting_tasks(reason="unit_timeout")

        self.assertIsNone(await server._store.get_next_task())
        with self.assertRaises(RuntimeError):
            await server.queue_task({"question": "during cleanup"})


if __name__ == "__main__":
    unittest.main()
