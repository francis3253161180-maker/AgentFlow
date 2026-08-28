import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agentflow.agentflow.engine.role_routing import (
    ACTOR_ROLE,
    BASE_ROLE,
    read_actor_route,
    resolve_role,
    write_actor_route,
)


class RoleRoutingUnitTest(unittest.TestCase):
    def test_base_isolated_and_actor_requires_route(self):
        self.assertEqual(resolve_role(BASE_ROLE)["adapter"], None)
        with self.assertRaises(LookupError):
            resolve_role(ACTOR_ROLE)
        resolved = resolve_role(
            ACTOR_ROLE,
            {"lora_name": "431", "lora_int_id": 431, "version": "v2"},
        )
        self.assertEqual(resolved["model"], ACTOR_ROLE)
        self.assertEqual(resolved["adapter"]["lora_int_id"], 431)

    def test_atomic_registry_refreshes_latest_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "routes.json")
            first_hash = write_actor_route(
                path, lora_name="101", lora_int_id=101, version="v1", base_model="qwen-base"
            )
            self.assertEqual(len(first_hash), 64)
            self.assertEqual(read_actor_route(path)["lora_int_id"], 101)
            write_actor_route(path, lora_name="202", lora_int_id=202, version="v2", base_model="qwen-base")
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            self.assertNotIn("question", payload)
            self.assertEqual(read_actor_route(path)["lora_int_id"], 202)


class _Models:
    def __init__(self):
        self.base_model_paths = [SimpleNamespace(name="models/Qwen2.5-7B-Instruct", model_path="/model")]
        self.lora_requests = []


class UnifiedServerRouteTest(unittest.IsolatedAsyncioTestCase):
    async def test_server_maps_actor_alias_and_keeps_base_without_adapter(self):
        from agentflow.verl.async_server import PatchedvLLMServer

        actor_class = PatchedvLLMServer.__ray_actor_class__
        server = object.__new__(actor_class)
        server._role_route_path = None
        server._actor_route_version = None
        server.openai_serving_chat = SimpleNamespace(models=_Models())
        server.config = SimpleNamespace(model=SimpleNamespace(path="/model"))
        server._role_route_path = tempfile.NamedTemporaryFile(delete=False).name
        route_path = Path(server._role_route_path)
        route_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "base_model": "qwen-base",
                    "actor": {"lora_name": "303", "lora_int_id": 303, "version": "v3"},
                }
            ),
            encoding="utf-8",
        )
        try:
            prepared = server._prepare_role_request({"model": ACTOR_ROLE})
            self.assertEqual(prepared["model"], ACTOR_ROLE)
            self.assertEqual(server.openai_serving_chat.models.lora_requests[0].lora_int_id, 303)
            self.assertEqual(server.openai_serving_chat.models.lora_requests[0].lora_path, "/model")
            server._prepare_role_request({"model": BASE_ROLE})
            self.assertEqual(len(server.openai_serving_chat.models.lora_requests), 1)
        finally:
            route_path.unlink(missing_ok=True)

    async def test_server_rejects_actor_before_first_sync(self):
        from agentflow.verl.async_server import PatchedvLLMServer

        actor_class = PatchedvLLMServer.__ray_actor_class__
        server = object.__new__(actor_class)
        server._role_route_path = "/path/that/does/not/exist"
        server._actor_route_version = None
        with self.assertRaises(RuntimeError):
            server._prepare_role_request({"model": ACTOR_ROLE})


if __name__ == "__main__":
    unittest.main()
