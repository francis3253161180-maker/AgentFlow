import ray
import asyncio
import logging
import os
import time
from copy import deepcopy

from agentflow.instrumentation.vllm import instrument_vllm, ChatCompletionResponsePatched
from starlette.requests import Request
from starlette.background import BackgroundTask
from starlette.responses import JSONResponse, StreamingResponse
from vllm.entrypoints.openai.protocol import ChatCompletionRequest, ErrorResponse
from vllm.entrypoints.openai.serving_models import BaseModelPath
from vllm.lora.request import LoRARequest
from verl.workers.rollout.vllm_rollout.vllm_async_server import AsyncvLLMServer

from agentflow.engine.role_routing import ACTOR_ROLE, BASE_ROLE, read_actor_route, route_state_path


logger = logging.getLogger(__name__)


def _unwrap_ray_remote(cls):
    if hasattr(cls, "__ray_actor_class__"):
        cls = cls.__ray_actor_class__
    return cls


@ray.remote(num_cpus=1)
class PatchedvLLMServer(_unwrap_ray_remote(AsyncvLLMServer)):

    def __init__(self, *args, **kwargs):
        instrument_vllm()
        super().__init__(*args, **kwargs)

        self.config = deepcopy(self.config)
        self.config.rollout.multi_turn.tool_config_path = "/dev/null"
        # vLLM 0.9.2 has no pause_generation()/wait_for_requests_to_drain().
        # These actor-local gates provide the missing lifecycle boundary while
        # preserving the existing OpenAI-compatible endpoint.
        self._request_state_lock = asyncio.Lock()
        self._cleanup_lock = asyncio.Lock()
        self._accepting_requests = True
        self._active_chat_requests = 0
        self._test_hold_requests_seen = 0
        self._test_hold_requests_used = 0
        self._role_route_path = route_state_path()
        self._base_model_name = None
        self._actor_route_version = None

    def _request_ids(self):
        """Return vLLM request ids using the 0.9.2 public-ish state surface."""
        output_processor = getattr(self.engine, "output_processor", None)
        request_states = getattr(output_processor, "request_states", None)
        if isinstance(request_states, dict):
            return list(request_states.keys())
        return []

    def _unfinished_requests(self) -> int:
        """Return a conservative outstanding-request count."""
        getter = getattr(self.engine, "get_num_unfinished_requests", None)
        if callable(getter):
            return int(getter())
        return len(self._request_ids())

    async def _release_chat_request(self):
        async with self._request_state_lock:
            self._active_chat_requests = max(0, self._active_chat_requests - 1)

    async def _begin_chat_request(self) -> bool:
        async with self._request_state_lock:
            if not self._accepting_requests:
                return False
            self._active_chat_requests += 1
            return True

    async def wake_up(self):
        await super().wake_up()
        async with self._request_state_lock:
            self._accepting_requests = True
        logger.info("VLLM_CLEANUP wake_up complete; request acceptance resumed")

    async def init_engine(self):
        await super().init_engine()
        # Keep the normal model id for VERL compatibility and add one explicit
        # base alias. Fixed AgentFlow roles use only this no-adapter alias.
        self._base_model_name = self.openai_serving_chat.models.base_model_paths[0].name
        if not any(model.name == BASE_ROLE for model in self.openai_serving_chat.models.base_model_paths):
            self.openai_serving_chat.models.base_model_paths.append(
                BaseModelPath(name=BASE_ROLE, model_path=self.config.model.path)
            )

    def _refresh_actor_route(self):
        """Expose the latest worker-loaded adapter through a stable HTTP alias."""

        route = read_actor_route(self._role_route_path)
        if not route or route.get("version") == self._actor_route_version:
            return route
        models = self.openai_serving_chat.models
        models.lora_requests = [item for item in models.lora_requests if item.lora_name != ACTOR_ROLE]
        models.lora_requests.append(
            LoRARequest(
                lora_name=ACTOR_ROLE,
                lora_int_id=route["lora_int_id"],
                # The adapter is registered in-memory by VERL's
                # TensorLoRARequest.  Point the registry's tokenizer lookup at
                # the already-loaded base model; it must not treat the
                # ephemeral numeric adapter id as a filesystem path.
                lora_path=str(self.config.model.path),
                base_model_name=BASE_ROLE,
            )
        )
        self._actor_route_version = route["version"]
        logger.info(
            "UNIFIED_ROLE_ROUTE adapter_registered role=%s adapter_id=%s version=%s",
            ACTOR_ROLE,
            route["lora_int_id"],
            route["version"],
        )
        return route

    def _prepare_role_request(self, request_json):
        requested_model = request_json.get("model")
        if requested_model == ACTOR_ROLE:
            route = self._refresh_actor_route()
            if not route:
                raise RuntimeError("qwen-actor requested before a synchronized LoRA adapter is available")
            logger.info(
                "UNIFIED_ROLE_ROUTE request role=%s adapter_id=%s version=%s",
                ACTOR_ROLE,
                route["lora_int_id"],
                route["version"],
            )
        elif requested_model == BASE_ROLE:
            # No adapter object is attached to this request: fixed roles are
            # isolated from the trainable planner adapter.
            logger.info("UNIFIED_ROLE_ROUTE request role=%s adapter_id=none", BASE_ROLE)
        return request_json

    async def _abort_and_drain(self, reason: str):
        """Abort vLLM 0.9.2 requests and wait for both layers to quiesce."""
        timeout = float(os.environ.get("AGENTFLOW_VLLM_CLEANUP_DRAIN_TIMEOUT_SECONDS", "30"))
        poll_interval = float(os.environ.get("AGENTFLOW_VLLM_CLEANUP_DRAIN_POLL_SECONDS", "0.05"))
        deadline = time.monotonic() + max(0.1, timeout)
        initial_ids = self._request_ids()
        async with self._request_state_lock:
            active_count = self._active_chat_requests
        initial_count = max(len(initial_ids), self._unfinished_requests(), active_count)
        logger.info(
            "VLLM_CLEANUP trigger=%s outstanding=%d active_http=%d abort_start=1",
            reason,
            initial_count,
            active_count,
        )

        abort_errors = []
        while time.monotonic() < deadline:
            request_ids = self._request_ids()
            for request_id in request_ids:
                try:
                    # vLLM 0.9.2 awaits EngineCore abort acknowledgement here.
                    await self.engine.abort(request_id)
                except Exception as exc:  # pragma: no cover - engine-specific
                    abort_errors.append(type(exc).__name__)
                    logger.warning("VLLM_CLEANUP abort_failed request=%s error=%s", request_id, type(exc).__name__)

            async with self._request_state_lock:
                active_count = self._active_chat_requests
            remaining = self._unfinished_requests()
            if not self._request_ids() and remaining == 0 and active_count == 0:
                elapsed = max(0.0, timeout - max(0.0, deadline - time.monotonic()))
                logger.info(
                    "VLLM_CLEANUP drain_complete=1 outstanding_before=%d abort_count=%d "
                    "abort_errors=%d drain_duration=%.3fs",
                    initial_count,
                    len(initial_ids),
                    len(abort_errors),
                    elapsed,
                )
                return {
                    "drained": True,
                    "outstanding_before": initial_count,
                    "abort_count": len(initial_ids),
                    "abort_errors": len(abort_errors),
                    "drain_duration_seconds": elapsed,
                }
            await asyncio.sleep(max(0.001, poll_interval))

        async with self._request_state_lock:
            active_count = self._active_chat_requests
        remaining = self._unfinished_requests()
        logger.error(
            "VLLM_CLEANUP drain_complete=0 outstanding_after=%d active_http=%d "
            "abort_errors=%d sleep=skipped reason=drain_timeout",
            remaining,
            active_count,
            len(abort_errors),
        )
        return {
            "drained": False,
            "outstanding_before": initial_count,
            "outstanding_after": remaining,
            "abort_count": len(initial_ids),
            "abort_errors": len(abort_errors),
            "drain_duration_seconds": timeout,
        }

    async def cleanup(self, reason: str = "manager_sleep"):
        """Safely drain vLLM before prefix-cache reset and optional sleep."""
        started = time.monotonic()
        async with self._cleanup_lock:
            async with self._request_state_lock:
                self._accepting_requests = False
            result = await self._abort_and_drain(reason)
            if not result["drained"]:
                result.update({"sleep_started": False, "cleanup_duration_seconds": time.monotonic() - started})
                return result

            logger.info("VLLM_CLEANUP reset_prefix_cache start=1")
            await self.engine.reset_prefix_cache()
            logger.info("VLLM_CLEANUP reset_prefix_cache complete=1")
            sleep_started = False
            if self.config.rollout.free_cache_engine:
                sleep_started = True
                logger.info("VLLM_CLEANUP sleep_start=1")
                await self.engine.sleep()
                logger.info("VLLM_CLEANUP sleep_complete=1")
            result.update({
                "sleep_started": sleep_started,
                "cleanup_duration_seconds": time.monotonic() - started,
            })
            logger.info(
                "VLLM_CLEANUP complete=1 drained=1 duration=%.3fs",
                result["cleanup_duration_seconds"],
            )
            return result

    async def sleep(self):
        """Compatibility entry point for the old AsyncLLMServerManager.sleep()."""
        return await self.cleanup(reason="manager_sleep")

    async def health_check(self):
        """Post-cleanup health check used by the forced-timeout smoke."""
        await self.engine.check_health()
        remaining = self._unfinished_requests()
        if remaining:
            raise RuntimeError(f"health check saw {remaining} unfinished requests")
        logger.info("VLLM_CLEANUP health_check=ok outstanding=0")
        return {"status": "ok", "outstanding": 0}

    async def chat_completion(self, raw_request: Request):
        """OpenAI-compatible HTTP endpoint.

        API reference: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
        """
        if not await self._begin_chat_request():
            return JSONResponse(
                content={
                    "error": {
                        "message": "rollout engine is draining",
                        "type": "server_error",
                    }
                },
                status_code=503,
            )

        request = None
        stream_response = False
        try:
            try:
                request_json = self._prepare_role_request(await raw_request.json())
            except RuntimeError as exc:
                logger.warning("UNIFIED_ROLE_ROUTE rejected request reason=%s", str(exc))
                return JSONResponse(
                    content={"error": {"message": str(exc), "type": "server_error"}},
                    status_code=503,
                )
            request = ChatCompletionRequest(**request_json)
            completion = self.openai_serving_chat.create_chat_completion(request, raw_request)
            test_hold = float(os.environ.get("AGENTFLOW_VLLM_TEST_REQUEST_HOLD_SECONDS", "0"))
            hold_limit = int(os.environ.get("AGENTFLOW_VLLM_TEST_REQUEST_HOLD_COUNT", "1"))
            hold_after = int(os.environ.get("AGENTFLOW_VLLM_TEST_REQUEST_HOLD_AFTER", "0"))
            async with self._request_state_lock:
                request_index = self._test_hold_requests_seen
                self._test_hold_requests_seen += 1
                should_hold = (
                    test_hold > 0
                    and request_index >= max(0, hold_after)
                    and self._test_hold_requests_used < max(0, hold_limit)
                )
                if should_hold:
                    self._test_hold_requests_used += 1
            generator = await completion
            if should_hold:
                # Smoke-only scheduling aid: vLLM has completed a real request;
                # the hold keeps its HTTP handler
                # active so timeout cleanup cannot race reset/sleep.
                logger.info("VLLM_CLEANUP test_request_hold=%.1fs", test_hold)
                await asyncio.sleep(test_hold)

            if isinstance(generator, ErrorResponse):
                return JSONResponse(content=generator.model_dump(), status_code=generator.code)
            if request.stream:
                # Keep the active-request gate closed until the stream is fully consumed.
                stream_response = True
                return StreamingResponse(
                    content=generator,
                    media_type="text/event-stream",
                    background=BackgroundTask(self._release_chat_request),
                )
            return JSONResponse(content=generator.model_dump())
        finally:
            if not stream_response:
                await self._release_chat_request()
