"""Centralized Debug and Tracing Engine for WorkOS.

Provides full-lifecycle observability into every request, LLM generation,
tool execution, memory operation, JSON parse attempt, and error.
Events are recorded both in an append-only JSONL/text log file and an
in-memory circular buffer for real-time inspection via the UI and API.
"""

from __future__ import annotations

import asyncio
import collections
import dataclasses
import json
import logging
import os
import threading
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

EventType = Literal[
    "LLM_PROMPT",
    "LLM_RESPONSE",
    "JSON_PARSE",
    "PLAN_FORMULATED",
    "STEP_START",
    "STEP_COMPLETE",
    "TOOL_CALL",
    "TOOL_RESULT",
    "MEMORY_OP",
    "CHAT_INTENT",
    "SYSTEM_ERROR",
    "INFRA_EVENT",
]


@dataclasses.dataclass
class TraceEvent:
    id: str
    timestamp: str
    level: str  # DEBUG, INFO, WARNING, ERROR
    event_type: EventType
    component: str
    message: str
    session_id: str | None = None
    payload: dict[str, Any] = dataclasses.field(default_factory=dict)
    duration_ms: float | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "level": self.level,
            "event_type": self.event_type,
            "component": self.component,
            "message": self.message,
            "session_id": self.session_id,
            "payload": self.payload,
            "duration_ms": self.duration_ms,
            "error": self.error,
        }


class DebugTraceManager:
    """Manages recording, buffering, and querying of execution traces."""

    def __init__(self, log_dir: str | None = None, buffer_size: int = 1000):
        self._lock = threading.Lock()
        self._buffer: collections.deque[TraceEvent] = collections.deque(maxlen=buffer_size)
        self._event_counter = 0
        self._subscribers: list[tuple[asyncio.Queue, str | None, asyncio.AbstractEventLoop | None]] = []

        target_dir = log_dir or os.getenv("WORKOS_LOG_DIR", "data/logs")
        self.log_dir = Path(target_dir).resolve()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "workos_debug.log"

    def record(
        self,
        event_type: EventType,
        component: str,
        message: str,
        level: str = "INFO",
        session_id: str | None = None,
        payload: dict[str, Any] | None = None,
        duration_ms: float | None = None,
        error: str | None = None,
    ) -> TraceEvent:
        """Records a structured trace event to the ring buffer and persistent log file."""
        now = datetime.now()
        ts_str = now.isoformat()

        with self._lock:
            self._event_counter += 1
            event_id = f"trc_{int(now.timestamp())}_{self._event_counter:04d}"

            event = TraceEvent(
                id=event_id,
                timestamp=ts_str,
                level=level.upper(),
                event_type=event_type,
                component=component,
                message=message,
                session_id=session_id,
                payload=payload or {},
                duration_ms=round(duration_ms, 2) if duration_ms is not None else None,
                error=error,
            )
            self._buffer.append(event)

            # Persist to disk
            try:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(event.to_dict()) + "\n")
            except Exception as e:
                logger.debug(f"Failed to append to debug log file: {e}")

            # Broadcast to live subscribers (e.g. SSE stream listeners)
            for q, sub_sess, loop in list(self._subscribers):
                if sub_sess is None or sub_sess == session_id or session_id is None:
                    try:
                        if loop and loop.is_running():
                            loop.call_soon_threadsafe(q.put_nowait, event)
                        else:
                            q.put_nowait(event)
                    except Exception:
                        pass

            return event

    def subscribe(
        self, session_id: str | None = None, loop: asyncio.AbstractEventLoop | None = None
    ) -> asyncio.Queue[TraceEvent]:
        """Subscribes an asyncio.Queue to real-time trace events (with optional session filter)."""
        q: asyncio.Queue[TraceEvent] = asyncio.Queue()
        if loop is None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
        with self._lock:
            self._subscribers.append((q, session_id, loop))
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        """Unregisters an active subscriber queue."""
        with self._lock:
            self._subscribers = [item for item in self._subscribers if item[0] is not q]

    def log_prompt(
        self,
        component: str,
        prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
        format_type: str | None = None,
        temperature: float | None = None,
        session_id: str | None = None,
    ) -> TraceEvent:
        """Logs an outgoing prompt to the LLM."""
        return self.record(
            event_type="LLM_PROMPT",
            component=component,
            message=f"Prompt dispatched to {model or 'LLM'}",
            level="DEBUG",
            session_id=session_id,
            payload={
                "model": model,
                "format": format_type,
                "temperature": temperature,
                "system_prompt": system_prompt,
                "user_prompt": prompt,
                "prompt_length_chars": len(prompt) + len(system_prompt or ""),
            },
        )

    def log_llm_response(
        self,
        component: str,
        response_text: str,
        model: str | None = None,
        duration_ms: float | None = None,
        session_id: str | None = None,
        error: str | None = None,
    ) -> TraceEvent:
        """Logs an incoming response from the LLM."""
        level = "ERROR" if error else "DEBUG"
        msg = (
            f"LLM generation failed: {error}"
            if error
            else f"Received response from {model or 'LLM'}"
        )
        return self.record(
            event_type="LLM_RESPONSE",
            component=component,
            message=msg,
            level=level,
            session_id=session_id,
            duration_ms=duration_ms,
            error=error,
            payload={
                "model": model,
                "response_text": response_text,
                "response_length_chars": len(response_text or ""),
            },
        )

    def log_json_parse(
        self,
        component: str,
        raw_text: str,
        success: bool,
        parsed_data: Any = None,
        strategy: str = "direct",
        error: str | None = None,
        session_id: str | None = None,
    ) -> TraceEvent:
        """Logs a JSON parse and any auto-repair attempts."""
        level = "DEBUG" if success else "WARNING"
        msg = f"JSON parsed via strategy '{strategy}'" if success else f"JSON parse failed: {error}"
        return self.record(
            event_type="JSON_PARSE",
            component=component,
            message=msg,
            level=level,
            session_id=session_id,
            error=error,
            payload={
                "strategy": strategy,
                "success": success,
                "raw_sample": (raw_text or "")[:400],
                "parsed_keys": list(parsed_data.keys())
                if isinstance(parsed_data, dict)
                else type(parsed_data).__name__,
            },
        )

    def log_plan(
        self,
        component: str,
        goal: str,
        steps: list[dict[str, Any]],
        fallback: bool = False,
        reason: str | None = None,
        session_id: str | None = None,
    ) -> TraceEvent:
        """Logs an execution plan."""
        level = "WARNING" if fallback else "INFO"
        msg = (
            f"Fallback plan generated ({reason})"
            if fallback
            else f"Formulated {len(steps)}-step execution plan"
        )
        return self.record(
            event_type="PLAN_FORMULATED",
            component=component,
            message=msg,
            level=level,
            session_id=session_id,
            payload={
                "goal": goal,
                "step_count": len(steps),
                "steps": steps,
                "is_fallback": fallback,
                "fallback_reason": reason,
            },
        )

    def log_tool_call(
        self,
        agent_name: str,
        tool_name: str,
        args: dict[str, Any],
        session_id: str | None = None,
    ) -> TraceEvent:
        """Logs a tool execution invocation."""
        return self.record(
            event_type="TOOL_CALL",
            component=agent_name,
            message=f"Dispatching tool '{tool_name}'",
            level="DEBUG",
            session_id=session_id,
            payload={"agent": agent_name, "tool": tool_name, "arguments": args},
        )

    def log_tool_result(
        self,
        agent_name: str,
        tool_name: str,
        success: bool,
        data: Any = None,
        error: str | None = None,
        duration_ms: float | None = None,
        session_id: str | None = None,
    ) -> TraceEvent:
        """Logs the outcome of a tool execution."""
        level = "DEBUG" if success else "WARNING"
        msg = (
            f"Tool '{tool_name}' finished successfully"
            if success
            else f"Tool '{tool_name}' failed: {error}"
        )
        return self.record(
            event_type="TOOL_RESULT",
            component=agent_name,
            message=msg,
            level=level,
            session_id=session_id,
            duration_ms=duration_ms,
            error=error,
            payload={
                "agent": agent_name,
                "tool": tool_name,
                "success": success,
                "data_preview": str(data)[:300] if data is not None else None,
            },
        )

    def log_error(
        self,
        component: str,
        error: Exception | str,
        context: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> TraceEvent:
        """Logs a system exception or failure."""
        err_msg = str(error)
        tb_str = traceback.format_exc() if isinstance(error, Exception) else None
        return self.record(
            event_type="SYSTEM_ERROR",
            component=component,
            message=f"Error in {component}: {err_msg}",
            level="ERROR",
            session_id=session_id,
            error=err_msg,
            payload={"traceback": tb_str, "context": context or {}},
        )

    def get_events(
        self,
        limit: int = 100,
        event_type: str | None = None,
        session_id: str | None = None,
        level: str | None = None,
        search: str | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieves filtered trace events from the in-memory ring buffer (newest first)."""
        with self._lock:
            events = list(self._buffer)

        # Apply filters
        filtered: list[TraceEvent] = []
        for ev in reversed(events):
            if event_type and ev.event_type.lower() != event_type.lower():
                continue
            if session_id and ev.session_id != session_id:
                continue
            if level and ev.level.lower() != level.lower():
                continue
            if search:
                s_lower = search.lower()
                combined = (
                    f"{ev.message} {ev.component} {ev.error or ''} {json.dumps(ev.payload)}".lower()
                )
                if s_lower not in combined:
                    continue
            filtered.append(ev)
            if len(filtered) >= limit:
                break

        return [e.to_dict() for e in filtered]

    def clear(self) -> int:
        """Clears the in-memory event buffer."""
        with self._lock:
            count = len(self._buffer)
            self._buffer.clear()
            return count

    def get_summary(self) -> dict[str, Any]:
        """Provides an aggregate summary of buffered trace events."""
        with self._lock:
            events = list(self._buffer)

        type_counts: dict[str, int] = collections.defaultdict(int)
        level_counts: dict[str, int] = collections.defaultdict(int)
        error_count = 0

        for ev in events:
            type_counts[ev.event_type] += 1
            level_counts[ev.level] += 1
            if ev.level == "ERROR" or ev.error:
                error_count += 1

        return {
            "total_events": len(events),
            "error_count": error_count,
            "type_counts": dict(type_counts),
            "level_counts": dict(level_counts),
            "log_file": str(self.log_file),
        }


# Global singleton instance
debug_tracer = DebugTraceManager()


def get_tracer() -> DebugTraceManager:
    """Returns the global debug tracer singleton."""
    return debug_tracer
