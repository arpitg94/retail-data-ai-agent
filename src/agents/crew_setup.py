from __future__ import annotations

import json
import os
from typing import Any

from crewai import Agent, Crew, LLM, Task
from dotenv import load_dotenv

from src.agents.router import extract_json_object, heuristic_plan
from src.prompts.system_prompts import (
    EXTRACTOR_SYSTEM_PROMPT,
    RESOLVER_SYSTEM_PROMPT,
    VALIDATOR_SYSTEM_PROMPT,
)
from src.tools.pandas_tools import PandasTools


class RetailInsightsCrew:
    def __init__(self, tools: PandasTools, model: str = "gpt-4o-mini"):
        self.tools = tools
        load_dotenv()

        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required to run this app.")

        llm = LLM(model=model, temperature=0, api_key=api_key)

        self.resolver_agent = Agent(
            role="Language-to-Query Resolution Agent",
            goal="Map user query to approved deterministic tool calls.",
            backstory=RESOLVER_SYSTEM_PROMPT,
            llm=llm,
            verbose=False,
        )

        self.extractor_agent = Agent(
            role="Data Extraction Agent",
            goal="Summarize already extracted tool outputs.",
            backstory=EXTRACTOR_SYSTEM_PROMPT,
            llm=llm,
            verbose=False,
        )

        self.validator_agent = Agent(
            role="Validation Agent",
            goal="Validate extracted findings and produce final answer.",
            backstory=VALIDATOR_SYSTEM_PROMPT,
            llm=llm,
            verbose=False,
        )

    def _resolve_plan(self, query: str, history: list[dict[str, str]]) -> dict[str, Any]:
        history_text = "\n".join(
            [f"{msg.get('role', 'user')}: {msg.get('content', '')}" for msg in history[-8:]]
        )
        resolver_task = Task(
            description=(
                f"User query: {query}\n"
                f"Conversation history:\n{history_text}\n\n"
                f"Allowed tools: {self.tools.list_available_tools()}\n"
                "Return ONLY JSON with intent, tool_calls, notes."
            ),
            expected_output="Strict JSON.",
            agent=self.resolver_agent,
        )
        crew = Crew(agents=[self.resolver_agent], tasks=[resolver_task], verbose=False)
        output = str(crew.kickoff())
        parsed = extract_json_object(output)
        if not parsed:
            return heuristic_plan(query)

        valid_tools = set(self.tools.list_available_tools())
        safe_calls = []
        for call in parsed.get("tool_calls", []):
            tool_name = call.get("tool_name")
            kwargs = call.get("kwargs", {})
            if tool_name in valid_tools and isinstance(kwargs, dict):
                safe_calls.append({"tool_name": tool_name, "kwargs": kwargs})

        if not safe_calls:
            return heuristic_plan(query)

        return {
            "intent": parsed.get("intent", "resolved_intent"),
            "tool_calls": safe_calls,
            "notes": parsed.get("notes", ""),
        }

    def _summarize_extraction(self, query: str, tool_results: list[dict[str, Any]]) -> str:
        extractor_task = Task(
            description=(
                f"User query: {query}\n"
                f"Tool outputs JSON:\n{json.dumps(tool_results, ensure_ascii=True)}\n"
                "Summarize extracted facts only."
            ),
            expected_output="Factual bullet summary.",
            agent=self.extractor_agent,
        )
        crew = Crew(agents=[self.extractor_agent], tasks=[extractor_task], verbose=False)
        return str(crew.kickoff())

    def _validate_answer(
        self,
        query: str,
        extraction_summary: str,
        tool_results: list[dict[str, Any]],
        mode: str,
    ) -> str:
        validator_task = Task(
            description=(
                f"Mode: {mode}\n"
                f"User query: {query}\n"
                f"Extraction summary:\n{extraction_summary}\n"
                f"Tool outputs JSON:\n{json.dumps(tool_results, ensure_ascii=True)}\n"
                "Return final business answer."
            ),
            expected_output="Business-friendly final answer.",
            agent=self.validator_agent,
        )
        crew = Crew(agents=[self.validator_agent], tasks=[validator_task], verbose=False)
        return str(crew.kickoff())

    def answer(self, query: str, history: list[dict[str, str]], mode: str) -> dict[str, Any]:
        plan = self._resolve_plan(query, history)

        tool_results: list[dict[str, Any]] = []
        for call in plan.get("tool_calls", []):
            result = self.tools.run_tool(call["tool_name"], **call.get("kwargs", {}))
            tool_results.append(result)

        extraction_summary = self._summarize_extraction(query, tool_results)

        # Simple post-filter for business phrasing question.
        if "blouse" in query.lower():
            for item in tool_results:
                if item.get("tool_name") == "units_sold_by_category":
                    blouse_rows = [
                        row
                        for row in item.get("records", [])
                        if "blouse" in str(row.get("Category", "")).lower()
                    ]
                    if blouse_rows:
                        extraction_summary += (
                            f"\nBlouse units found: {blouse_rows[0].get('Qty', 0)}."
                        )

        final_answer = self._validate_answer(query, extraction_summary, tool_results, mode)
        return {
            "plan": plan,
            "tool_results": tool_results,
            "extraction_summary": extraction_summary,
            "final_answer": final_answer,
        }

