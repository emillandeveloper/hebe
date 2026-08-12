from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Callable


def _plain(value: Any) -> Any:
    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, set):
        return sorted(_plain(item) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "__dict__"):
        return _plain(vars(value))
    return str(value)


@dataclass(slots=True)
class CognitiveStateSnapshot:
    runtime: dict[str, Any] = field(default_factory=dict)
    stream_session: dict[str, Any] = field(default_factory=dict)
    current_scene: dict[str, Any] = field(default_factory=dict)
    pending: dict[str, Any] = field(default_factory=dict)
    conversation: dict[str, Any] = field(default_factory=dict)
    open_threads: list[dict[str, Any]] = field(default_factory=list)
    memory: dict[str, Any] = field(default_factory=dict)
    beliefs: dict[str, Any] = field(default_factory=dict)
    belief_evidence: list[dict[str, Any]] = field(default_factory=list)
    retrieval: dict[str, Any] = field(default_factory=dict)
    memory_compatibility: dict[str, Any] = field(default_factory=dict)
    game_state: dict[str, Any] = field(default_factory=dict)
    social_state: dict[str, Any] = field(default_factory=dict)
    learning: dict[str, Any] = field(default_factory=dict)
    self_model: dict[str, Any] = field(default_factory=dict)
    owner_preferences: list[dict[str, Any]] = field(default_factory=list)
    leo_language: dict[str, Any] = field(default_factory=dict)
    temporal: dict[str, Any] = field(default_factory=dict)
    schedule: dict[str, Any] = field(default_factory=dict)
    action_ledger: dict[str, Any] = field(default_factory=dict)
    scene_transitions: dict[str, Any] = field(default_factory=dict)
    continuity_context: dict[str, Any] = field(default_factory=dict)
    promotion_profiles: list[dict[str, Any]] = field(default_factory=list)
    actions: dict[str, Any] = field(default_factory=dict)
    receipts: list[dict[str, Any]] = field(default_factory=list)
    emitted_outputs: list[dict[str, Any]] = field(default_factory=list)
    final_emission_results: list[dict[str, Any]] = field(default_factory=list)
    database_watermarks: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _plain(asdict(self))


class CognitiveStateProbe:
    """Read-only state projection used by assertions and reports."""

    SAFE_COUNT_TABLES = (
        "chat_log", "memory_facts", "memory_chunks", "stream_sessions",
        "stream_chat_messages", "stream_events", "live_session_timeline",
        "promotion_events", "viewer_promotion_profiles", "schema_migrations",
        "conversations", "open_threads", "beliefs", "belief_evidence", "scene_assertions",
        "game_identities", "game_runs", "game_run_sessions", "game_run_events",
        "game_knowledge_facts", "game_knowledge_v2_gaps",
        "people", "person_identities", "person_sessions", "social_episodes",
        "shared_culture_items", "shared_culture_evidence",
        "consolidation_runs", "consolidation_deltas", "action_ledger",
        "temporal_maintenance_audit", "learning_observations", "scene_transitions",
        "schedule_observations", "schedule_hypotheses",
    )

    def __init__(
        self,
        *,
        connection_factory: Callable[[], sqlite3.Connection],
        actions: list[dict[str, Any]],
        speech_requests: list[dict[str, Any]],
        final_emissions: list[dict[str, Any]],
        model_calls: list[dict[str, Any]],
        research_calls: list[dict[str, Any]],
    ) -> None:
        self.connection_factory = connection_factory
        self.actions = actions
        self.speech_requests = speech_requests
        self.final_emissions = final_emissions
        self.model_calls = model_calls
        self.research_calls = research_calls

    def snapshot(self, engine: Any) -> CognitiveStateSnapshot:
        state = getattr(getattr(engine, "runtime", None), "state", None)
        stream = getattr(state, "stream", None)
        pending = getattr(state, "pending_clarification", None)
        pending_turn = getattr(state, "pending_conversation_turn", None)
        rows = self._db_rows()
        trace = dict(getattr(engine, "_last_policy_trace", {}) or {})
        cognitive = dict(getattr(engine, "_last_cognitive_trace", {}) or {})
        runtime = {
            "mode": getattr(state, "mode", None),
            "hebe_sleeping": bool(getattr(state, "hebe_sleeping", False)),
            "is_running": bool(getattr(state, "is_running", False)),
            "last_input_source": getattr(state, "last_input_source", None),
            "last_intent": getattr(state, "last_intent", None),
            "last_firewall": {
                "source": dict(getattr(engine, "_last_input_firewall", {}) or {}).get("source"),
                "authority": dict(getattr(engine, "_last_input_firewall", {}) or {}).get("authority"),
                "decision": dict(getattr(engine, "_last_input_firewall", {}) or {}).get("firewall_decision"),
                "reason": dict(getattr(engine, "_last_input_firewall", {}) or {}).get("reason"),
            },
            "last_policy": {
                "source": trace.get("source"),
                "authority": trace.get("authority"),
                "decision": trace.get("policy_decision"),
                "reason": trace.get("reason"),
            },
        }
        stream_session = {
            "enabled": bool(getattr(stream, "enabled", False)),
            "is_live": bool(getattr(stream, "is_live", False)),
            "live_status_known": bool(getattr(stream, "live_status_known", False)),
            "active_stream_session_id": getattr(stream, "active_stream_session_id", None),
            "last_transition": getattr(stream, "last_stream_live_transition", None),
            "title": getattr(stream, "current_stream_title", None),
            "game": getattr(stream, "current_game", None),
            "category": getattr(stream, "current_category", None),
        }
        scene_value = getattr(stream, "current_scene_timeline", None)
        scene = _plain(scene_value) if scene_value else {}
        game_run = getattr(state, "game_run_state", None)
        game_state = _plain(game_run) if game_run is not None else {}
        game_state.update({
            "current_game": getattr(stream, "current_game", None),
            "current_objective": getattr(stream, "current_run_objective", None),
            "current_location": getattr(stream, "current_run_location", None),
            "recent_run_context_facts": _plain(list(getattr(stream, "recent_run_context_facts", []) or [])),
        })
        game_context=_plain(getattr(getattr(engine,"game_context_resolver",None),"last_context",{}) or {})
        context_diag=_plain(getattr(getattr(engine,"game_context_resolver",None),"diagnostics",lambda:{})())
        game_state.update({
            "identity":game_context.get("game_identity") or {},
            "context":game_context,
            "active_run":game_context.get("active_run") or next((item for item in rows["game_runs"] if item["status"]=="ACTIVE"),{}),
            "runs":rows["game_runs"],"session_links":rows["game_run_sessions"],"run_events":rows["game_run_events"],
            "run_beliefs":{
                "current":[item for item in rows["beliefs"] if item["namespace"]=="game_run" and item["epistemic_status"] in {"KNOWN","INFERRED","SUSPECTED"} and not item["superseded_by"]],
                "inferred":[item for item in rows["beliefs"] if item["namespace"]=="game_run" and item["epistemic_status"]=="INFERRED" and not item["superseded_by"]],
                "superseded":[item for item in rows["beliefs"] if item["namespace"]=="game_run" and item["epistemic_status"]=="SUPERSEDED"],
            },
            "knowledge":{"selected":game_context.get("knowledge_claims") or [],"rejected":game_context.get("rejected_knowledge") or [],"spoiler_blocked":[item for item in game_context.get("rejected_knowledge") or [] if item.get("rejection_reason")=="spoiler_blocked"],"all":rows["game_knowledge_facts"]},
            "gaps":rows["game_knowledge_gaps"],
            "research":{**context_diag,"fixture_calls":_plain(self.research_calls),"status":game_context.get("research_status") or ""},
            "compatibility":_plain(getattr(getattr(engine,"legacy_game_adapter",None),"telemetry",{}) or {}),
            "provenance_manifest":game_context.get("provenance_manifest") or [],
            "advice_allowed":game_context.get("advice_allowed"),"reaction_allowed":game_context.get("reaction_allowed"),
            "performance":_plain(getattr(getattr(engine,"game_v2_repository",None),"performance",lambda:{})()),
            "context_performance":context_diag.get("context_performance") or {},"manifest_size_bytes":game_context.get("manifest_size_bytes") or 0,
            "last_run_resolution":_plain(getattr(getattr(engine,"game_run_service",None),"last_resolution",{}) or {}),
            "run_resolution_performance":_plain(getattr(getattr(engine,"game_run_service",None),"performance",lambda:{})()),
        })
        social_state = {
            "recent_active_users": list(getattr(stream, "recent_active_users", []) or []),
            "recent_chat_count": len(list(getattr(stream, "recent_chat_messages", []) or [])),
            "last_raid": _plain(getattr(stream, "last_raid_event", None) or {}),
            "last_cheer": _plain(getattr(stream, "last_cheer_event", None) or {}),
            "people":rows["people"],"identities":rows["person_identities"],"recent_episodes":rows["social_episodes"],
            "active_hypotheses":[item for item in rows["beliefs"] if item["namespace"]=="social" and item["epistemic_status"] in {"KNOWN","INFERRED","SUSPECTED"} and not item["superseded_by"]],
            "historical_hypotheses":[item for item in rows["beliefs"] if item["namespace"]=="social" and item["epistemic_status"] in {"HISTORICAL","SUPERSEDED"}],
            "open_threads":[item for item in rows["open_threads"] if item["scope_kind"]=="person"],
            "relationships":[{"person_id":item["person_id"],**_plain(getattr(getattr(engine,"social_world_repository",None),"familiarity",lambda _:{ })(item["person_id"]))} for item in rows["people"]],
            "shared_culture":{
                "all":rows["shared_culture_items"],"candidates":[x for x in rows["shared_culture_items"] if x["status"]=="CANDIDATE"],"active":[x for x in rows["shared_culture_items"] if x["status"]=="ACTIVE"],"weakening":[x for x in rows["shared_culture_items"] if x["status"]=="WEAKENING"],"retired":[x for x in rows["shared_culture_items"] if x["status"]=="RETIRED"],"reactions":rows["shared_culture_evidence"],"selection":_plain(getattr(getattr(engine,"social_world",None),"last_culture_selection",{}))},
            "retrieval":_plain(getattr(getattr(engine,"social_world",None),"last_context",{})),
            "opportunities":_plain(getattr(getattr(engine,"social_world",None),"last_opportunities",[])),
            "resolution":_plain(getattr(getattr(engine,"social_world",None),"last_resolution",{})),
            "rejected_writes":_plain(getattr(getattr(engine,"social_world",None),"rejections",[])),
            "compatibility":_plain(getattr(getattr(engine,"legacy_social_adapter",None),"telemetry",{})),
            "performance":_plain(getattr(getattr(engine,"social_world_repository",None),"performance",lambda:{})()),
            "belief_lookup_performance":_plain(getattr(getattr(engine,"belief_repository",None),"performance",lambda:{})()),
        }
        final_response = ""
        final_response = str(cognitive.get("final_response") or trace.get("final_response") or trace.get("hebe_response") or "")
        emitted = [self._minimal_emission(item) for item in self.final_emissions]
        learning_repo=getattr(engine,"learning_repository",None)
        consolidation_runs=learning_repo.rows("consolidation_runs",order="started_at,id") if learning_repo else []
        consolidation_deltas=learning_repo.rows("consolidation_deltas",order="created_at,id") if learning_repo else []
        action_rows=learning_repo.rows("action_ledger",order="requested_at,id") if learning_repo else []
        maintenance_rows=learning_repo.rows("temporal_maintenance_audit",order="changed_at,id") if learning_repo else []
        scene_rows=learning_repo.rows("scene_transitions",order="created_at,id") if learning_repo else []
        self_active=[item for item in rows["beliefs"] if item["namespace"]=="hebe_self" and item["epistemic_status"] in {"KNOWN","INFERRED","SUSPECTED"} and not item["superseded_by"]]
        self_old=[item for item in rows["beliefs"] if item["namespace"]=="hebe_self" and item["epistemic_status"] in {"HISTORICAL","SUPERSEDED"}]
        owner_prefs=[item for item in rows["beliefs"] if item["namespace"]=="owner_preference" and item["epistemic_status"] in {"KNOWN","INFERRED"} and not item["superseded_by"]]
        language_items=[item for item in rows["beliefs"] if item["namespace"]=="leo_language" and item["epistemic_status"] in {"KNOWN","INFERRED"} and not item["superseded_by"]]
        return CognitiveStateSnapshot(
            runtime=runtime,
            stream_session=stream_session,
            current_scene=scene,
            pending={"clarification": _plain(pending), "conversation_turn": _plain(pending_turn)},
            conversation={
                "active": rows["active_conversation"],
                "latest": rows["conversations"][0] if rows["conversations"] else {},
                "all": rows["conversations"],
                "last_resolution": _plain(getattr(engine, "_last_continuity_resolution", {}) or {}),
                "legacy_pending_projection": _plain(
                    getattr(getattr(engine, "legacy_pending_adapter", None), "last_projection", {}) or {}
                ),
                "continuity_shadow_diff": _plain(getattr(engine, "_last_continuity_shadow_diff", {}) or {}),
                "shadow_metrics": _plain(
                    getattr(getattr(engine, "conversation_continuity", None), "shadow_metrics", lambda: {})()
                ),
                "performance": _plain(
                    getattr(getattr(engine, "conversation_continuity", None), "performance", lambda: {})()
                ),
            },
            open_threads=rows["open_threads"],
            memory={"facts_count": rows["counts"].get("memory_facts", 0), "chunks_count": rows["counts"].get("memory_chunks", 0)},
            beliefs={
                "active": [item for item in rows["beliefs"] if item["epistemic_status"] in {"KNOWN","INFERRED","SUSPECTED"} and not item["superseded_by"]],
                "historical": [item for item in rows["beliefs"] if item["epistemic_status"] == "HISTORICAL"],
                "superseded": [item for item in rows["beliefs"] if item["epistemic_status"] == "SUPERSEDED"],
                "suspected": [item for item in rows["beliefs"] if item["epistemic_status"] == "SUSPECTED"],
                "all": rows["beliefs"],
                "last_transition": _plain(getattr(getattr(engine,"belief_lifecycle",None),"last_transition",{}) or {}),
            },
            belief_evidence=rows["belief_evidence"],
            retrieval={
                "last_request": _plain(getattr(getattr(engine,"memory_retrieval",None),"last_request",{}) or {}),
                **_plain(getattr(getattr(engine,"memory_retrieval",None),"last_result",{}) or {}),
                "performance": _plain(getattr(getattr(engine,"memory_retrieval",None),"performance",lambda:{})()),
                "write_performance": _plain(getattr(getattr(engine,"belief_lifecycle",None),"performance",lambda:{})()),
                "repository_performance": _plain(getattr(getattr(engine,"belief_repository",None),"performance",lambda:{})()),
            },
            memory_compatibility=_plain(getattr(getattr(engine,"legacy_memory_fact_adapter",None),"telemetry",{}) or {}),
            game_state=game_state,
            social_state=social_state,
            learning={"consolidation_runs":consolidation_runs,"deltas":consolidation_deltas,"rejected_deltas":[x for x in consolidation_deltas if x["validator_result"]=="REJECTED"],"watermarks":[{"session_id":x["session_id"],"start":x["input_start_event"],"end":x["input_end_event"],"status":x["status"]} for x in consolidation_runs],"last_result":_plain(getattr(getattr(engine,"session_consolidator",None),"last_result",{})),"stable_core_version":getattr(getattr(engine,"stable_hebe_core",None),"version",""),"performance":{"repository":_plain(getattr(learning_repo,"performance",lambda:{})()),"consolidation":_plain(getattr(getattr(engine,"session_consolidator",None),"performance",lambda:{})()),"temporal":_plain(getattr(getattr(engine,"temporal_relevance_service",None),"performance",lambda:{})()),"action_history":_plain(getattr(getattr(engine,"historical_action_ledger",None),"performance",lambda:{})()),"owner_preferences":_plain(getattr(getattr(engine,"owner_procedural_preferences",None),"performance",lambda:{})()),"hebe_self":_plain(getattr(getattr(engine,"hebe_self_model",None),"performance",lambda:{})()),"context":_plain(getattr(getattr(engine,"continuity_context_builder",None),"performance",lambda:{})())}},
            self_model={"stable_core_version":getattr(getattr(engine,"stable_hebe_core",None),"version",""),"evolving_preferences":[x for x in self_active if x["predicate"].startswith("preference.")],"opinions":self_active,"superseded_opinions":self_old},
            owner_preferences=owner_prefs,
            leo_language={"beliefs":language_items,"interpretation_aliases":_plain(getattr(getattr(engine,"leo_language_model",None),"interpretation_aliases",lambda:{})())},
            temporal={"expired":[x for x in maintenance_rows if x["new_status"]=="EXPIRED"],"archived":[x for x in maintenance_rows if x["new_status"]=="ARCHIVED"],"weakened":[x for x in maintenance_rows if x["new_status"]=="WEAKENING"],"maintenance_actions":maintenance_rows,"last_actions":_plain(getattr(getattr(engine,"temporal_relevance_service",None),"last_actions",[]))},
            schedule={"observations":rows.get("schedule_observations",[]),"hypotheses":rows.get("schedule_hypotheses",[]),"observed_current_state":{"game":getattr(stream,"current_game",None),"title":getattr(stream,"current_stream_title",None)},"precedence":"observed_twitch_metadata" if getattr(stream,"current_game",None) else "schedule_prediction"},
            action_ledger={"entries":action_rows,"last_claim_validation":_plain(getattr(getattr(engine,"historical_action_ledger",None),"last_decision",{}))},
            scene_transitions={"all":scene_rows,"last":_plain(getattr(getattr(engine,"scene_consequence_reducer",None),"last_transition",{}))},
            continuity_context=_plain(getattr(getattr(engine,"continuity_context_builder",None),"last_context",{})),
            promotion_profiles=rows["promotion_profiles"],
            actions={
                "attempts": _plain(self.actions),
                "speech_requests": [{"language": item.get("language"), "text_digest": _digest(item.get("text"))} for item in self.speech_requests],
                "model_calls": _plain(self.model_calls),
                "research_calls": _plain(self.research_calls),
            },
            receipts=rows["promotion_events"],
            emitted_outputs=emitted,
            final_emission_results=emitted,
            database_watermarks={
                "counts": rows["counts"],
                "schema_migrations": rows["schema_migrations"],
                "final_response_digest": _digest(final_response),
                "final_response_present": bool(final_response),
            },
        )

    def _db_rows(self) -> dict[str, Any]:
        conn = self.connection_factory()
        conn.row_factory = sqlite3.Row
        try:
            existing = {str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            counts = {
                table: int(conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
                for table in self.SAFE_COUNT_TABLES if table in existing
            }
            profiles = []
            if "viewer_promotion_profiles" in existing:
                profiles = [dict(row) for row in conn.execute(
                    "SELECT twitch_user_id, current_login, display_name, auto_promo_mode, created_by, last_promoted_stream_id, owner_locked, active FROM viewer_promotion_profiles ORDER BY twitch_user_id"
                )]
            promotions = []
            if "promotion_events" in existing:
                promotions = [dict(row) for row in conn.execute(
                    "SELECT id, stream_session_id, source_event_id, requested_by, resolved_twitch_user_id, resolved_login, trigger_type, execution_status, twitch_message_id, failure_reason FROM promotion_events ORDER BY created_at, id"
                )]
            migrations = []
            if "schema_migrations" in existing:
                migrations = [dict(row) for row in conn.execute(
                    "SELECT component, version, name, checksum, applied_at FROM schema_migrations ORDER BY component, version"
                )]
            conversations = []
            if "conversations" in existing:
                conversations = [dict(row) for row in conn.execute(
                    """SELECT id,context_kind,context_id,participants_json,attention_state,turn_owner,
                        expected_reply_type,topic,origin_event_id,last_event_id,opened_at,last_turn_at,
                        expires_at,status,closure_reason,version,domain_payload_json,consumed_event_ids_json
                        FROM conversations ORDER BY opened_at DESC,id DESC"""
                )]
                for item in conversations:
                    item["participants"] = json.loads(item.pop("participants_json") or "[]")
                    item["domain_payload"] = json.loads(item.pop("domain_payload_json") or "{}")
                    item["consumed_event_ids"] = json.loads(item.pop("consumed_event_ids_json") or "[]")
            open_threads = []
            if "open_threads" in existing:
                open_threads = [dict(row) for row in conn.execute(
                    """SELECT id,thread_type,scope_kind,scope_id,participant_ids_json,subject_ref,
                        summary,origin_event_id,latest_event_id,status,priority,created_at,relevance_until,
                        valid_until,resolved_at,resolution_event_id,sensitivity,version
                        FROM open_threads ORDER BY created_at DESC,id DESC"""
                )]
                for item in open_threads:
                    item["participant_ids"] = json.loads(item.pop("participant_ids_json") or "[]")
            active = next((item for item in conversations if item["status"] in {"OPEN", "WAITING_ON_LEO", "WAITING_ON_HEBE"}), {})
            beliefs=[]
            if "beliefs" in existing:
                beliefs=[dict(row) for row in conn.execute("SELECT * FROM beliefs ORDER BY created_at DESC,id")]
                for item in beliefs:
                    item["object"]=json.loads(item.pop("object_json") or "null")
                    item["owner_confirmed"]=bool(item["owner_confirmed"])
                    item["evidence_ids"]=[str(r[0]) for r in conn.execute("SELECT id FROM belief_evidence WHERE belief_id=? ORDER BY observed_at,id",(item["id"],))]
            belief_evidence=[]
            if "belief_evidence" in existing:
                belief_evidence=[dict(row) for row in conn.execute("SELECT * FROM belief_evidence ORDER BY observed_at,id")]
                for item in belief_evidence:item["literal_span"]=json.loads(item.pop("literal_span_json") or "{}")
            game_runs=[]
            if "game_runs" in existing:
                game_runs=[dict(row) for row in conn.execute("SELECT * FROM game_runs ORDER BY started_at,id")]
                for item in game_runs:item["rules"]=json.loads(item.pop("rules_json") or "{}")
            game_run_sessions=[dict(row) for row in conn.execute("SELECT * FROM game_run_sessions ORDER BY started_at,id")] if "game_run_sessions" in existing else []
            game_run_events=[]
            if "game_run_events" in existing:
                game_run_events=[dict(row) for row in conn.execute("SELECT * FROM game_run_events ORDER BY observed_at,id")]
                for item in game_run_events:item["object"]=json.loads(item.pop("object_json") or "null")
            game_knowledge_facts=[dict(row) for row in conn.execute("SELECT * FROM game_knowledge_facts ORDER BY created_at,id")] if "game_knowledge_facts" in existing else []
            game_knowledge_gaps=[]
            if "game_knowledge_v2_gaps" in existing:
                game_knowledge_gaps=[dict(row) for row in conn.execute("SELECT * FROM game_knowledge_v2_gaps ORDER BY created_at,id")]
                for item in game_knowledge_gaps:item["resolved_fact_ids"]=json.loads(item.pop("resolved_fact_ids_json") or "[]")
            people=[dict(row) for row in conn.execute("SELECT * FROM people ORDER BY created_at,person_id")] if "people" in existing else []
            person_identities=[dict(row) for row in conn.execute("SELECT * FROM person_identities ORDER BY first_seen_at,id")] if "person_identities" in existing else []
            for item in person_identities:item["aliases"]=json.loads(item.pop("aliases_json") or "[]")
            social_episodes=[dict(row) for row in conn.execute("SELECT * FROM social_episodes ORDER BY created_at,id")] if "social_episodes" in existing else []
            for item in social_episodes:
                item["participant_ids"]=json.loads(item.pop("participant_ids_json") or "[]");item["related_event_ids"]=json.loads(item.pop("related_event_ids_json") or "[]");item["tone_observations"]=json.loads(item.pop("tone_observations_json") or "[]")
            shared_culture_items=[dict(row) for row in conn.execute("SELECT * FROM shared_culture_items ORDER BY created_at,id")] if "shared_culture_items" in existing else []
            for item in shared_culture_items:item["participant_ids"]=json.loads(item.pop("participant_ids_json") or "[]")
            shared_culture_evidence=[dict(row) for row in conn.execute("SELECT * FROM shared_culture_evidence ORDER BY observed_at,id")] if "shared_culture_evidence" in existing else []
            schedule_observations=[dict(row) for row in conn.execute("SELECT * FROM schedule_observations ORDER BY observed_at,id")] if "schedule_observations" in existing else []
            schedule_hypotheses=[dict(row) for row in conn.execute("SELECT * FROM schedule_hypotheses ORDER BY last_observed_at,id")] if "schedule_hypotheses" in existing else []
            return {
                "counts": counts, "promotion_profiles": profiles, "promotion_events": promotions,
                "schema_migrations": migrations, "conversations": conversations,
                "active_conversation": active, "open_threads": open_threads,
                "beliefs":beliefs,"belief_evidence":belief_evidence,"game_runs":game_runs,
                "game_run_sessions":game_run_sessions,"game_run_events":game_run_events,
                "game_knowledge_facts":game_knowledge_facts,"game_knowledge_gaps":game_knowledge_gaps,
                "people":people,"person_identities":person_identities,"social_episodes":social_episodes,"shared_culture_items":shared_culture_items,"shared_culture_evidence":shared_culture_evidence,
                "schedule_observations":schedule_observations,"schedule_hypotheses":schedule_hypotheses,
            }
        finally:
            conn.close()

    @staticmethod
    def _minimal_emission(item: dict[str, Any]) -> dict[str, Any]:
        data = dict(item or {})
        return {
            "event_id": data.get("event_id"),
            "emitted": bool(data.get("emitted")),
            "route": data.get("output_route") or data.get("route"),
            "targets": list(data.get("output_targets") or data.get("targets") or []),
            "reason": data.get("reason") or data.get("suppress_reason") or "",
            "text_digest": _digest(data.get("final_response") or data.get("text")),
            "text_present": bool(data.get("final_response") or data.get("text")),
        }


def _digest(value: Any) -> str:
    import hashlib

    text = str(value or "")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16] if text else ""
