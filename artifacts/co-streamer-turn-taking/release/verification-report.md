# Cognitive Replay Verification Report

Overall status: **VERIFIED**
Phase result: **VERIFIED**

## Repository and environment

- Commit: `465bada4117e27486243b1b94c25b5e8e68faa79`
- Working tree: `d11d58ad1bff`
- Platform: `Windows-10-10.0.26200-SP0`
- Python: `3.11.0`

## Commands

- `C:\Program Files\Python311\python.exe -m unittest backend.tests.test_cognitive_replay backend.tests.test_voice_command_pipeline backend.tests.test_cognitive_twitch backend.tests.test_stream_presence backend.tests.test_hebe_live_v1 backend.tests.test_hebe_live_v11 backend.tests.test_hebe_live_v12 backend.tests.test_hebe_live_v12_followup backend.tests.test_hebe_live_20260809_followup backend.tests.test_final_emission_gate backend.tests.test_cognitive_execution_guard backend.tests.test_game_knowledge backend.tests.test_stream_session_primer backend.tests.test_live_session_brain backend.tests.test_conversation_continuity_phase1 backend.tests.test_epistemic_beliefs_phase2 backend.tests.test_game_guidance_routing backend.tests.test_game_context_phase3 backend.tests.test_social_world_phase4 backend.tests.test_learning_continuity_phase5 backend.tests.test_architecture_consolidation_phase6 backend.tests.test_stream_companion_loop backend.tests.test_co_streamer_turn_taking` → exit 1 (77.568063s)
- `python -m app.replay --suite co-streamer-turn-taking --run-phase-tests --baseline-differential artifacts/cognitive-continuity-phase6/release/baseline-differential.json --output artifacts/co-streamer-turn-taking/release` → exit 0 (78.7492s)

## Tests

```json
{
  "unit_integration_regression": {
    "passed": 577,
    "failed": 17,
    "skipped": 0,
    "total": 594,
    "duration_seconds": 77.568063,
    "expected_failures": 0,
    "failing_tests": [
      "test_fallback_chat_blocks_ungrounded_walkthrough_claim (backend.tests.test_game_guidance_routing.GameGuidanceRoutingTests.test_fallback_chat_blocks_ungrounded_walkthrough_claim)",
      "test_fallback_chat_is_blocked_while_game_pending_is_active (backend.tests.test_game_guidance_routing.GameGuidanceRoutingTests.test_fallback_chat_is_blocked_while_game_pending_is_active)",
      "test_high_value_game_tip_can_reply_without_hebe_mention (backend.tests.test_voice_command_pipeline.VoiceCommandPipelineTests.test_high_value_game_tip_can_reply_without_hebe_mention)",
      "test_no_wake_whitelisted_app_command_routes_while_stream_offline (backend.tests.test_voice_command_pipeline.VoiceCommandPipelineTests.test_no_wake_whitelisted_app_command_routes_while_stream_offline)",
      "test_obs_path_missing_returns_structured_action_result_not_generic_advice (backend.tests.test_voice_command_pipeline.VoiceCommandPipelineTests.test_obs_path_missing_returns_structured_action_result_not_generic_advice)",
      "test_response_synthesizer_handles_game_knowledge_command_result (backend.tests.test_game_knowledge.GameKnowledgeTests.test_response_synthesizer_handles_game_knowledge_command_result)",
      "test_stt_answer_without_wake_is_owner_followup (backend.tests.test_game_guidance_routing.GameGuidanceRoutingTests.test_stt_answer_without_wake_is_owner_followup)",
      "test_stt_canonical_melonds_command_executes_once (backend.tests.test_voice_command_pipeline.VoiceCommandPipelineTests.test_stt_canonical_melonds_command_executes_once) (transcript='Ebe, abre Melón DS')",
      "test_stt_canonical_melonds_command_executes_once (backend.tests.test_voice_command_pipeline.VoiceCommandPipelineTests.test_stt_canonical_melonds_command_executes_once) (transcript='Eve, abre Melón de Ese')",
      "test_stt_canonical_melonds_command_executes_once (backend.tests.test_voice_command_pipeline.VoiceCommandPipelineTests.test_stt_canonical_melonds_command_executes_once) (transcript='Hebe, abre melonDS')",
      "test_stt_hebe_abre_obs_uses_same_open_application_pipeline (backend.tests.test_voice_command_pipeline.VoiceCommandPipelineTests.test_stt_hebe_abre_obs_uses_same_open_application_pipeline)",
      "test_successful_state_update_mutates_runtime_game_run_state (backend.tests.test_game_guidance_routing.GameGuidanceRoutingTests.test_successful_state_update_mutates_runtime_game_run_state)",
      "test_twitch_normal_no_mention_chat_reaches_presence_observe (backend.tests.test_voice_command_pipeline.VoiceCommandPipelineTests.test_twitch_normal_no_mention_chat_reaches_presence_observe)",
      "test_twitch_pipeline_health_counts_messages (backend.tests.test_voice_command_pipeline.VoiceCommandPipelineTests.test_twitch_pipeline_health_counts_messages)",
      "test_ui_abre_obs_creates_open_application_when_awake_and_whitelisted (backend.tests.test_voice_command_pipeline.VoiceCommandPipelineTests.test_ui_abre_obs_creates_open_application_when_awake_and_whitelisted)",
      "test_ui_hebe_abre_obs_creates_open_application_action_plan (backend.tests.test_voice_command_pipeline.VoiceCommandPipelineTests.test_ui_hebe_abre_obs_creates_open_application_action_plan)",
      "test_unrelated_action_during_pending_conversation_still_uses_action_flow (backend.tests.test_voice_command_pipeline.VoiceCommandPipelineTests.test_unrelated_action_during_pending_conversation_still_uses_action_flow)"
    ],
    "required_layer_missing": false,
    "output_digest": "a73ed8a752a43226"
  },
  "replay": {
    "passed": 1,
    "failed": 0,
    "skipped": 0,
    "expected_future_gaps": 0,
    "expected_failures": 0,
    "duration_seconds": 1.050375
  },
  "failed": 17,
  "required_layer_missing": false
}
```

## Replay scenarios

### co_streamer_realistic_cadence

- Status: **VERIFIED**
- Events: 11
- Restarts: 0
- Duration: 1.050375s
- Assertions passed/failed/skipped: 10/0/0


#### Checkpoint state

```json
{
  "stream-start": {
    "runtime": {
      "mode": "active",
      "hebe_sleeping": false,
      "is_running": false,
      "last_input_source": null,
      "last_intent": null,
      "last_firewall": {
        "source": null,
        "authority": null,
        "decision": null,
        "reason": null
      },
      "last_policy": {
        "source": null,
        "authority": null,
        "decision": null,
        "reason": null
      }
    },
    "stream_session": {
      "enabled": true,
      "is_live": true,
      "live_status_known": true,
      "active_stream_session_id": 1,
      "last_transition": "online",
      "title": "Boss attempts",
      "game": "Test RPG",
      "category": "Test RPG"
    },
    "current_scene": {},
    "pending": {
      "clarification": null,
      "conversation_turn": null
    },
    "conversation": {
      "active": {},
      "latest": {},
      "all": [],
      "last_resolution": {},
      "legacy_pending_projection": {},
      "continuity_shadow_diff": {},
      "shadow_metrics": {
        "total": 0,
        "matches": 0,
        "differences": 0,
        "match_rate": 1.0,
        "difference_reasons": {}
      },
      "performance": {
        "count": 0,
        "p50_ms": 0.0,
        "p95_ms": 0.0
      }
    },
    "open_threads": [],
    "memory": {
      "facts_count": 0,
      "chunks_count": 0
    },
    "beliefs": {
      "active": [],
      "historical": [],
      "superseded": [],
      "suspected": [],
      "all": [],
      "last_transition": {}
    },
    "belief_evidence": [],
    "retrieval": {
      "last_request": {},
      "performance": {
        "count": 0,
        "p50_ms": 0.0,
        "p95_ms": 0.0
      },
      "write_performance": {
        "count": 0,
        "p50_ms": 0.0,
        "p95_ms": 0.0
      },
      "repository_performance": {
        "belief_lookup": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "evidence_lookup": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "sqlite_write": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        }
      }
    },
    "memory_compatibility": {
      "legacy_to_v2": [],
      "v2_to_legacy": [],
      "shadow_diffs": [],
      "backfill": {
        "safe": 0,
        "compatibility_only": 0,
        "ambiguous": 0,
        "invalid_stale": 0
      }
    },
    "game_state": {
      "game": "Test RPG",
      "platform_version": "",
      "playthrough_type": "casual",
      "spoiler_policy": "no_spoilers",
      "current_location": null,
      "current_character": "",
      "party_members": [],
      "last_confirmed_progress": "",
      "current_objective": null,
      "challenge": "",
      "known_constraints": [],
      "last_updated": 1786726800.0,
      "provenance": "stream_context_sync",
      "confidence": 0.75,
      "current_game": "Test RPG",
      "recent_run_context_facts": [],
      "identity": {},
      "context": {},
      "active_run": {},
      "runs": [],
      "session_links": [],
      "run_events": [],
      "run_beliefs": {
        "current": [],
        "inferred": [],
        "superseded": []
      },
      "knowledge": {
        "selected": [],
        "rejected": [],
        "spoiler_blocked": [],
        "all": []
      },
      "gaps": [],
      "research": {
        "research_calls": 0,
        "cache_hits": 0,
        "failures": [],
        "context_performance": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "fixture_calls": [],
        "status": ""
      },
      "compatibility": {
        "legacy_progress": [],
        "dossier": [],
        "legacy_run": [],
        "shadow_diffs": [],
        "backfill": {
          "validated": 0,
          "compatibility_only": 0,
          "ambiguous": 0,
          "stale": 0
        }
      },
      "provenance_manifest": [],
      "advice_allowed": null,
      "reaction_allowed": null,
      "performance": {
        "run": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "run_fact": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "knowledge": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "research_gap": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "db_write": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        }
      },
      "context_performance": {
        "count": 0,
        "p50_ms": 0.0,
        "p95_ms": 0.0
      },
      "manifest_size_bytes": 0,
      "last_run_resolution": {},
      "run_resolution_performance": {
        "count": 0,
        "p50_ms": 0.0,
        "p95_ms": 0.0
      }
    },
    "social_state": {
      "recent_active_users": [],
      "recent_chat_count": 0,
      "last_raid": {},
      "last_cheer": {},
      "people": [],
      "identities": [],
      "recent_episodes": [],
      "active_hypotheses": [],
      "historical_hypotheses": [],
      "open_threads": [],
      "relationships": [],
      "shared_culture": {
        "all": [],
        "candidates": [],
        "active": [],
        "weakening": [],
        "retired": [],
        "reactions": [],
        "selection": {
          "selected": [],
          "rejected": []
        }
      },
      "retrieval": {},
      "opportunities": [],
      "resolution": {},
      "rejected_writes": [],
      "compatibility": {
        "chatter_presence": [],
        "chatter_profiles": [],
        "chatter_facts": [],
        "stream_chatter_summaries": [],
        "viewer_profiles": [],
        "social_events": [],
        "promotion_profiles": [],
        "backfill": {
          "explicit_observation": 0,
          "safe_episode": 0,
          "inferred_compatibility_only": 0,
          "ambiguous": 0,
          "sensitive": 0,
          "stale": 0
        },
        "shadow_diffs": []
      },
      "performance": {
        "identity": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "episode_write": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "thread_lookup": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "culture_select": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "context": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "db_write": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        }
      },
      "belief_lookup_performance": {
        "belief_lookup": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "evidence_lookup": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "sqlite_write": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        }
      }
    },
    "learning": {
      "consolidation_runs": [],
      "deltas": [],
      "rejected_deltas": [],
      "watermarks": [],
      "last_result": {},
      "stable_core_version": "a1a58e51882b0c88",
      "performance": {
        "repository": {
          "lookup": {
            "count": 5,
            "p50_ms": 0.8359,
            "p95_ms": 0.9587
          },
          "write": {
            "count": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0
          },
          "context": {
            "count": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0
          }
        },
        "consolidation": {
          "consolidation_duration": {
            "count": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0
          },
          "candidate_validation": {
            "count": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0
          }
        },
        "temporal": {
          "temporal_maintenance": {
            "count": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0
          }
        },
        "action_history": {
          "action_ledger_lookup": {
            "count": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0
          }
        },
        "owner_preferences": {
          "owner_preference_lookup": {
            "count": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0
          }
        },
        "hebe_self": {
          "hebe_self_lookup": {
            "count": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0
          }
        },
        "context": {
          "continuity_context_build": {
            "count": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0
          }
        }
      }
    },
    "self_model": {
      "stable_core_version": "a1a58e51882b0c88",
      "evolving_preferences": [],
      "opinions": [],
      "superseded_opinions": []
    },
    "owner_preferences": [],
    "leo_language": {
      "beliefs": [],
      "interpretation_aliases": {}
    },
    "temporal": {
      "expired": [],
      "archived": [],
      "weakened": [],
      "maintenance_actions": [],
      "last_actions": []
    },
    "schedule": {
      "observations": [
        {
          "id": 1,
          "stream_session_id": "1",
          "weekday": "friday",
          "time_window": "night",
          "canonical_content": "Test RPG",
          "content_key": "test rpg",
          "stream_format": "game_playthrough",
          "source": "observed",
          "observed_at": "2026-08-14T04:35:32.628934+02:00"
        }
      ],
      "hypotheses": [
        {
          "id": 1,
          "weekday": "monday",
          "time_window": "any",
          "canonical_content": "FINAL FANTASY IX",
          "content_key": "final fantasy ix",
          "stream_format": "challenge_run",
          "source": "owner_declared",
          "confidence": 0.9,
          "evidence_count": 1,
          "consecutive_matches": 1,
          "consecutive_misses": 0,
          "first_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "last_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "status": "active",
          "superseded_by": null
        },
        {
          "id": 2,
          "weekday": "tuesday",
          "time_window": "any",
          "canonical_content": "Persona 5 Royal",
          "content_key": "persona 5 royal",
          "stream_format": "game_playthrough",
          "source": "owner_declared",
          "confidence": 0.9,
          "evidence_count": 1,
          "consecutive_matches": 1,
          "consecutive_misses": 0,
          "first_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "last_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "status": "active",
          "superseded_by": null
        },
        {
          "id": 3,
          "weekday": "wednesday",
          "time_window": "any",
          "canonical_content": "Persona 5 Royal",
          "content_key": "persona 5 royal",
          "stream_format": "game_playthrough",
          "source": "owner_declared",
          "confidence": 0.9,
          "evidence_count": 1,
          "consecutive_matches": 1,
          "consecutive_misses": 0,
          "first_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "last_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "status": "active",
          "superseded_by": null
        },
        {
          "id": 4,
          "weekday": "thursday",
          "time_window": "any",
          "canonical_content": "Baldur's Gate 3",
          "content_key": "baldur s gate 3",
          "stream_format": "game_playthrough",
          "source": "owner_declared",
          "confidence": 0.9,
          "evidence_count": 1,
          "consecutive_matches": 1,
          "consecutive_misses": 0,
          "first_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "last_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "status": "active",
          "superseded_by": null
        },
        {
          "id": 6,
          "weekday": "saturday",
          "time_window": "any",
          "canonical_content": "Retro Weekend",
          "content_key": "retro weekend",
          "stream_format": "retro",
          "source": "owner_declared",
          "confidence": 0.9,
          "evidence_count": 1,
          "consecutive_matches": 1,
          "consecutive_misses": 0,
          "first_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "last_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "status": "active",
          "superseded_by": null
        },
        {
          "id": 7,
          "weekday": "sunday",
          "time_window": "any",
          "canonical_content": "Retro Weekend",
          "content_key": "retro weekend",
          "stream_format": "retro",
          "source": "owner_declared",
          "confidence": 0.9,
          "evidence_count": 1,
          "consecutive_matches": 1,
          "consecutive_misses": 0,
          "first_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "last_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "status": "active",
          "superseded_by": null
        },
        {
          "id": 5,
          "weekday": "friday",
          "time_window": "any",
          "canonical_content": "Baldur's Gate 3",
          "content_key": "baldur s gate 3",
          "stream_format": "game_playthrough",
          "source": "owner_declared",
          "confidence": 0.76,
          "evidence_count": 1,
          "consecutive_matches": 0,
          "consecutive_misses": 1,
          "first_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "last_observed_at": "2026-08-14T04:35:32.628934+02:00",
          "status": "weakening",
          "superseded_by": null
        }
      ],
      "observed_current_state": {
        "game": "Test RPG",
        "title": "Boss attempts"
      },
      "precedence": "observed_twitch_metadata"
    },
    "action_ledger": {
      "entries": [],
      "last_claim_validation": {}
    },
    "scene_transitions": {
      "all": [],
      "last": {}
    },
    "continuity_context": {},
    "promotion_profiles": [],
    "actions": {
      "attempts": [],
      "speech_requests": [],
      "model_calls": [],
      "research_calls": []
    },
    "receipts": [],
    "emitted_outputs": [],
    "final_emission_results": [],
    "speech_intents": {
      "active": [],
      "all": [],
      "metrics": {
        "pending": 0,
        "time_created_to_emit": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "turn_gap_before_emit": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "intent_creation": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "pending_queue_operation": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "turn_arbitration": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "presence_turn_decision": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        }
      }
    },
    "database_watermarks": {
      "counts": {
        "chat_log": 0,
        "memory_facts": 0,
        "memory_chunks": 0,
        "stream_sessions": 1,
        "stream_chat_messages": 0,
        "stream_events": 1,
        "live_session_timeline": 2,
        "promotion_events": 0,
        "viewer_promotion_profiles": 0,
        "schema_migrations": 7,
        "conversations": 0,
        "open_threads": 0,
        "beliefs": 0,
        "belief_evidence": 0,
        "scene_assertions": 0,
        "game_identities": 0,
        "game_runs": 0,
        "game_run_sessions": 0,
        "game_run_events": 0,
        "game_knowledge_facts": 0,
        "game_knowledge_v2_gaps": 0,
        "people": 0,
        "person_identities": 0,
        "person_sessions": 0,
        "social_episodes": 0,
        "shared_culture_items": 0,
        "shared_culture_evidence": 0,
        "consolidation_runs": 0,
        "consolidation_deltas": 0,
        "action_ledger": 0,
        "temporal_maintenance_audit": 0,
        "learning_observations": 0,
        "scene_transitions": 0,
        "schedule_observations": 1,
        "schedule_hypotheses": 7
      },
      "schema_migrations": [
        {
          "component": "architecture_consolidation",
          "version": 1,
          "name": "audit_hygiene_and_cutover_state",
          "checksum": "944b3ad6532151d4204ed59f663668c9b0ac62df54ece7602d26b91acddd5b7d",
          "applied_at": "2026-08-14T02:35:32.541524+00:00"
        },
        {
          "component": "belief_v2",
          "version": 1,
          "name": "beliefs_evidence_and_compatibility_columns",
          "checksum": "f0df6f1288caccaf6bb47670b38f9cca747f14c84916d6c1ca75beb687200507",
          "applied_at": "2026-08-14T02:35:32.399139+00:00"
        },
        {
          "component": "cognitive_replay",
          "version": 1,
          "name": "replay_metadata",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "applied_at": "2026-08-14T02:35:32.177746+00:00"
        },
        {
          "component": "conversation_continuity",
          "version": 1,
          "name": "conversation_and_open_threads",
          "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
          "applied_at": "2026-08-14T02:35:32.312278+00:00"
        },
        {
          "component": "game_context_v2",
          "version": 1,
          "name": "durable_runs_knowledge_and_gaps",
          "checksum": "08e342acaae00d5d24c1a6dbccad5aee41f753ed90d9ec26282a5fdf042d0a75",
          "applied_at": "2026-08-14T02:35:32.450003+00:00"
        },
        {
          "component": "learning_v2",
          "version": 1,
          "name": "consolidation_temporal_action_and_scene",
          "checksum": "6a86e2d1c7c03167f3b20c328bc97b73fb92e03903f64e64d326fb08f9e3b942",
          "applied_at": "2026-08-14T02:35:32.526410+00:00"
        },
        {
          "component": "social_world_v2",
          "version": 1,
          "name": "people_episodes_and_shared_culture",
          "checksum": "b02adc2cd7f298f1af228dc52c4ba44ae15999fb4c815ea004c40f68b78cbfa5",
          "applied_at": "2026-08-14T02:35:32.488704+00:00"
        }
      ],
      "final_response_digest": "",
      "final_response_present": false
    }
  },
  "past-grace": {
    "runtime": {
      "mode": "active",
      "hebe_sleeping": false,
      "is_running": false,
      "last_input_source": null,
      "last_intent": null,
      "last_firewall": {
        "source": null,
        "authority": null,
        "decision": null,
        "reason": null
      },
      "last_policy": {
        "source": null,
        "authority": null,
        "decision": null,
        "reason": null
      }
    },
    "stream_session": {
      "enabled": true,
      "is_live": true,
      "live_status_known": true,
      "active_stream_session_id": 1,
      "last_transition": "online",
      "title": "Boss attempts",
      "game": "Test RPG",
      "category": "Test RPG"
    },
    "current_scene": {},
    "pending": {
      "clarification": null,
      "conversation_turn": null
    },
    "conversation": {
      "active": {},
      "latest": {},
      "all": [],
      "last_resolution": {},
      "legacy_pending_projection": {},
      "continuity_shadow_diff": {},
      "shadow_metrics": {
        "total": 0,
        "matches": 0,
        "differences": 0,
        "match_rate": 1.0,
        "difference_reasons": {}
      },
      "performance": {
        "count": 0,
        "p50_ms": 0.0,
        "p95_ms": 0.0
      }
    },
    "open_threads": [],
    "memory": {
      "facts_count": 0,
      "chunks_count": 0
    },
    "beliefs": {
      "active": [],
      "historical": [],
      "superseded": [],
      "suspected": [],
      "all": [],
      "last_transition": {}
    },
    "belief_evidence": [],
    "retrieval": {
      "last_request": {},
      "performance": {
        "count": 0,
        "p50_ms": 0.0,
        "p95_ms": 0.0
      },
      "write_performance": {
        "count": 0,
        "p50_ms": 0.0,
        "p95_ms": 0.0
      },
      "repository_performance": {
        "belief_lookup": {
          "count": 2,
          "p50_ms": 1.3846,
          "p95_ms": 1.8886
        },
        "evidence_lookup": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "sqlite_write": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        }
      }
    },
    "memory_compatibility": {
      "legacy_to_v2": [],
      "v2_to_legacy": [],
      "shadow_diffs": [],
      "backfill": {
        "safe": 0,
        "compatibility_only": 0,
        "ambiguous": 0,
        "invalid_stale": 0
      }
    },
    "game_state": {
      "game": "Test RPG",
      "platform_version": "",
      "playthrough_type": "casual",
      "spoiler_policy": "no_spoilers",
      "current_location": null,
      "current_character": "",
      "party_members": [],
      "last_confirmed_progress": "",
      "current_objective": null,
      "challenge": "",
      "known_constraints": [],
      "last_updated": 1786727100.0,
      "provenance": "stream_context_sync",
      "confidence": 0.75,
      "current_game": "Test RPG",
      "recent_run_context_facts": [],
      "identity": {},
      "context": {},
      "active_run": {},
      "runs": [],
      "session_links": [],
      "run_events": [],
      "run_beliefs": {
        "current": [],
        "inferred": [],
        "superseded": []
      },
      "knowledge": {
        "selected": [],
        "rejected": [],
        "spoiler_blocked": [],
        "all": []
      },
      "gaps": [],
      "research": {
        "research_calls": 0,
        "cache_hits": 0,
        "failures": [],
        "context_performance": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "fixture_calls": [],
        "status": ""
      },
      "compatibility": {
        "legacy_progress": [],
        "dossier": [],
        "legacy_run": [],
        "shadow_diffs": [],
        "backfill": {
          "validated": 0,
          "compatibility_only": 0,
          "ambiguous": 0,
          "stale": 0
        }
      },
      "provenance_manifest": [],
      "advice_allowed": null,
      "reaction_allowed": null,
      "performance": {
        "run": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "run_fact": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "knowledge": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "research_gap": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "db_write": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        }
      },
      "context_performance": {
        "count": 0,
        "p50_ms": 0.0,
        "p95_ms": 0.0
      },
      "manifest_size_bytes": 0,
      "last_run_resolution": {},
      "run_resolution_performance": {
        "count": 0,
        "p50_ms": 0.0,
        "p95_ms": 0.0
      }
    },
    "social_state": {
      "recent_active_users": [],
      "recent_chat_count": 0,
      "last_raid": {},
      "last_cheer": {},
      "people": [],
      "identities": [],
      "recent_episodes": [],
      "active_hypotheses": [],
      "historical_hypotheses": [],
      "open_threads": [],
      "relationships": [],
      "shared_culture": {
        "all": [],
        "candidates": [],
        "active": [],
        "weakening": [],
        "retired": [],
        "reactions": [],
        "selection": {
          "selected": [],
          "rejected": []
        }
      },
      "retrieval": {},
      "opportunities": [],
      "resolution": {},
      "rejected_writes": [],
      "compatibility": {
        "chatter_presence": [],
        "chatter_profiles": [],
        "chatter_facts": [],
        "stream_chatter_summaries": [],
        "viewer_profiles": [],
        "social_events": [],
        "promotion_profiles": [],
        "backfill": {
          "explicit_observation": 0,
          "safe_episode": 0,
          "inferred_compatibility_only": 0,
          "ambiguous": 0,
          "sensitive": 0,
          "stale": 0
        },
        "shadow_diffs": []
      },
      "performance": {
        "identity": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "episode_write": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "thread_lookup": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "culture_select": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "context": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "db_write": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        }
      },
      "belief_lookup_performance": {
        "belief_lookup": {
          "count": 2,
          "p50_ms": 1.3846,
          "p95_ms": 1.8886
        },
        "evidence_lookup": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "sqlite_write": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        }
      }
    },
    "learning": {
      "consolidation_runs": [],
      "deltas": [],
      "rejected_deltas": [],
      "watermarks": [],
      "last_result": {},
      "stable_core_version": "a1a58e51882b0c88",
      "performance": {
        "repository": {
          "lookup": {
            "count": 10,
            "p50_ms": 0.844,
            "p95_ms": 1.1089
          },
          "write": {
            "count": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0
          },
          "context": {
            "count": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0
          }
        },
        "consolidation": {
          "consolidation_duration": {
            "count": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0
          },
          "candidate_validation": {
            "count": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0
          }
        },
        "temporal": {
          "temporal_maintenance": {
            "count": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0
          }
        },
        "action_history": {
          "action_ledger_lookup": {
            "count": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0
          }
        },
        "owner_preferences": {
          "owner_preference_lookup": {
            "count": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0
          }
        },
        "hebe_self": {
          "hebe_self_lookup": {
            "count": 1,
            "p50_ms": 1.8985,
            "p95_ms": 1.8985
          }
        },
        "context": {
          "continuity_context_build": {
            "count": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0
          }
        }
      }
    },
    "self_model": {
      "stable_core_version": "a1a58e51882b0c88",
      "evolving_preferences": [],
      "opinions": [],
      "superseded_opinions": []
    },
    "owner_preferences": [],
    "leo_language": {
      "beliefs": [],
      "interpretation_aliases": {}
    },
    "temporal": {
      "expired": [],
      "archived": [],
      "weakened": [],
      "maintenance_actions": [],
      "last_actions": []
    },
    "schedule": {
      "observations": [
        {
          "id": 1,
          "stream_session_id": "1",
          "weekday": "friday",
          "time_window": "night",
          "canonical_content": "Test RPG",
          "content_key": "test rpg",
          "stream_format": "game_playthrough",
          "source": "observed",
          "observed_at": "2026-08-14T04:35:32.628934+02:00"
        }
      ],
      "hypotheses": [
        {
          "id": 1,
          "weekday": "monday",
          "time_window": "any",
          "canonical_content": "FINAL FANTASY IX",
          "content_key": "final fantasy ix",
          "stream_format": "challenge_run",
          "source": "owner_declared",
          "confidence": 0.9,
          "evidence_count": 1,
          "consecutive_matches": 1,
          "consecutive_misses": 0,
          "first_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "last_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "status": "active",
          "superseded_by": null
        },
        {
          "id": 2,
          "weekday": "tuesday",
          "time_window": "any",
          "canonical_content": "Persona 5 Royal",
          "content_key": "persona 5 royal",
          "stream_format": "game_playthrough",
          "source": "owner_declared",
          "confidence": 0.9,
          "evidence_count": 1,
          "consecutive_matches": 1,
          "consecutive_misses": 0,
          "first_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "last_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "status": "active",
          "superseded_by": null
        },
        {
          "id": 3,
          "weekday": "wednesday",
          "time_window": "any",
          "canonical_content": "Persona 5 Royal",
          "content_key": "persona 5 royal",
          "stream_format": "game_playthrough",
          "source": "owner_declared",
          "confidence": 0.9,
          "evidence_count": 1,
          "consecutive_matches": 1,
          "consecutive_misses": 0,
          "first_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "last_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "status": "active",
          "superseded_by": null
        },
        {
          "id": 4,
          "weekday": "thursday",
          "time_window": "any",
          "canonical_content": "Baldur's Gate 3",
          "content_key": "baldur s gate 3",
          "stream_format": "game_playthrough",
          "source": "owner_declared",
          "confidence": 0.9,
          "evidence_count": 1,
          "consecutive_matches": 1,
          "consecutive_misses": 0,
          "first_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "last_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "status": "active",
          "superseded_by": null
        },
        {
          "id": 6,
          "weekday": "saturday",
          "time_window": "any",
          "canonical_content": "Retro Weekend",
          "content_key": "retro weekend",
          "stream_format": "retro",
          "source": "owner_declared",
          "confidence": 0.9,
          "evidence_count": 1,
          "consecutive_matches": 1,
          "consecutive_misses": 0,
          "first_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "last_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "status": "active",
          "superseded_by": null
        },
        {
          "id": 7,
          "weekday": "sunday",
          "time_window": "any",
          "canonical_content": "Retro Weekend",
          "content_key": "retro weekend",
          "stream_format": "retro",
          "source": "owner_declared",
          "confidence": 0.9,
          "evidence_count": 1,
          "consecutive_matches": 1,
          "consecutive_misses": 0,
          "first_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "last_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "status": "active",
          "superseded_by": null
        },
        {
          "id": 5,
          "weekday": "friday",
          "time_window": "any",
          "canonical_content": "Baldur's Gate 3",
          "content_key": "baldur s gate 3",
          "stream_format": "game_playthrough",
          "source": "owner_declared",
          "confidence": 0.76,
          "evidence_count": 1,
          "consecutive_matches": 0,
          "consecutive_misses": 1,
          "first_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "last_observed_at": "2026-08-14T04:35:32.628934+02:00",
          "status": "weakening",
          "superseded_by": null
        }
      ],
      "observed_current_state": {
        "game": "Test RPG",
        "title": "Boss attempts"
      },
      "precedence": "observed_twitch_metadata"
    },
    "action_ledger": {
      "entries": [],
      "last_claim_validation": {}
    },
    "scene_transitions": {
      "all": [],
      "last": {}
    },
    "continuity_context": {},
    "promotion_profiles": [],
    "actions": {
      "attempts": [],
      "speech_requests": [],
      "model_calls": [],
      "research_calls": []
    },
    "receipts": [],
    "emitted_outputs": [],
    "final_emission_results": [],
    "speech_intents": {
      "active": [],
      "all": [],
      "metrics": {
        "pending": 0,
        "time_created_to_emit": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "turn_gap_before_emit": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "intent_creation": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "pending_queue_operation": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "turn_arbitration": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "presence_turn_decision": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        }
      }
    },
    "database_watermarks": {
      "counts": {
        "chat_log": 0,
        "memory_facts": 0,
        "memory_chunks": 0,
        "stream_sessions": 1,
        "stream_chat_messages": 0,
        "stream_events": 1,
        "live_session_timeline": 3,
        "promotion_events": 0,
        "viewer_promotion_profiles": 0,
        "schema_migrations": 7,
        "conversations": 0,
        "open_threads": 0,
        "beliefs": 0,
        "belief_evidence": 0,
        "scene_assertions": 0,
        "game_identities": 0,
        "game_runs": 0,
        "game_run_sessions": 0,
        "game_run_events": 0,
        "game_knowledge_facts": 0,
        "game_knowledge_v2_gaps": 0,
        "people": 0,
        "person_identities": 0,
        "person_sessions": 0,
        "social_episodes": 0,
        "shared_culture_items": 0,
        "shared_culture_evidence": 0,
        "consolidation_runs": 0,
        "consolidation_deltas": 0,
        "action_ledger": 0,
        "temporal_maintenance_audit": 0,
        "learning_observations": 0,
        "scene_transitions": 0,
        "schedule_observations": 1,
        "schedule_hypotheses": 7
      },
      "schema_migrations": [
        {
          "component": "architecture_consolidation",
          "version": 1,
          "name": "audit_hygiene_and_cutover_state",
          "checksum": "944b3ad6532151d4204ed59f663668c9b0ac62df54ece7602d26b91acddd5b7d",
          "applied_at": "2026-08-14T02:35:32.541524+00:00"
        },
        {
          "component": "belief_v2",
          "version": 1,
          "name": "beliefs_evidence_and_compatibility_columns",
          "checksum": "f0df6f1288caccaf6bb47670b38f9cca747f14c84916d6c1ca75beb687200507",
          "applied_at": "2026-08-14T02:35:32.399139+00:00"
        },
        {
          "component": "cognitive_replay",
          "version": 1,
          "name": "replay_metadata",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "applied_at": "2026-08-14T02:35:32.177746+00:00"
        },
        {
          "component": "conversation_continuity",
          "version": 1,
          "name": "conversation_and_open_threads",
          "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
          "applied_at": "2026-08-14T02:35:32.312278+00:00"
        },
        {
          "component": "game_context_v2",
          "version": 1,
          "name": "durable_runs_knowledge_and_gaps",
          "checksum": "08e342acaae00d5d24c1a6dbccad5aee41f753ed90d9ec26282a5fdf042d0a75",
          "applied_at": "2026-08-14T02:35:32.450003+00:00"
        },
        {
          "component": "learning_v2",
          "version": 1,
          "name": "consolidation_temporal_action_and_scene",
          "checksum": "6a86e2d1c7c03167f3b20c328bc97b73fb92e03903f64e64d326fb08f9e3b942",
          "applied_at": "2026-08-14T02:35:32.526410+00:00"
        },
        {
          "component": "social_world_v2",
          "version": 1,
          "name": "people_episodes_and_shared_culture",
          "checksum": "b02adc2cd7f298f1af228dc52c4ba44ae15999fb4c815ea004c40f68b78cbfa5",
          "applied_at": "2026-08-14T02:35:32.488704+00:00"
        }
      ],
      "final_response_digest": "",
      "final_response_present": false
    }
  },
  "leo-speaking": {
    "runtime": {
      "mode": "active",
      "hebe_sleeping": false,
      "is_running": false,
      "last_input_source": null,
      "last_intent": null,
      "last_firewall": {
        "source": null,
        "authority": null,
        "decision": null,
        "reason": null
      },
      "last_policy": {
        "source": null,
        "authority": null,
        "decision": null,
        "reason": null
      }
    },
    "stream_session": {
      "enabled": true,
      "is_live": true,
      "live_status_known": true,
      "active_stream_session_id": 1,
      "last_transition": "online",
      "title": "Boss attempts",
      "game": "Test RPG",
      "category": "Test RPG"
    },
    "current_scene": {},
    "pending": {
      "clarification": null,
      "conversation_turn": null
    },
    "conversation": {
      "active": {},
      "latest": {},
      "all": [],
      "last_resolution": {},
      "legacy_pending_projection": {},
      "continuity_shadow_diff": {},
      "shadow_metrics": {
        "total": 0,
        "matches": 0,
        "differences": 0,
        "match_rate": 1.0,
        "difference_reasons": {}
      },
      "performance": {
        "count": 0,
        "p50_ms": 0.0,
        "p95_ms": 0.0
      }
    },
    "open_threads": [],
    "memory": {
      "facts_count": 0,
      "chunks_count": 0
    },
    "beliefs": {
      "active": [],
      "historical": [],
      "superseded": [],
      "suspected": [],
      "all": [],
      "last_transition": {}
    },
    "belief_evidence": [],
    "retrieval": {
      "last_request": {},
      "performance": {
        "count": 0,
        "p50_ms": 0.0,
        "p95_ms": 0.0
      },
      "write_performance": {
        "count": 0,
        "p50_ms": 0.0,
        "p95_ms": 0.0
      },
      "repository_performance": {
        "belief_lookup": {
          "count": 3,
          "p50_ms": 0.8806,
          "p95_ms": 1.8886
        },
        "evidence_lookup": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "sqlite_write": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        }
      }
    },
    "memory_compatibility": {
      "legacy_to_v2": [],
      "v2_to_legacy": [],
      "shadow_diffs": [],
      "backfill": {
        "safe": 0,
        "compatibility_only": 0,
        "ambiguous": 0,
        "invalid_stale": 0
      }
    },
    "game_state": {
      "game": "Test RPG",
      "platform_version": "",
      "playthrough_type": "casual",
      "spoiler_policy": "no_spoilers",
      "current_location": null,
      "current_character": "",
      "party_members": [],
      "last_confirmed_progress": "",
      "current_objective": null,
      "challenge": "",
      "known_constraints": [],
      "last_updated": 1786727100.0,
      "provenance": "stream_context_sync",
      "confidence": 0.75,
      "current_game": "Test RPG",
      "recent_run_context_facts": [],
      "identity": {},
      "context": {},
      "active_run": {},
      "runs": [],
      "session_links": [],
      "run_events": [],
      "run_beliefs": {
        "current": [],
        "inferred": [],
        "superseded": []
      },
      "knowledge": {
        "selected": [],
        "rejected": [],
        "spoiler_blocked": [],
        "all": []
      },
      "gaps": [],
      "research": {
        "research_calls": 0,
        "cache_hits": 0,
        "failures": [],
        "context_performance": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "fixture_calls": [],
        "status": ""
      },
      "compatibility": {
        "legacy_progress": [],
        "dossier": [],
        "legacy_run": [],
        "shadow_diffs": [],
        "backfill": {
          "validated": 0,
          "compatibility_only": 0,
          "ambiguous": 0,
          "stale": 0
        }
      },
      "provenance_manifest": [],
      "advice_allowed": null,
      "reaction_allowed": null,
      "performance": {
        "run": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "run_fact": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "knowledge": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "research_gap": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "db_write": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        }
      },
      "context_performance": {
        "count": 0,
        "p50_ms": 0.0,
        "p95_ms": 0.0
      },
      "manifest_size_bytes": 0,
      "last_run_resolution": {},
      "run_resolution_performance": {
        "count": 0,
        "p50_ms": 0.0,
        "p95_ms": 0.0
      }
    },
    "social_state": {
      "recent_active_users": [],
      "recent_chat_count": 0,
      "last_raid": {},
      "last_cheer": {},
      "people": [],
      "identities": [],
      "recent_episodes": [],
      "active_hypotheses": [],
      "historical_hypotheses": [],
      "open_threads": [],
      "relationships": [],
      "shared_culture": {
        "all": [],
        "candidates": [],
        "active": [],
        "weakening": [],
        "retired": [],
        "reactions": [],
        "selection": {
          "selected": [],
          "rejected": []
        }
      },
      "retrieval": {},
      "opportunities": [],
      "resolution": {},
      "rejected_writes": [],
      "compatibility": {
        "chatter_presence": [],
        "chatter_profiles": [],
        "chatter_facts": [],
        "stream_chatter_summaries": [],
        "viewer_profiles": [],
        "social_events": [],
        "promotion_profiles": [],
        "backfill": {
          "explicit_observation": 0,
          "safe_episode": 0,
          "inferred_compatibility_only": 0,
          "ambiguous": 0,
          "sensitive": 0,
          "stale": 0
        },
        "shadow_diffs": []
      },
      "performance": {
        "identity": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "episode_write": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "thread_lookup": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "culture_select": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "context": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "db_write": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        }
      },
      "belief_lookup_performance": {
        "belief_lookup": {
          "count": 3,
          "p50_ms": 0.8806,
          "p95_ms": 1.8886
        },
        "evidence_lookup": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "sqlite_write": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        }
      }
    },
    "learning": {
      "consolidation_runs": [],
      "deltas": [],
      "rejected_deltas": [],
      "watermarks": [],
      "last_result": {},
      "stable_core_version": "a1a58e51882b0c88",
      "performance": {
        "repository": {
          "lookup": {
            "count": 15,
            "p50_ms": 0.8474,
            "p95_ms": 1.1089
          },
          "write": {
            "count": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0
          },
          "context": {
            "count": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0
          }
        },
        "consolidation": {
          "consolidation_duration": {
            "count": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0
          },
          "candidate_validation": {
            "count": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0
          }
        },
        "temporal": {
          "temporal_maintenance": {
            "count": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0
          }
        },
        "action_history": {
          "action_ledger_lookup": {
            "count": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0
          }
        },
        "owner_preferences": {
          "owner_preference_lookup": {
            "count": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0
          }
        },
        "hebe_self": {
          "hebe_self_lookup": {
            "count": 1,
            "p50_ms": 1.8985,
            "p95_ms": 1.8985
          }
        },
        "context": {
          "continuity_context_build": {
            "count": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0
          }
        }
      }
    },
    "self_model": {
      "stable_core_version": "a1a58e51882b0c88",
      "evolving_preferences": [],
      "opinions": [],
      "superseded_opinions": []
    },
    "owner_preferences": [],
    "leo_language": {
      "beliefs": [],
      "interpretation_aliases": {}
    },
    "temporal": {
      "expired": [],
      "archived": [],
      "weakened": [],
      "maintenance_actions": [],
      "last_actions": []
    },
    "schedule": {
      "observations": [
        {
          "id": 1,
          "stream_session_id": "1",
          "weekday": "friday",
          "time_window": "night",
          "canonical_content": "Test RPG",
          "content_key": "test rpg",
          "stream_format": "game_playthrough",
          "source": "observed",
          "observed_at": "2026-08-14T04:35:32.628934+02:00"
        }
      ],
      "hypotheses": [
        {
          "id": 1,
          "weekday": "monday",
          "time_window": "any",
          "canonical_content": "FINAL FANTASY IX",
          "content_key": "final fantasy ix",
          "stream_format": "challenge_run",
          "source": "owner_declared",
          "confidence": 0.9,
          "evidence_count": 1,
          "consecutive_matches": 1,
          "consecutive_misses": 0,
          "first_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "last_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "status": "active",
          "superseded_by": null
        },
        {
          "id": 2,
          "weekday": "tuesday",
          "time_window": "any",
          "canonical_content": "Persona 5 Royal",
          "content_key": "persona 5 royal",
          "stream_format": "game_playthrough",
          "source": "owner_declared",
          "confidence": 0.9,
          "evidence_count": 1,
          "consecutive_matches": 1,
          "consecutive_misses": 0,
          "first_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "last_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "status": "active",
          "superseded_by": null
        },
        {
          "id": 3,
          "weekday": "wednesday",
          "time_window": "any",
          "canonical_content": "Persona 5 Royal",
          "content_key": "persona 5 royal",
          "stream_format": "game_playthrough",
          "source": "owner_declared",
          "confidence": 0.9,
          "evidence_count": 1,
          "consecutive_matches": 1,
          "consecutive_misses": 0,
          "first_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "last_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "status": "active",
          "superseded_by": null
        },
        {
          "id": 4,
          "weekday": "thursday",
          "time_window": "any",
          "canonical_content": "Baldur's Gate 3",
          "content_key": "baldur s gate 3",
          "stream_format": "game_playthrough",
          "source": "owner_declared",
          "confidence": 0.9,
          "evidence_count": 1,
          "consecutive_matches": 1,
          "consecutive_misses": 0,
          "first_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "last_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "status": "active",
          "superseded_by": null
        },
        {
          "id": 6,
          "weekday": "saturday",
          "time_window": "any",
          "canonical_content": "Retro Weekend",
          "content_key": "retro weekend",
          "stream_format": "retro",
          "source": "owner_declared",
          "confidence": 0.9,
          "evidence_count": 1,
          "consecutive_matches": 1,
          "consecutive_misses": 0,
          "first_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "last_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "status": "active",
          "superseded_by": null
        },
        {
          "id": 7,
          "weekday": "sunday",
          "time_window": "any",
          "canonical_content": "Retro Weekend",
          "content_key": "retro weekend",
          "stream_format": "retro",
          "source": "owner_declared",
          "confidence": 0.9,
          "evidence_count": 1,
          "consecutive_matches": 1,
          "consecutive_misses": 0,
          "first_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "last_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "status": "active",
          "superseded_by": null
        },
        {
          "id": 5,
          "weekday": "friday",
          "time_window": "any",
          "canonical_content": "Baldur's Gate 3",
          "content_key": "baldur s gate 3",
          "stream_format": "game_playthrough",
          "source": "owner_declared",
          "confidence": 0.76,
          "evidence_count": 1,
          "consecutive_matches": 0,
          "consecutive_misses": 1,
          "first_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "last_observed_at": "2026-08-14T04:35:32.628934+02:00",
          "status": "weakening",
          "superseded_by": null
        }
      ],
      "observed_current_state": {
        "game": "Test RPG",
        "title": "Boss attempts"
      },
      "precedence": "observed_twitch_metadata"
    },
    "action_ledger": {
      "entries": [],
      "last_claim_validation": {}
    },
    "scene_transitions": {
      "all": [],
      "last": {}
    },
    "continuity_context": {},
    "promotion_profiles": [],
    "actions": {
      "attempts": [],
      "speech_requests": [],
      "model_calls": [],
      "research_calls": []
    },
    "receipts": [],
    "emitted_outputs": [],
    "final_emission_results": [],
    "speech_intents": {
      "active": [],
      "all": [],
      "metrics": {
        "pending": 0,
        "time_created_to_emit": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "turn_gap_before_emit": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "intent_creation": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "pending_queue_operation": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "turn_arbitration": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "presence_turn_decision": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        }
      }
    },
    "database_watermarks": {
      "counts": {
        "chat_log": 0,
        "memory_facts": 0,
        "memory_chunks": 0,
        "stream_sessions": 1,
        "stream_chat_messages": 0,
        "stream_events": 1,
        "live_session_timeline": 3,
        "promotion_events": 0,
        "viewer_promotion_profiles": 0,
        "schema_migrations": 7,
        "conversations": 0,
        "open_threads": 0,
        "beliefs": 0,
        "belief_evidence": 0,
        "scene_assertions": 0,
        "game_identities": 0,
        "game_runs": 0,
        "game_run_sessions": 0,
        "game_run_events": 0,
        "game_knowledge_facts": 0,
        "game_knowledge_v2_gaps": 0,
        "people": 0,
        "person_identities": 0,
        "person_sessions": 0,
        "social_episodes": 0,
        "shared_culture_items": 0,
        "shared_culture_evidence": 0,
        "consolidation_runs": 0,
        "consolidation_deltas": 0,
        "action_ledger": 0,
        "temporal_maintenance_audit": 0,
        "learning_observations": 0,
        "scene_transitions": 0,
        "schedule_observations": 1,
        "schedule_hypotheses": 7
      },
      "schema_migrations": [
        {
          "component": "architecture_consolidation",
          "version": 1,
          "name": "audit_hygiene_and_cutover_state",
          "checksum": "944b3ad6532151d4204ed59f663668c9b0ac62df54ece7602d26b91acddd5b7d",
          "applied_at": "2026-08-14T02:35:32.541524+00:00"
        },
        {
          "component": "belief_v2",
          "version": 1,
          "name": "beliefs_evidence_and_compatibility_columns",
          "checksum": "f0df6f1288caccaf6bb47670b38f9cca747f14c84916d6c1ca75beb687200507",
          "applied_at": "2026-08-14T02:35:32.399139+00:00"
        },
        {
          "component": "cognitive_replay",
          "version": 1,
          "name": "replay_metadata",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "applied_at": "2026-08-14T02:35:32.177746+00:00"
        },
        {
          "component": "conversation_continuity",
          "version": 1,
          "name": "conversation_and_open_threads",
          "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
          "applied_at": "2026-08-14T02:35:32.312278+00:00"
        },
        {
          "component": "game_context_v2",
          "version": 1,
          "name": "durable_runs_knowledge_and_gaps",
          "checksum": "08e342acaae00d5d24c1a6dbccad5aee41f753ed90d9ec26282a5fdf042d0a75",
          "applied_at": "2026-08-14T02:35:32.450003+00:00"
        },
        {
          "component": "learning_v2",
          "version": 1,
          "name": "consolidation_temporal_action_and_scene",
          "checksum": "6a86e2d1c7c03167f3b20c328bc97b73fb92e03903f64e64d326fb08f9e3b942",
          "applied_at": "2026-08-14T02:35:32.526410+00:00"
        },
        {
          "component": "social_world_v2",
          "version": 1,
          "name": "people_episodes_and_shared_culture",
          "checksum": "b02adc2cd7f298f1af228dc52c4ba44ae15999fb4c815ea004c40f68b78cbfa5",
          "applied_at": "2026-08-14T02:35:32.488704+00:00"
        }
      ],
      "final_response_digest": "",
      "final_response_present": false
    }
  },
  "reaction-material": {
    "runtime": {
      "mode": "active",
      "hebe_sleeping": false,
      "is_running": false,
      "last_input_source": null,
      "last_intent": null,
      "last_firewall": {
        "source": "ambient_stt",
        "authority": "ambient",
        "decision": "allow_context_only",
        "reason": "ambient_context_only"
      },
      "last_policy": {
        "source": null,
        "authority": null,
        "decision": null,
        "reason": null
      }
    },
    "stream_session": {
      "enabled": true,
      "is_live": true,
      "live_status_known": true,
      "active_stream_session_id": 1,
      "last_transition": "online",
      "title": "Boss attempts",
      "game": "Test RPG",
      "category": "Test RPG"
    },
    "current_scene": {
      "scene_id": "scene_2f0690b8ded85d18ba449ec4e37abb13",
      "topic_id": "topic_dc448af4c39f",
      "entity": "unknown",
      "current_state": "active",
      "state_version": 1,
      "supporting_event_ids": [
        "ambient:rng_dependency:1786727102"
      ],
      "superseded_event_ids": [],
      "terminal": false,
      "updated_at": 1786727102.0
    },
    "pending": {
      "clarification": null,
      "conversation_turn": null
    },
    "conversation": {
      "active": {},
      "latest": {},
      "all": [],
      "last_resolution": {
        "consumed": false,
        "decision": "no_conversation",
        "reason": "no_compatible_active_conversation",
        "conversation_id": "",
        "reply_act": "UNKNOWN",
        "payload": {},
        "conversation": null,
        "latency_ms": 3.095800057053566
      },
      "legacy_pending_projection": {},
      "continuity_shadow_diff": {
        "legacy_result": false,
        "v2_result": false,
        "match": true,
        "difference_reason": ""
      },
      "shadow_metrics": {
        "total": 1,
        "matches": 1,
        "differences": 0,
        "match_rate": 1.0,
        "difference_reasons": {}
      },
      "performance": {
        "count": 1,
        "p50_ms": 3.0958,
        "p95_ms": 3.0958
      }
    },
    "open_threads": [],
    "memory": {
      "facts_count": 0,
      "chunks_count": 0
    },
    "beliefs": {
      "active": [],
      "historical": [],
      "superseded": [],
      "suspected": [],
      "all": [],
      "last_transition": {}
    },
    "belief_evidence": [],
    "retrieval": {
      "last_request": {},
      "performance": {
        "count": 0,
        "p50_ms": 0.0,
        "p95_ms": 0.0
      },
      "write_performance": {
        "count": 0,
        "p50_ms": 0.0,
        "p95_ms": 0.0
      },
      "repository_performance": {
        "belief_lookup": {
          "count": 4,
          "p50_ms": 0.91625,
          "p95_ms": 1.8886
        },
        "evidence_lookup": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "sqlite_write": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        }
      }
    },
    "memory_compatibility": {
      "legacy_to_v2": [],
      "v2_to_legacy": [],
      "shadow_diffs": [],
      "backfill": {
        "safe": 0,
        "compatibility_only": 0,
        "ambiguous": 0,
        "invalid_stale": 0
      }
    },
    "game_state": {
      "game": "Test RPG",
      "platform_version": "",
      "playthrough_type": "casual",
      "spoiler_policy": "no_spoilers",
      "current_location": null,
      "current_character": "",
      "party_members": [],
      "last_confirmed_progress": "",
      "current_objective": null,
      "challenge": "",
      "known_constraints": [],
      "last_updated": 1786727100.0,
      "provenance": "stream_context_sync",
      "confidence": 0.75,
      "current_game": "Test RPG",
      "recent_run_context_facts": [
        {
          "kind": "rng_dependency",
          "text": "Leo framed the current situation as dependent on RNG or luck.",
          "category": "rng_dependency",
          "summary": "Leo framed the current situation as dependent on RNG or luck.",
          "id": "ambient:rng_dependency:1786727102",
          "fact_id": "ambient:rng_dependency:1786727102",
          "raw_text": "<redacted:2320c1febf71>",
          "conservative_normalized_text": "que ha sido demasiada suerte dios",
          "utterance_role": "owner_question_to_stream",
          "timestamp": 1786727102.0,
          "topic_id": "topic_dc448af4c39f",
          "heuristic_category": "rng_dependency",
          "extracted_subject": "unknown",
          "subject": "unknown",
          "extracted_object": "",
          "object": "",
          "extracted_predicate": "",
          "predicate": "",
          "confidence": 0.86,
          "referent_confidence": 0.86,
          "inference_level": "heuristic",
          "supported_claims": [
            "que ha sido demasiada suerte dios"
          ],
          "directly_supported_claims": [
            "que ha sido demasiada suerte dios"
          ],
          "inferred_claims": [],
          "unsupported_claims": [],
          "evidence_span": "que ha sido demasiada suerte dios",
          "evidence_tokens": "<redacted:323fcf3ad517>",
          "semantic_rule": "ambient_category:rng_dependency",
          "model_reason": "Leo framed the current situation as dependent on RNG or luck.",
          "expires_at": 1786727222.0,
          "scene_id": "scene_2f0690b8ded85d18ba449ec4e37abb13",
          "game": null,
          "source": "stt_voice",
          "raw_evidence": "que ha sido demasiada suerte dios",
          "normalized_text": "que ha sido demasiada suerte dios",
          "normalized_evidence": "que ha sido demasiada suerte dios",
          "language": "es",
          "ttl_sec": 120,
          "data": {
            "category": "rng_dependency",
            "raw_text": "<redacted:2320c1febf71>",
            "normalized_text": "que ha sido demasiada suerte dios",
            "mood": "rng tension",
            "extracted_subject": "unknown",
            "extracted_object": "",
            "extracted_predicate": "",
            "inference_level": "heuristic",
            "supported_claims": [
              "que ha sido demasiada suerte dios"
            ],
            "inferred_claims": [],
            "unsupported_claims": [],
            "evidence_span": "que ha sido demasiada suerte dios",
            "evidence_tokens": "<redacted:323fcf3ad517>",
            "semantic_rule": "ambient_category:rng_dependency",
            "model_reason": "Leo framed the current situation as dependent on RNG or luck.",
            "scene_id": "scene_2f0690b8ded85d18ba449ec4e37abb13"
          },
          "age_seconds": 0.0,
          "superseded": false,
          "state_version": 1,
          "current_state": "active",
          "terminal": false,
          "currentness_score": 1.0
        }
      ],
      "identity": {},
      "context": {},
      "active_run": {},
      "runs": [],
      "session_links": [],
      "run_events": [],
      "run_beliefs": {
        "current": [],
        "inferred": [],
        "superseded": []
      },
      "knowledge": {
        "selected": [],
        "rejected": [],
        "spoiler_blocked": [],
        "all": []
      },
      "gaps": [],
      "research": {
        "research_calls": 0,
        "cache_hits": 0,
        "failures": [],
        "context_performance": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "fixture_calls": [],
        "status": ""
      },
      "compatibility": {
        "legacy_progress": [],
        "dossier": [],
        "legacy_run": [],
        "shadow_diffs": [],
        "backfill": {
          "validated": 0,
          "compatibility_only": 0,
          "ambiguous": 0,
          "stale": 0
        }
      },
      "provenance_manifest": [],
      "advice_allowed": null,
      "reaction_allowed": null,
      "performance": {
        "run": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "run_fact": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "knowledge": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "research_gap": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "db_write": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        }
      },
      "context_performance": {
        "count": 0,
        "p50_ms": 0.0,
        "p95_ms": 0.0
      },
      "manifest_size_bytes": 0,
      "last_run_resolution": {},
      "run_resolution_performance": {
        "count": 0,
        "p50_ms": 0.0,
        "p95_ms": 0.0
      }
    },
    "social_state": {
      "recent_active_users": [],
      "recent_chat_count": 0,
      "last_raid": {},
      "last_cheer": {},
      "people": [],
      "identities": [],
      "recent_episodes": [],
      "active_hypotheses": [],
      "historical_hypotheses": [],
      "open_threads": [],
      "relationships": [],
      "shared_culture": {
        "all": [],
        "candidates": [],
        "active": [],
        "weakening": [],
        "retired": [],
        "reactions": [],
        "selection": {
          "selected": [],
          "rejected": []
        }
      },
      "retrieval": {},
      "opportunities": [],
      "resolution": {},
      "rejected_writes": [],
      "compatibility": {
        "chatter_presence": [],
        "chatter_profiles": [],
        "chatter_facts": [],
        "stream_chatter_summaries": [],
        "viewer_profiles": [],
        "social_events": [],
        "promotion_profiles": [],
        "backfill": {
          "explicit_observation": 0,
          "safe_episode": 0,
          "inferred_compatibility_only": 0,
          "ambiguous": 0,
          "sensitive": 0,
          "stale": 0
        },
        "shadow_diffs": []
      },
      "performance": {
        "identity": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "episode_write": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "thread_lookup": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "culture_select": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "context": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "db_write": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        }
      },
      "belief_lookup_performance": {
        "belief_lookup": {
          "count": 4,
          "p50_ms": 0.91625,
          "p95_ms": 1.8886
        },
        "evidence_lookup": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "sqlite_write": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        }
      }
    },
    "learning": {
      "consolidation_runs": [],
      "deltas": [],
      "rejected_deltas": [],
      "watermarks": [],
      "last_result": {},
      "stable_core_version": "a1a58e51882b0c88",
      "performance": {
        "repository": {
          "lookup": {
            "count": 20,
            "p50_ms": 0.8607,
            "p95_ms": 1.1089
          },
          "write": {
            "count": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0
          },
          "context": {
            "count": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0
          }
        },
        "consolidation": {
          "consolidation_duration": {
            "count": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0
          },
          "candidate_validation": {
            "count": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0
          }
        },
        "temporal": {
          "temporal_maintenance": {
            "count": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0
          }
        },
        "action_history": {
          "action_ledger_lookup": {
            "count": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0
          }
        },
        "owner_preferences": {
          "owner_preference_lookup": {
            "count": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0
          }
        },
        "hebe_self": {
          "hebe_self_lookup": {
            "count": 1,
            "p50_ms": 1.8985,
            "p95_ms": 1.8985
          }
        },
        "context": {
          "continuity_context_build": {
            "count": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0
          }
        }
      }
    },
    "self_model": {
      "stable_core_version": "a1a58e51882b0c88",
      "evolving_preferences": [],
      "opinions": [],
      "superseded_opinions": []
    },
    "owner_preferences": [],
    "leo_language": {
      "beliefs": [],
      "interpretation_aliases": {}
    },
    "temporal": {
      "expired": [],
      "archived": [],
      "weakened": [],
      "maintenance_actions": [],
      "last_actions": []
    },
    "schedule": {
      "observations": [
        {
          "id": 1,
          "stream_session_id": "1",
          "weekday": "friday",
          "time_window": "night",
          "canonical_content": "Test RPG",
          "content_key": "test rpg",
          "stream_format": "game_playthrough",
          "source": "observed",
          "observed_at": "2026-08-14T04:35:32.628934+02:00"
        }
      ],
      "hypotheses": [
        {
          "id": 1,
          "weekday": "monday",
          "time_window": "any",
          "canonical_content": "FINAL FANTASY IX",
          "content_key": "final fantasy ix",
          "stream_format": "challenge_run",
          "source": "owner_declared",
          "confidence": 0.9,
          "evidence_count": 1,
          "consecutive_matches": 1,
          "consecutive_misses": 0,
          "first_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "last_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "status": "active",
          "superseded_by": null
        },
        {
          "id": 2,
          "weekday": "tuesday",
          "time_window": "any",
          "canonical_content": "Persona 5 Royal",
          "content_key": "persona 5 royal",
          "stream_format": "game_playthrough",
          "source": "owner_declared",
          "confidence": 0.9,
          "evidence_count": 1,
          "consecutive_matches": 1,
          "consecutive_misses": 0,
          "first_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "last_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "status": "active",
          "superseded_by": null
        },
        {
          "id": 3,
          "weekday": "wednesday",
          "time_window": "any",
          "canonical_content": "Persona 5 Royal",
          "content_key": "persona 5 royal",
          "stream_format": "game_playthrough",
          "source": "owner_declared",
          "confidence": 0.9,
          "evidence_count": 1,
          "consecutive_matches": 1,
          "consecutive_misses": 0,
          "first_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "last_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "status": "active",
          "superseded_by": null
        },
        {
          "id": 4,
          "weekday": "thursday",
          "time_window": "any",
          "canonical_content": "Baldur's Gate 3",
          "content_key": "baldur s gate 3",
          "stream_format": "game_playthrough",
          "source": "owner_declared",
          "confidence": 0.9,
          "evidence_count": 1,
          "consecutive_matches": 1,
          "consecutive_misses": 0,
          "first_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "last_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "status": "active",
          "superseded_by": null
        },
        {
          "id": 6,
          "weekday": "saturday",
          "time_window": "any",
          "canonical_content": "Retro Weekend",
          "content_key": "retro weekend",
          "stream_format": "retro",
          "source": "owner_declared",
          "confidence": 0.9,
          "evidence_count": 1,
          "consecutive_matches": 1,
          "consecutive_misses": 0,
          "first_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "last_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "status": "active",
          "superseded_by": null
        },
        {
          "id": 7,
          "weekday": "sunday",
          "time_window": "any",
          "canonical_content": "Retro Weekend",
          "content_key": "retro weekend",
          "stream_format": "retro",
          "source": "owner_declared",
          "confidence": 0.9,
          "evidence_count": 1,
          "consecutive_matches": 1,
          "consecutive_misses": 0,
          "first_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "last_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "status": "active",
          "superseded_by": null
        },
        {
          "id": 5,
          "weekday": "friday",
          "time_window": "any",
          "canonical_content": "Baldur's Gate 3",
          "content_key": "baldur s gate 3",
          "stream_format": "game_playthrough",
          "source": "owner_declared",
          "confidence": 0.76,
          "evidence_count": 1,
          "consecutive_matches": 0,
          "consecutive_misses": 1,
          "first_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "last_observed_at": "2026-08-14T04:35:32.628934+02:00",
          "status": "weakening",
          "superseded_by": null
        }
      ],
      "observed_current_state": {
        "game": "Test RPG",
        "title": "Boss attempts"
      },
      "precedence": "observed_twitch_metadata"
    },
    "action_ledger": {
      "entries": [],
      "last_claim_validation": {}
    },
    "scene_transitions": {
      "all": [],
      "last": {}
    },
    "continuity_context": {},
    "promotion_profiles": [],
    "actions": {
      "attempts": [],
      "speech_requests": [],
      "model_calls": [],
      "research_calls": []
    },
    "receipts": [],
    "emitted_outputs": [],
    "final_emission_results": [],
    "speech_intents": {
      "active": [],
      "all": [],
      "metrics": {
        "pending": 0,
        "time_created_to_emit": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "turn_gap_before_emit": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "intent_creation": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "pending_queue_operation": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "turn_arbitration": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "presence_turn_decision": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        }
      }
    },
    "database_watermarks": {
      "counts": {
        "chat_log": 0,
        "memory_facts": 0,
        "memory_chunks": 0,
        "stream_sessions": 1,
        "stream_chat_messages": 0,
        "stream_events": 1,
        "live_session_timeline": 4,
        "promotion_events": 0,
        "viewer_promotion_profiles": 0,
        "schema_migrations": 7,
        "conversations": 0,
        "open_threads": 0,
        "beliefs": 0,
        "belief_evidence": 0,
        "scene_assertions": 0,
        "game_identities": 0,
        "game_runs": 0,
        "game_run_sessions": 0,
        "game_run_events": 0,
        "game_knowledge_facts": 0,
        "game_knowledge_v2_gaps": 0,
        "people": 0,
        "person_identities": 0,
        "person_sessions": 0,
        "social_episodes": 0,
        "shared_culture_items": 0,
        "shared_culture_evidence": 0,
        "consolidation_runs": 0,
        "consolidation_deltas": 0,
        "action_ledger": 0,
        "temporal_maintenance_audit": 0,
        "learning_observations": 0,
        "scene_transitions": 0,
        "schedule_observations": 1,
        "schedule_hypotheses": 7
      },
      "schema_migrations": [
        {
          "component": "architecture_consolidation",
          "version": 1,
          "name": "audit_hygiene_and_cutover_state",
          "checksum": "944b3ad6532151d4204ed59f663668c9b0ac62df54ece7602d26b91acddd5b7d",
          "applied_at": "2026-08-14T02:35:32.541524+00:00"
        },
        {
          "component": "belief_v2",
          "version": 1,
          "name": "beliefs_evidence_and_compatibility_columns",
          "checksum": "f0df6f1288caccaf6bb47670b38f9cca747f14c84916d6c1ca75beb687200507",
          "applied_at": "2026-08-14T02:35:32.399139+00:00"
        },
        {
          "component": "cognitive_replay",
          "version": 1,
          "name": "replay_metadata",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "applied_at": "2026-08-14T02:35:32.177746+00:00"
        },
        {
          "component": "conversation_continuity",
          "version": 1,
          "name": "conversation_and_open_threads",
          "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
          "applied_at": "2026-08-14T02:35:32.312278+00:00"
        },
        {
          "component": "game_context_v2",
          "version": 1,
          "name": "durable_runs_knowledge_and_gaps",
          "checksum": "08e342acaae00d5d24c1a6dbccad5aee41f753ed90d9ec26282a5fdf042d0a75",
          "applied_at": "2026-08-14T02:35:32.450003+00:00"
        },
        {
          "component": "learning_v2",
          "version": 1,
          "name": "consolidation_temporal_action_and_scene",
          "checksum": "6a86e2d1c7c03167f3b20c328bc97b73fb92e03903f64e64d326fb08f9e3b942",
          "applied_at": "2026-08-14T02:35:32.526410+00:00"
        },
        {
          "component": "social_world_v2",
          "version": 1,
          "name": "people_episodes_and_shared_culture",
          "checksum": "b02adc2cd7f298f1af228dc52c4ba44ae15999fb4c815ea004c40f68b78cbfa5",
          "applied_at": "2026-08-14T02:35:32.488704+00:00"
        }
      ],
      "final_response_digest": "",
      "final_response_present": false
    }
  },
  "blocked-active-voice": {
    "runtime": {
      "mode": "active",
      "hebe_sleeping": false,
      "is_running": false,
      "last_input_source": null,
      "last_intent": null,
      "last_firewall": {
        "source": "ambient_stt",
        "authority": "ambient",
        "decision": "allow_context_only",
        "reason": "ambient_context_only"
      },
      "last_policy": {
        "source": null,
        "authority": null,
        "decision": null,
        "reason": null
      }
    },
    "stream_session": {
      "enabled": true,
      "is_live": true,
      "live_status_known": true,
      "active_stream_session_id": 1,
      "last_transition": "online",
      "title": "Boss attempts",
      "game": "Test RPG",
      "category": "Test RPG"
    },
    "current_scene": {
      "scene_id": "scene_2f0690b8ded85d18ba449ec4e37abb13",
      "topic_id": "topic_dc448af4c39f",
      "entity": "unknown",
      "current_state": "active",
      "state_version": 1,
      "supporting_event_ids": [
        "ambient:rng_dependency:1786727102"
      ],
      "superseded_event_ids": [],
      "terminal": false,
      "updated_at": 1786727102.0
    },
    "pending": {
      "clarification": null,
      "conversation_turn": null
    },
    "conversation": {
      "active": {},
      "latest": {},
      "all": [],
      "last_resolution": {
        "consumed": false,
        "decision": "no_conversation",
        "reason": "no_compatible_active_conversation",
        "conversation_id": "",
        "reply_act": "UNKNOWN",
        "payload": {},
        "conversation": null,
        "latency_ms": 3.095800057053566
      },
      "legacy_pending_projection": {},
      "continuity_shadow_diff": {
        "legacy_result": false,
        "v2_result": false,
        "match": true,
        "difference_reason": ""
      },
      "shadow_metrics": {
        "total": 1,
        "matches": 1,
        "differences": 0,
        "match_rate": 1.0,
        "difference_reasons": {}
      },
      "performance": {
        "count": 1,
        "p50_ms": 3.0958,
        "p95_ms": 3.0958
      }
    },
    "open_threads": [],
    "memory": {
      "facts_count": 0,
      "chunks_count": 0
    },
    "beliefs": {
      "active": [],
      "historical": [],
      "superseded": [],
      "suspected": [],
      "all": [],
      "last_transition": {}
    },
    "belief_evidence": [],
    "retrieval": {
      "last_request": {},
      "performance": {
        "count": 0,
        "p50_ms": 0.0,
        "p95_ms": 0.0
      },
      "write_performance": {
        "count": 0,
        "p50_ms": 0.0,
        "p95_ms": 0.0
      },
      "repository_performance": {
        "belief_lookup": {
          "count": 5,
          "p50_ms": 0.8806,
          "p95_ms": 1.8886
        },
        "evidence_lookup": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "sqlite_write": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        }
      }
    },
    "memory_compatibility": {
      "legacy_to_v2": [],
      "v2_to_legacy": [],
      "shadow_diffs": [],
      "backfill": {
        "safe": 0,
        "compatibility_only": 0,
        "ambiguous": 0,
        "invalid_stale": 0
      }
    },
    "game_state": {
      "game": "Test RPG",
      "platform_version": "",
      "playthrough_type": "casual",
      "spoiler_policy": "no_spoilers",
      "current_location": null,
      "current_character": "",
      "party_members": [],
      "last_confirmed_progress": "",
      "current_objective": null,
      "challenge": "",
      "known_constraints": [],
      "last_updated": 1786727100.0,
      "provenance": "stream_context_sync",
      "confidence": 0.75,
      "current_game": "Test RPG",
      "recent_run_context_facts": [
        {
          "kind": "rng_dependency",
          "text": "Leo framed the current situation as dependent on RNG or luck.",
          "category": "rng_dependency",
          "summary": "Leo framed the current situation as dependent on RNG or luck.",
          "id": "ambient:rng_dependency:1786727102",
          "fact_id": "ambient:rng_dependency:1786727102",
          "raw_text": "<redacted:2320c1febf71>",
          "conservative_normalized_text": "que ha sido demasiada suerte dios",
          "utterance_role": "owner_question_to_stream",
          "timestamp": 1786727102.0,
          "topic_id": "topic_dc448af4c39f",
          "heuristic_category": "rng_dependency",
          "extracted_subject": "unknown",
          "subject": "unknown",
          "extracted_object": "",
          "object": "",
          "extracted_predicate": "",
          "predicate": "",
          "confidence": 0.86,
          "referent_confidence": 0.86,
          "inference_level": "heuristic",
          "supported_claims": [
            "que ha sido demasiada suerte dios"
          ],
          "directly_supported_claims": [
            "que ha sido demasiada suerte dios"
          ],
          "inferred_claims": [],
          "unsupported_claims": [],
          "evidence_span": "que ha sido demasiada suerte dios",
          "evidence_tokens": "<redacted:323fcf3ad517>",
          "semantic_rule": "ambient_category:rng_dependency",
          "model_reason": "Leo framed the current situation as dependent on RNG or luck.",
          "expires_at": 1786727222.0,
          "scene_id": "scene_2f0690b8ded85d18ba449ec4e37abb13",
          "game": null,
          "source": "stt_voice",
          "raw_evidence": "que ha sido demasiada suerte dios",
          "normalized_text": "que ha sido demasiada suerte dios",
          "normalized_evidence": "que ha sido demasiada suerte dios",
          "language": "es",
          "ttl_sec": 120,
          "data": {
            "category": "rng_dependency",
            "raw_text": "<redacted:2320c1febf71>",
            "normalized_text": "que ha sido demasiada suerte dios",
            "mood": "rng tension",
            "extracted_subject": "unknown",
            "extracted_object": "",
            "extracted_predicate": "",
            "inference_level": "heuristic",
            "supported_claims": [
              "que ha sido demasiada suerte dios"
            ],
            "inferred_claims": [],
            "unsupported_claims": [],
            "evidence_span": "que ha sido demasiada suerte dios",
            "evidence_tokens": "<redacted:323fcf3ad517>",
            "semantic_rule": "ambient_category:rng_dependency",
            "model_reason": "Leo framed the current situation as dependent on RNG or luck.",
            "scene_id": "scene_2f0690b8ded85d18ba449ec4e37abb13"
          },
          "age_seconds": 0.0,
          "superseded": false,
          "state_version": 1,
          "current_state": "active",
          "terminal": false,
          "currentness_score": 1.0
        }
      ],
      "identity": {},
      "context": {},
      "active_run": {},
      "runs": [],
      "session_links": [],
      "run_events": [],
      "run_beliefs": {
        "current": [],
        "inferred": [],
        "superseded": []
      },
      "knowledge": {
        "selected": [],
        "rejected": [],
        "spoiler_blocked": [],
        "all": []
      },
      "gaps": [],
      "research": {
        "research_calls": 0,
        "cache_hits": 0,
        "failures": [],
        "context_performance": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "fixture_calls": [],
        "status": ""
      },
      "compatibility": {
        "legacy_progress": [],
        "dossier": [],
        "legacy_run": [],
        "shadow_diffs": [],
        "backfill": {
          "validated": 0,
          "compatibility_only": 0,
          "ambiguous": 0,
          "stale": 0
        }
      },
      "provenance_manifest": [],
      "advice_allowed": null,
      "reaction_allowed": null,
      "performance": {
        "run": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "run_fact": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "knowledge": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "research_gap": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "db_write": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        }
      },
      "context_performance": {
        "count": 0,
        "p50_ms": 0.0,
        "p95_ms": 0.0
      },
      "manifest_size_bytes": 0,
      "last_run_resolution": {},
      "run_resolution_performance": {
        "count": 0,
        "p50_ms": 0.0,
        "p95_ms": 0.0
      }
    },
    "social_state": {
      "recent_active_users": [],
      "recent_chat_count": 0,
      "last_raid": {},
      "last_cheer": {},
      "people": [],
      "identities": [],
      "recent_episodes": [],
      "active_hypotheses": [],
      "historical_hypotheses": [],
      "open_threads": [],
      "relationships": [],
      "shared_culture": {
        "all": [],
        "candidates": [],
        "active": [],
        "weakening": [],
        "retired": [],
        "reactions": [],
        "selection": {
          "selected": [],
          "rejected": []
        }
      },
      "retrieval": {},
      "opportunities": [],
      "resolution": {},
      "rejected_writes": [],
      "compatibility": {
        "chatter_presence": [],
        "chatter_profiles": [],
        "chatter_facts": [],
        "stream_chatter_summaries": [],
        "viewer_profiles": [],
        "social_events": [],
        "promotion_profiles": [],
        "backfill": {
          "explicit_observation": 0,
          "safe_episode": 0,
          "inferred_compatibility_only": 0,
          "ambiguous": 0,
          "sensitive": 0,
          "stale": 0
        },
        "shadow_diffs": []
      },
      "performance": {
        "identity": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "episode_write": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "thread_lookup": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "culture_select": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "context": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "db_write": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        }
      },
      "belief_lookup_performance": {
        "belief_lookup": {
          "count": 5,
          "p50_ms": 0.8806,
          "p95_ms": 1.8886
        },
        "evidence_lookup": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "sqlite_write": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        }
      }
    },
    "learning": {
      "consolidation_runs": [],
      "deltas": [],
      "rejected_deltas": [],
      "watermarks": [],
      "last_result": {},
      "stable_core_version": "a1a58e51882b0c88",
      "performance": {
        "repository": {
          "lookup": {
            "count": 25,
            "p50_ms": 0.8703,
            "p95_ms": 1.4161
          },
          "write": {
            "count": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0
          },
          "context": {
            "count": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0
          }
        },
        "consolidation": {
          "consolidation_duration": {
            "count": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0
          },
          "candidate_validation": {
            "count": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0
          }
        },
        "temporal": {
          "temporal_maintenance": {
            "count": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0
          }
        },
        "action_history": {
          "action_ledger_lookup": {
            "count": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0
          }
        },
        "owner_preferences": {
          "owner_preference_lookup": {
            "count": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0
          }
        },
        "hebe_self": {
          "hebe_self_lookup": {
            "count": 1,
            "p50_ms": 1.8985,
            "p95_ms": 1.8985
          }
        },
        "context": {
          "continuity_context_build": {
            "count": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0
          }
        }
      }
    },
    "self_model": {
      "stable_core_version": "a1a58e51882b0c88",
      "evolving_preferences": [],
      "opinions": [],
      "superseded_opinions": []
    },
    "owner_preferences": [],
    "leo_language": {
      "beliefs": [],
      "interpretation_aliases": {}
    },
    "temporal": {
      "expired": [],
      "archived": [],
      "weakened": [],
      "maintenance_actions": [],
      "last_actions": []
    },
    "schedule": {
      "observations": [
        {
          "id": 1,
          "stream_session_id": "1",
          "weekday": "friday",
          "time_window": "night",
          "canonical_content": "Test RPG",
          "content_key": "test rpg",
          "stream_format": "game_playthrough",
          "source": "observed",
          "observed_at": "2026-08-14T04:35:32.628934+02:00"
        }
      ],
      "hypotheses": [
        {
          "id": 1,
          "weekday": "monday",
          "time_window": "any",
          "canonical_content": "FINAL FANTASY IX",
          "content_key": "final fantasy ix",
          "stream_format": "challenge_run",
          "source": "owner_declared",
          "confidence": 0.9,
          "evidence_count": 1,
          "consecutive_matches": 1,
          "consecutive_misses": 0,
          "first_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "last_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "status": "active",
          "superseded_by": null
        },
        {
          "id": 2,
          "weekday": "tuesday",
          "time_window": "any",
          "canonical_content": "Persona 5 Royal",
          "content_key": "persona 5 royal",
          "stream_format": "game_playthrough",
          "source": "owner_declared",
          "confidence": 0.9,
          "evidence_count": 1,
          "consecutive_matches": 1,
          "consecutive_misses": 0,
          "first_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "last_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "status": "active",
          "superseded_by": null
        },
        {
          "id": 3,
          "weekday": "wednesday",
          "time_window": "any",
          "canonical_content": "Persona 5 Royal",
          "content_key": "persona 5 royal",
          "stream_format": "game_playthrough",
          "source": "owner_declared",
          "confidence": 0.9,
          "evidence_count": 1,
          "consecutive_matches": 1,
          "consecutive_misses": 0,
          "first_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "last_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "status": "active",
          "superseded_by": null
        },
        {
          "id": 4,
          "weekday": "thursday",
          "time_window": "any",
          "canonical_content": "Baldur's Gate 3",
          "content_key": "baldur s gate 3",
          "stream_format": "game_playthrough",
          "source": "owner_declared",
          "confidence": 0.9,
          "evidence_count": 1,
          "consecutive_matches": 1,
          "consecutive_misses": 0,
          "first_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "last_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "status": "active",
          "superseded_by": null
        },
        {
          "id": 6,
          "weekday": "saturday",
          "time_window": "any",
          "canonical_content": "Retro Weekend",
          "content_key": "retro weekend",
          "stream_format": "retro",
          "source": "owner_declared",
          "confidence": 0.9,
          "evidence_count": 1,
          "consecutive_matches": 1,
          "consecutive_misses": 0,
          "first_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "last_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "status": "active",
          "superseded_by": null
        },
        {
          "id": 7,
          "weekday": "sunday",
          "time_window": "any",
          "canonical_content": "Retro Weekend",
          "content_key": "retro weekend",
          "stream_format": "retro",
          "source": "owner_declared",
          "confidence": 0.9,
          "evidence_count": 1,
          "consecutive_matches": 1,
          "consecutive_misses": 0,
          "first_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "last_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "status": "active",
          "superseded_by": null
        },
        {
          "id": 5,
          "weekday": "friday",
          "time_window": "any",
          "canonical_content": "Baldur's Gate 3",
          "content_key": "baldur s gate 3",
          "stream_format": "game_playthrough",
          "source": "owner_declared",
          "confidence": 0.76,
          "evidence_count": 1,
          "consecutive_matches": 0,
          "consecutive_misses": 1,
          "first_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "last_observed_at": "2026-08-14T04:35:32.628934+02:00",
          "status": "weakening",
          "superseded_by": null
        }
      ],
      "observed_current_state": {
        "game": "Test RPG",
        "title": "Boss attempts"
      },
      "precedence": "observed_twitch_metadata"
    },
    "action_ledger": {
      "entries": [],
      "last_claim_validation": {}
    },
    "scene_transitions": {
      "all": [],
      "last": {}
    },
    "continuity_context": {},
    "promotion_profiles": [],
    "actions": {
      "attempts": [],
      "speech_requests": [],
      "model_calls": [],
      "research_calls": []
    },
    "receipts": [],
    "emitted_outputs": [],
    "final_emission_results": [],
    "speech_intents": {
      "active": [
        {
          "id": "intent_23c0d3235c5150eb9db77c46247f8c22",
          "type": "REACTION",
          "source_event_ids": [],
          "anchor_ids": [
            "ambient:rng_dependency:1786727102"
          ],
          "topic": "rng_dependency",
          "subject_ref": "unknown",
          "value": 0.86,
          "urgency": 0.85,
          "freshness": 0.998333332935969,
          "created_at": 1786727102.2,
          "expires_at": 1786727110.2,
          "interruptibility": "yield_before_tts_commit",
          "minimum_turn_gap": 1.2,
          "maximum_turn_delay": 8.0,
          "scene_relevance": {
            "scene_id": "scene_2f0690b8ded85d18ba449ec4e37abb13",
            "state_version": 1,
            "current_state": "active",
            "terminal": false
          },
          "status": "PENDING",
          "suppression_reason": "",
          "reserved_at": 0.0,
          "emitted_at": 0.0,
          "contribution_material": {
            "anchor": {
              "id": "ambient:rng_dependency:1786727102",
              "type": "rng_dependency",
              "quality": 0.86,
              "reason": "recent_ambient_context",
              "evidence": {
                "anchor_id": "ambient:rng_dependency:1786727102",
                "anchor_type": "rng_dependency",
                "raw_owner_fragments": [
                  "que ha sido demasiada suerte dios"
                ],
                "exact_supported_claims": [
                  "que ha sido demasiada suerte dios"
                ],
                "timestamps": [
                  1786727102.0
                ],
                "topic_id": "topic_dc448af4c39f",
                "currentness": 0.998333332935969,
                "confidence": 0.86,
                "allowed_contribution_types": [
                  "contextual_reaction",
                  "emotional_banter",
                  "concise_observation"
                ],
                "forbidden_claims": [
                  "unsupported strategy",
                  "save instruction",
                  "unrelated mechanic",
                  "stale topic fusion"
                ],
                "expires_at": 1786727222.0,
                "scene_id": "scene_2f0690b8ded85d18ba449ec4e37abb13",
                "state_version": 1,
                "current_state": "active",
                "terminal": false,
                "extracted_subject": "unknown",
                "extracted_object": "",
                "extracted_predicate": "",
                "supported_claims": [
                  "que ha sido demasiada suerte dios"
                ],
                "unsupported_claims": []
              },
              "scene_guard": {
                "scene_id": "scene_2f0690b8ded85d18ba449ec4e37abb13",
                "state_version": 1,
                "current_state": "active",
                "terminal": false
              }
            },
            "readiness_topic": null
          }
        }
      ],
      "all": [
        {
          "id": "intent_23c0d3235c5150eb9db77c46247f8c22",
          "type": "REACTION",
          "source_event_ids": [],
          "anchor_ids": [
            "ambient:rng_dependency:1786727102"
          ],
          "topic": "rng_dependency",
          "subject_ref": "unknown",
          "value": 0.86,
          "urgency": 0.85,
          "freshness": 0.998333332935969,
          "created_at": 1786727102.2,
          "expires_at": 1786727110.2,
          "interruptibility": "yield_before_tts_commit",
          "minimum_turn_gap": 1.2,
          "maximum_turn_delay": 8.0,
          "scene_relevance": {
            "scene_id": "scene_2f0690b8ded85d18ba449ec4e37abb13",
            "state_version": 1,
            "current_state": "active",
            "terminal": false
          },
          "status": "PENDING",
          "suppression_reason": "",
          "reserved_at": 0.0,
          "emitted_at": 0.0,
          "contribution_material": {
            "anchor": {
              "id": "ambient:rng_dependency:1786727102",
              "type": "rng_dependency",
              "quality": 0.86,
              "reason": "recent_ambient_context",
              "evidence": {
                "anchor_id": "ambient:rng_dependency:1786727102",
                "anchor_type": "rng_dependency",
                "raw_owner_fragments": [
                  "que ha sido demasiada suerte dios"
                ],
                "exact_supported_claims": [
                  "que ha sido demasiada suerte dios"
                ],
                "timestamps": [
                  1786727102.0
                ],
                "topic_id": "topic_dc448af4c39f",
                "currentness": 0.998333332935969,
                "confidence": 0.86,
                "allowed_contribution_types": [
                  "contextual_reaction",
                  "emotional_banter",
                  "concise_observation"
                ],
                "forbidden_claims": [
                  "unsupported strategy",
                  "save instruction",
                  "unrelated mechanic",
                  "stale topic fusion"
                ],
                "expires_at": 1786727222.0,
                "scene_id": "scene_2f0690b8ded85d18ba449ec4e37abb13",
                "state_version": 1,
                "current_state": "active",
                "terminal": false,
                "extracted_subject": "unknown",
                "extracted_object": "",
                "extracted_predicate": "",
                "supported_claims": [
                  "que ha sido demasiada suerte dios"
                ],
                "unsupported_claims": []
              },
              "scene_guard": {
                "scene_id": "scene_2f0690b8ded85d18ba449ec4e37abb13",
                "state_version": 1,
                "current_state": "active",
                "terminal": false
              }
            },
            "readiness_topic": null
          }
        }
      ],
      "metrics": {
        "intents_created": 1,
        "created:REACTION": 1,
        "pending_due_owner_voice_active": 1,
        "pending": 1,
        "time_created_to_emit": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "turn_gap_before_emit": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "intent_creation": {
          "count": 1,
          "p50_ms": 0.073,
          "p95_ms": 0.073
        },
        "pending_queue_operation": {
          "count": 1,
          "p50_ms": 0.073,
          "p95_ms": 0.073
        },
        "turn_arbitration": {
          "count": 1,
          "p50_ms": 0.012,
          "p95_ms": 0.012
        },
        "presence_turn_decision": {
          "count": 1,
          "p50_ms": 1.476,
          "p95_ms": 1.476
        }
      }
    },
    "database_watermarks": {
      "counts": {
        "chat_log": 0,
        "memory_facts": 0,
        "memory_chunks": 0,
        "stream_sessions": 1,
        "stream_chat_messages": 0,
        "stream_events": 1,
        "live_session_timeline": 4,
        "promotion_events": 0,
        "viewer_promotion_profiles": 0,
        "schema_migrations": 7,
        "conversations": 0,
        "open_threads": 0,
        "beliefs": 0,
        "belief_evidence": 0,
        "scene_assertions": 0,
        "game_identities": 0,
        "game_runs": 0,
        "game_run_sessions": 0,
        "game_run_events": 0,
        "game_knowledge_facts": 0,
        "game_knowledge_v2_gaps": 0,
        "people": 0,
        "person_identities": 0,
        "person_sessions": 0,
        "social_episodes": 0,
        "shared_culture_items": 0,
        "shared_culture_evidence": 0,
        "consolidation_runs": 0,
        "consolidation_deltas": 0,
        "action_ledger": 0,
        "temporal_maintenance_audit": 0,
        "learning_observations": 0,
        "scene_transitions": 0,
        "schedule_observations": 1,
        "schedule_hypotheses": 7
      },
      "schema_migrations": [
        {
          "component": "architecture_consolidation",
          "version": 1,
          "name": "audit_hygiene_and_cutover_state",
          "checksum": "944b3ad6532151d4204ed59f663668c9b0ac62df54ece7602d26b91acddd5b7d",
          "applied_at": "2026-08-14T02:35:32.541524+00:00"
        },
        {
          "component": "belief_v2",
          "version": 1,
          "name": "beliefs_evidence_and_compatibility_columns",
          "checksum": "f0df6f1288caccaf6bb47670b38f9cca747f14c84916d6c1ca75beb687200507",
          "applied_at": "2026-08-14T02:35:32.399139+00:00"
        },
        {
          "component": "cognitive_replay",
          "version": 1,
          "name": "replay_metadata",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "applied_at": "2026-08-14T02:35:32.177746+00:00"
        },
        {
          "component": "conversation_continuity",
          "version": 1,
          "name": "conversation_and_open_threads",
          "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
          "applied_at": "2026-08-14T02:35:32.312278+00:00"
        },
        {
          "component": "game_context_v2",
          "version": 1,
          "name": "durable_runs_knowledge_and_gaps",
          "checksum": "08e342acaae00d5d24c1a6dbccad5aee41f753ed90d9ec26282a5fdf042d0a75",
          "applied_at": "2026-08-14T02:35:32.450003+00:00"
        },
        {
          "component": "learning_v2",
          "version": 1,
          "name": "consolidation_temporal_action_and_scene",
          "checksum": "6a86e2d1c7c03167f3b20c328bc97b73fb92e03903f64e64d326fb08f9e3b942",
          "applied_at": "2026-08-14T02:35:32.526410+00:00"
        },
        {
          "component": "social_world_v2",
          "version": 1,
          "name": "people_episodes_and_shared_culture",
          "checksum": "b02adc2cd7f298f1af228dc52c4ba44ae15999fb4c815ea004c40f68b78cbfa5",
          "applied_at": "2026-08-14T02:35:32.488704+00:00"
        }
      ],
      "final_response_digest": "",
      "final_response_present": false
    }
  },
  "leo-ends": {
    "runtime": {
      "mode": "active",
      "hebe_sleeping": false,
      "is_running": false,
      "last_input_source": null,
      "last_intent": null,
      "last_firewall": {
        "source": "ambient_stt",
        "authority": "ambient",
        "decision": "allow_context_only",
        "reason": "ambient_context_only"
      },
      "last_policy": {
        "source": null,
        "authority": null,
        "decision": null,
        "reason": null
      }
    },
    "stream_session": {
      "enabled": true,
      "is_live": true,
      "live_status_known": true,
      "active_stream_session_id": 1,
      "last_transition": "online",
      "title": "Boss attempts",
      "game": "Test RPG",
      "category": "Test RPG"
    },
    "current_scene": {
      "scene_id": "scene_2f0690b8ded85d18ba449ec4e37abb13",
      "topic_id": "topic_dc448af4c39f",
      "entity": "unknown",
      "current_state": "active",
      "state_version": 1,
      "supporting_event_ids": [
        "ambient:rng_dependency:1786727102"
      ],
      "superseded_event_ids": [],
      "terminal": false,
      "updated_at": 1786727102.0
    },
    "pending": {
      "clarification": null,
      "conversation_turn": null
    },
    "conversation": {
      "active": {},
      "latest": {},
      "all": [],
      "last_resolution": {
        "consumed": false,
        "decision": "no_conversation",
        "reason": "no_compatible_active_conversation",
        "conversation_id": "",
        "reply_act": "UNKNOWN",
        "payload": {},
        "conversation": null,
        "latency_ms": 3.095800057053566
      },
      "legacy_pending_projection": {},
      "continuity_shadow_diff": {
        "legacy_result": false,
        "v2_result": false,
        "match": true,
        "difference_reason": ""
      },
      "shadow_metrics": {
        "total": 1,
        "matches": 1,
        "differences": 0,
        "match_rate": 1.0,
        "difference_reasons": {}
      },
      "performance": {
        "count": 1,
        "p50_ms": 3.0958,
        "p95_ms": 3.0958
      }
    },
    "open_threads": [],
    "memory": {
      "facts_count": 0,
      "chunks_count": 0
    },
    "beliefs": {
      "active": [],
      "historical": [],
      "superseded": [],
      "suspected": [],
      "all": [],
      "last_transition": {}
    },
    "belief_evidence": [],
    "retrieval": {
      "last_request": {},
      "performance": {
        "count": 0,
        "p50_ms": 0.0,
        "p95_ms": 0.0
      },
      "write_performance": {
        "count": 0,
        "p50_ms": 0.0,
        "p95_ms": 0.0
      },
      "repository_performance": {
        "belief_lookup": {
          "count": 6,
          "p50_ms": 0.88655,
          "p95_ms": 1.8886
        },
        "evidence_lookup": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "sqlite_write": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        }
      }
    },
    "memory_compatibility": {
      "legacy_to_v2": [],
      "v2_to_legacy": [],
      "shadow_diffs": [],
      "backfill": {
        "safe": 0,
        "compatibility_only": 0,
        "ambiguous": 0,
        "invalid_stale": 0
      }
    },
    "game_state": {
      "game": "Test RPG",
      "platform_version": "",
      "playthrough_type": "casual",
      "spoiler_policy": "no_spoilers",
      "current_location": null,
      "current_character": "",
      "party_members": [],
      "last_confirmed_progress": "",
      "current_objective": null,
      "challenge": "",
      "known_constraints": [],
      "last_updated": 1786727100.0,
      "provenance": "stream_context_sync",
      "confidence": 0.75,
      "current_game": "Test RPG",
      "recent_run_context_facts": [
        {
          "kind": "rng_dependency",
          "text": "Leo framed the current situation as dependent on RNG or luck.",
          "category": "rng_dependency",
          "summary": "Leo framed the current situation as dependent on RNG or luck.",
          "id": "ambient:rng_dependency:1786727102",
          "fact_id": "ambient:rng_dependency:1786727102",
          "raw_text": "<redacted:2320c1febf71>",
          "conservative_normalized_text": "que ha sido demasiada suerte dios",
          "utterance_role": "owner_question_to_stream",
          "timestamp": 1786727102.0,
          "topic_id": "topic_dc448af4c39f",
          "heuristic_category": "rng_dependency",
          "extracted_subject": "unknown",
          "subject": "unknown",
          "extracted_object": "",
          "object": "",
          "extracted_predicate": "",
          "predicate": "",
          "confidence": 0.86,
          "referent_confidence": 0.86,
          "inference_level": "heuristic",
          "supported_claims": [
            "que ha sido demasiada suerte dios"
          ],
          "directly_supported_claims": [
            "que ha sido demasiada suerte dios"
          ],
          "inferred_claims": [],
          "unsupported_claims": [],
          "evidence_span": "que ha sido demasiada suerte dios",
          "evidence_tokens": "<redacted:323fcf3ad517>",
          "semantic_rule": "ambient_category:rng_dependency",
          "model_reason": "Leo framed the current situation as dependent on RNG or luck.",
          "expires_at": 1786727222.0,
          "scene_id": "scene_2f0690b8ded85d18ba449ec4e37abb13",
          "game": null,
          "source": "stt_voice",
          "raw_evidence": "que ha sido demasiada suerte dios",
          "normalized_text": "que ha sido demasiada suerte dios",
          "normalized_evidence": "que ha sido demasiada suerte dios",
          "language": "es",
          "ttl_sec": 120,
          "data": {
            "category": "rng_dependency",
            "raw_text": "<redacted:2320c1febf71>",
            "normalized_text": "que ha sido demasiada suerte dios",
            "mood": "rng tension",
            "extracted_subject": "unknown",
            "extracted_object": "",
            "extracted_predicate": "",
            "inference_level": "heuristic",
            "supported_claims": [
              "que ha sido demasiada suerte dios"
            ],
            "inferred_claims": [],
            "unsupported_claims": [],
            "evidence_span": "que ha sido demasiada suerte dios",
            "evidence_tokens": "<redacted:323fcf3ad517>",
            "semantic_rule": "ambient_category:rng_dependency",
            "model_reason": "Leo framed the current situation as dependent on RNG or luck.",
            "scene_id": "scene_2f0690b8ded85d18ba449ec4e37abb13"
          },
          "age_seconds": 0.0,
          "superseded": false,
          "state_version": 1,
          "current_state": "active",
          "terminal": false,
          "currentness_score": 1.0
        }
      ],
      "identity": {},
      "context": {},
      "active_run": {},
      "runs": [],
      "session_links": [],
      "run_events": [],
      "run_beliefs": {
        "current": [],
        "inferred": [],
        "superseded": []
      },
      "knowledge": {
        "selected": [],
        "rejected": [],
        "spoiler_blocked": [],
        "all": []
      },
      "gaps": [],
      "research": {
        "research_calls": 0,
        "cache_hits": 0,
        "failures": [],
        "context_performance": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "fixture_calls": [],
        "status": ""
      },
      "compatibility": {
        "legacy_progress": [],
        "dossier": [],
        "legacy_run": [],
        "shadow_diffs": [],
        "backfill": {
          "validated": 0,
          "compatibility_only": 0,
          "ambiguous": 0,
          "stale": 0
        }
      },
      "provenance_manifest": [],
      "advice_allowed": null,
      "reaction_allowed": null,
      "performance": {
        "run": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "run_fact": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "knowledge": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "research_gap": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "db_write": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        }
      },
      "context_performance": {
        "count": 0,
        "p50_ms": 0.0,
        "p95_ms": 0.0
      },
      "manifest_size_bytes": 0,
      "last_run_resolution": {},
      "run_resolution_performance": {
        "count": 0,
        "p50_ms": 0.0,
        "p95_ms": 0.0
      }
    },
    "social_state": {
      "recent_active_users": [],
      "recent_chat_count": 0,
      "last_raid": {},
      "last_cheer": {},
      "people": [],
      "identities": [],
      "recent_episodes": [],
      "active_hypotheses": [],
      "historical_hypotheses": [],
      "open_threads": [],
      "relationships": [],
      "shared_culture": {
        "all": [],
        "candidates": [],
        "active": [],
        "weakening": [],
        "retired": [],
        "reactions": [],
        "selection": {
          "selected": [],
          "rejected": []
        }
      },
      "retrieval": {},
      "opportunities": [],
      "resolution": {},
      "rejected_writes": [],
      "compatibility": {
        "chatter_presence": [],
        "chatter_profiles": [],
        "chatter_facts": [],
        "stream_chatter_summaries": [],
        "viewer_profiles": [],
        "social_events": [],
        "promotion_profiles": [],
        "backfill": {
          "explicit_observation": 0,
          "safe_episode": 0,
          "inferred_compatibility_only": 0,
          "ambiguous": 0,
          "sensitive": 0,
          "stale": 0
        },
        "shadow_diffs": []
      },
      "performance": {
        "identity": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "episode_write": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "thread_lookup": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "culture_select": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "context": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "db_write": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        }
      },
      "belief_lookup_performance": {
        "belief_lookup": {
          "count": 6,
          "p50_ms": 0.88655,
          "p95_ms": 1.8886
        },
        "evidence_lookup": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "sqlite_write": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        }
      }
    },
    "learning": {
      "consolidation_runs": [],
      "deltas": [],
      "rejected_deltas": [],
      "watermarks": [],
      "last_result": {},
      "stable_core_version": "a1a58e51882b0c88",
      "performance": {
        "repository": {
          "lookup": {
            "count": 30,
            "p50_ms": 0.8782,
            "p95_ms": 1.4252
          },
          "write": {
            "count": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0
          },
          "context": {
            "count": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0
          }
        },
        "consolidation": {
          "consolidation_duration": {
            "count": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0
          },
          "candidate_validation": {
            "count": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0
          }
        },
        "temporal": {
          "temporal_maintenance": {
            "count": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0
          }
        },
        "action_history": {
          "action_ledger_lookup": {
            "count": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0
          }
        },
        "owner_preferences": {
          "owner_preference_lookup": {
            "count": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0
          }
        },
        "hebe_self": {
          "hebe_self_lookup": {
            "count": 1,
            "p50_ms": 1.8985,
            "p95_ms": 1.8985
          }
        },
        "context": {
          "continuity_context_build": {
            "count": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0
          }
        }
      }
    },
    "self_model": {
      "stable_core_version": "a1a58e51882b0c88",
      "evolving_preferences": [],
      "opinions": [],
      "superseded_opinions": []
    },
    "owner_preferences": [],
    "leo_language": {
      "beliefs": [],
      "interpretation_aliases": {}
    },
    "temporal": {
      "expired": [],
      "archived": [],
      "weakened": [],
      "maintenance_actions": [],
      "last_actions": []
    },
    "schedule": {
      "observations": [
        {
          "id": 1,
          "stream_session_id": "1",
          "weekday": "friday",
          "time_window": "night",
          "canonical_content": "Test RPG",
          "content_key": "test rpg",
          "stream_format": "game_playthrough",
          "source": "observed",
          "observed_at": "2026-08-14T04:35:32.628934+02:00"
        }
      ],
      "hypotheses": [
        {
          "id": 1,
          "weekday": "monday",
          "time_window": "any",
          "canonical_content": "FINAL FANTASY IX",
          "content_key": "final fantasy ix",
          "stream_format": "challenge_run",
          "source": "owner_declared",
          "confidence": 0.9,
          "evidence_count": 1,
          "consecutive_matches": 1,
          "consecutive_misses": 0,
          "first_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "last_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "status": "active",
          "superseded_by": null
        },
        {
          "id": 2,
          "weekday": "tuesday",
          "time_window": "any",
          "canonical_content": "Persona 5 Royal",
          "content_key": "persona 5 royal",
          "stream_format": "game_playthrough",
          "source": "owner_declared",
          "confidence": 0.9,
          "evidence_count": 1,
          "consecutive_matches": 1,
          "consecutive_misses": 0,
          "first_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "last_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "status": "active",
          "superseded_by": null
        },
        {
          "id": 3,
          "weekday": "wednesday",
          "time_window": "any",
          "canonical_content": "Persona 5 Royal",
          "content_key": "persona 5 royal",
          "stream_format": "game_playthrough",
          "source": "owner_declared",
          "confidence": 0.9,
          "evidence_count": 1,
          "consecutive_matches": 1,
          "consecutive_misses": 0,
          "first_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "last_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "status": "active",
          "superseded_by": null
        },
        {
          "id": 4,
          "weekday": "thursday",
          "time_window": "any",
          "canonical_content": "Baldur's Gate 3",
          "content_key": "baldur s gate 3",
          "stream_format": "game_playthrough",
          "source": "owner_declared",
          "confidence": 0.9,
          "evidence_count": 1,
          "consecutive_matches": 1,
          "consecutive_misses": 0,
          "first_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "last_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "status": "active",
          "superseded_by": null
        },
        {
          "id": 6,
          "weekday": "saturday",
          "time_window": "any",
          "canonical_content": "Retro Weekend",
          "content_key": "retro weekend",
          "stream_format": "retro",
          "source": "owner_declared",
          "confidence": 0.9,
          "evidence_count": 1,
          "consecutive_matches": 1,
          "consecutive_misses": 0,
          "first_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "last_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "status": "active",
          "superseded_by": null
        },
        {
          "id": 7,
          "weekday": "sunday",
          "time_window": "any",
          "canonical_content": "Retro Weekend",
          "content_key": "retro weekend",
          "stream_format": "retro",
          "source": "owner_declared",
          "confidence": 0.9,
          "evidence_count": 1,
          "consecutive_matches": 1,
          "consecutive_misses": 0,
          "first_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "last_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "status": "active",
          "superseded_by": null
        },
        {
          "id": 5,
          "weekday": "friday",
          "time_window": "any",
          "canonical_content": "Baldur's Gate 3",
          "content_key": "baldur s gate 3",
          "stream_format": "game_playthrough",
          "source": "owner_declared",
          "confidence": 0.76,
          "evidence_count": 1,
          "consecutive_matches": 0,
          "consecutive_misses": 1,
          "first_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "last_observed_at": "2026-08-14T04:35:32.628934+02:00",
          "status": "weakening",
          "superseded_by": null
        }
      ],
      "observed_current_state": {
        "game": "Test RPG",
        "title": "Boss attempts"
      },
      "precedence": "observed_twitch_metadata"
    },
    "action_ledger": {
      "entries": [],
      "last_claim_validation": {}
    },
    "scene_transitions": {
      "all": [],
      "last": {}
    },
    "continuity_context": {},
    "promotion_profiles": [],
    "actions": {
      "attempts": [],
      "speech_requests": [],
      "model_calls": [],
      "research_calls": []
    },
    "receipts": [],
    "emitted_outputs": [],
    "final_emission_results": [],
    "speech_intents": {
      "active": [
        {
          "id": "intent_23c0d3235c5150eb9db77c46247f8c22",
          "type": "REACTION",
          "source_event_ids": [],
          "anchor_ids": [
            "ambient:rng_dependency:1786727102"
          ],
          "topic": "rng_dependency",
          "subject_ref": "unknown",
          "value": 0.86,
          "urgency": 0.85,
          "freshness": 0.998333332935969,
          "created_at": 1786727102.2,
          "expires_at": 1786727110.2,
          "interruptibility": "yield_before_tts_commit",
          "minimum_turn_gap": 1.2,
          "maximum_turn_delay": 8.0,
          "scene_relevance": {
            "scene_id": "scene_2f0690b8ded85d18ba449ec4e37abb13",
            "state_version": 1,
            "current_state": "active",
            "terminal": false
          },
          "status": "PENDING",
          "suppression_reason": "",
          "reserved_at": 0.0,
          "emitted_at": 0.0,
          "contribution_material": {
            "anchor": {
              "id": "ambient:rng_dependency:1786727102",
              "type": "rng_dependency",
              "quality": 0.86,
              "reason": "recent_ambient_context",
              "evidence": {
                "anchor_id": "ambient:rng_dependency:1786727102",
                "anchor_type": "rng_dependency",
                "raw_owner_fragments": [
                  "que ha sido demasiada suerte dios"
                ],
                "exact_supported_claims": [
                  "que ha sido demasiada suerte dios"
                ],
                "timestamps": [
                  1786727102.0
                ],
                "topic_id": "topic_dc448af4c39f",
                "currentness": 0.998333332935969,
                "confidence": 0.86,
                "allowed_contribution_types": [
                  "contextual_reaction",
                  "emotional_banter",
                  "concise_observation"
                ],
                "forbidden_claims": [
                  "unsupported strategy",
                  "save instruction",
                  "unrelated mechanic",
                  "stale topic fusion"
                ],
                "expires_at": 1786727222.0,
                "scene_id": "scene_2f0690b8ded85d18ba449ec4e37abb13",
                "state_version": 1,
                "current_state": "active",
                "terminal": false,
                "extracted_subject": "unknown",
                "extracted_object": "",
                "extracted_predicate": "",
                "supported_claims": [
                  "que ha sido demasiada suerte dios"
                ],
                "unsupported_claims": []
              },
              "scene_guard": {
                "scene_id": "scene_2f0690b8ded85d18ba449ec4e37abb13",
                "state_version": 1,
                "current_state": "active",
                "terminal": false
              }
            },
            "readiness_topic": null
          }
        }
      ],
      "all": [
        {
          "id": "intent_23c0d3235c5150eb9db77c46247f8c22",
          "type": "REACTION",
          "source_event_ids": [],
          "anchor_ids": [
            "ambient:rng_dependency:1786727102"
          ],
          "topic": "rng_dependency",
          "subject_ref": "unknown",
          "value": 0.86,
          "urgency": 0.85,
          "freshness": 0.998333332935969,
          "created_at": 1786727102.2,
          "expires_at": 1786727110.2,
          "interruptibility": "yield_before_tts_commit",
          "minimum_turn_gap": 1.2,
          "maximum_turn_delay": 8.0,
          "scene_relevance": {
            "scene_id": "scene_2f0690b8ded85d18ba449ec4e37abb13",
            "state_version": 1,
            "current_state": "active",
            "terminal": false
          },
          "status": "PENDING",
          "suppression_reason": "",
          "reserved_at": 0.0,
          "emitted_at": 0.0,
          "contribution_material": {
            "anchor": {
              "id": "ambient:rng_dependency:1786727102",
              "type": "rng_dependency",
              "quality": 0.86,
              "reason": "recent_ambient_context",
              "evidence": {
                "anchor_id": "ambient:rng_dependency:1786727102",
                "anchor_type": "rng_dependency",
                "raw_owner_fragments": [
                  "que ha sido demasiada suerte dios"
                ],
                "exact_supported_claims": [
                  "que ha sido demasiada suerte dios"
                ],
                "timestamps": [
                  1786727102.0
                ],
                "topic_id": "topic_dc448af4c39f",
                "currentness": 0.998333332935969,
                "confidence": 0.86,
                "allowed_contribution_types": [
                  "contextual_reaction",
                  "emotional_banter",
                  "concise_observation"
                ],
                "forbidden_claims": [
                  "unsupported strategy",
                  "save instruction",
                  "unrelated mechanic",
                  "stale topic fusion"
                ],
                "expires_at": 1786727222.0,
                "scene_id": "scene_2f0690b8ded85d18ba449ec4e37abb13",
                "state_version": 1,
                "current_state": "active",
                "terminal": false,
                "extracted_subject": "unknown",
                "extracted_object": "",
                "extracted_predicate": "",
                "supported_claims": [
                  "que ha sido demasiada suerte dios"
                ],
                "unsupported_claims": []
              },
              "scene_guard": {
                "scene_id": "scene_2f0690b8ded85d18ba449ec4e37abb13",
                "state_version": 1,
                "current_state": "active",
                "terminal": false
              }
            },
            "readiness_topic": null
          }
        }
      ],
      "metrics": {
        "intents_created": 1,
        "created:REACTION": 1,
        "pending_due_owner_voice_active": 1,
        "pending": 1,
        "time_created_to_emit": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "turn_gap_before_emit": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "intent_creation": {
          "count": 1,
          "p50_ms": 0.073,
          "p95_ms": 0.073
        },
        "pending_queue_operation": {
          "count": 1,
          "p50_ms": 0.073,
          "p95_ms": 0.073
        },
        "turn_arbitration": {
          "count": 1,
          "p50_ms": 0.012,
          "p95_ms": 0.012
        },
        "presence_turn_decision": {
          "count": 1,
          "p50_ms": 1.476,
          "p95_ms": 1.476
        }
      }
    },
    "database_watermarks": {
      "counts": {
        "chat_log": 0,
        "memory_facts": 0,
        "memory_chunks": 0,
        "stream_sessions": 1,
        "stream_chat_messages": 0,
        "stream_events": 1,
        "live_session_timeline": 4,
        "promotion_events": 0,
        "viewer_promotion_profiles": 0,
        "schema_migrations": 7,
        "conversations": 0,
        "open_threads": 0,
        "beliefs": 0,
        "belief_evidence": 0,
        "scene_assertions": 0,
        "game_identities": 0,
        "game_runs": 0,
        "game_run_sessions": 0,
        "game_run_events": 0,
        "game_knowledge_facts": 0,
        "game_knowledge_v2_gaps": 0,
        "people": 0,
        "person_identities": 0,
        "person_sessions": 0,
        "social_episodes": 0,
        "shared_culture_items": 0,
        "shared_culture_evidence": 0,
        "consolidation_runs": 0,
        "consolidation_deltas": 0,
        "action_ledger": 0,
        "temporal_maintenance_audit": 0,
        "learning_observations": 0,
        "scene_transitions": 0,
        "schedule_observations": 1,
        "schedule_hypotheses": 7
      },
      "schema_migrations": [
        {
          "component": "architecture_consolidation",
          "version": 1,
          "name": "audit_hygiene_and_cutover_state",
          "checksum": "944b3ad6532151d4204ed59f663668c9b0ac62df54ece7602d26b91acddd5b7d",
          "applied_at": "2026-08-14T02:35:32.541524+00:00"
        },
        {
          "component": "belief_v2",
          "version": 1,
          "name": "beliefs_evidence_and_compatibility_columns",
          "checksum": "f0df6f1288caccaf6bb47670b38f9cca747f14c84916d6c1ca75beb687200507",
          "applied_at": "2026-08-14T02:35:32.399139+00:00"
        },
        {
          "component": "cognitive_replay",
          "version": 1,
          "name": "replay_metadata",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "applied_at": "2026-08-14T02:35:32.177746+00:00"
        },
        {
          "component": "conversation_continuity",
          "version": 1,
          "name": "conversation_and_open_threads",
          "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
          "applied_at": "2026-08-14T02:35:32.312278+00:00"
        },
        {
          "component": "game_context_v2",
          "version": 1,
          "name": "durable_runs_knowledge_and_gaps",
          "checksum": "08e342acaae00d5d24c1a6dbccad5aee41f753ed90d9ec26282a5fdf042d0a75",
          "applied_at": "2026-08-14T02:35:32.450003+00:00"
        },
        {
          "component": "learning_v2",
          "version": 1,
          "name": "consolidation_temporal_action_and_scene",
          "checksum": "6a86e2d1c7c03167f3b20c328bc97b73fb92e03903f64e64d326fb08f9e3b942",
          "applied_at": "2026-08-14T02:35:32.526410+00:00"
        },
        {
          "component": "social_world_v2",
          "version": 1,
          "name": "people_episodes_and_shared_culture",
          "checksum": "b02adc2cd7f298f1af228dc52c4ba44ae15999fb4c815ea004c40f68b78cbfa5",
          "applied_at": "2026-08-14T02:35:32.488704+00:00"
        }
      ],
      "final_response_digest": "",
      "final_response_present": false
    }
  },
  "gap-too-short": {
    "runtime": {
      "mode": "active",
      "hebe_sleeping": false,
      "is_running": false,
      "last_input_source": null,
      "last_intent": null,
      "last_firewall": {
        "source": "ambient_stt",
        "authority": "ambient",
        "decision": "allow_context_only",
        "reason": "ambient_context_only"
      },
      "last_policy": {
        "source": null,
        "authority": null,
        "decision": null,
        "reason": null
      }
    },
    "stream_session": {
      "enabled": true,
      "is_live": true,
      "live_status_known": true,
      "active_stream_session_id": 1,
      "last_transition": "online",
      "title": "Boss attempts",
      "game": "Test RPG",
      "category": "Test RPG"
    },
    "current_scene": {
      "scene_id": "scene_2f0690b8ded85d18ba449ec4e37abb13",
      "topic_id": "topic_dc448af4c39f",
      "entity": "unknown",
      "current_state": "active",
      "state_version": 1,
      "supporting_event_ids": [
        "ambient:rng_dependency:1786727102"
      ],
      "superseded_event_ids": [],
      "terminal": false,
      "updated_at": 1786727102.0
    },
    "pending": {
      "clarification": null,
      "conversation_turn": null
    },
    "conversation": {
      "active": {},
      "latest": {},
      "all": [],
      "last_resolution": {
        "consumed": false,
        "decision": "no_conversation",
        "reason": "no_compatible_active_conversation",
        "conversation_id": "",
        "reply_act": "UNKNOWN",
        "payload": {},
        "conversation": null,
        "latency_ms": 3.095800057053566
      },
      "legacy_pending_projection": {},
      "continuity_shadow_diff": {
        "legacy_result": false,
        "v2_result": false,
        "match": true,
        "difference_reason": ""
      },
      "shadow_metrics": {
        "total": 1,
        "matches": 1,
        "differences": 0,
        "match_rate": 1.0,
        "difference_reasons": {}
      },
      "performance": {
        "count": 1,
        "p50_ms": 3.0958,
        "p95_ms": 3.0958
      }
    },
    "open_threads": [],
    "memory": {
      "facts_count": 0,
      "chunks_count": 0
    },
    "beliefs": {
      "active": [],
      "historical": [],
      "superseded": [],
      "suspected": [],
      "all": [],
      "last_transition": {}
    },
    "belief_evidence": [],
    "retrieval": {
      "last_request": {},
      "performance": {
        "count": 0,
        "p50_ms": 0.0,
        "p95_ms": 0.0
      },
      "write_performance": {
        "count": 0,
        "p50_ms": 0.0,
        "p95_ms": 0.0
      },
      "repository_performance": {
        "belief_lookup": {
          "count": 7,
          "p50_ms": 0.8925,
          "p95_ms": 1.8886
        },
        "evidence_lookup": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "sqlite_write": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        }
      }
    },
    "memory_compatibility": {
      "legacy_to_v2": [],
      "v2_to_legacy": [],
      "shadow_diffs": [],
      "backfill": {
        "safe": 0,
        "compatibility_only": 0,
        "ambiguous": 0,
        "invalid_stale": 0
      }
    },
    "game_state": {
      "game": "Test RPG",
      "platform_version": "",
      "playthrough_type": "casual",
      "spoiler_policy": "no_spoilers",
      "current_location": null,
      "current_character": "",
      "party_members": [],
      "last_confirmed_progress": "",
      "current_objective": null,
      "challenge": "",
      "known_constraints": [],
      "last_updated": 1786727100.0,
      "provenance": "stream_context_sync",
      "confidence": 0.75,
      "current_game": "Test RPG",
      "recent_run_context_facts": [
        {
          "kind": "rng_dependency",
          "text": "Leo framed the current situation as dependent on RNG or luck.",
          "category": "rng_dependency",
          "summary": "Leo framed the current situation as dependent on RNG or luck.",
          "id": "ambient:rng_dependency:1786727102",
          "fact_id": "ambient:rng_dependency:1786727102",
          "raw_text": "<redacted:2320c1febf71>",
          "conservative_normalized_text": "que ha sido demasiada suerte dios",
          "utterance_role": "owner_question_to_stream",
          "timestamp": 1786727102.0,
          "topic_id": "topic_dc448af4c39f",
          "heuristic_category": "rng_dependency",
          "extracted_subject": "unknown",
          "subject": "unknown",
          "extracted_object": "",
          "object": "",
          "extracted_predicate": "",
          "predicate": "",
          "confidence": 0.86,
          "referent_confidence": 0.86,
          "inference_level": "heuristic",
          "supported_claims": [
            "que ha sido demasiada suerte dios"
          ],
          "directly_supported_claims": [
            "que ha sido demasiada suerte dios"
          ],
          "inferred_claims": [],
          "unsupported_claims": [],
          "evidence_span": "que ha sido demasiada suerte dios",
          "evidence_tokens": "<redacted:323fcf3ad517>",
          "semantic_rule": "ambient_category:rng_dependency",
          "model_reason": "Leo framed the current situation as dependent on RNG or luck.",
          "expires_at": 1786727222.0,
          "scene_id": "scene_2f0690b8ded85d18ba449ec4e37abb13",
          "game": null,
          "source": "stt_voice",
          "raw_evidence": "que ha sido demasiada suerte dios",
          "normalized_text": "que ha sido demasiada suerte dios",
          "normalized_evidence": "que ha sido demasiada suerte dios",
          "language": "es",
          "ttl_sec": 120,
          "data": {
            "category": "rng_dependency",
            "raw_text": "<redacted:2320c1febf71>",
            "normalized_text": "que ha sido demasiada suerte dios",
            "mood": "rng tension",
            "extracted_subject": "unknown",
            "extracted_object": "",
            "extracted_predicate": "",
            "inference_level": "heuristic",
            "supported_claims": [
              "que ha sido demasiada suerte dios"
            ],
            "inferred_claims": [],
            "unsupported_claims": [],
            "evidence_span": "que ha sido demasiada suerte dios",
            "evidence_tokens": "<redacted:323fcf3ad517>",
            "semantic_rule": "ambient_category:rng_dependency",
            "model_reason": "Leo framed the current situation as dependent on RNG or luck.",
            "scene_id": "scene_2f0690b8ded85d18ba449ec4e37abb13"
          },
          "age_seconds": 0.0,
          "superseded": false,
          "state_version": 1,
          "current_state": "active",
          "terminal": false,
          "currentness_score": 1.0
        }
      ],
      "identity": {},
      "context": {},
      "active_run": {},
      "runs": [],
      "session_links": [],
      "run_events": [],
      "run_beliefs": {
        "current": [],
        "inferred": [],
        "superseded": []
      },
      "knowledge": {
        "selected": [],
        "rejected": [],
        "spoiler_blocked": [],
        "all": []
      },
      "gaps": [],
      "research": {
        "research_calls": 0,
        "cache_hits": 0,
        "failures": [],
        "context_performance": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "fixture_calls": [],
        "status": ""
      },
      "compatibility": {
        "legacy_progress": [],
        "dossier": [],
        "legacy_run": [],
        "shadow_diffs": [],
        "backfill": {
          "validated": 0,
          "compatibility_only": 0,
          "ambiguous": 0,
          "stale": 0
        }
      },
      "provenance_manifest": [],
      "advice_allowed": null,
      "reaction_allowed": null,
      "performance": {
        "run": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "run_fact": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "knowledge": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "research_gap": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "db_write": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        }
      },
      "context_performance": {
        "count": 0,
        "p50_ms": 0.0,
        "p95_ms": 0.0
      },
      "manifest_size_bytes": 0,
      "last_run_resolution": {},
      "run_resolution_performance": {
        "count": 0,
        "p50_ms": 0.0,
        "p95_ms": 0.0
      }
    },
    "social_state": {
      "recent_active_users": [],
      "recent_chat_count": 0,
      "last_raid": {},
      "last_cheer": {},
      "people": [],
      "identities": [],
      "recent_episodes": [],
      "active_hypotheses": [],
      "historical_hypotheses": [],
      "open_threads": [],
      "relationships": [],
      "shared_culture": {
        "all": [],
        "candidates": [],
        "active": [],
        "weakening": [],
        "retired": [],
        "reactions": [],
        "selection": {
          "selected": [],
          "rejected": []
        }
      },
      "retrieval": {},
      "opportunities": [],
      "resolution": {},
      "rejected_writes": [],
      "compatibility": {
        "chatter_presence": [],
        "chatter_profiles": [],
        "chatter_facts": [],
        "stream_chatter_summaries": [],
        "viewer_profiles": [],
        "social_events": [],
        "promotion_profiles": [],
        "backfill": {
          "explicit_observation": 0,
          "safe_episode": 0,
          "inferred_compatibility_only": 0,
          "ambiguous": 0,
          "sensitive": 0,
          "stale": 0
        },
        "shadow_diffs": []
      },
      "performance": {
        "identity": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "episode_write": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "thread_lookup": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "culture_select": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "context": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "db_write": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        }
      },
      "belief_lookup_performance": {
        "belief_lookup": {
          "count": 7,
          "p50_ms": 0.8925,
          "p95_ms": 1.8886
        },
        "evidence_lookup": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "sqlite_write": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        }
      }
    },
    "learning": {
      "consolidation_runs": [],
      "deltas": [],
      "rejected_deltas": [],
      "watermarks": [],
      "last_result": {},
      "stable_core_version": "a1a58e51882b0c88",
      "performance": {
        "repository": {
          "lookup": {
            "count": 35,
            "p50_ms": 0.8885,
            "p95_ms": 1.4359
          },
          "write": {
            "count": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0
          },
          "context": {
            "count": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0
          }
        },
        "consolidation": {
          "consolidation_duration": {
            "count": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0
          },
          "candidate_validation": {
            "count": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0
          }
        },
        "temporal": {
          "temporal_maintenance": {
            "count": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0
          }
        },
        "action_history": {
          "action_ledger_lookup": {
            "count": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0
          }
        },
        "owner_preferences": {
          "owner_preference_lookup": {
            "count": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0
          }
        },
        "hebe_self": {
          "hebe_self_lookup": {
            "count": 1,
            "p50_ms": 1.8985,
            "p95_ms": 1.8985
          }
        },
        "context": {
          "continuity_context_build": {
            "count": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0
          }
        }
      }
    },
    "self_model": {
      "stable_core_version": "a1a58e51882b0c88",
      "evolving_preferences": [],
      "opinions": [],
      "superseded_opinions": []
    },
    "owner_preferences": [],
    "leo_language": {
      "beliefs": [],
      "interpretation_aliases": {}
    },
    "temporal": {
      "expired": [],
      "archived": [],
      "weakened": [],
      "maintenance_actions": [],
      "last_actions": []
    },
    "schedule": {
      "observations": [
        {
          "id": 1,
          "stream_session_id": "1",
          "weekday": "friday",
          "time_window": "night",
          "canonical_content": "Test RPG",
          "content_key": "test rpg",
          "stream_format": "game_playthrough",
          "source": "observed",
          "observed_at": "2026-08-14T04:35:32.628934+02:00"
        }
      ],
      "hypotheses": [
        {
          "id": 1,
          "weekday": "monday",
          "time_window": "any",
          "canonical_content": "FINAL FANTASY IX",
          "content_key": "final fantasy ix",
          "stream_format": "challenge_run",
          "source": "owner_declared",
          "confidence": 0.9,
          "evidence_count": 1,
          "consecutive_matches": 1,
          "consecutive_misses": 0,
          "first_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "last_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "status": "active",
          "superseded_by": null
        },
        {
          "id": 2,
          "weekday": "tuesday",
          "time_window": "any",
          "canonical_content": "Persona 5 Royal",
          "content_key": "persona 5 royal",
          "stream_format": "game_playthrough",
          "source": "owner_declared",
          "confidence": 0.9,
          "evidence_count": 1,
          "consecutive_matches": 1,
          "consecutive_misses": 0,
          "first_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "last_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "status": "active",
          "superseded_by": null
        },
        {
          "id": 3,
          "weekday": "wednesday",
          "time_window": "any",
          "canonical_content": "Persona 5 Royal",
          "content_key": "persona 5 royal",
          "stream_format": "game_playthrough",
          "source": "owner_declared",
          "confidence": 0.9,
          "evidence_count": 1,
          "consecutive_matches": 1,
          "consecutive_misses": 0,
          "first_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "last_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "status": "active",
          "superseded_by": null
        },
        {
          "id": 4,
          "weekday": "thursday",
          "time_window": "any",
          "canonical_content": "Baldur's Gate 3",
          "content_key": "baldur s gate 3",
          "stream_format": "game_playthrough",
          "source": "owner_declared",
          "confidence": 0.9,
          "evidence_count": 1,
          "consecutive_matches": 1,
          "consecutive_misses": 0,
          "first_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "last_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "status": "active",
          "superseded_by": null
        },
        {
          "id": 6,
          "weekday": "saturday",
          "time_window": "any",
          "canonical_content": "Retro Weekend",
          "content_key": "retro weekend",
          "stream_format": "retro",
          "source": "owner_declared",
          "confidence": 0.9,
          "evidence_count": 1,
          "consecutive_matches": 1,
          "consecutive_misses": 0,
          "first_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "last_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "status": "active",
          "superseded_by": null
        },
        {
          "id": 7,
          "weekday": "sunday",
          "time_window": "any",
          "canonical_content": "Retro Weekend",
          "content_key": "retro weekend",
          "stream_format": "retro",
          "source": "owner_declared",
          "confidence": 0.9,
          "evidence_count": 1,
          "consecutive_matches": 1,
          "consecutive_misses": 0,
          "first_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "last_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "status": "active",
          "superseded_by": null
        },
        {
          "id": 5,
          "weekday": "friday",
          "time_window": "any",
          "canonical_content": "Baldur's Gate 3",
          "content_key": "baldur s gate 3",
          "stream_format": "game_playthrough",
          "source": "owner_declared",
          "confidence": 0.76,
          "evidence_count": 1,
          "consecutive_matches": 0,
          "consecutive_misses": 1,
          "first_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "last_observed_at": "2026-08-14T04:35:32.628934+02:00",
          "status": "weakening",
          "superseded_by": null
        }
      ],
      "observed_current_state": {
        "game": "Test RPG",
        "title": "Boss attempts"
      },
      "precedence": "observed_twitch_metadata"
    },
    "action_ledger": {
      "entries": [],
      "last_claim_validation": {}
    },
    "scene_transitions": {
      "all": [],
      "last": {}
    },
    "continuity_context": {},
    "promotion_profiles": [],
    "actions": {
      "attempts": [],
      "speech_requests": [],
      "model_calls": [],
      "research_calls": []
    },
    "receipts": [],
    "emitted_outputs": [],
    "final_emission_results": [],
    "speech_intents": {
      "active": [
        {
          "id": "intent_23c0d3235c5150eb9db77c46247f8c22",
          "type": "REACTION",
          "source_event_ids": [],
          "anchor_ids": [
            "ambient:rng_dependency:1786727102"
          ],
          "topic": "rng_dependency",
          "subject_ref": "unknown",
          "value": 0.86,
          "urgency": 0.85,
          "freshness": 0.998333332935969,
          "created_at": 1786727102.2,
          "expires_at": 1786727110.2,
          "interruptibility": "yield_before_tts_commit",
          "minimum_turn_gap": 1.2,
          "maximum_turn_delay": 8.0,
          "scene_relevance": {
            "scene_id": "scene_2f0690b8ded85d18ba449ec4e37abb13",
            "state_version": 1,
            "current_state": "active",
            "terminal": false
          },
          "status": "PENDING",
          "suppression_reason": "",
          "reserved_at": 0.0,
          "emitted_at": 0.0,
          "contribution_material": {
            "anchor": {
              "id": "ambient:rng_dependency:1786727102",
              "type": "rng_dependency",
              "quality": 0.86,
              "reason": "recent_ambient_context",
              "evidence": {
                "anchor_id": "ambient:rng_dependency:1786727102",
                "anchor_type": "rng_dependency",
                "raw_owner_fragments": [
                  "que ha sido demasiada suerte dios"
                ],
                "exact_supported_claims": [
                  "que ha sido demasiada suerte dios"
                ],
                "timestamps": [
                  1786727102.0
                ],
                "topic_id": "topic_dc448af4c39f",
                "currentness": 0.998333332935969,
                "confidence": 0.86,
                "allowed_contribution_types": [
                  "contextual_reaction",
                  "emotional_banter",
                  "concise_observation"
                ],
                "forbidden_claims": [
                  "unsupported strategy",
                  "save instruction",
                  "unrelated mechanic",
                  "stale topic fusion"
                ],
                "expires_at": 1786727222.0,
                "scene_id": "scene_2f0690b8ded85d18ba449ec4e37abb13",
                "state_version": 1,
                "current_state": "active",
                "terminal": false,
                "extracted_subject": "unknown",
                "extracted_object": "",
                "extracted_predicate": "",
                "supported_claims": [
                  "que ha sido demasiada suerte dios"
                ],
                "unsupported_claims": []
              },
              "scene_guard": {
                "scene_id": "scene_2f0690b8ded85d18ba449ec4e37abb13",
                "state_version": 1,
                "current_state": "active",
                "terminal": false
              }
            },
            "readiness_topic": null
          }
        }
      ],
      "all": [
        {
          "id": "intent_23c0d3235c5150eb9db77c46247f8c22",
          "type": "REACTION",
          "source_event_ids": [],
          "anchor_ids": [
            "ambient:rng_dependency:1786727102"
          ],
          "topic": "rng_dependency",
          "subject_ref": "unknown",
          "value": 0.86,
          "urgency": 0.85,
          "freshness": 0.998333332935969,
          "created_at": 1786727102.2,
          "expires_at": 1786727110.2,
          "interruptibility": "yield_before_tts_commit",
          "minimum_turn_gap": 1.2,
          "maximum_turn_delay": 8.0,
          "scene_relevance": {
            "scene_id": "scene_2f0690b8ded85d18ba449ec4e37abb13",
            "state_version": 1,
            "current_state": "active",
            "terminal": false
          },
          "status": "PENDING",
          "suppression_reason": "",
          "reserved_at": 0.0,
          "emitted_at": 0.0,
          "contribution_material": {
            "anchor": {
              "id": "ambient:rng_dependency:1786727102",
              "type": "rng_dependency",
              "quality": 0.86,
              "reason": "recent_ambient_context",
              "evidence": {
                "anchor_id": "ambient:rng_dependency:1786727102",
                "anchor_type": "rng_dependency",
                "raw_owner_fragments": [
                  "que ha sido demasiada suerte dios"
                ],
                "exact_supported_claims": [
                  "que ha sido demasiada suerte dios"
                ],
                "timestamps": [
                  1786727102.0
                ],
                "topic_id": "topic_dc448af4c39f",
                "currentness": 0.998333332935969,
                "confidence": 0.86,
                "allowed_contribution_types": [
                  "contextual_reaction",
                  "emotional_banter",
                  "concise_observation"
                ],
                "forbidden_claims": [
                  "unsupported strategy",
                  "save instruction",
                  "unrelated mechanic",
                  "stale topic fusion"
                ],
                "expires_at": 1786727222.0,
                "scene_id": "scene_2f0690b8ded85d18ba449ec4e37abb13",
                "state_version": 1,
                "current_state": "active",
                "terminal": false,
                "extracted_subject": "unknown",
                "extracted_object": "",
                "extracted_predicate": "",
                "supported_claims": [
                  "que ha sido demasiada suerte dios"
                ],
                "unsupported_claims": []
              },
              "scene_guard": {
                "scene_id": "scene_2f0690b8ded85d18ba449ec4e37abb13",
                "state_version": 1,
                "current_state": "active",
                "terminal": false
              }
            },
            "readiness_topic": null
          }
        }
      ],
      "metrics": {
        "intents_created": 1,
        "created:REACTION": 1,
        "pending_due_owner_voice_active": 1,
        "pending": 1,
        "time_created_to_emit": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "turn_gap_before_emit": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "intent_creation": {
          "count": 1,
          "p50_ms": 0.073,
          "p95_ms": 0.073
        },
        "pending_queue_operation": {
          "count": 1,
          "p50_ms": 0.073,
          "p95_ms": 0.073
        },
        "turn_arbitration": {
          "count": 2,
          "p50_ms": 0.012,
          "p95_ms": 0.012
        },
        "presence_turn_decision": {
          "count": 2,
          "p50_ms": 1.476,
          "p95_ms": 1.476
        }
      }
    },
    "database_watermarks": {
      "counts": {
        "chat_log": 0,
        "memory_facts": 0,
        "memory_chunks": 0,
        "stream_sessions": 1,
        "stream_chat_messages": 0,
        "stream_events": 1,
        "live_session_timeline": 4,
        "promotion_events": 0,
        "viewer_promotion_profiles": 0,
        "schema_migrations": 7,
        "conversations": 0,
        "open_threads": 0,
        "beliefs": 0,
        "belief_evidence": 0,
        "scene_assertions": 0,
        "game_identities": 0,
        "game_runs": 0,
        "game_run_sessions": 0,
        "game_run_events": 0,
        "game_knowledge_facts": 0,
        "game_knowledge_v2_gaps": 0,
        "people": 0,
        "person_identities": 0,
        "person_sessions": 0,
        "social_episodes": 0,
        "shared_culture_items": 0,
        "shared_culture_evidence": 0,
        "consolidation_runs": 0,
        "consolidation_deltas": 0,
        "action_ledger": 0,
        "temporal_maintenance_audit": 0,
        "learning_observations": 0,
        "scene_transitions": 0,
        "schedule_observations": 1,
        "schedule_hypotheses": 7
      },
      "schema_migrations": [
        {
          "component": "architecture_consolidation",
          "version": 1,
          "name": "audit_hygiene_and_cutover_state",
          "checksum": "944b3ad6532151d4204ed59f663668c9b0ac62df54ece7602d26b91acddd5b7d",
          "applied_at": "2026-08-14T02:35:32.541524+00:00"
        },
        {
          "component": "belief_v2",
          "version": 1,
          "name": "beliefs_evidence_and_compatibility_columns",
          "checksum": "f0df6f1288caccaf6bb47670b38f9cca747f14c84916d6c1ca75beb687200507",
          "applied_at": "2026-08-14T02:35:32.399139+00:00"
        },
        {
          "component": "cognitive_replay",
          "version": 1,
          "name": "replay_metadata",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "applied_at": "2026-08-14T02:35:32.177746+00:00"
        },
        {
          "component": "conversation_continuity",
          "version": 1,
          "name": "conversation_and_open_threads",
          "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
          "applied_at": "2026-08-14T02:35:32.312278+00:00"
        },
        {
          "component": "game_context_v2",
          "version": 1,
          "name": "durable_runs_knowledge_and_gaps",
          "checksum": "08e342acaae00d5d24c1a6dbccad5aee41f753ed90d9ec26282a5fdf042d0a75",
          "applied_at": "2026-08-14T02:35:32.450003+00:00"
        },
        {
          "component": "learning_v2",
          "version": 1,
          "name": "consolidation_temporal_action_and_scene",
          "checksum": "6a86e2d1c7c03167f3b20c328bc97b73fb92e03903f64e64d326fb08f9e3b942",
          "applied_at": "2026-08-14T02:35:32.526410+00:00"
        },
        {
          "component": "social_world_v2",
          "version": 1,
          "name": "people_episodes_and_shared_culture",
          "checksum": "b02adc2cd7f298f1af228dc52c4ba44ae15999fb4c815ea004c40f68b78cbfa5",
          "applied_at": "2026-08-14T02:35:32.488704+00:00"
        }
      ],
      "final_response_digest": "",
      "final_response_present": false
    }
  },
  "natural-turn": {
    "runtime": {
      "mode": "active",
      "hebe_sleeping": false,
      "is_running": false,
      "last_input_source": null,
      "last_intent": null,
      "last_firewall": {
        "source": "twitch_system",
        "authority": "system",
        "decision": "allow",
        "reason": "system_event"
      },
      "last_policy": {
        "source": null,
        "authority": null,
        "decision": null,
        "reason": null
      }
    },
    "stream_session": {
      "enabled": true,
      "is_live": true,
      "live_status_known": true,
      "active_stream_session_id": 1,
      "last_transition": "online",
      "title": "Boss attempts",
      "game": "Test RPG",
      "category": "Test RPG"
    },
    "current_scene": {
      "scene_id": "scene_2f0690b8ded85d18ba449ec4e37abb13",
      "topic_id": "topic_dc448af4c39f",
      "entity": "unknown",
      "current_state": "active",
      "state_version": 1,
      "supporting_event_ids": [
        "ambient:rng_dependency:1786727102"
      ],
      "superseded_event_ids": [],
      "terminal": false,
      "updated_at": 1786727102.0
    },
    "pending": {
      "clarification": null,
      "conversation_turn": null
    },
    "conversation": {
      "active": {},
      "latest": {},
      "all": [],
      "last_resolution": {
        "consumed": false,
        "decision": "no_conversation",
        "reason": "no_compatible_active_conversation",
        "conversation_id": "",
        "reply_act": "UNKNOWN",
        "payload": {},
        "conversation": null,
        "latency_ms": 3.095800057053566
      },
      "legacy_pending_projection": {},
      "continuity_shadow_diff": {
        "legacy_result": false,
        "v2_result": false,
        "match": true,
        "difference_reason": ""
      },
      "shadow_metrics": {
        "total": 1,
        "matches": 1,
        "differences": 0,
        "match_rate": 1.0,
        "difference_reasons": {}
      },
      "performance": {
        "count": 1,
        "p50_ms": 3.0958,
        "p95_ms": 3.0958
      }
    },
    "open_threads": [],
    "memory": {
      "facts_count": 0,
      "chunks_count": 0
    },
    "beliefs": {
      "active": [],
      "historical": [],
      "superseded": [],
      "suspected": [],
      "all": [],
      "last_transition": {}
    },
    "belief_evidence": [],
    "retrieval": {
      "last_request": {},
      "performance": {
        "count": 0,
        "p50_ms": 0.0,
        "p95_ms": 0.0
      },
      "write_performance": {
        "count": 0,
        "p50_ms": 0.0,
        "p95_ms": 0.0
      },
      "repository_performance": {
        "belief_lookup": {
          "count": 8,
          "p50_ms": 0.9222,
          "p95_ms": 1.8886
        },
        "evidence_lookup": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "sqlite_write": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        }
      }
    },
    "memory_compatibility": {
      "legacy_to_v2": [],
      "v2_to_legacy": [],
      "shadow_diffs": [],
      "backfill": {
        "safe": 0,
        "compatibility_only": 0,
        "ambiguous": 0,
        "invalid_stale": 0
      }
    },
    "game_state": {
      "game": "Test RPG",
      "platform_version": "",
      "playthrough_type": "casual",
      "spoiler_policy": "no_spoilers",
      "current_location": null,
      "current_character": "",
      "party_members": [],
      "last_confirmed_progress": "",
      "current_objective": null,
      "challenge": "",
      "known_constraints": [],
      "last_updated": 1786727100.0,
      "provenance": "stream_context_sync",
      "confidence": 0.75,
      "current_game": "Test RPG",
      "recent_run_context_facts": [
        {
          "kind": "rng_dependency",
          "text": "Leo framed the current situation as dependent on RNG or luck.",
          "category": "rng_dependency",
          "summary": "Leo framed the current situation as dependent on RNG or luck.",
          "id": "ambient:rng_dependency:1786727102",
          "fact_id": "ambient:rng_dependency:1786727102",
          "raw_text": "<redacted:2320c1febf71>",
          "conservative_normalized_text": "que ha sido demasiada suerte dios",
          "utterance_role": "owner_question_to_stream",
          "timestamp": 1786727102.0,
          "topic_id": "topic_dc448af4c39f",
          "heuristic_category": "rng_dependency",
          "extracted_subject": "unknown",
          "subject": "unknown",
          "extracted_object": "",
          "object": "",
          "extracted_predicate": "",
          "predicate": "",
          "confidence": 0.86,
          "referent_confidence": 0.86,
          "inference_level": "heuristic",
          "supported_claims": [
            "que ha sido demasiada suerte dios"
          ],
          "directly_supported_claims": [
            "que ha sido demasiada suerte dios"
          ],
          "inferred_claims": [],
          "unsupported_claims": [],
          "evidence_span": "que ha sido demasiada suerte dios",
          "evidence_tokens": "<redacted:323fcf3ad517>",
          "semantic_rule": "ambient_category:rng_dependency",
          "model_reason": "Leo framed the current situation as dependent on RNG or luck.",
          "expires_at": 1786727222.0,
          "scene_id": "scene_2f0690b8ded85d18ba449ec4e37abb13",
          "game": null,
          "source": "stt_voice",
          "raw_evidence": "que ha sido demasiada suerte dios",
          "normalized_text": "que ha sido demasiada suerte dios",
          "normalized_evidence": "que ha sido demasiada suerte dios",
          "language": "es",
          "ttl_sec": 120,
          "data": {
            "category": "rng_dependency",
            "raw_text": "<redacted:2320c1febf71>",
            "normalized_text": "que ha sido demasiada suerte dios",
            "mood": "rng tension",
            "extracted_subject": "unknown",
            "extracted_object": "",
            "extracted_predicate": "",
            "inference_level": "heuristic",
            "supported_claims": [
              "que ha sido demasiada suerte dios"
            ],
            "inferred_claims": [],
            "unsupported_claims": [],
            "evidence_span": "que ha sido demasiada suerte dios",
            "evidence_tokens": "<redacted:323fcf3ad517>",
            "semantic_rule": "ambient_category:rng_dependency",
            "model_reason": "Leo framed the current situation as dependent on RNG or luck.",
            "scene_id": "scene_2f0690b8ded85d18ba449ec4e37abb13"
          },
          "age_seconds": 0.0,
          "superseded": false,
          "state_version": 1,
          "current_state": "active",
          "terminal": false,
          "currentness_score": 1.0
        }
      ],
      "identity": {},
      "context": {},
      "active_run": {},
      "runs": [],
      "session_links": [],
      "run_events": [],
      "run_beliefs": {
        "current": [],
        "inferred": [],
        "superseded": []
      },
      "knowledge": {
        "selected": [],
        "rejected": [],
        "spoiler_blocked": [],
        "all": []
      },
      "gaps": [],
      "research": {
        "research_calls": 0,
        "cache_hits": 0,
        "failures": [],
        "context_performance": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "fixture_calls": [],
        "status": ""
      },
      "compatibility": {
        "legacy_progress": [],
        "dossier": [],
        "legacy_run": [],
        "shadow_diffs": [],
        "backfill": {
          "validated": 0,
          "compatibility_only": 0,
          "ambiguous": 0,
          "stale": 0
        }
      },
      "provenance_manifest": [],
      "advice_allowed": null,
      "reaction_allowed": null,
      "performance": {
        "run": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "run_fact": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "knowledge": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "research_gap": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "db_write": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        }
      },
      "context_performance": {
        "count": 0,
        "p50_ms": 0.0,
        "p95_ms": 0.0
      },
      "manifest_size_bytes": 0,
      "last_run_resolution": {},
      "run_resolution_performance": {
        "count": 0,
        "p50_ms": 0.0,
        "p95_ms": 0.0
      }
    },
    "social_state": {
      "recent_active_users": [],
      "recent_chat_count": 0,
      "last_raid": {},
      "last_cheer": {},
      "people": [],
      "identities": [],
      "recent_episodes": [],
      "active_hypotheses": [],
      "historical_hypotheses": [],
      "open_threads": [],
      "relationships": [],
      "shared_culture": {
        "all": [],
        "candidates": [],
        "active": [],
        "weakening": [],
        "retired": [],
        "reactions": [],
        "selection": {
          "selected": [],
          "rejected": []
        }
      },
      "retrieval": {},
      "opportunities": [],
      "resolution": {},
      "rejected_writes": [],
      "compatibility": {
        "chatter_presence": [],
        "chatter_profiles": [],
        "chatter_facts": [],
        "stream_chatter_summaries": [],
        "viewer_profiles": [],
        "social_events": [],
        "promotion_profiles": [],
        "backfill": {
          "explicit_observation": 0,
          "safe_episode": 0,
          "inferred_compatibility_only": 0,
          "ambiguous": 0,
          "sensitive": 0,
          "stale": 0
        },
        "shadow_diffs": []
      },
      "performance": {
        "identity": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "episode_write": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "thread_lookup": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "culture_select": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "context": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "db_write": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        }
      },
      "belief_lookup_performance": {
        "belief_lookup": {
          "count": 8,
          "p50_ms": 0.9222,
          "p95_ms": 1.8886
        },
        "evidence_lookup": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "sqlite_write": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        }
      }
    },
    "learning": {
      "consolidation_runs": [],
      "deltas": [],
      "rejected_deltas": [],
      "watermarks": [],
      "last_result": {},
      "stable_core_version": "a1a58e51882b0c88",
      "performance": {
        "repository": {
          "lookup": {
            "count": 40,
            "p50_ms": 0.89735,
            "p95_ms": 1.4252
          },
          "write": {
            "count": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0
          },
          "context": {
            "count": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0
          }
        },
        "consolidation": {
          "consolidation_duration": {
            "count": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0
          },
          "candidate_validation": {
            "count": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0
          }
        },
        "temporal": {
          "temporal_maintenance": {
            "count": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0
          }
        },
        "action_history": {
          "action_ledger_lookup": {
            "count": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0
          }
        },
        "owner_preferences": {
          "owner_preference_lookup": {
            "count": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0
          }
        },
        "hebe_self": {
          "hebe_self_lookup": {
            "count": 1,
            "p50_ms": 1.8985,
            "p95_ms": 1.8985
          }
        },
        "context": {
          "continuity_context_build": {
            "count": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0
          }
        }
      }
    },
    "self_model": {
      "stable_core_version": "a1a58e51882b0c88",
      "evolving_preferences": [],
      "opinions": [],
      "superseded_opinions": []
    },
    "owner_preferences": [],
    "leo_language": {
      "beliefs": [],
      "interpretation_aliases": {}
    },
    "temporal": {
      "expired": [],
      "archived": [],
      "weakened": [],
      "maintenance_actions": [],
      "last_actions": []
    },
    "schedule": {
      "observations": [
        {
          "id": 1,
          "stream_session_id": "1",
          "weekday": "friday",
          "time_window": "night",
          "canonical_content": "Test RPG",
          "content_key": "test rpg",
          "stream_format": "game_playthrough",
          "source": "observed",
          "observed_at": "2026-08-14T04:35:32.628934+02:00"
        }
      ],
      "hypotheses": [
        {
          "id": 1,
          "weekday": "monday",
          "time_window": "any",
          "canonical_content": "FINAL FANTASY IX",
          "content_key": "final fantasy ix",
          "stream_format": "challenge_run",
          "source": "owner_declared",
          "confidence": 0.9,
          "evidence_count": 1,
          "consecutive_matches": 1,
          "consecutive_misses": 0,
          "first_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "last_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "status": "active",
          "superseded_by": null
        },
        {
          "id": 2,
          "weekday": "tuesday",
          "time_window": "any",
          "canonical_content": "Persona 5 Royal",
          "content_key": "persona 5 royal",
          "stream_format": "game_playthrough",
          "source": "owner_declared",
          "confidence": 0.9,
          "evidence_count": 1,
          "consecutive_matches": 1,
          "consecutive_misses": 0,
          "first_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "last_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "status": "active",
          "superseded_by": null
        },
        {
          "id": 3,
          "weekday": "wednesday",
          "time_window": "any",
          "canonical_content": "Persona 5 Royal",
          "content_key": "persona 5 royal",
          "stream_format": "game_playthrough",
          "source": "owner_declared",
          "confidence": 0.9,
          "evidence_count": 1,
          "consecutive_matches": 1,
          "consecutive_misses": 0,
          "first_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "last_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "status": "active",
          "superseded_by": null
        },
        {
          "id": 4,
          "weekday": "thursday",
          "time_window": "any",
          "canonical_content": "Baldur's Gate 3",
          "content_key": "baldur s gate 3",
          "stream_format": "game_playthrough",
          "source": "owner_declared",
          "confidence": 0.9,
          "evidence_count": 1,
          "consecutive_matches": 1,
          "consecutive_misses": 0,
          "first_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "last_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "status": "active",
          "superseded_by": null
        },
        {
          "id": 6,
          "weekday": "saturday",
          "time_window": "any",
          "canonical_content": "Retro Weekend",
          "content_key": "retro weekend",
          "stream_format": "retro",
          "source": "owner_declared",
          "confidence": 0.9,
          "evidence_count": 1,
          "consecutive_matches": 1,
          "consecutive_misses": 0,
          "first_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "last_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "status": "active",
          "superseded_by": null
        },
        {
          "id": 7,
          "weekday": "sunday",
          "time_window": "any",
          "canonical_content": "Retro Weekend",
          "content_key": "retro weekend",
          "stream_format": "retro",
          "source": "owner_declared",
          "confidence": 0.9,
          "evidence_count": 1,
          "consecutive_matches": 1,
          "consecutive_misses": 0,
          "first_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "last_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "status": "active",
          "superseded_by": null
        },
        {
          "id": 5,
          "weekday": "friday",
          "time_window": "any",
          "canonical_content": "Baldur's Gate 3",
          "content_key": "baldur s gate 3",
          "stream_format": "game_playthrough",
          "source": "owner_declared",
          "confidence": 0.76,
          "evidence_count": 1,
          "consecutive_matches": 0,
          "consecutive_misses": 1,
          "first_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "last_observed_at": "2026-08-14T04:35:32.628934+02:00",
          "status": "weakening",
          "superseded_by": null
        }
      ],
      "observed_current_state": {
        "game": "Test RPG",
        "title": "Boss attempts"
      },
      "precedence": "observed_twitch_metadata"
    },
    "action_ledger": {
      "entries": [],
      "last_claim_validation": {}
    },
    "scene_transitions": {
      "all": [],
      "last": {}
    },
    "continuity_context": {},
    "promotion_profiles": [],
    "actions": {
      "attempts": [
        {
          "operation": "twitch.send_message",
          "payload": {
            "text": "Uf, qué tensión."
          },
          "outcome": {
            "success": true,
            "status": "sent"
          }
        }
      ],
      "speech_requests": [],
      "model_calls": [
        {
          "key": "promotion_clarification:v1:none",
          "method": "chat"
        }
      ],
      "research_calls": []
    },
    "receipts": [],
    "emitted_outputs": [
      {
        "event_id": "twitch_job_1786727104300000",
        "emitted": true,
        "route": "twitch_text_reply",
        "targets": [
          "twitch_chat"
        ],
        "reason": "",
        "text_digest": "",
        "text_present": false
      }
    ],
    "final_emission_results": [
      {
        "event_id": "twitch_job_1786727104300000",
        "emitted": true,
        "route": "twitch_text_reply",
        "targets": [
          "twitch_chat"
        ],
        "reason": "",
        "text_digest": "",
        "text_present": false
      }
    ],
    "speech_intents": {
      "active": [],
      "all": [
        {
          "id": "intent_23c0d3235c5150eb9db77c46247f8c22",
          "type": "REACTION",
          "source_event_ids": [],
          "anchor_ids": [
            "ambient:rng_dependency:1786727102"
          ],
          "topic": "rng_dependency",
          "subject_ref": "unknown",
          "value": 0.86,
          "urgency": 0.85,
          "freshness": 0.998333332935969,
          "created_at": 1786727102.2,
          "expires_at": 1786727110.2,
          "interruptibility": "yield_before_tts_commit",
          "minimum_turn_gap": 1.2,
          "maximum_turn_delay": 8.0,
          "scene_relevance": {
            "scene_id": "scene_2f0690b8ded85d18ba449ec4e37abb13",
            "state_version": 1,
            "current_state": "active",
            "terminal": false
          },
          "status": "EMITTED",
          "suppression_reason": "",
          "reserved_at": 1786727104.3,
          "emitted_at": 1786727104.3,
          "contribution_material": {
            "anchor": {
              "id": "ambient:rng_dependency:1786727102",
              "type": "rng_dependency",
              "quality": 0.86,
              "reason": "recent_ambient_context",
              "evidence": {
                "anchor_id": "ambient:rng_dependency:1786727102",
                "anchor_type": "rng_dependency",
                "raw_owner_fragments": [
                  "que ha sido demasiada suerte dios"
                ],
                "exact_supported_claims": [
                  "que ha sido demasiada suerte dios"
                ],
                "timestamps": [
                  1786727102.0
                ],
                "topic_id": "topic_dc448af4c39f",
                "currentness": 0.998333332935969,
                "confidence": 0.86,
                "allowed_contribution_types": [
                  "contextual_reaction",
                  "emotional_banter",
                  "concise_observation"
                ],
                "forbidden_claims": [
                  "unsupported strategy",
                  "save instruction",
                  "unrelated mechanic",
                  "stale topic fusion"
                ],
                "expires_at": 1786727222.0,
                "scene_id": "scene_2f0690b8ded85d18ba449ec4e37abb13",
                "state_version": 1,
                "current_state": "active",
                "terminal": false,
                "extracted_subject": "unknown",
                "extracted_object": "",
                "extracted_predicate": "",
                "supported_claims": [
                  "que ha sido demasiada suerte dios"
                ],
                "unsupported_claims": []
              },
              "scene_guard": {
                "scene_id": "scene_2f0690b8ded85d18ba449ec4e37abb13",
                "state_version": 1,
                "current_state": "active",
                "terminal": false
              }
            },
            "readiness_topic": null
          }
        }
      ],
      "metrics": {
        "intents_created": 1,
        "created:REACTION": 1,
        "pending_due_owner_voice_active": 1,
        "turns_reserved": 1,
        "intents_emitted": 1,
        "emitted:REACTION": 1,
        "pending": 0,
        "time_created_to_emit": {
          "count": 1,
          "p50_ms": 2100.0,
          "p95_ms": 2100.0
        },
        "turn_gap_before_emit": {
          "count": 1,
          "p50_ms": 1300.0,
          "p95_ms": 1300.0
        },
        "intent_creation": {
          "count": 1,
          "p50_ms": 0.073,
          "p95_ms": 0.073
        },
        "pending_queue_operation": {
          "count": 1,
          "p50_ms": 0.073,
          "p95_ms": 0.073
        },
        "turn_arbitration": {
          "count": 3,
          "p50_ms": 0.024,
          "p95_ms": 0.024
        },
        "presence_turn_decision": {
          "count": 3,
          "p50_ms": 1.799,
          "p95_ms": 1.799
        }
      }
    },
    "database_watermarks": {
      "counts": {
        "chat_log": 0,
        "memory_facts": 0,
        "memory_chunks": 0,
        "stream_sessions": 1,
        "stream_chat_messages": 0,
        "stream_events": 1,
        "live_session_timeline": 7,
        "promotion_events": 0,
        "viewer_promotion_profiles": 0,
        "schema_migrations": 7,
        "conversations": 0,
        "open_threads": 0,
        "beliefs": 0,
        "belief_evidence": 0,
        "scene_assertions": 0,
        "game_identities": 0,
        "game_runs": 0,
        "game_run_sessions": 0,
        "game_run_events": 0,
        "game_knowledge_facts": 0,
        "game_knowledge_v2_gaps": 0,
        "people": 0,
        "person_identities": 0,
        "person_sessions": 0,
        "social_episodes": 0,
        "shared_culture_items": 0,
        "shared_culture_evidence": 0,
        "consolidation_runs": 0,
        "consolidation_deltas": 0,
        "action_ledger": 0,
        "temporal_maintenance_audit": 0,
        "learning_observations": 0,
        "scene_transitions": 0,
        "schedule_observations": 1,
        "schedule_hypotheses": 7
      },
      "schema_migrations": [
        {
          "component": "architecture_consolidation",
          "version": 1,
          "name": "audit_hygiene_and_cutover_state",
          "checksum": "944b3ad6532151d4204ed59f663668c9b0ac62df54ece7602d26b91acddd5b7d",
          "applied_at": "2026-08-14T02:35:32.541524+00:00"
        },
        {
          "component": "belief_v2",
          "version": 1,
          "name": "beliefs_evidence_and_compatibility_columns",
          "checksum": "f0df6f1288caccaf6bb47670b38f9cca747f14c84916d6c1ca75beb687200507",
          "applied_at": "2026-08-14T02:35:32.399139+00:00"
        },
        {
          "component": "cognitive_replay",
          "version": 1,
          "name": "replay_metadata",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "applied_at": "2026-08-14T02:35:32.177746+00:00"
        },
        {
          "component": "conversation_continuity",
          "version": 1,
          "name": "conversation_and_open_threads",
          "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
          "applied_at": "2026-08-14T02:35:32.312278+00:00"
        },
        {
          "component": "game_context_v2",
          "version": 1,
          "name": "durable_runs_knowledge_and_gaps",
          "checksum": "08e342acaae00d5d24c1a6dbccad5aee41f753ed90d9ec26282a5fdf042d0a75",
          "applied_at": "2026-08-14T02:35:32.450003+00:00"
        },
        {
          "component": "learning_v2",
          "version": 1,
          "name": "consolidation_temporal_action_and_scene",
          "checksum": "6a86e2d1c7c03167f3b20c328bc97b73fb92e03903f64e64d326fb08f9e3b942",
          "applied_at": "2026-08-14T02:35:32.526410+00:00"
        },
        {
          "component": "social_world_v2",
          "version": 1,
          "name": "people_episodes_and_shared_culture",
          "checksum": "b02adc2cd7f298f1af228dc52c4ba44ae15999fb4c815ea004c40f68b78cbfa5",
          "applied_at": "2026-08-14T02:35:32.488704+00:00"
        }
      ],
      "final_response_digest": "",
      "final_response_present": false
    }
  },
  "leo-resumes": {
    "runtime": {
      "mode": "active",
      "hebe_sleeping": false,
      "is_running": false,
      "last_input_source": null,
      "last_intent": null,
      "last_firewall": {
        "source": "twitch_system",
        "authority": "system",
        "decision": "allow",
        "reason": "system_event"
      },
      "last_policy": {
        "source": null,
        "authority": null,
        "decision": null,
        "reason": null
      }
    },
    "stream_session": {
      "enabled": true,
      "is_live": true,
      "live_status_known": true,
      "active_stream_session_id": 1,
      "last_transition": "online",
      "title": "Boss attempts",
      "game": "Test RPG",
      "category": "Test RPG"
    },
    "current_scene": {
      "scene_id": "scene_2f0690b8ded85d18ba449ec4e37abb13",
      "topic_id": "topic_dc448af4c39f",
      "entity": "unknown",
      "current_state": "active",
      "state_version": 1,
      "supporting_event_ids": [
        "ambient:rng_dependency:1786727102"
      ],
      "superseded_event_ids": [],
      "terminal": false,
      "updated_at": 1786727102.0
    },
    "pending": {
      "clarification": null,
      "conversation_turn": null
    },
    "conversation": {
      "active": {},
      "latest": {},
      "all": [],
      "last_resolution": {
        "consumed": false,
        "decision": "no_conversation",
        "reason": "no_compatible_active_conversation",
        "conversation_id": "",
        "reply_act": "UNKNOWN",
        "payload": {},
        "conversation": null,
        "latency_ms": 3.095800057053566
      },
      "legacy_pending_projection": {},
      "continuity_shadow_diff": {
        "legacy_result": false,
        "v2_result": false,
        "match": true,
        "difference_reason": ""
      },
      "shadow_metrics": {
        "total": 1,
        "matches": 1,
        "differences": 0,
        "match_rate": 1.0,
        "difference_reasons": {}
      },
      "performance": {
        "count": 1,
        "p50_ms": 3.0958,
        "p95_ms": 3.0958
      }
    },
    "open_threads": [],
    "memory": {
      "facts_count": 0,
      "chunks_count": 0
    },
    "beliefs": {
      "active": [],
      "historical": [],
      "superseded": [],
      "suspected": [],
      "all": [],
      "last_transition": {}
    },
    "belief_evidence": [],
    "retrieval": {
      "last_request": {},
      "performance": {
        "count": 0,
        "p50_ms": 0.0,
        "p95_ms": 0.0
      },
      "write_performance": {
        "count": 0,
        "p50_ms": 0.0,
        "p95_ms": 0.0
      },
      "repository_performance": {
        "belief_lookup": {
          "count": 9,
          "p50_ms": 0.9519,
          "p95_ms": 1.8886
        },
        "evidence_lookup": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "sqlite_write": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        }
      }
    },
    "memory_compatibility": {
      "legacy_to_v2": [],
      "v2_to_legacy": [],
      "shadow_diffs": [],
      "backfill": {
        "safe": 0,
        "compatibility_only": 0,
        "ambiguous": 0,
        "invalid_stale": 0
      }
    },
    "game_state": {
      "game": "Test RPG",
      "platform_version": "",
      "playthrough_type": "casual",
      "spoiler_policy": "no_spoilers",
      "current_location": null,
      "current_character": "",
      "party_members": [],
      "last_confirmed_progress": "",
      "current_objective": null,
      "challenge": "",
      "known_constraints": [],
      "last_updated": 1786727100.0,
      "provenance": "stream_context_sync",
      "confidence": 0.75,
      "current_game": "Test RPG",
      "recent_run_context_facts": [
        {
          "kind": "rng_dependency",
          "text": "Leo framed the current situation as dependent on RNG or luck.",
          "category": "rng_dependency",
          "summary": "Leo framed the current situation as dependent on RNG or luck.",
          "id": "ambient:rng_dependency:1786727102",
          "fact_id": "ambient:rng_dependency:1786727102",
          "raw_text": "<redacted:2320c1febf71>",
          "conservative_normalized_text": "que ha sido demasiada suerte dios",
          "utterance_role": "owner_question_to_stream",
          "timestamp": 1786727102.0,
          "topic_id": "topic_dc448af4c39f",
          "heuristic_category": "rng_dependency",
          "extracted_subject": "unknown",
          "subject": "unknown",
          "extracted_object": "",
          "object": "",
          "extracted_predicate": "",
          "predicate": "",
          "confidence": 0.86,
          "referent_confidence": 0.86,
          "inference_level": "heuristic",
          "supported_claims": [
            "que ha sido demasiada suerte dios"
          ],
          "directly_supported_claims": [
            "que ha sido demasiada suerte dios"
          ],
          "inferred_claims": [],
          "unsupported_claims": [],
          "evidence_span": "que ha sido demasiada suerte dios",
          "evidence_tokens": "<redacted:323fcf3ad517>",
          "semantic_rule": "ambient_category:rng_dependency",
          "model_reason": "Leo framed the current situation as dependent on RNG or luck.",
          "expires_at": 1786727222.0,
          "scene_id": "scene_2f0690b8ded85d18ba449ec4e37abb13",
          "game": null,
          "source": "stt_voice",
          "raw_evidence": "que ha sido demasiada suerte dios",
          "normalized_text": "que ha sido demasiada suerte dios",
          "normalized_evidence": "que ha sido demasiada suerte dios",
          "language": "es",
          "ttl_sec": 120,
          "data": {
            "category": "rng_dependency",
            "raw_text": "<redacted:2320c1febf71>",
            "normalized_text": "que ha sido demasiada suerte dios",
            "mood": "rng tension",
            "extracted_subject": "unknown",
            "extracted_object": "",
            "extracted_predicate": "",
            "inference_level": "heuristic",
            "supported_claims": [
              "que ha sido demasiada suerte dios"
            ],
            "inferred_claims": [],
            "unsupported_claims": [],
            "evidence_span": "que ha sido demasiada suerte dios",
            "evidence_tokens": "<redacted:323fcf3ad517>",
            "semantic_rule": "ambient_category:rng_dependency",
            "model_reason": "Leo framed the current situation as dependent on RNG or luck.",
            "scene_id": "scene_2f0690b8ded85d18ba449ec4e37abb13"
          },
          "age_seconds": 0.0,
          "superseded": false,
          "state_version": 1,
          "current_state": "active",
          "terminal": false,
          "currentness_score": 1.0
        }
      ],
      "identity": {},
      "context": {},
      "active_run": {},
      "runs": [],
      "session_links": [],
      "run_events": [],
      "run_beliefs": {
        "current": [],
        "inferred": [],
        "superseded": []
      },
      "knowledge": {
        "selected": [],
        "rejected": [],
        "spoiler_blocked": [],
        "all": []
      },
      "gaps": [],
      "research": {
        "research_calls": 0,
        "cache_hits": 0,
        "failures": [],
        "context_performance": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "fixture_calls": [],
        "status": ""
      },
      "compatibility": {
        "legacy_progress": [],
        "dossier": [],
        "legacy_run": [],
        "shadow_diffs": [],
        "backfill": {
          "validated": 0,
          "compatibility_only": 0,
          "ambiguous": 0,
          "stale": 0
        }
      },
      "provenance_manifest": [],
      "advice_allowed": null,
      "reaction_allowed": null,
      "performance": {
        "run": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "run_fact": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "knowledge": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "research_gap": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "db_write": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        }
      },
      "context_performance": {
        "count": 0,
        "p50_ms": 0.0,
        "p95_ms": 0.0
      },
      "manifest_size_bytes": 0,
      "last_run_resolution": {},
      "run_resolution_performance": {
        "count": 0,
        "p50_ms": 0.0,
        "p95_ms": 0.0
      }
    },
    "social_state": {
      "recent_active_users": [],
      "recent_chat_count": 0,
      "last_raid": {},
      "last_cheer": {},
      "people": [],
      "identities": [],
      "recent_episodes": [],
      "active_hypotheses": [],
      "historical_hypotheses": [],
      "open_threads": [],
      "relationships": [],
      "shared_culture": {
        "all": [],
        "candidates": [],
        "active": [],
        "weakening": [],
        "retired": [],
        "reactions": [],
        "selection": {
          "selected": [],
          "rejected": []
        }
      },
      "retrieval": {},
      "opportunities": [],
      "resolution": {},
      "rejected_writes": [],
      "compatibility": {
        "chatter_presence": [],
        "chatter_profiles": [],
        "chatter_facts": [],
        "stream_chatter_summaries": [],
        "viewer_profiles": [],
        "social_events": [],
        "promotion_profiles": [],
        "backfill": {
          "explicit_observation": 0,
          "safe_episode": 0,
          "inferred_compatibility_only": 0,
          "ambiguous": 0,
          "sensitive": 0,
          "stale": 0
        },
        "shadow_diffs": []
      },
      "performance": {
        "identity": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "episode_write": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "thread_lookup": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "culture_select": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "context": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "db_write": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        }
      },
      "belief_lookup_performance": {
        "belief_lookup": {
          "count": 9,
          "p50_ms": 0.9519,
          "p95_ms": 1.8886
        },
        "evidence_lookup": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "sqlite_write": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        }
      }
    },
    "learning": {
      "consolidation_runs": [],
      "deltas": [],
      "rejected_deltas": [],
      "watermarks": [],
      "last_result": {},
      "stable_core_version": "a1a58e51882b0c88",
      "performance": {
        "repository": {
          "lookup": {
            "count": 45,
            "p50_ms": 0.8861,
            "p95_ms": 1.4252
          },
          "write": {
            "count": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0
          },
          "context": {
            "count": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0
          }
        },
        "consolidation": {
          "consolidation_duration": {
            "count": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0
          },
          "candidate_validation": {
            "count": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0
          }
        },
        "temporal": {
          "temporal_maintenance": {
            "count": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0
          }
        },
        "action_history": {
          "action_ledger_lookup": {
            "count": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0
          }
        },
        "owner_preferences": {
          "owner_preference_lookup": {
            "count": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0
          }
        },
        "hebe_self": {
          "hebe_self_lookup": {
            "count": 1,
            "p50_ms": 1.8985,
            "p95_ms": 1.8985
          }
        },
        "context": {
          "continuity_context_build": {
            "count": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0
          }
        }
      }
    },
    "self_model": {
      "stable_core_version": "a1a58e51882b0c88",
      "evolving_preferences": [],
      "opinions": [],
      "superseded_opinions": []
    },
    "owner_preferences": [],
    "leo_language": {
      "beliefs": [],
      "interpretation_aliases": {}
    },
    "temporal": {
      "expired": [],
      "archived": [],
      "weakened": [],
      "maintenance_actions": [],
      "last_actions": []
    },
    "schedule": {
      "observations": [
        {
          "id": 1,
          "stream_session_id": "1",
          "weekday": "friday",
          "time_window": "night",
          "canonical_content": "Test RPG",
          "content_key": "test rpg",
          "stream_format": "game_playthrough",
          "source": "observed",
          "observed_at": "2026-08-14T04:35:32.628934+02:00"
        }
      ],
      "hypotheses": [
        {
          "id": 1,
          "weekday": "monday",
          "time_window": "any",
          "canonical_content": "FINAL FANTASY IX",
          "content_key": "final fantasy ix",
          "stream_format": "challenge_run",
          "source": "owner_declared",
          "confidence": 0.9,
          "evidence_count": 1,
          "consecutive_matches": 1,
          "consecutive_misses": 0,
          "first_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "last_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "status": "active",
          "superseded_by": null
        },
        {
          "id": 2,
          "weekday": "tuesday",
          "time_window": "any",
          "canonical_content": "Persona 5 Royal",
          "content_key": "persona 5 royal",
          "stream_format": "game_playthrough",
          "source": "owner_declared",
          "confidence": 0.9,
          "evidence_count": 1,
          "consecutive_matches": 1,
          "consecutive_misses": 0,
          "first_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "last_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "status": "active",
          "superseded_by": null
        },
        {
          "id": 3,
          "weekday": "wednesday",
          "time_window": "any",
          "canonical_content": "Persona 5 Royal",
          "content_key": "persona 5 royal",
          "stream_format": "game_playthrough",
          "source": "owner_declared",
          "confidence": 0.9,
          "evidence_count": 1,
          "consecutive_matches": 1,
          "consecutive_misses": 0,
          "first_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "last_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "status": "active",
          "superseded_by": null
        },
        {
          "id": 4,
          "weekday": "thursday",
          "time_window": "any",
          "canonical_content": "Baldur's Gate 3",
          "content_key": "baldur s gate 3",
          "stream_format": "game_playthrough",
          "source": "owner_declared",
          "confidence": 0.9,
          "evidence_count": 1,
          "consecutive_matches": 1,
          "consecutive_misses": 0,
          "first_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "last_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "status": "active",
          "superseded_by": null
        },
        {
          "id": 6,
          "weekday": "saturday",
          "time_window": "any",
          "canonical_content": "Retro Weekend",
          "content_key": "retro weekend",
          "stream_format": "retro",
          "source": "owner_declared",
          "confidence": 0.9,
          "evidence_count": 1,
          "consecutive_matches": 1,
          "consecutive_misses": 0,
          "first_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "last_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "status": "active",
          "superseded_by": null
        },
        {
          "id": 7,
          "weekday": "sunday",
          "time_window": "any",
          "canonical_content": "Retro Weekend",
          "content_key": "retro weekend",
          "stream_format": "retro",
          "source": "owner_declared",
          "confidence": 0.9,
          "evidence_count": 1,
          "consecutive_matches": 1,
          "consecutive_misses": 0,
          "first_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "last_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "status": "active",
          "superseded_by": null
        },
        {
          "id": 5,
          "weekday": "friday",
          "time_window": "any",
          "canonical_content": "Baldur's Gate 3",
          "content_key": "baldur s gate 3",
          "stream_format": "game_playthrough",
          "source": "owner_declared",
          "confidence": 0.76,
          "evidence_count": 1,
          "consecutive_matches": 0,
          "consecutive_misses": 1,
          "first_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "last_observed_at": "2026-08-14T04:35:32.628934+02:00",
          "status": "weakening",
          "superseded_by": null
        }
      ],
      "observed_current_state": {
        "game": "Test RPG",
        "title": "Boss attempts"
      },
      "precedence": "observed_twitch_metadata"
    },
    "action_ledger": {
      "entries": [],
      "last_claim_validation": {}
    },
    "scene_transitions": {
      "all": [],
      "last": {}
    },
    "continuity_context": {},
    "promotion_profiles": [],
    "actions": {
      "attempts": [
        {
          "operation": "twitch.send_message",
          "payload": {
            "text": "Uf, qué tensión."
          },
          "outcome": {
            "success": true,
            "status": "sent"
          }
        }
      ],
      "speech_requests": [],
      "model_calls": [
        {
          "key": "promotion_clarification:v1:none",
          "method": "chat"
        }
      ],
      "research_calls": []
    },
    "receipts": [],
    "emitted_outputs": [
      {
        "event_id": "twitch_job_1786727104300000",
        "emitted": true,
        "route": "twitch_text_reply",
        "targets": [
          "twitch_chat"
        ],
        "reason": "",
        "text_digest": "",
        "text_present": false
      }
    ],
    "final_emission_results": [
      {
        "event_id": "twitch_job_1786727104300000",
        "emitted": true,
        "route": "twitch_text_reply",
        "targets": [
          "twitch_chat"
        ],
        "reason": "",
        "text_digest": "",
        "text_present": false
      }
    ],
    "speech_intents": {
      "active": [],
      "all": [
        {
          "id": "intent_23c0d3235c5150eb9db77c46247f8c22",
          "type": "REACTION",
          "source_event_ids": [],
          "anchor_ids": [
            "ambient:rng_dependency:1786727102"
          ],
          "topic": "rng_dependency",
          "subject_ref": "unknown",
          "value": 0.86,
          "urgency": 0.85,
          "freshness": 0.998333332935969,
          "created_at": 1786727102.2,
          "expires_at": 1786727110.2,
          "interruptibility": "yield_before_tts_commit",
          "minimum_turn_gap": 1.2,
          "maximum_turn_delay": 8.0,
          "scene_relevance": {
            "scene_id": "scene_2f0690b8ded85d18ba449ec4e37abb13",
            "state_version": 1,
            "current_state": "active",
            "terminal": false
          },
          "status": "EMITTED",
          "suppression_reason": "",
          "reserved_at": 1786727104.3,
          "emitted_at": 1786727104.3,
          "contribution_material": {
            "anchor": {
              "id": "ambient:rng_dependency:1786727102",
              "type": "rng_dependency",
              "quality": 0.86,
              "reason": "recent_ambient_context",
              "evidence": {
                "anchor_id": "ambient:rng_dependency:1786727102",
                "anchor_type": "rng_dependency",
                "raw_owner_fragments": [
                  "que ha sido demasiada suerte dios"
                ],
                "exact_supported_claims": [
                  "que ha sido demasiada suerte dios"
                ],
                "timestamps": [
                  1786727102.0
                ],
                "topic_id": "topic_dc448af4c39f",
                "currentness": 0.998333332935969,
                "confidence": 0.86,
                "allowed_contribution_types": [
                  "contextual_reaction",
                  "emotional_banter",
                  "concise_observation"
                ],
                "forbidden_claims": [
                  "unsupported strategy",
                  "save instruction",
                  "unrelated mechanic",
                  "stale topic fusion"
                ],
                "expires_at": 1786727222.0,
                "scene_id": "scene_2f0690b8ded85d18ba449ec4e37abb13",
                "state_version": 1,
                "current_state": "active",
                "terminal": false,
                "extracted_subject": "unknown",
                "extracted_object": "",
                "extracted_predicate": "",
                "supported_claims": [
                  "que ha sido demasiada suerte dios"
                ],
                "unsupported_claims": []
              },
              "scene_guard": {
                "scene_id": "scene_2f0690b8ded85d18ba449ec4e37abb13",
                "state_version": 1,
                "current_state": "active",
                "terminal": false
              }
            },
            "readiness_topic": null
          }
        }
      ],
      "metrics": {
        "intents_created": 1,
        "created:REACTION": 1,
        "pending_due_owner_voice_active": 1,
        "turns_reserved": 1,
        "intents_emitted": 1,
        "emitted:REACTION": 1,
        "pending": 0,
        "time_created_to_emit": {
          "count": 1,
          "p50_ms": 2100.0,
          "p95_ms": 2100.0
        },
        "turn_gap_before_emit": {
          "count": 1,
          "p50_ms": 1300.0,
          "p95_ms": 1300.0
        },
        "intent_creation": {
          "count": 1,
          "p50_ms": 0.073,
          "p95_ms": 0.073
        },
        "pending_queue_operation": {
          "count": 1,
          "p50_ms": 0.073,
          "p95_ms": 0.073
        },
        "turn_arbitration": {
          "count": 3,
          "p50_ms": 0.024,
          "p95_ms": 0.024
        },
        "presence_turn_decision": {
          "count": 3,
          "p50_ms": 1.799,
          "p95_ms": 1.799
        }
      }
    },
    "database_watermarks": {
      "counts": {
        "chat_log": 0,
        "memory_facts": 0,
        "memory_chunks": 0,
        "stream_sessions": 1,
        "stream_chat_messages": 0,
        "stream_events": 1,
        "live_session_timeline": 7,
        "promotion_events": 0,
        "viewer_promotion_profiles": 0,
        "schema_migrations": 7,
        "conversations": 0,
        "open_threads": 0,
        "beliefs": 0,
        "belief_evidence": 0,
        "scene_assertions": 0,
        "game_identities": 0,
        "game_runs": 0,
        "game_run_sessions": 0,
        "game_run_events": 0,
        "game_knowledge_facts": 0,
        "game_knowledge_v2_gaps": 0,
        "people": 0,
        "person_identities": 0,
        "person_sessions": 0,
        "social_episodes": 0,
        "shared_culture_items": 0,
        "shared_culture_evidence": 0,
        "consolidation_runs": 0,
        "consolidation_deltas": 0,
        "action_ledger": 0,
        "temporal_maintenance_audit": 0,
        "learning_observations": 0,
        "scene_transitions": 0,
        "schedule_observations": 1,
        "schedule_hypotheses": 7
      },
      "schema_migrations": [
        {
          "component": "architecture_consolidation",
          "version": 1,
          "name": "audit_hygiene_and_cutover_state",
          "checksum": "944b3ad6532151d4204ed59f663668c9b0ac62df54ece7602d26b91acddd5b7d",
          "applied_at": "2026-08-14T02:35:32.541524+00:00"
        },
        {
          "component": "belief_v2",
          "version": 1,
          "name": "beliefs_evidence_and_compatibility_columns",
          "checksum": "f0df6f1288caccaf6bb47670b38f9cca747f14c84916d6c1ca75beb687200507",
          "applied_at": "2026-08-14T02:35:32.399139+00:00"
        },
        {
          "component": "cognitive_replay",
          "version": 1,
          "name": "replay_metadata",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "applied_at": "2026-08-14T02:35:32.177746+00:00"
        },
        {
          "component": "conversation_continuity",
          "version": 1,
          "name": "conversation_and_open_threads",
          "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
          "applied_at": "2026-08-14T02:35:32.312278+00:00"
        },
        {
          "component": "game_context_v2",
          "version": 1,
          "name": "durable_runs_knowledge_and_gaps",
          "checksum": "08e342acaae00d5d24c1a6dbccad5aee41f753ed90d9ec26282a5fdf042d0a75",
          "applied_at": "2026-08-14T02:35:32.450003+00:00"
        },
        {
          "component": "learning_v2",
          "version": 1,
          "name": "consolidation_temporal_action_and_scene",
          "checksum": "6a86e2d1c7c03167f3b20c328bc97b73fb92e03903f64e64d326fb08f9e3b942",
          "applied_at": "2026-08-14T02:35:32.526410+00:00"
        },
        {
          "component": "social_world_v2",
          "version": 1,
          "name": "people_episodes_and_shared_culture",
          "checksum": "b02adc2cd7f298f1af228dc52c4ba44ae15999fb4c815ea004c40f68b78cbfa5",
          "applied_at": "2026-08-14T02:35:32.488704+00:00"
        }
      ],
      "final_response_digest": "",
      "final_response_present": false
    }
  },
  "second-comment": {
    "runtime": {
      "mode": "active",
      "hebe_sleeping": false,
      "is_running": false,
      "last_input_source": null,
      "last_intent": null,
      "last_firewall": {
        "source": "twitch_system",
        "authority": "system",
        "decision": "allow",
        "reason": "system_event"
      },
      "last_policy": {
        "source": null,
        "authority": null,
        "decision": null,
        "reason": null
      }
    },
    "stream_session": {
      "enabled": true,
      "is_live": true,
      "live_status_known": true,
      "active_stream_session_id": 1,
      "last_transition": "online",
      "title": "Boss attempts",
      "game": "Test RPG",
      "category": "Test RPG"
    },
    "current_scene": {
      "scene_id": "scene_2f0690b8ded85d18ba449ec4e37abb13",
      "topic_id": "topic_dc448af4c39f",
      "entity": "unknown",
      "current_state": "active",
      "state_version": 1,
      "supporting_event_ids": [
        "ambient:rng_dependency:1786727102"
      ],
      "superseded_event_ids": [],
      "terminal": false,
      "updated_at": 1786727102.0
    },
    "pending": {
      "clarification": null,
      "conversation_turn": null
    },
    "conversation": {
      "active": {},
      "latest": {},
      "all": [],
      "last_resolution": {
        "consumed": false,
        "decision": "no_conversation",
        "reason": "no_compatible_active_conversation",
        "conversation_id": "",
        "reply_act": "UNKNOWN",
        "payload": {},
        "conversation": null,
        "latency_ms": 3.095800057053566
      },
      "legacy_pending_projection": {},
      "continuity_shadow_diff": {
        "legacy_result": false,
        "v2_result": false,
        "match": true,
        "difference_reason": ""
      },
      "shadow_metrics": {
        "total": 1,
        "matches": 1,
        "differences": 0,
        "match_rate": 1.0,
        "difference_reasons": {}
      },
      "performance": {
        "count": 1,
        "p50_ms": 3.0958,
        "p95_ms": 3.0958
      }
    },
    "open_threads": [],
    "memory": {
      "facts_count": 0,
      "chunks_count": 0
    },
    "beliefs": {
      "active": [],
      "historical": [],
      "superseded": [],
      "suspected": [],
      "all": [],
      "last_transition": {}
    },
    "belief_evidence": [],
    "retrieval": {
      "last_request": {},
      "performance": {
        "count": 0,
        "p50_ms": 0.0,
        "p95_ms": 0.0
      },
      "write_performance": {
        "count": 0,
        "p50_ms": 0.0,
        "p95_ms": 0.0
      },
      "repository_performance": {
        "belief_lookup": {
          "count": 10,
          "p50_ms": 0.92585,
          "p95_ms": 1.8886
        },
        "evidence_lookup": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "sqlite_write": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        }
      }
    },
    "memory_compatibility": {
      "legacy_to_v2": [],
      "v2_to_legacy": [],
      "shadow_diffs": [],
      "backfill": {
        "safe": 0,
        "compatibility_only": 0,
        "ambiguous": 0,
        "invalid_stale": 0
      }
    },
    "game_state": {
      "game": "Test RPG",
      "platform_version": "",
      "playthrough_type": "casual",
      "spoiler_policy": "no_spoilers",
      "current_location": null,
      "current_character": "",
      "party_members": [],
      "last_confirmed_progress": "",
      "current_objective": null,
      "challenge": "",
      "known_constraints": [],
      "last_updated": 1786727100.0,
      "provenance": "stream_context_sync",
      "confidence": 0.75,
      "current_game": "Test RPG",
      "recent_run_context_facts": [
        {
          "kind": "rng_dependency",
          "text": "Leo framed the current situation as dependent on RNG or luck.",
          "category": "rng_dependency",
          "summary": "Leo framed the current situation as dependent on RNG or luck.",
          "id": "ambient:rng_dependency:1786727102",
          "fact_id": "ambient:rng_dependency:1786727102",
          "raw_text": "<redacted:2320c1febf71>",
          "conservative_normalized_text": "que ha sido demasiada suerte dios",
          "utterance_role": "owner_question_to_stream",
          "timestamp": 1786727102.0,
          "topic_id": "topic_dc448af4c39f",
          "heuristic_category": "rng_dependency",
          "extracted_subject": "unknown",
          "subject": "unknown",
          "extracted_object": "",
          "object": "",
          "extracted_predicate": "",
          "predicate": "",
          "confidence": 0.86,
          "referent_confidence": 0.86,
          "inference_level": "heuristic",
          "supported_claims": [
            "que ha sido demasiada suerte dios"
          ],
          "directly_supported_claims": [
            "que ha sido demasiada suerte dios"
          ],
          "inferred_claims": [],
          "unsupported_claims": [],
          "evidence_span": "que ha sido demasiada suerte dios",
          "evidence_tokens": "<redacted:323fcf3ad517>",
          "semantic_rule": "ambient_category:rng_dependency",
          "model_reason": "Leo framed the current situation as dependent on RNG or luck.",
          "expires_at": 1786727222.0,
          "scene_id": "scene_2f0690b8ded85d18ba449ec4e37abb13",
          "game": null,
          "source": "stt_voice",
          "raw_evidence": "que ha sido demasiada suerte dios",
          "normalized_text": "que ha sido demasiada suerte dios",
          "normalized_evidence": "que ha sido demasiada suerte dios",
          "language": "es",
          "ttl_sec": 120,
          "data": {
            "category": "rng_dependency",
            "raw_text": "<redacted:2320c1febf71>",
            "normalized_text": "que ha sido demasiada suerte dios",
            "mood": "rng tension",
            "extracted_subject": "unknown",
            "extracted_object": "",
            "extracted_predicate": "",
            "inference_level": "heuristic",
            "supported_claims": [
              "que ha sido demasiada suerte dios"
            ],
            "inferred_claims": [],
            "unsupported_claims": [],
            "evidence_span": "que ha sido demasiada suerte dios",
            "evidence_tokens": "<redacted:323fcf3ad517>",
            "semantic_rule": "ambient_category:rng_dependency",
            "model_reason": "Leo framed the current situation as dependent on RNG or luck.",
            "scene_id": "scene_2f0690b8ded85d18ba449ec4e37abb13"
          },
          "age_seconds": 0.0,
          "superseded": false,
          "state_version": 1,
          "current_state": "active",
          "terminal": false,
          "currentness_score": 1.0
        }
      ],
      "identity": {},
      "context": {},
      "active_run": {},
      "runs": [],
      "session_links": [],
      "run_events": [],
      "run_beliefs": {
        "current": [],
        "inferred": [],
        "superseded": []
      },
      "knowledge": {
        "selected": [],
        "rejected": [],
        "spoiler_blocked": [],
        "all": []
      },
      "gaps": [],
      "research": {
        "research_calls": 0,
        "cache_hits": 0,
        "failures": [],
        "context_performance": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "fixture_calls": [],
        "status": ""
      },
      "compatibility": {
        "legacy_progress": [],
        "dossier": [],
        "legacy_run": [],
        "shadow_diffs": [],
        "backfill": {
          "validated": 0,
          "compatibility_only": 0,
          "ambiguous": 0,
          "stale": 0
        }
      },
      "provenance_manifest": [],
      "advice_allowed": null,
      "reaction_allowed": null,
      "performance": {
        "run": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "run_fact": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "knowledge": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "research_gap": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "db_write": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        }
      },
      "context_performance": {
        "count": 0,
        "p50_ms": 0.0,
        "p95_ms": 0.0
      },
      "manifest_size_bytes": 0,
      "last_run_resolution": {},
      "run_resolution_performance": {
        "count": 0,
        "p50_ms": 0.0,
        "p95_ms": 0.0
      }
    },
    "social_state": {
      "recent_active_users": [],
      "recent_chat_count": 0,
      "last_raid": {},
      "last_cheer": {},
      "people": [],
      "identities": [],
      "recent_episodes": [],
      "active_hypotheses": [],
      "historical_hypotheses": [],
      "open_threads": [],
      "relationships": [],
      "shared_culture": {
        "all": [],
        "candidates": [],
        "active": [],
        "weakening": [],
        "retired": [],
        "reactions": [],
        "selection": {
          "selected": [],
          "rejected": []
        }
      },
      "retrieval": {},
      "opportunities": [],
      "resolution": {},
      "rejected_writes": [],
      "compatibility": {
        "chatter_presence": [],
        "chatter_profiles": [],
        "chatter_facts": [],
        "stream_chatter_summaries": [],
        "viewer_profiles": [],
        "social_events": [],
        "promotion_profiles": [],
        "backfill": {
          "explicit_observation": 0,
          "safe_episode": 0,
          "inferred_compatibility_only": 0,
          "ambiguous": 0,
          "sensitive": 0,
          "stale": 0
        },
        "shadow_diffs": []
      },
      "performance": {
        "identity": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "episode_write": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "thread_lookup": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "culture_select": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "context": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "db_write": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        }
      },
      "belief_lookup_performance": {
        "belief_lookup": {
          "count": 10,
          "p50_ms": 0.92585,
          "p95_ms": 1.8886
        },
        "evidence_lookup": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "sqlite_write": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        }
      }
    },
    "learning": {
      "consolidation_runs": [],
      "deltas": [],
      "rejected_deltas": [],
      "watermarks": [],
      "last_result": {},
      "stable_core_version": "a1a58e51882b0c88",
      "performance": {
        "repository": {
          "lookup": {
            "count": 50,
            "p50_ms": 0.8782,
            "p95_ms": 1.4359
          },
          "write": {
            "count": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0
          },
          "context": {
            "count": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0
          }
        },
        "consolidation": {
          "consolidation_duration": {
            "count": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0
          },
          "candidate_validation": {
            "count": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0
          }
        },
        "temporal": {
          "temporal_maintenance": {
            "count": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0
          }
        },
        "action_history": {
          "action_ledger_lookup": {
            "count": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0
          }
        },
        "owner_preferences": {
          "owner_preference_lookup": {
            "count": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0
          }
        },
        "hebe_self": {
          "hebe_self_lookup": {
            "count": 1,
            "p50_ms": 1.8985,
            "p95_ms": 1.8985
          }
        },
        "context": {
          "continuity_context_build": {
            "count": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0
          }
        }
      }
    },
    "self_model": {
      "stable_core_version": "a1a58e51882b0c88",
      "evolving_preferences": [],
      "opinions": [],
      "superseded_opinions": []
    },
    "owner_preferences": [],
    "leo_language": {
      "beliefs": [],
      "interpretation_aliases": {}
    },
    "temporal": {
      "expired": [],
      "archived": [],
      "weakened": [],
      "maintenance_actions": [],
      "last_actions": []
    },
    "schedule": {
      "observations": [
        {
          "id": 1,
          "stream_session_id": "1",
          "weekday": "friday",
          "time_window": "night",
          "canonical_content": "Test RPG",
          "content_key": "test rpg",
          "stream_format": "game_playthrough",
          "source": "observed",
          "observed_at": "2026-08-14T04:35:32.628934+02:00"
        }
      ],
      "hypotheses": [
        {
          "id": 1,
          "weekday": "monday",
          "time_window": "any",
          "canonical_content": "FINAL FANTASY IX",
          "content_key": "final fantasy ix",
          "stream_format": "challenge_run",
          "source": "owner_declared",
          "confidence": 0.9,
          "evidence_count": 1,
          "consecutive_matches": 1,
          "consecutive_misses": 0,
          "first_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "last_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "status": "active",
          "superseded_by": null
        },
        {
          "id": 2,
          "weekday": "tuesday",
          "time_window": "any",
          "canonical_content": "Persona 5 Royal",
          "content_key": "persona 5 royal",
          "stream_format": "game_playthrough",
          "source": "owner_declared",
          "confidence": 0.9,
          "evidence_count": 1,
          "consecutive_matches": 1,
          "consecutive_misses": 0,
          "first_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "last_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "status": "active",
          "superseded_by": null
        },
        {
          "id": 3,
          "weekday": "wednesday",
          "time_window": "any",
          "canonical_content": "Persona 5 Royal",
          "content_key": "persona 5 royal",
          "stream_format": "game_playthrough",
          "source": "owner_declared",
          "confidence": 0.9,
          "evidence_count": 1,
          "consecutive_matches": 1,
          "consecutive_misses": 0,
          "first_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "last_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "status": "active",
          "superseded_by": null
        },
        {
          "id": 4,
          "weekday": "thursday",
          "time_window": "any",
          "canonical_content": "Baldur's Gate 3",
          "content_key": "baldur s gate 3",
          "stream_format": "game_playthrough",
          "source": "owner_declared",
          "confidence": 0.9,
          "evidence_count": 1,
          "consecutive_matches": 1,
          "consecutive_misses": 0,
          "first_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "last_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "status": "active",
          "superseded_by": null
        },
        {
          "id": 6,
          "weekday": "saturday",
          "time_window": "any",
          "canonical_content": "Retro Weekend",
          "content_key": "retro weekend",
          "stream_format": "retro",
          "source": "owner_declared",
          "confidence": 0.9,
          "evidence_count": 1,
          "consecutive_matches": 1,
          "consecutive_misses": 0,
          "first_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "last_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "status": "active",
          "superseded_by": null
        },
        {
          "id": 7,
          "weekday": "sunday",
          "time_window": "any",
          "canonical_content": "Retro Weekend",
          "content_key": "retro weekend",
          "stream_format": "retro",
          "source": "owner_declared",
          "confidence": 0.9,
          "evidence_count": 1,
          "consecutive_matches": 1,
          "consecutive_misses": 0,
          "first_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "last_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "status": "active",
          "superseded_by": null
        },
        {
          "id": 5,
          "weekday": "friday",
          "time_window": "any",
          "canonical_content": "Baldur's Gate 3",
          "content_key": "baldur s gate 3",
          "stream_format": "game_playthrough",
          "source": "owner_declared",
          "confidence": 0.76,
          "evidence_count": 1,
          "consecutive_matches": 0,
          "consecutive_misses": 1,
          "first_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "last_observed_at": "2026-08-14T04:35:32.628934+02:00",
          "status": "weakening",
          "superseded_by": null
        }
      ],
      "observed_current_state": {
        "game": "Test RPG",
        "title": "Boss attempts"
      },
      "precedence": "observed_twitch_metadata"
    },
    "action_ledger": {
      "entries": [],
      "last_claim_validation": {}
    },
    "scene_transitions": {
      "all": [],
      "last": {}
    },
    "continuity_context": {},
    "promotion_profiles": [],
    "actions": {
      "attempts": [
        {
          "operation": "twitch.send_message",
          "payload": {
            "text": "Uf, qué tensión."
          },
          "outcome": {
            "success": true,
            "status": "sent"
          }
        }
      ],
      "speech_requests": [],
      "model_calls": [
        {
          "key": "promotion_clarification:v1:none",
          "method": "chat"
        }
      ],
      "research_calls": []
    },
    "receipts": [],
    "emitted_outputs": [
      {
        "event_id": "twitch_job_1786727104300000",
        "emitted": true,
        "route": "twitch_text_reply",
        "targets": [
          "twitch_chat"
        ],
        "reason": "",
        "text_digest": "",
        "text_present": false
      }
    ],
    "final_emission_results": [
      {
        "event_id": "twitch_job_1786727104300000",
        "emitted": true,
        "route": "twitch_text_reply",
        "targets": [
          "twitch_chat"
        ],
        "reason": "",
        "text_digest": "",
        "text_present": false
      }
    ],
    "speech_intents": {
      "active": [],
      "all": [
        {
          "id": "intent_23c0d3235c5150eb9db77c46247f8c22",
          "type": "REACTION",
          "source_event_ids": [],
          "anchor_ids": [
            "ambient:rng_dependency:1786727102"
          ],
          "topic": "rng_dependency",
          "subject_ref": "unknown",
          "value": 0.86,
          "urgency": 0.85,
          "freshness": 0.998333332935969,
          "created_at": 1786727102.2,
          "expires_at": 1786727110.2,
          "interruptibility": "yield_before_tts_commit",
          "minimum_turn_gap": 1.2,
          "maximum_turn_delay": 8.0,
          "scene_relevance": {
            "scene_id": "scene_2f0690b8ded85d18ba449ec4e37abb13",
            "state_version": 1,
            "current_state": "active",
            "terminal": false
          },
          "status": "EMITTED",
          "suppression_reason": "",
          "reserved_at": 1786727104.3,
          "emitted_at": 1786727104.3,
          "contribution_material": {
            "anchor": {
              "id": "ambient:rng_dependency:1786727102",
              "type": "rng_dependency",
              "quality": 0.86,
              "reason": "recent_ambient_context",
              "evidence": {
                "anchor_id": "ambient:rng_dependency:1786727102",
                "anchor_type": "rng_dependency",
                "raw_owner_fragments": [
                  "que ha sido demasiada suerte dios"
                ],
                "exact_supported_claims": [
                  "que ha sido demasiada suerte dios"
                ],
                "timestamps": [
                  1786727102.0
                ],
                "topic_id": "topic_dc448af4c39f",
                "currentness": 0.998333332935969,
                "confidence": 0.86,
                "allowed_contribution_types": [
                  "contextual_reaction",
                  "emotional_banter",
                  "concise_observation"
                ],
                "forbidden_claims": [
                  "unsupported strategy",
                  "save instruction",
                  "unrelated mechanic",
                  "stale topic fusion"
                ],
                "expires_at": 1786727222.0,
                "scene_id": "scene_2f0690b8ded85d18ba449ec4e37abb13",
                "state_version": 1,
                "current_state": "active",
                "terminal": false,
                "extracted_subject": "unknown",
                "extracted_object": "",
                "extracted_predicate": "",
                "supported_claims": [
                  "que ha sido demasiada suerte dios"
                ],
                "unsupported_claims": []
              },
              "scene_guard": {
                "scene_id": "scene_2f0690b8ded85d18ba449ec4e37abb13",
                "state_version": 1,
                "current_state": "active",
                "terminal": false
              }
            },
            "readiness_topic": null
          }
        }
      ],
      "metrics": {
        "intents_created": 1,
        "created:REACTION": 1,
        "pending_due_owner_voice_active": 1,
        "turns_reserved": 1,
        "intents_emitted": 1,
        "emitted:REACTION": 1,
        "pending": 0,
        "time_created_to_emit": {
          "count": 1,
          "p50_ms": 2100.0,
          "p95_ms": 2100.0
        },
        "turn_gap_before_emit": {
          "count": 1,
          "p50_ms": 1300.0,
          "p95_ms": 1300.0
        },
        "intent_creation": {
          "count": 1,
          "p50_ms": 0.073,
          "p95_ms": 0.073
        },
        "pending_queue_operation": {
          "count": 1,
          "p50_ms": 0.073,
          "p95_ms": 0.073
        },
        "turn_arbitration": {
          "count": 3,
          "p50_ms": 0.024,
          "p95_ms": 0.024
        },
        "presence_turn_decision": {
          "count": 3,
          "p50_ms": 1.799,
          "p95_ms": 1.799
        }
      }
    },
    "database_watermarks": {
      "counts": {
        "chat_log": 0,
        "memory_facts": 0,
        "memory_chunks": 0,
        "stream_sessions": 1,
        "stream_chat_messages": 0,
        "stream_events": 1,
        "live_session_timeline": 7,
        "promotion_events": 0,
        "viewer_promotion_profiles": 0,
        "schema_migrations": 7,
        "conversations": 0,
        "open_threads": 0,
        "beliefs": 0,
        "belief_evidence": 0,
        "scene_assertions": 0,
        "game_identities": 0,
        "game_runs": 0,
        "game_run_sessions": 0,
        "game_run_events": 0,
        "game_knowledge_facts": 0,
        "game_knowledge_v2_gaps": 0,
        "people": 0,
        "person_identities": 0,
        "person_sessions": 0,
        "social_episodes": 0,
        "shared_culture_items": 0,
        "shared_culture_evidence": 0,
        "consolidation_runs": 0,
        "consolidation_deltas": 0,
        "action_ledger": 0,
        "temporal_maintenance_audit": 0,
        "learning_observations": 0,
        "scene_transitions": 0,
        "schedule_observations": 1,
        "schedule_hypotheses": 7
      },
      "schema_migrations": [
        {
          "component": "architecture_consolidation",
          "version": 1,
          "name": "audit_hygiene_and_cutover_state",
          "checksum": "944b3ad6532151d4204ed59f663668c9b0ac62df54ece7602d26b91acddd5b7d",
          "applied_at": "2026-08-14T02:35:32.541524+00:00"
        },
        {
          "component": "belief_v2",
          "version": 1,
          "name": "beliefs_evidence_and_compatibility_columns",
          "checksum": "f0df6f1288caccaf6bb47670b38f9cca747f14c84916d6c1ca75beb687200507",
          "applied_at": "2026-08-14T02:35:32.399139+00:00"
        },
        {
          "component": "cognitive_replay",
          "version": 1,
          "name": "replay_metadata",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "applied_at": "2026-08-14T02:35:32.177746+00:00"
        },
        {
          "component": "conversation_continuity",
          "version": 1,
          "name": "conversation_and_open_threads",
          "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
          "applied_at": "2026-08-14T02:35:32.312278+00:00"
        },
        {
          "component": "game_context_v2",
          "version": 1,
          "name": "durable_runs_knowledge_and_gaps",
          "checksum": "08e342acaae00d5d24c1a6dbccad5aee41f753ed90d9ec26282a5fdf042d0a75",
          "applied_at": "2026-08-14T02:35:32.450003+00:00"
        },
        {
          "component": "learning_v2",
          "version": 1,
          "name": "consolidation_temporal_action_and_scene",
          "checksum": "6a86e2d1c7c03167f3b20c328bc97b73fb92e03903f64e64d326fb08f9e3b942",
          "applied_at": "2026-08-14T02:35:32.526410+00:00"
        },
        {
          "component": "social_world_v2",
          "version": 1,
          "name": "people_episodes_and_shared_culture",
          "checksum": "b02adc2cd7f298f1af228dc52c4ba44ae15999fb4c815ea004c40f68b78cbfa5",
          "applied_at": "2026-08-14T02:35:32.488704+00:00"
        }
      ],
      "final_response_digest": "",
      "final_response_present": false
    }
  },
  "yield-second": {
    "runtime": {
      "mode": "active",
      "hebe_sleeping": false,
      "is_running": false,
      "last_input_source": null,
      "last_intent": null,
      "last_firewall": {
        "source": "twitch_system",
        "authority": "system",
        "decision": "allow",
        "reason": "system_event"
      },
      "last_policy": {
        "source": null,
        "authority": null,
        "decision": null,
        "reason": null
      }
    },
    "stream_session": {
      "enabled": true,
      "is_live": true,
      "live_status_known": true,
      "active_stream_session_id": 1,
      "last_transition": "online",
      "title": "Boss attempts",
      "game": "Test RPG",
      "category": "Test RPG"
    },
    "current_scene": {
      "scene_id": "scene_2f0690b8ded85d18ba449ec4e37abb13",
      "topic_id": "topic_dc448af4c39f",
      "entity": "unknown",
      "current_state": "active",
      "state_version": 1,
      "supporting_event_ids": [
        "ambient:rng_dependency:1786727102"
      ],
      "superseded_event_ids": [],
      "terminal": false,
      "updated_at": 1786727102.0
    },
    "pending": {
      "clarification": null,
      "conversation_turn": null
    },
    "conversation": {
      "active": {},
      "latest": {},
      "all": [],
      "last_resolution": {
        "consumed": false,
        "decision": "no_conversation",
        "reason": "no_compatible_active_conversation",
        "conversation_id": "",
        "reply_act": "UNKNOWN",
        "payload": {},
        "conversation": null,
        "latency_ms": 3.095800057053566
      },
      "legacy_pending_projection": {},
      "continuity_shadow_diff": {
        "legacy_result": false,
        "v2_result": false,
        "match": true,
        "difference_reason": ""
      },
      "shadow_metrics": {
        "total": 1,
        "matches": 1,
        "differences": 0,
        "match_rate": 1.0,
        "difference_reasons": {}
      },
      "performance": {
        "count": 1,
        "p50_ms": 3.0958,
        "p95_ms": 3.0958
      }
    },
    "open_threads": [],
    "memory": {
      "facts_count": 0,
      "chunks_count": 0
    },
    "beliefs": {
      "active": [],
      "historical": [],
      "superseded": [],
      "suspected": [],
      "all": [],
      "last_transition": {}
    },
    "belief_evidence": [],
    "retrieval": {
      "last_request": {},
      "performance": {
        "count": 0,
        "p50_ms": 0.0,
        "p95_ms": 0.0
      },
      "write_performance": {
        "count": 0,
        "p50_ms": 0.0,
        "p95_ms": 0.0
      },
      "repository_performance": {
        "belief_lookup": {
          "count": 11,
          "p50_ms": 0.8998,
          "p95_ms": 1.8886
        },
        "evidence_lookup": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "sqlite_write": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        }
      }
    },
    "memory_compatibility": {
      "legacy_to_v2": [],
      "v2_to_legacy": [],
      "shadow_diffs": [],
      "backfill": {
        "safe": 0,
        "compatibility_only": 0,
        "ambiguous": 0,
        "invalid_stale": 0
      }
    },
    "game_state": {
      "game": "Test RPG",
      "platform_version": "",
      "playthrough_type": "casual",
      "spoiler_policy": "no_spoilers",
      "current_location": null,
      "current_character": "",
      "party_members": [],
      "last_confirmed_progress": "",
      "current_objective": null,
      "challenge": "",
      "known_constraints": [],
      "last_updated": 1786727100.0,
      "provenance": "stream_context_sync",
      "confidence": 0.75,
      "current_game": "Test RPG",
      "recent_run_context_facts": [
        {
          "kind": "rng_dependency",
          "text": "Leo framed the current situation as dependent on RNG or luck.",
          "category": "rng_dependency",
          "summary": "Leo framed the current situation as dependent on RNG or luck.",
          "id": "ambient:rng_dependency:1786727102",
          "fact_id": "ambient:rng_dependency:1786727102",
          "raw_text": "<redacted:2320c1febf71>",
          "conservative_normalized_text": "que ha sido demasiada suerte dios",
          "utterance_role": "owner_question_to_stream",
          "timestamp": 1786727102.0,
          "topic_id": "topic_dc448af4c39f",
          "heuristic_category": "rng_dependency",
          "extracted_subject": "unknown",
          "subject": "unknown",
          "extracted_object": "",
          "object": "",
          "extracted_predicate": "",
          "predicate": "",
          "confidence": 0.86,
          "referent_confidence": 0.86,
          "inference_level": "heuristic",
          "supported_claims": [
            "que ha sido demasiada suerte dios"
          ],
          "directly_supported_claims": [
            "que ha sido demasiada suerte dios"
          ],
          "inferred_claims": [],
          "unsupported_claims": [],
          "evidence_span": "que ha sido demasiada suerte dios",
          "evidence_tokens": "<redacted:323fcf3ad517>",
          "semantic_rule": "ambient_category:rng_dependency",
          "model_reason": "Leo framed the current situation as dependent on RNG or luck.",
          "expires_at": 1786727222.0,
          "scene_id": "scene_2f0690b8ded85d18ba449ec4e37abb13",
          "game": null,
          "source": "stt_voice",
          "raw_evidence": "que ha sido demasiada suerte dios",
          "normalized_text": "que ha sido demasiada suerte dios",
          "normalized_evidence": "que ha sido demasiada suerte dios",
          "language": "es",
          "ttl_sec": 120,
          "data": {
            "category": "rng_dependency",
            "raw_text": "<redacted:2320c1febf71>",
            "normalized_text": "que ha sido demasiada suerte dios",
            "mood": "rng tension",
            "extracted_subject": "unknown",
            "extracted_object": "",
            "extracted_predicate": "",
            "inference_level": "heuristic",
            "supported_claims": [
              "que ha sido demasiada suerte dios"
            ],
            "inferred_claims": [],
            "unsupported_claims": [],
            "evidence_span": "que ha sido demasiada suerte dios",
            "evidence_tokens": "<redacted:323fcf3ad517>",
            "semantic_rule": "ambient_category:rng_dependency",
            "model_reason": "Leo framed the current situation as dependent on RNG or luck.",
            "scene_id": "scene_2f0690b8ded85d18ba449ec4e37abb13"
          },
          "age_seconds": 0.0,
          "superseded": false,
          "state_version": 1,
          "current_state": "active",
          "terminal": false,
          "currentness_score": 1.0
        }
      ],
      "identity": {},
      "context": {},
      "active_run": {},
      "runs": [],
      "session_links": [],
      "run_events": [],
      "run_beliefs": {
        "current": [],
        "inferred": [],
        "superseded": []
      },
      "knowledge": {
        "selected": [],
        "rejected": [],
        "spoiler_blocked": [],
        "all": []
      },
      "gaps": [],
      "research": {
        "research_calls": 0,
        "cache_hits": 0,
        "failures": [],
        "context_performance": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "fixture_calls": [],
        "status": ""
      },
      "compatibility": {
        "legacy_progress": [],
        "dossier": [],
        "legacy_run": [],
        "shadow_diffs": [],
        "backfill": {
          "validated": 0,
          "compatibility_only": 0,
          "ambiguous": 0,
          "stale": 0
        }
      },
      "provenance_manifest": [],
      "advice_allowed": null,
      "reaction_allowed": null,
      "performance": {
        "run": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "run_fact": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "knowledge": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "research_gap": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "db_write": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        }
      },
      "context_performance": {
        "count": 0,
        "p50_ms": 0.0,
        "p95_ms": 0.0
      },
      "manifest_size_bytes": 0,
      "last_run_resolution": {},
      "run_resolution_performance": {
        "count": 0,
        "p50_ms": 0.0,
        "p95_ms": 0.0
      }
    },
    "social_state": {
      "recent_active_users": [],
      "recent_chat_count": 0,
      "last_raid": {},
      "last_cheer": {},
      "people": [],
      "identities": [],
      "recent_episodes": [],
      "active_hypotheses": [],
      "historical_hypotheses": [],
      "open_threads": [],
      "relationships": [],
      "shared_culture": {
        "all": [],
        "candidates": [],
        "active": [],
        "weakening": [],
        "retired": [],
        "reactions": [],
        "selection": {
          "selected": [],
          "rejected": []
        }
      },
      "retrieval": {},
      "opportunities": [],
      "resolution": {},
      "rejected_writes": [],
      "compatibility": {
        "chatter_presence": [],
        "chatter_profiles": [],
        "chatter_facts": [],
        "stream_chatter_summaries": [],
        "viewer_profiles": [],
        "social_events": [],
        "promotion_profiles": [],
        "backfill": {
          "explicit_observation": 0,
          "safe_episode": 0,
          "inferred_compatibility_only": 0,
          "ambiguous": 0,
          "sensitive": 0,
          "stale": 0
        },
        "shadow_diffs": []
      },
      "performance": {
        "identity": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "episode_write": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "thread_lookup": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "culture_select": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "context": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "db_write": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        }
      },
      "belief_lookup_performance": {
        "belief_lookup": {
          "count": 11,
          "p50_ms": 0.8998,
          "p95_ms": 1.8886
        },
        "evidence_lookup": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "sqlite_write": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        }
      }
    },
    "learning": {
      "consolidation_runs": [],
      "deltas": [],
      "rejected_deltas": [],
      "watermarks": [],
      "last_result": {},
      "stable_core_version": "a1a58e51882b0c88",
      "performance": {
        "repository": {
          "lookup": {
            "count": 55,
            "p50_ms": 0.8703,
            "p95_ms": 1.4359
          },
          "write": {
            "count": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0
          },
          "context": {
            "count": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0
          }
        },
        "consolidation": {
          "consolidation_duration": {
            "count": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0
          },
          "candidate_validation": {
            "count": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0
          }
        },
        "temporal": {
          "temporal_maintenance": {
            "count": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0
          }
        },
        "action_history": {
          "action_ledger_lookup": {
            "count": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0
          }
        },
        "owner_preferences": {
          "owner_preference_lookup": {
            "count": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0
          }
        },
        "hebe_self": {
          "hebe_self_lookup": {
            "count": 1,
            "p50_ms": 1.8985,
            "p95_ms": 1.8985
          }
        },
        "context": {
          "continuity_context_build": {
            "count": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0
          }
        }
      }
    },
    "self_model": {
      "stable_core_version": "a1a58e51882b0c88",
      "evolving_preferences": [],
      "opinions": [],
      "superseded_opinions": []
    },
    "owner_preferences": [],
    "leo_language": {
      "beliefs": [],
      "interpretation_aliases": {}
    },
    "temporal": {
      "expired": [],
      "archived": [],
      "weakened": [],
      "maintenance_actions": [],
      "last_actions": []
    },
    "schedule": {
      "observations": [
        {
          "id": 1,
          "stream_session_id": "1",
          "weekday": "friday",
          "time_window": "night",
          "canonical_content": "Test RPG",
          "content_key": "test rpg",
          "stream_format": "game_playthrough",
          "source": "observed",
          "observed_at": "2026-08-14T04:35:32.628934+02:00"
        }
      ],
      "hypotheses": [
        {
          "id": 1,
          "weekday": "monday",
          "time_window": "any",
          "canonical_content": "FINAL FANTASY IX",
          "content_key": "final fantasy ix",
          "stream_format": "challenge_run",
          "source": "owner_declared",
          "confidence": 0.9,
          "evidence_count": 1,
          "consecutive_matches": 1,
          "consecutive_misses": 0,
          "first_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "last_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "status": "active",
          "superseded_by": null
        },
        {
          "id": 2,
          "weekday": "tuesday",
          "time_window": "any",
          "canonical_content": "Persona 5 Royal",
          "content_key": "persona 5 royal",
          "stream_format": "game_playthrough",
          "source": "owner_declared",
          "confidence": 0.9,
          "evidence_count": 1,
          "consecutive_matches": 1,
          "consecutive_misses": 0,
          "first_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "last_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "status": "active",
          "superseded_by": null
        },
        {
          "id": 3,
          "weekday": "wednesday",
          "time_window": "any",
          "canonical_content": "Persona 5 Royal",
          "content_key": "persona 5 royal",
          "stream_format": "game_playthrough",
          "source": "owner_declared",
          "confidence": 0.9,
          "evidence_count": 1,
          "consecutive_matches": 1,
          "consecutive_misses": 0,
          "first_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "last_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "status": "active",
          "superseded_by": null
        },
        {
          "id": 4,
          "weekday": "thursday",
          "time_window": "any",
          "canonical_content": "Baldur's Gate 3",
          "content_key": "baldur s gate 3",
          "stream_format": "game_playthrough",
          "source": "owner_declared",
          "confidence": 0.9,
          "evidence_count": 1,
          "consecutive_matches": 1,
          "consecutive_misses": 0,
          "first_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "last_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "status": "active",
          "superseded_by": null
        },
        {
          "id": 6,
          "weekday": "saturday",
          "time_window": "any",
          "canonical_content": "Retro Weekend",
          "content_key": "retro weekend",
          "stream_format": "retro",
          "source": "owner_declared",
          "confidence": 0.9,
          "evidence_count": 1,
          "consecutive_matches": 1,
          "consecutive_misses": 0,
          "first_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "last_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "status": "active",
          "superseded_by": null
        },
        {
          "id": 7,
          "weekday": "sunday",
          "time_window": "any",
          "canonical_content": "Retro Weekend",
          "content_key": "retro weekend",
          "stream_format": "retro",
          "source": "owner_declared",
          "confidence": 0.9,
          "evidence_count": 1,
          "consecutive_matches": 1,
          "consecutive_misses": 0,
          "first_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "last_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "status": "active",
          "superseded_by": null
        },
        {
          "id": 5,
          "weekday": "friday",
          "time_window": "any",
          "canonical_content": "Baldur's Gate 3",
          "content_key": "baldur s gate 3",
          "stream_format": "game_playthrough",
          "source": "owner_declared",
          "confidence": 0.76,
          "evidence_count": 1,
          "consecutive_matches": 0,
          "consecutive_misses": 1,
          "first_observed_at": "2026-08-14T04:35:32.625425+02:00",
          "last_observed_at": "2026-08-14T04:35:32.628934+02:00",
          "status": "weakening",
          "superseded_by": null
        }
      ],
      "observed_current_state": {
        "game": "Test RPG",
        "title": "Boss attempts"
      },
      "precedence": "observed_twitch_metadata"
    },
    "action_ledger": {
      "entries": [],
      "last_claim_validation": {}
    },
    "scene_transitions": {
      "all": [],
      "last": {}
    },
    "continuity_context": {},
    "promotion_profiles": [],
    "actions": {
      "attempts": [
        {
          "operation": "twitch.send_message",
          "payload": {
            "text": "Uf, qué tensión."
          },
          "outcome": {
            "success": true,
            "status": "sent"
          }
        }
      ],
      "speech_requests": [],
      "model_calls": [
        {
          "key": "promotion_clarification:v1:none",
          "method": "chat"
        }
      ],
      "research_calls": []
    },
    "receipts": [],
    "emitted_outputs": [
      {
        "event_id": "twitch_job_1786727104300000",
        "emitted": true,
        "route": "twitch_text_reply",
        "targets": [
          "twitch_chat"
        ],
        "reason": "",
        "text_digest": "",
        "text_present": false
      }
    ],
    "final_emission_results": [
      {
        "event_id": "twitch_job_1786727104300000",
        "emitted": true,
        "route": "twitch_text_reply",
        "targets": [
          "twitch_chat"
        ],
        "reason": "",
        "text_digest": "",
        "text_present": false
      }
    ],
    "speech_intents": {
      "active": [
        {
          "id": "intent_1c5ea004e10154bc90ef7fce1d7896b8",
          "type": "GAME_COMMENT",
          "source_event_ids": [],
          "anchor_ids": [],
          "topic": "game,title,title_markers,run_context,recent_voice_event",
          "subject_ref": "",
          "value": 0.58,
          "urgency": 0.55,
          "freshness": 1.0,
          "created_at": 1786727105.7,
          "expires_at": 1786727119.7,
          "interruptibility": "yield_before_tts_commit",
          "minimum_turn_gap": 1.8,
          "maximum_turn_delay": 14.0,
          "scene_relevance": {},
          "status": "PENDING",
          "suppression_reason": "",
          "reserved_at": 0.0,
          "emitted_at": 0.0,
          "contribution_material": {
            "anchor": {
              "id": "",
              "type": "game,title,title_markers,run_context,recent_voice_event",
              "quality": 0.58,
              "reason": "stream_context"
            },
            "readiness_topic": null
          }
        },
        {
          "id": "intent_a1ac9c1e5e345442b692b3d6c1119607",
          "type": "OPINION",
          "source_event_ids": [
            "second-comment"
          ],
          "anchor_ids": [],
          "topic": "boss-design",
          "subject_ref": "",
          "value": 0.73,
          "urgency": 0.3,
          "freshness": 1.0,
          "created_at": 1786727105.7,
          "expires_at": 1786727130.7,
          "interruptibility": "yield_before_tts_commit",
          "minimum_turn_gap": 2.5,
          "maximum_turn_delay": 25.0,
          "scene_relevance": {
            "scene_id": "scene_2f0690b8ded85d18ba449ec4e37abb13",
            "topic_id": "topic_dc448af4c39f",
            "entity": "unknown",
            "current_state": "active",
            "state_version": 1,
            "supporting_event_ids": [
              "ambient:rng_dependency:1786727102"
            ],
            "superseded_event_ids": [],
            "terminal": false,
            "updated_at": 1786727102.0
          },
          "status": "PENDING",
          "suppression_reason": "",
          "reserved_at": 0.0,
          "emitted_at": 0.0,
          "contribution_material": {
            "cognitive_candidate": true
          }
        }
      ],
      "all": [
        {
          "id": "intent_23c0d3235c5150eb9db77c46247f8c22",
          "type": "REACTION",
          "source_event_ids": [],
          "anchor_ids": [
            "ambient:rng_dependency:1786727102"
          ],
          "topic": "rng_dependency",
          "subject_ref": "unknown",
          "value": 0.86,
          "urgency": 0.85,
          "freshness": 0.998333332935969,
          "created_at": 1786727102.2,
          "expires_at": 1786727110.2,
          "interruptibility": "yield_before_tts_commit",
          "minimum_turn_gap": 1.2,
          "maximum_turn_delay": 8.0,
          "scene_relevance": {
            "scene_id": "scene_2f0690b8ded85d18ba449ec4e37abb13",
            "state_version": 1,
            "current_state": "active",
            "terminal": false
          },
          "status": "EMITTED",
          "suppression_reason": "",
          "reserved_at": 1786727104.3,
          "emitted_at": 1786727104.3,
          "contribution_material": {
            "anchor": {
              "id": "ambient:rng_dependency:1786727102",
              "type": "rng_dependency",
              "quality": 0.86,
              "reason": "recent_ambient_context",
              "evidence": {
                "anchor_id": "ambient:rng_dependency:1786727102",
                "anchor_type": "rng_dependency",
                "raw_owner_fragments": [
                  "que ha sido demasiada suerte dios"
                ],
                "exact_supported_claims": [
                  "que ha sido demasiada suerte dios"
                ],
                "timestamps": [
                  1786727102.0
                ],
                "topic_id": "topic_dc448af4c39f",
                "currentness": 0.998333332935969,
                "confidence": 0.86,
                "allowed_contribution_types": [
                  "contextual_reaction",
                  "emotional_banter",
                  "concise_observation"
                ],
                "forbidden_claims": [
                  "unsupported strategy",
                  "save instruction",
                  "unrelated mechanic",
                  "stale topic fusion"
                ],
                "expires_at": 1786727222.0,
                "scene_id": "scene_2f0690b8ded85d18ba449ec4e37abb13",
                "state_version": 1,
                "current_state": "active",
                "terminal": false,
                "extracted_subject": "unknown",
                "extracted_object": "",
                "extracted_predicate": "",
                "supported_claims": [
                  "que ha sido demasiada suerte dios"
                ],
                "unsupported_claims": []
              },
              "scene_guard": {
                "scene_id": "scene_2f0690b8ded85d18ba449ec4e37abb13",
                "state_version": 1,
                "current_state": "active",
                "terminal": false
              }
            },
            "readiness_topic": null
          }
        },
        {
          "id": "intent_1c5ea004e10154bc90ef7fce1d7896b8",
          "type": "GAME_COMMENT",
          "source_event_ids": [],
          "anchor_ids": [],
          "topic": "game,title,title_markers,run_context,recent_voice_event",
          "subject_ref": "",
          "value": 0.58,
          "urgency": 0.55,
          "freshness": 1.0,
          "created_at": 1786727105.7,
          "expires_at": 1786727119.7,
          "interruptibility": "yield_before_tts_commit",
          "minimum_turn_gap": 1.8,
          "maximum_turn_delay": 14.0,
          "scene_relevance": {},
          "status": "PENDING",
          "suppression_reason": "",
          "reserved_at": 0.0,
          "emitted_at": 0.0,
          "contribution_material": {
            "anchor": {
              "id": "",
              "type": "game,title,title_markers,run_context,recent_voice_event",
              "quality": 0.58,
              "reason": "stream_context"
            },
            "readiness_topic": null
          }
        },
        {
          "id": "intent_a1ac9c1e5e345442b692b3d6c1119607",
          "type": "OPINION",
          "source_event_ids": [
            "second-comment"
          ],
          "anchor_ids": [],
          "topic": "boss-design",
          "subject_ref": "",
          "value": 0.73,
          "urgency": 0.3,
          "freshness": 1.0,
          "created_at": 1786727105.7,
          "expires_at": 1786727130.7,
          "interruptibility": "yield_before_tts_commit",
          "minimum_turn_gap": 2.5,
          "maximum_turn_delay": 25.0,
          "scene_relevance": {
            "scene_id": "scene_2f0690b8ded85d18ba449ec4e37abb13",
            "topic_id": "topic_dc448af4c39f",
            "entity": "unknown",
            "current_state": "active",
            "state_version": 1,
            "supporting_event_ids": [
              "ambient:rng_dependency:1786727102"
            ],
            "superseded_event_ids": [],
            "terminal": false,
            "updated_at": 1786727102.0
          },
          "status": "PENDING",
          "suppression_reason": "",
          "reserved_at": 0.0,
          "emitted_at": 0.0,
          "contribution_material": {
            "cognitive_candidate": true
          }
        }
      ],
      "metrics": {
        "intents_created": 3,
        "created:REACTION": 1,
        "pending_due_owner_voice_active": 3,
        "turns_reserved": 1,
        "intents_emitted": 1,
        "emitted:REACTION": 1,
        "created:GAME_COMMENT": 1,
        "created:OPINION": 1,
        "pending": 2,
        "time_created_to_emit": {
          "count": 1,
          "p50_ms": 2100.0,
          "p95_ms": 2100.0
        },
        "turn_gap_before_emit": {
          "count": 1,
          "p50_ms": 1300.0,
          "p95_ms": 1300.0
        },
        "intent_creation": {
          "count": 3,
          "p50_ms": 0.057,
          "p95_ms": 0.057
        },
        "pending_queue_operation": {
          "count": 3,
          "p50_ms": 0.057,
          "p95_ms": 0.057
        },
        "turn_arbitration": {
          "count": 4,
          "p50_ms": 0.02,
          "p95_ms": 0.024
        },
        "presence_turn_decision": {
          "count": 4,
          "p50_ms": 1.799,
          "p95_ms": 2.037
        }
      }
    },
    "database_watermarks": {
      "counts": {
        "chat_log": 0,
        "memory_facts": 0,
        "memory_chunks": 0,
        "stream_sessions": 1,
        "stream_chat_messages": 0,
        "stream_events": 1,
        "live_session_timeline": 7,
        "promotion_events": 0,
        "viewer_promotion_profiles": 0,
        "schema_migrations": 7,
        "conversations": 0,
        "open_threads": 0,
        "beliefs": 0,
        "belief_evidence": 0,
        "scene_assertions": 0,
        "game_identities": 0,
        "game_runs": 0,
        "game_run_sessions": 0,
        "game_run_events": 0,
        "game_knowledge_facts": 0,
        "game_knowledge_v2_gaps": 0,
        "people": 0,
        "person_identities": 0,
        "person_sessions": 0,
        "social_episodes": 0,
        "shared_culture_items": 0,
        "shared_culture_evidence": 0,
        "consolidation_runs": 0,
        "consolidation_deltas": 0,
        "action_ledger": 0,
        "temporal_maintenance_audit": 0,
        "learning_observations": 0,
        "scene_transitions": 0,
        "schedule_observations": 1,
        "schedule_hypotheses": 7
      },
      "schema_migrations": [
        {
          "component": "architecture_consolidation",
          "version": 1,
          "name": "audit_hygiene_and_cutover_state",
          "checksum": "944b3ad6532151d4204ed59f663668c9b0ac62df54ece7602d26b91acddd5b7d",
          "applied_at": "2026-08-14T02:35:32.541524+00:00"
        },
        {
          "component": "belief_v2",
          "version": 1,
          "name": "beliefs_evidence_and_compatibility_columns",
          "checksum": "f0df6f1288caccaf6bb47670b38f9cca747f14c84916d6c1ca75beb687200507",
          "applied_at": "2026-08-14T02:35:32.399139+00:00"
        },
        {
          "component": "cognitive_replay",
          "version": 1,
          "name": "replay_metadata",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "applied_at": "2026-08-14T02:35:32.177746+00:00"
        },
        {
          "component": "conversation_continuity",
          "version": 1,
          "name": "conversation_and_open_threads",
          "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
          "applied_at": "2026-08-14T02:35:32.312278+00:00"
        },
        {
          "component": "game_context_v2",
          "version": 1,
          "name": "durable_runs_knowledge_and_gaps",
          "checksum": "08e342acaae00d5d24c1a6dbccad5aee41f753ed90d9ec26282a5fdf042d0a75",
          "applied_at": "2026-08-14T02:35:32.450003+00:00"
        },
        {
          "component": "learning_v2",
          "version": 1,
          "name": "consolidation_temporal_action_and_scene",
          "checksum": "6a86e2d1c7c03167f3b20c328bc97b73fb92e03903f64e64d326fb08f9e3b942",
          "applied_at": "2026-08-14T02:35:32.526410+00:00"
        },
        {
          "component": "social_world_v2",
          "version": 1,
          "name": "people_episodes_and_shared_culture",
          "checksum": "b02adc2cd7f298f1af228dc52c4ba44ae15999fb4c815ea004c40f68b78cbfa5",
          "applied_at": "2026-08-14T02:35:32.488704+00:00"
        }
      ],
      "final_response_digest": "",
      "final_response_present": false
    }
  }
}
```

#### Final state and side effects

```json
{
  "runtime": {
    "mode": "active",
    "hebe_sleeping": false,
    "is_running": false,
    "last_input_source": null,
    "last_intent": null,
    "last_firewall": {
      "source": "twitch_system",
      "authority": "system",
      "decision": "allow",
      "reason": "system_event"
    },
    "last_policy": {
      "source": null,
      "authority": null,
      "decision": null,
      "reason": null
    }
  },
  "stream_session": {
    "enabled": true,
    "is_live": true,
    "live_status_known": true,
    "active_stream_session_id": 1,
    "last_transition": "online",
    "title": "Boss attempts",
    "game": "Test RPG",
    "category": "Test RPG"
  },
  "current_scene": {
    "scene_id": "scene_2f0690b8ded85d18ba449ec4e37abb13",
    "topic_id": "topic_dc448af4c39f",
    "entity": "unknown",
    "current_state": "active",
    "state_version": 1,
    "supporting_event_ids": [
      "ambient:rng_dependency:1786727102"
    ],
    "superseded_event_ids": [],
    "terminal": false,
    "updated_at": 1786727102.0
  },
  "pending": {
    "clarification": null,
    "conversation_turn": null
  },
  "conversation": {
    "active": {},
    "latest": {},
    "all": [],
    "last_resolution": {
      "consumed": false,
      "decision": "no_conversation",
      "reason": "no_compatible_active_conversation",
      "conversation_id": "",
      "reply_act": "UNKNOWN",
      "payload": {},
      "conversation": null,
      "latency_ms": 3.095800057053566
    },
    "legacy_pending_projection": {},
    "continuity_shadow_diff": {
      "legacy_result": false,
      "v2_result": false,
      "match": true,
      "difference_reason": ""
    },
    "shadow_metrics": {
      "total": 1,
      "matches": 1,
      "differences": 0,
      "match_rate": 1.0,
      "difference_reasons": {}
    },
    "performance": {
      "count": 1,
      "p50_ms": 3.0958,
      "p95_ms": 3.0958
    }
  },
  "open_threads": [],
  "memory": {
    "facts_count": 0,
    "chunks_count": 0
  },
  "beliefs": {
    "active": [],
    "historical": [],
    "superseded": [],
    "suspected": [],
    "all": [],
    "last_transition": {}
  },
  "belief_evidence": [],
  "retrieval": {
    "last_request": {},
    "performance": {
      "count": 0,
      "p50_ms": 0.0,
      "p95_ms": 0.0
    },
    "write_performance": {
      "count": 0,
      "p50_ms": 0.0,
      "p95_ms": 0.0
    },
    "repository_performance": {
      "belief_lookup": {
        "count": 12,
        "p50_ms": 0.89615,
        "p95_ms": 1.8886
      },
      "evidence_lookup": {
        "count": 0,
        "p50_ms": 0.0,
        "p95_ms": 0.0
      },
      "sqlite_write": {
        "count": 0,
        "p50_ms": 0.0,
        "p95_ms": 0.0
      }
    }
  },
  "memory_compatibility": {
    "legacy_to_v2": [],
    "v2_to_legacy": [],
    "shadow_diffs": [],
    "backfill": {
      "safe": 0,
      "compatibility_only": 0,
      "ambiguous": 0,
      "invalid_stale": 0
    }
  },
  "game_state": {
    "game": "Test RPG",
    "platform_version": "",
    "playthrough_type": "casual",
    "spoiler_policy": "no_spoilers",
    "current_location": null,
    "current_character": "",
    "party_members": [],
    "last_confirmed_progress": "",
    "current_objective": null,
    "challenge": "",
    "known_constraints": [],
    "last_updated": 1786727100.0,
    "provenance": "stream_context_sync",
    "confidence": 0.75,
    "current_game": "Test RPG",
    "recent_run_context_facts": [
      {
        "kind": "rng_dependency",
        "text": "Leo framed the current situation as dependent on RNG or luck.",
        "category": "rng_dependency",
        "summary": "Leo framed the current situation as dependent on RNG or luck.",
        "id": "ambient:rng_dependency:1786727102",
        "fact_id": "ambient:rng_dependency:1786727102",
        "raw_text": "<redacted:2320c1febf71>",
        "conservative_normalized_text": "que ha sido demasiada suerte dios",
        "utterance_role": "owner_question_to_stream",
        "timestamp": 1786727102.0,
        "topic_id": "topic_dc448af4c39f",
        "heuristic_category": "rng_dependency",
        "extracted_subject": "unknown",
        "subject": "unknown",
        "extracted_object": "",
        "object": "",
        "extracted_predicate": "",
        "predicate": "",
        "confidence": 0.86,
        "referent_confidence": 0.86,
        "inference_level": "heuristic",
        "supported_claims": [
          "que ha sido demasiada suerte dios"
        ],
        "directly_supported_claims": [
          "que ha sido demasiada suerte dios"
        ],
        "inferred_claims": [],
        "unsupported_claims": [],
        "evidence_span": "que ha sido demasiada suerte dios",
        "evidence_tokens": "<redacted:323fcf3ad517>",
        "semantic_rule": "ambient_category:rng_dependency",
        "model_reason": "Leo framed the current situation as dependent on RNG or luck.",
        "expires_at": 1786727222.0,
        "scene_id": "scene_2f0690b8ded85d18ba449ec4e37abb13",
        "game": null,
        "source": "stt_voice",
        "raw_evidence": "que ha sido demasiada suerte dios",
        "normalized_text": "que ha sido demasiada suerte dios",
        "normalized_evidence": "que ha sido demasiada suerte dios",
        "language": "es",
        "ttl_sec": 120,
        "data": {
          "category": "rng_dependency",
          "raw_text": "<redacted:2320c1febf71>",
          "normalized_text": "que ha sido demasiada suerte dios",
          "mood": "rng tension",
          "extracted_subject": "unknown",
          "extracted_object": "",
          "extracted_predicate": "",
          "inference_level": "heuristic",
          "supported_claims": [
            "que ha sido demasiada suerte dios"
          ],
          "inferred_claims": [],
          "unsupported_claims": [],
          "evidence_span": "que ha sido demasiada suerte dios",
          "evidence_tokens": "<redacted:323fcf3ad517>",
          "semantic_rule": "ambient_category:rng_dependency",
          "model_reason": "Leo framed the current situation as dependent on RNG or luck.",
          "scene_id": "scene_2f0690b8ded85d18ba449ec4e37abb13"
        },
        "age_seconds": 0.0,
        "superseded": false,
        "state_version": 1,
        "current_state": "active",
        "terminal": false,
        "currentness_score": 1.0
      }
    ],
    "identity": {},
    "context": {},
    "active_run": {},
    "runs": [],
    "session_links": [],
    "run_events": [],
    "run_beliefs": {
      "current": [],
      "inferred": [],
      "superseded": []
    },
    "knowledge": {
      "selected": [],
      "rejected": [],
      "spoiler_blocked": [],
      "all": []
    },
    "gaps": [],
    "research": {
      "research_calls": 0,
      "cache_hits": 0,
      "failures": [],
      "context_performance": {
        "count": 0,
        "p50_ms": 0.0,
        "p95_ms": 0.0
      },
      "fixture_calls": [],
      "status": ""
    },
    "compatibility": {
      "legacy_progress": [],
      "dossier": [],
      "legacy_run": [],
      "shadow_diffs": [],
      "backfill": {
        "validated": 0,
        "compatibility_only": 0,
        "ambiguous": 0,
        "stale": 0
      }
    },
    "provenance_manifest": [],
    "advice_allowed": null,
    "reaction_allowed": null,
    "performance": {
      "run": {
        "count": 0,
        "p50_ms": 0.0,
        "p95_ms": 0.0
      },
      "run_fact": {
        "count": 0,
        "p50_ms": 0.0,
        "p95_ms": 0.0
      },
      "knowledge": {
        "count": 0,
        "p50_ms": 0.0,
        "p95_ms": 0.0
      },
      "research_gap": {
        "count": 0,
        "p50_ms": 0.0,
        "p95_ms": 0.0
      },
      "db_write": {
        "count": 0,
        "p50_ms": 0.0,
        "p95_ms": 0.0
      }
    },
    "context_performance": {
      "count": 0,
      "p50_ms": 0.0,
      "p95_ms": 0.0
    },
    "manifest_size_bytes": 0,
    "last_run_resolution": {},
    "run_resolution_performance": {
      "count": 0,
      "p50_ms": 0.0,
      "p95_ms": 0.0
    }
  },
  "social_state": {
    "recent_active_users": [],
    "recent_chat_count": 0,
    "last_raid": {},
    "last_cheer": {},
    "people": [],
    "identities": [],
    "recent_episodes": [],
    "active_hypotheses": [],
    "historical_hypotheses": [],
    "open_threads": [],
    "relationships": [],
    "shared_culture": {
      "all": [],
      "candidates": [],
      "active": [],
      "weakening": [],
      "retired": [],
      "reactions": [],
      "selection": {
        "selected": [],
        "rejected": []
      }
    },
    "retrieval": {},
    "opportunities": [],
    "resolution": {},
    "rejected_writes": [],
    "compatibility": {
      "chatter_presence": [],
      "chatter_profiles": [],
      "chatter_facts": [],
      "stream_chatter_summaries": [],
      "viewer_profiles": [],
      "social_events": [],
      "promotion_profiles": [],
      "backfill": {
        "explicit_observation": 0,
        "safe_episode": 0,
        "inferred_compatibility_only": 0,
        "ambiguous": 0,
        "sensitive": 0,
        "stale": 0
      },
      "shadow_diffs": []
    },
    "performance": {
      "identity": {
        "count": 0,
        "p50_ms": 0.0,
        "p95_ms": 0.0
      },
      "episode_write": {
        "count": 0,
        "p50_ms": 0.0,
        "p95_ms": 0.0
      },
      "thread_lookup": {
        "count": 0,
        "p50_ms": 0.0,
        "p95_ms": 0.0
      },
      "culture_select": {
        "count": 0,
        "p50_ms": 0.0,
        "p95_ms": 0.0
      },
      "context": {
        "count": 0,
        "p50_ms": 0.0,
        "p95_ms": 0.0
      },
      "db_write": {
        "count": 0,
        "p50_ms": 0.0,
        "p95_ms": 0.0
      }
    },
    "belief_lookup_performance": {
      "belief_lookup": {
        "count": 12,
        "p50_ms": 0.89615,
        "p95_ms": 1.8886
      },
      "evidence_lookup": {
        "count": 0,
        "p50_ms": 0.0,
        "p95_ms": 0.0
      },
      "sqlite_write": {
        "count": 0,
        "p50_ms": 0.0,
        "p95_ms": 0.0
      }
    }
  },
  "learning": {
    "consolidation_runs": [],
    "deltas": [],
    "rejected_deltas": [],
    "watermarks": [],
    "last_result": {},
    "stable_core_version": "a1a58e51882b0c88",
    "performance": {
      "repository": {
        "lookup": {
          "count": 60,
          "p50_ms": 0.86295,
          "p95_ms": 1.4252
        },
        "write": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "context": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        }
      },
      "consolidation": {
        "consolidation_duration": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        },
        "candidate_validation": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        }
      },
      "temporal": {
        "temporal_maintenance": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        }
      },
      "action_history": {
        "action_ledger_lookup": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        }
      },
      "owner_preferences": {
        "owner_preference_lookup": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        }
      },
      "hebe_self": {
        "hebe_self_lookup": {
          "count": 1,
          "p50_ms": 1.8985,
          "p95_ms": 1.8985
        }
      },
      "context": {
        "continuity_context_build": {
          "count": 0,
          "p50_ms": 0.0,
          "p95_ms": 0.0
        }
      }
    }
  },
  "self_model": {
    "stable_core_version": "a1a58e51882b0c88",
    "evolving_preferences": [],
    "opinions": [],
    "superseded_opinions": []
  },
  "owner_preferences": [],
  "leo_language": {
    "beliefs": [],
    "interpretation_aliases": {}
  },
  "temporal": {
    "expired": [],
    "archived": [],
    "weakened": [],
    "maintenance_actions": [],
    "last_actions": []
  },
  "schedule": {
    "observations": [
      {
        "id": 1,
        "stream_session_id": "1",
        "weekday": "friday",
        "time_window": "night",
        "canonical_content": "Test RPG",
        "content_key": "test rpg",
        "stream_format": "game_playthrough",
        "source": "observed",
        "observed_at": "2026-08-14T04:35:32.628934+02:00"
      }
    ],
    "hypotheses": [
      {
        "id": 1,
        "weekday": "monday",
        "time_window": "any",
        "canonical_content": "FINAL FANTASY IX",
        "content_key": "final fantasy ix",
        "stream_format": "challenge_run",
        "source": "owner_declared",
        "confidence": 0.9,
        "evidence_count": 1,
        "consecutive_matches": 1,
        "consecutive_misses": 0,
        "first_observed_at": "2026-08-14T04:35:32.625425+02:00",
        "last_observed_at": "2026-08-14T04:35:32.625425+02:00",
        "status": "active",
        "superseded_by": null
      },
      {
        "id": 2,
        "weekday": "tuesday",
        "time_window": "any",
        "canonical_content": "Persona 5 Royal",
        "content_key": "persona 5 royal",
        "stream_format": "game_playthrough",
        "source": "owner_declared",
        "confidence": 0.9,
        "evidence_count": 1,
        "consecutive_matches": 1,
        "consecutive_misses": 0,
        "first_observed_at": "2026-08-14T04:35:32.625425+02:00",
        "last_observed_at": "2026-08-14T04:35:32.625425+02:00",
        "status": "active",
        "superseded_by": null
      },
      {
        "id": 3,
        "weekday": "wednesday",
        "time_window": "any",
        "canonical_content": "Persona 5 Royal",
        "content_key": "persona 5 royal",
        "stream_format": "game_playthrough",
        "source": "owner_declared",
        "confidence": 0.9,
        "evidence_count": 1,
        "consecutive_matches": 1,
        "consecutive_misses": 0,
        "first_observed_at": "2026-08-14T04:35:32.625425+02:00",
        "last_observed_at": "2026-08-14T04:35:32.625425+02:00",
        "status": "active",
        "superseded_by": null
      },
      {
        "id": 4,
        "weekday": "thursday",
        "time_window": "any",
        "canonical_content": "Baldur's Gate 3",
        "content_key": "baldur s gate 3",
        "stream_format": "game_playthrough",
        "source": "owner_declared",
        "confidence": 0.9,
        "evidence_count": 1,
        "consecutive_matches": 1,
        "consecutive_misses": 0,
        "first_observed_at": "2026-08-14T04:35:32.625425+02:00",
        "last_observed_at": "2026-08-14T04:35:32.625425+02:00",
        "status": "active",
        "superseded_by": null
      },
      {
        "id": 6,
        "weekday": "saturday",
        "time_window": "any",
        "canonical_content": "Retro Weekend",
        "content_key": "retro weekend",
        "stream_format": "retro",
        "source": "owner_declared",
        "confidence": 0.9,
        "evidence_count": 1,
        "consecutive_matches": 1,
        "consecutive_misses": 0,
        "first_observed_at": "2026-08-14T04:35:32.625425+02:00",
        "last_observed_at": "2026-08-14T04:35:32.625425+02:00",
        "status": "active",
        "superseded_by": null
      },
      {
        "id": 7,
        "weekday": "sunday",
        "time_window": "any",
        "canonical_content": "Retro Weekend",
        "content_key": "retro weekend",
        "stream_format": "retro",
        "source": "owner_declared",
        "confidence": 0.9,
        "evidence_count": 1,
        "consecutive_matches": 1,
        "consecutive_misses": 0,
        "first_observed_at": "2026-08-14T04:35:32.625425+02:00",
        "last_observed_at": "2026-08-14T04:35:32.625425+02:00",
        "status": "active",
        "superseded_by": null
      },
      {
        "id": 5,
        "weekday": "friday",
        "time_window": "any",
        "canonical_content": "Baldur's Gate 3",
        "content_key": "baldur s gate 3",
        "stream_format": "game_playthrough",
        "source": "owner_declared",
        "confidence": 0.76,
        "evidence_count": 1,
        "consecutive_matches": 0,
        "consecutive_misses": 1,
        "first_observed_at": "2026-08-14T04:35:32.625425+02:00",
        "last_observed_at": "2026-08-14T04:35:32.628934+02:00",
        "status": "weakening",
        "superseded_by": null
      }
    ],
    "observed_current_state": {
      "game": "Test RPG",
      "title": "Boss attempts"
    },
    "precedence": "observed_twitch_metadata"
  },
  "action_ledger": {
    "entries": [],
    "last_claim_validation": {}
  },
  "scene_transitions": {
    "all": [],
    "last": {}
  },
  "continuity_context": {},
  "promotion_profiles": [],
  "actions": {
    "attempts": [
      {
        "operation": "twitch.send_message",
        "payload": {
          "text": "Uf, qué tensión."
        },
        "outcome": {
          "success": true,
          "status": "sent"
        }
      }
    ],
    "speech_requests": [],
    "model_calls": [
      {
        "key": "promotion_clarification:v1:none",
        "method": "chat"
      }
    ],
    "research_calls": []
  },
  "receipts": [],
  "emitted_outputs": [
    {
      "event_id": "twitch_job_1786727104300000",
      "emitted": true,
      "route": "twitch_text_reply",
      "targets": [
        "twitch_chat"
      ],
      "reason": "",
      "text_digest": "",
      "text_present": false
    }
  ],
  "final_emission_results": [
    {
      "event_id": "twitch_job_1786727104300000",
      "emitted": true,
      "route": "twitch_text_reply",
      "targets": [
        "twitch_chat"
      ],
      "reason": "",
      "text_digest": "",
      "text_present": false
    }
  ],
  "speech_intents": {
    "active": [
      {
        "id": "intent_1c5ea004e10154bc90ef7fce1d7896b8",
        "type": "GAME_COMMENT",
        "source_event_ids": [],
        "anchor_ids": [],
        "topic": "game,title,title_markers,run_context,recent_voice_event",
        "subject_ref": "",
        "value": 0.58,
        "urgency": 0.55,
        "freshness": 1.0,
        "created_at": 1786727105.7,
        "expires_at": 1786727119.7,
        "interruptibility": "yield_before_tts_commit",
        "minimum_turn_gap": 1.8,
        "maximum_turn_delay": 14.0,
        "scene_relevance": {},
        "status": "PENDING",
        "suppression_reason": "",
        "reserved_at": 0.0,
        "emitted_at": 0.0,
        "contribution_material": {
          "anchor": {
            "id": "",
            "type": "game,title,title_markers,run_context,recent_voice_event",
            "quality": 0.58,
            "reason": "stream_context"
          },
          "readiness_topic": null
        }
      },
      {
        "id": "intent_a1ac9c1e5e345442b692b3d6c1119607",
        "type": "OPINION",
        "source_event_ids": [
          "second-comment"
        ],
        "anchor_ids": [],
        "topic": "boss-design",
        "subject_ref": "",
        "value": 0.73,
        "urgency": 0.3,
        "freshness": 1.0,
        "created_at": 1786727105.7,
        "expires_at": 1786727130.7,
        "interruptibility": "yield_before_tts_commit",
        "minimum_turn_gap": 2.5,
        "maximum_turn_delay": 25.0,
        "scene_relevance": {
          "scene_id": "scene_2f0690b8ded85d18ba449ec4e37abb13",
          "topic_id": "topic_dc448af4c39f",
          "entity": "unknown",
          "current_state": "active",
          "state_version": 1,
          "supporting_event_ids": [
            "ambient:rng_dependency:1786727102"
          ],
          "superseded_event_ids": [],
          "terminal": false,
          "updated_at": 1786727102.0
        },
        "status": "PENDING",
        "suppression_reason": "",
        "reserved_at": 0.0,
        "emitted_at": 0.0,
        "contribution_material": {
          "cognitive_candidate": true
        }
      }
    ],
    "all": [
      {
        "id": "intent_23c0d3235c5150eb9db77c46247f8c22",
        "type": "REACTION",
        "source_event_ids": [],
        "anchor_ids": [
          "ambient:rng_dependency:1786727102"
        ],
        "topic": "rng_dependency",
        "subject_ref": "unknown",
        "value": 0.86,
        "urgency": 0.85,
        "freshness": 0.998333332935969,
        "created_at": 1786727102.2,
        "expires_at": 1786727110.2,
        "interruptibility": "yield_before_tts_commit",
        "minimum_turn_gap": 1.2,
        "maximum_turn_delay": 8.0,
        "scene_relevance": {
          "scene_id": "scene_2f0690b8ded85d18ba449ec4e37abb13",
          "state_version": 1,
          "current_state": "active",
          "terminal": false
        },
        "status": "EMITTED",
        "suppression_reason": "",
        "reserved_at": 1786727104.3,
        "emitted_at": 1786727104.3,
        "contribution_material": {
          "anchor": {
            "id": "ambient:rng_dependency:1786727102",
            "type": "rng_dependency",
            "quality": 0.86,
            "reason": "recent_ambient_context",
            "evidence": {
              "anchor_id": "ambient:rng_dependency:1786727102",
              "anchor_type": "rng_dependency",
              "raw_owner_fragments": [
                "que ha sido demasiada suerte dios"
              ],
              "exact_supported_claims": [
                "que ha sido demasiada suerte dios"
              ],
              "timestamps": [
                1786727102.0
              ],
              "topic_id": "topic_dc448af4c39f",
              "currentness": 0.998333332935969,
              "confidence": 0.86,
              "allowed_contribution_types": [
                "contextual_reaction",
                "emotional_banter",
                "concise_observation"
              ],
              "forbidden_claims": [
                "unsupported strategy",
                "save instruction",
                "unrelated mechanic",
                "stale topic fusion"
              ],
              "expires_at": 1786727222.0,
              "scene_id": "scene_2f0690b8ded85d18ba449ec4e37abb13",
              "state_version": 1,
              "current_state": "active",
              "terminal": false,
              "extracted_subject": "unknown",
              "extracted_object": "",
              "extracted_predicate": "",
              "supported_claims": [
                "que ha sido demasiada suerte dios"
              ],
              "unsupported_claims": []
            },
            "scene_guard": {
              "scene_id": "scene_2f0690b8ded85d18ba449ec4e37abb13",
              "state_version": 1,
              "current_state": "active",
              "terminal": false
            }
          },
          "readiness_topic": null
        }
      },
      {
        "id": "intent_1c5ea004e10154bc90ef7fce1d7896b8",
        "type": "GAME_COMMENT",
        "source_event_ids": [],
        "anchor_ids": [],
        "topic": "game,title,title_markers,run_context,recent_voice_event",
        "subject_ref": "",
        "value": 0.58,
        "urgency": 0.55,
        "freshness": 1.0,
        "created_at": 1786727105.7,
        "expires_at": 1786727119.7,
        "interruptibility": "yield_before_tts_commit",
        "minimum_turn_gap": 1.8,
        "maximum_turn_delay": 14.0,
        "scene_relevance": {},
        "status": "PENDING",
        "suppression_reason": "",
        "reserved_at": 0.0,
        "emitted_at": 0.0,
        "contribution_material": {
          "anchor": {
            "id": "",
            "type": "game,title,title_markers,run_context,recent_voice_event",
            "quality": 0.58,
            "reason": "stream_context"
          },
          "readiness_topic": null
        }
      },
      {
        "id": "intent_a1ac9c1e5e345442b692b3d6c1119607",
        "type": "OPINION",
        "source_event_ids": [
          "second-comment"
        ],
        "anchor_ids": [],
        "topic": "boss-design",
        "subject_ref": "",
        "value": 0.73,
        "urgency": 0.3,
        "freshness": 1.0,
        "created_at": 1786727105.7,
        "expires_at": 1786727130.7,
        "interruptibility": "yield_before_tts_commit",
        "minimum_turn_gap": 2.5,
        "maximum_turn_delay": 25.0,
        "scene_relevance": {
          "scene_id": "scene_2f0690b8ded85d18ba449ec4e37abb13",
          "topic_id": "topic_dc448af4c39f",
          "entity": "unknown",
          "current_state": "active",
          "state_version": 1,
          "supporting_event_ids": [
            "ambient:rng_dependency:1786727102"
          ],
          "superseded_event_ids": [],
          "terminal": false,
          "updated_at": 1786727102.0
        },
        "status": "PENDING",
        "suppression_reason": "",
        "reserved_at": 0.0,
        "emitted_at": 0.0,
        "contribution_material": {
          "cognitive_candidate": true
        }
      }
    ],
    "metrics": {
      "intents_created": 3,
      "created:REACTION": 1,
      "pending_due_owner_voice_active": 3,
      "turns_reserved": 1,
      "intents_emitted": 1,
      "emitted:REACTION": 1,
      "created:GAME_COMMENT": 1,
      "created:OPINION": 1,
      "pending": 2,
      "time_created_to_emit": {
        "count": 1,
        "p50_ms": 2100.0,
        "p95_ms": 2100.0
      },
      "turn_gap_before_emit": {
        "count": 1,
        "p50_ms": 1300.0,
        "p95_ms": 1300.0
      },
      "intent_creation": {
        "count": 3,
        "p50_ms": 0.057,
        "p95_ms": 0.057
      },
      "pending_queue_operation": {
        "count": 3,
        "p50_ms": 0.057,
        "p95_ms": 0.057
      },
      "turn_arbitration": {
        "count": 4,
        "p50_ms": 0.02,
        "p95_ms": 0.024
      },
      "presence_turn_decision": {
        "count": 4,
        "p50_ms": 1.799,
        "p95_ms": 2.037
      }
    }
  },
  "database_watermarks": {
    "counts": {
      "chat_log": 0,
      "memory_facts": 0,
      "memory_chunks": 0,
      "stream_sessions": 1,
      "stream_chat_messages": 0,
      "stream_events": 1,
      "live_session_timeline": 7,
      "promotion_events": 0,
      "viewer_promotion_profiles": 0,
      "schema_migrations": 7,
      "conversations": 0,
      "open_threads": 0,
      "beliefs": 0,
      "belief_evidence": 0,
      "scene_assertions": 0,
      "game_identities": 0,
      "game_runs": 0,
      "game_run_sessions": 0,
      "game_run_events": 0,
      "game_knowledge_facts": 0,
      "game_knowledge_v2_gaps": 0,
      "people": 0,
      "person_identities": 0,
      "person_sessions": 0,
      "social_episodes": 0,
      "shared_culture_items": 0,
      "shared_culture_evidence": 0,
      "consolidation_runs": 0,
      "consolidation_deltas": 0,
      "action_ledger": 0,
      "temporal_maintenance_audit": 0,
      "learning_observations": 0,
      "scene_transitions": 0,
      "schedule_observations": 1,
      "schedule_hypotheses": 7
    },
    "schema_migrations": [
      {
        "component": "architecture_consolidation",
        "version": 1,
        "name": "audit_hygiene_and_cutover_state",
        "checksum": "944b3ad6532151d4204ed59f663668c9b0ac62df54ece7602d26b91acddd5b7d",
        "applied_at": "2026-08-14T02:35:32.541524+00:00"
      },
      {
        "component": "belief_v2",
        "version": 1,
        "name": "beliefs_evidence_and_compatibility_columns",
        "checksum": "f0df6f1288caccaf6bb47670b38f9cca747f14c84916d6c1ca75beb687200507",
        "applied_at": "2026-08-14T02:35:32.399139+00:00"
      },
      {
        "component": "cognitive_replay",
        "version": 1,
        "name": "replay_metadata",
        "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
        "applied_at": "2026-08-14T02:35:32.177746+00:00"
      },
      {
        "component": "conversation_continuity",
        "version": 1,
        "name": "conversation_and_open_threads",
        "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
        "applied_at": "2026-08-14T02:35:32.312278+00:00"
      },
      {
        "component": "game_context_v2",
        "version": 1,
        "name": "durable_runs_knowledge_and_gaps",
        "checksum": "08e342acaae00d5d24c1a6dbccad5aee41f753ed90d9ec26282a5fdf042d0a75",
        "applied_at": "2026-08-14T02:35:32.450003+00:00"
      },
      {
        "component": "learning_v2",
        "version": 1,
        "name": "consolidation_temporal_action_and_scene",
        "checksum": "6a86e2d1c7c03167f3b20c328bc97b73fb92e03903f64e64d326fb08f9e3b942",
        "applied_at": "2026-08-14T02:35:32.526410+00:00"
      },
      {
        "component": "social_world_v2",
        "version": 1,
        "name": "people_episodes_and_shared_culture",
        "checksum": "b02adc2cd7f298f1af228dc52c4ba44ae15999fb4c815ea004c40f68b78cbfa5",
        "applied_at": "2026-08-14T02:35:32.488704+00:00"
      }
    ],
    "final_response_digest": "",
    "final_response_present": false
  }
}
```

#### Restart evidence

```json
[]
```

## External boundaries

- twitch: `fake`
- tts_audio: `fake`
- desktop: `fake`
- game_research_web: `fixture`
- llm_model: `fixture`
- network: `blocked_by_design`

## Persistence

```json
{
  "database_type": "isolated_sqlite",
  "database_paths": [
    "C:\\Users\\Leo Nifelheim\\Documents\\Hebe\\hebe-ui\\artifacts\\co-streamer-turn-taking\\release\\workspaces\\co_streamer_realistic_cadence\\hebe-replay.sqlite3"
  ],
  "restart_points": 0,
  "schema_migrations": [
    [
      {
        "component": "architecture_consolidation",
        "version": 1,
        "name": "audit_hygiene_and_cutover_state",
        "checksum": "944b3ad6532151d4204ed59f663668c9b0ac62df54ece7602d26b91acddd5b7d",
        "applied_at": "2026-08-14T02:35:32.541524+00:00"
      },
      {
        "component": "belief_v2",
        "version": 1,
        "name": "beliefs_evidence_and_compatibility_columns",
        "checksum": "f0df6f1288caccaf6bb47670b38f9cca747f14c84916d6c1ca75beb687200507",
        "applied_at": "2026-08-14T02:35:32.399139+00:00"
      },
      {
        "component": "cognitive_replay",
        "version": 1,
        "name": "replay_metadata",
        "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
        "applied_at": "2026-08-14T02:35:32.177746+00:00"
      },
      {
        "component": "conversation_continuity",
        "version": 1,
        "name": "conversation_and_open_threads",
        "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
        "applied_at": "2026-08-14T02:35:32.312278+00:00"
      },
      {
        "component": "game_context_v2",
        "version": 1,
        "name": "durable_runs_knowledge_and_gaps",
        "checksum": "08e342acaae00d5d24c1a6dbccad5aee41f753ed90d9ec26282a5fdf042d0a75",
        "applied_at": "2026-08-14T02:35:32.450003+00:00"
      },
      {
        "component": "learning_v2",
        "version": 1,
        "name": "consolidation_temporal_action_and_scene",
        "checksum": "6a86e2d1c7c03167f3b20c328bc97b73fb92e03903f64e64d326fb08f9e3b942",
        "applied_at": "2026-08-14T02:35:32.526410+00:00"
      },
      {
        "component": "social_world_v2",
        "version": 1,
        "name": "people_episodes_and_shared_culture",
        "checksum": "b02adc2cd7f298f1af228dc52c4ba44ae15999fb4c815ea004c40f68b78cbfa5",
        "applied_at": "2026-08-14T02:35:32.488704+00:00"
      }
    ]
  ]
}
```

## Limitations

- datetime.now() reads in legacy persistence remain wall-clock based; behavioral TTL/cooldown time.time() reads are controlled during replay dispatch
- faster-whisper audio decoding is outside the cognitive replay boundary and requires its separate integration suite

## Baseline differential

```json
{
  "baseline_commit": "5b3cc1b",
  "baseline_phase": 5,
  "baseline_tests_total": 557,
  "baseline_tests_passed": 540,
  "baseline_tests_failed": 17,
  "phase_5_tests_failed": 17,
  "phase_6_tests_total": 573,
  "phase_6_tests_passed": 556,
  "phase_6_tests_failed": 17,
  "shared_passing": 540,
  "phase_6_new_passing": 16,
  "pre_existing_failures": 17,
  "fixed_existing_failures": 0,
  "changed_failures": 0,
  "new_regressions": 0,
  "NEW_PHASE_6_REGRESSION": 0,
  "classification_counts": {
    "PASS_BOTH": 540,
    "NEW_PHASE_6_PASS": 16,
    "PRE_EXISTING_FAILURE": 17,
    "FIXED_BY_PHASE_6": 0,
    "FAILURE_CHANGED": 0,
    "NEW_PHASE_6_REGRESSION": 0
  },
  "evidence": "Failure identifiers in the Phase 5 committed report and Phase 6 completed report are identical."
}
```

## Human evaluation boundary

This harness verifies cognitive/state prerequisites. Naturalness, personality, comedic timing, and social appropriateness still require human judgment.
