# Cognitive Replay Verification Report

Overall status: **VERIFIED**
Phase result: **PHASE 1 VERIFIED**

## Repository and environment

- Commit: `a1e05d626c0ac2335f590b2179b6eaca6d5af4d1`
- Working tree: `42f24b556d36`
- Platform: `Windows-10-10.0.26200-SP0`
- Python: `3.11.0`

## Commands

- `C:\Program Files\Python311\python.exe -m unittest backend.tests.test_cognitive_replay backend.tests.test_voice_command_pipeline backend.tests.test_cognitive_twitch backend.tests.test_stream_presence backend.tests.test_hebe_live_v1 backend.tests.test_hebe_live_v11 backend.tests.test_hebe_live_v12 backend.tests.test_hebe_live_v12_followup backend.tests.test_hebe_live_20260809_followup backend.tests.test_final_emission_gate backend.tests.test_cognitive_execution_guard backend.tests.test_game_knowledge backend.tests.test_stream_session_primer backend.tests.test_live_session_brain backend.tests.test_conversation_continuity_phase1` → exit 1 (54.291623s)
- `python -m app.replay --suite cognitive-v2-phase1 --run-phase-tests --baseline-differential artifacts/cognitive-continuity-phase1/final/baseline-differential.json --output artifacts/cognitive-continuity-phase1/final` → exit 0 (86.709625s)

## Tests

```json
{
  "unit_integration_regression": {
    "passed": 480,
    "failed": 13,
    "skipped": 0,
    "total": 493,
    "duration_seconds": 54.291623,
    "expected_failures": 0,
    "failing_tests": [
      "test_high_value_game_tip_can_reply_without_hebe_mention (backend.tests.test_voice_command_pipeline.VoiceCommandPipelineTests.test_high_value_game_tip_can_reply_without_hebe_mention)",
      "test_no_wake_whitelisted_app_command_routes_while_stream_offline (backend.tests.test_voice_command_pipeline.VoiceCommandPipelineTests.test_no_wake_whitelisted_app_command_routes_while_stream_offline)",
      "test_obs_path_missing_returns_structured_action_result_not_generic_advice (backend.tests.test_voice_command_pipeline.VoiceCommandPipelineTests.test_obs_path_missing_returns_structured_action_result_not_generic_advice)",
      "test_response_synthesizer_handles_game_knowledge_command_result (backend.tests.test_game_knowledge.GameKnowledgeTests.test_response_synthesizer_handles_game_knowledge_command_result)",
      "test_stt_canonical_melonds_command_executes_once (backend.tests.test_voice_command_pipeline.VoiceCommandPipelineTests.test_stt_canonical_melonds_command_executes_once) (transcript='Ebe, abre Melón DS')",
      "test_stt_canonical_melonds_command_executes_once (backend.tests.test_voice_command_pipeline.VoiceCommandPipelineTests.test_stt_canonical_melonds_command_executes_once) (transcript='Eve, abre Melón de Ese')",
      "test_stt_canonical_melonds_command_executes_once (backend.tests.test_voice_command_pipeline.VoiceCommandPipelineTests.test_stt_canonical_melonds_command_executes_once) (transcript='Hebe, abre melonDS')",
      "test_stt_hebe_abre_obs_uses_same_open_application_pipeline (backend.tests.test_voice_command_pipeline.VoiceCommandPipelineTests.test_stt_hebe_abre_obs_uses_same_open_application_pipeline)",
      "test_twitch_normal_no_mention_chat_reaches_presence_observe (backend.tests.test_voice_command_pipeline.VoiceCommandPipelineTests.test_twitch_normal_no_mention_chat_reaches_presence_observe)",
      "test_twitch_pipeline_health_counts_messages (backend.tests.test_voice_command_pipeline.VoiceCommandPipelineTests.test_twitch_pipeline_health_counts_messages)",
      "test_ui_abre_obs_creates_open_application_when_awake_and_whitelisted (backend.tests.test_voice_command_pipeline.VoiceCommandPipelineTests.test_ui_abre_obs_creates_open_application_when_awake_and_whitelisted)",
      "test_ui_hebe_abre_obs_creates_open_application_action_plan (backend.tests.test_voice_command_pipeline.VoiceCommandPipelineTests.test_ui_hebe_abre_obs_creates_open_application_action_plan)",
      "test_unrelated_action_during_pending_conversation_still_uses_action_flow (backend.tests.test_voice_command_pipeline.VoiceCommandPipelineTests.test_unrelated_action_during_pending_conversation_still_uses_action_flow)"
    ],
    "required_layer_missing": false,
    "output_digest": "6d4e8a6f7d2c67f0"
  },
  "replay": {
    "passed": 15,
    "failed": 0,
    "skipped": 4,
    "expected_future_gaps": 4,
    "expected_failures": 0,
    "duration_seconds": 31.013601
  },
  "failed": 13,
  "required_layer_missing": false
}
```

## Replay scenarios

### ambient_false_positive_foundation

- Status: **VERIFIED**
- Events: 4
- Restarts: 0
- Duration: 0.457889s
- Assertions passed/failed/skipped: 5/0/0


#### Checkpoint state

```json
{
  "start": {
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
      "title": null,
      "game": null,
      "category": null
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
    "beliefs": [],
    "game_state": {
      "game": "",
      "platform_version": "",
      "playthrough_type": "casual",
      "spoiler_policy": "spoiler_safe_hints",
      "current_location": null,
      "current_character": "",
      "party_members": [],
      "last_confirmed_progress": "",
      "current_objective": null,
      "challenge": "",
      "known_constraints": [],
      "last_updated": 1786437793.5096016,
      "provenance": "inferred",
      "confidence": 0.0,
      "current_game": null,
      "recent_run_context_facts": []
    },
    "social_state": {
      "recent_active_users": [],
      "recent_chat_count": 0,
      "last_raid": {},
      "last_cheer": {}
    },
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
        "schema_migrations": 2,
        "conversations": 0,
        "open_threads": 0
      },
      "schema_migrations": [
        {
          "component": "cognitive_replay",
          "version": 1,
          "name": "replay_metadata",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "applied_at": "2026-08-11T08:43:13.502031+00:00"
        },
        {
          "component": "conversation_continuity",
          "version": 1,
          "name": "conversation_and_open_threads",
          "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
          "applied_at": "2026-08-11T08:43:13.615668+00:00"
        }
      ],
      "final_response_digest": "",
      "final_response_present": false
    }
  },
  "owner": {
    "runtime": {
      "mode": "active",
      "hebe_sleeping": false,
      "is_running": false,
      "last_input_source": null,
      "last_intent": null,
      "last_firewall": {
        "source": "owner_stt_direct",
        "authority": "owner",
        "decision": "allow",
        "reason": "owner_direct"
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
      "title": null,
      "game": null,
      "category": null
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
      "last_resolution": {
        "consumed": false,
        "decision": "no_conversation",
        "reason": "no_compatible_active_conversation",
        "conversation_id": "",
        "reply_act": "UNKNOWN",
        "payload": {},
        "conversation": null,
        "latency_ms": 1.303899975027889
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
        "p50_ms": 1.3039,
        "p95_ms": 1.3039
      }
    },
    "open_threads": [],
    "memory": {
      "facts_count": 0,
      "chunks_count": 0
    },
    "beliefs": [],
    "game_state": {
      "game": "",
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
      "last_updated": 1786437793.5096016,
      "provenance": "current_live_session",
      "confidence": 0.0,
      "current_game": null,
      "recent_run_context_facts": []
    },
    "social_state": {
      "recent_active_users": [],
      "recent_chat_count": 0,
      "last_raid": {},
      "last_cheer": {}
    },
    "promotion_profiles": [],
    "actions": {
      "attempts": [],
      "speech_requests": [],
      "model_calls": [
        {
          "key": "stream_response:v1:none",
          "method": "chat"
        }
      ],
      "research_calls": []
    },
    "receipts": [],
    "emitted_outputs": [
      {
        "event_id": "1786437793.6790042",
        "emitted": true,
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "reason": "",
        "text_digest": "",
        "text_present": false
      }
    ],
    "final_emission_results": [
      {
        "event_id": "1786437793.6790042",
        "emitted": true,
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "reason": "",
        "text_digest": "",
        "text_present": false
      }
    ],
    "database_watermarks": {
      "counts": {
        "chat_log": 1,
        "memory_facts": 0,
        "memory_chunks": 0,
        "stream_sessions": 1,
        "stream_chat_messages": 0,
        "stream_events": 1,
        "live_session_timeline": 4,
        "promotion_events": 0,
        "viewer_promotion_profiles": 0,
        "schema_migrations": 2,
        "conversations": 0,
        "open_threads": 0
      },
      "schema_migrations": [
        {
          "component": "cognitive_replay",
          "version": 1,
          "name": "replay_metadata",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "applied_at": "2026-08-11T08:43:13.502031+00:00"
        },
        {
          "component": "conversation_continuity",
          "version": 1,
          "name": "conversation_and_open_threads",
          "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
          "applied_at": "2026-08-11T08:43:13.615668+00:00"
        }
      ],
      "final_response_digest": "7e07c6a5322789f6",
      "final_response_present": true
    }
  },
  "ambient": {
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
      "title": null,
      "game": null,
      "category": null
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
      "last_resolution": {
        "consumed": false,
        "decision": "no_conversation",
        "reason": "no_compatible_active_conversation",
        "conversation_id": "",
        "reply_act": "UNKNOWN",
        "payload": {},
        "conversation": null,
        "latency_ms": 1.279400021303445
      },
      "legacy_pending_projection": {},
      "continuity_shadow_diff": {
        "legacy_result": false,
        "v2_result": false,
        "match": true,
        "difference_reason": ""
      },
      "shadow_metrics": {
        "total": 2,
        "matches": 2,
        "differences": 0,
        "match_rate": 1.0,
        "difference_reasons": {}
      },
      "performance": {
        "count": 2,
        "p50_ms": 1.29165,
        "p95_ms": 1.3039
      }
    },
    "open_threads": [],
    "memory": {
      "facts_count": 0,
      "chunks_count": 0
    },
    "beliefs": [],
    "game_state": {
      "game": "",
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
      "last_updated": 1786437793.5096016,
      "provenance": "current_live_session",
      "confidence": 0.0,
      "current_game": null,
      "recent_run_context_facts": []
    },
    "social_state": {
      "recent_active_users": [],
      "recent_chat_count": 0,
      "last_raid": {},
      "last_cheer": {}
    },
    "promotion_profiles": [],
    "actions": {
      "attempts": [],
      "speech_requests": [],
      "model_calls": [
        {
          "key": "stream_response:v1:none",
          "method": "chat"
        }
      ],
      "research_calls": []
    },
    "receipts": [],
    "emitted_outputs": [
      {
        "event_id": "1786437793.6790042",
        "emitted": true,
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "reason": "",
        "text_digest": "",
        "text_present": false
      }
    ],
    "final_emission_results": [
      {
        "event_id": "1786437793.6790042",
        "emitted": true,
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "reason": "",
        "text_digest": "",
        "text_present": false
      }
    ],
    "database_watermarks": {
      "counts": {
        "chat_log": 1,
        "memory_facts": 0,
        "memory_chunks": 0,
        "stream_sessions": 1,
        "stream_chat_messages": 0,
        "stream_events": 1,
        "live_session_timeline": 4,
        "promotion_events": 0,
        "viewer_promotion_profiles": 0,
        "schema_migrations": 2,
        "conversations": 0,
        "open_threads": 0
      },
      "schema_migrations": [
        {
          "component": "cognitive_replay",
          "version": 1,
          "name": "replay_metadata",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "applied_at": "2026-08-11T08:43:13.502031+00:00"
        },
        {
          "component": "conversation_continuity",
          "version": 1,
          "name": "conversation_and_open_threads",
          "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
          "applied_at": "2026-08-11T08:43:13.615668+00:00"
        }
      ],
      "final_response_digest": "7e07c6a5322789f6",
      "final_response_present": true
    }
  },
  "time": {
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
      "title": null,
      "game": null,
      "category": null
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
      "last_resolution": {
        "consumed": false,
        "decision": "no_conversation",
        "reason": "no_compatible_active_conversation",
        "conversation_id": "",
        "reply_act": "UNKNOWN",
        "payload": {},
        "conversation": null,
        "latency_ms": 1.279400021303445
      },
      "legacy_pending_projection": {},
      "continuity_shadow_diff": {
        "legacy_result": false,
        "v2_result": false,
        "match": true,
        "difference_reason": ""
      },
      "shadow_metrics": {
        "total": 2,
        "matches": 2,
        "differences": 0,
        "match_rate": 1.0,
        "difference_reasons": {}
      },
      "performance": {
        "count": 2,
        "p50_ms": 1.29165,
        "p95_ms": 1.3039
      }
    },
    "open_threads": [],
    "memory": {
      "facts_count": 0,
      "chunks_count": 0
    },
    "beliefs": [],
    "game_state": {
      "game": "",
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
      "last_updated": 1786437793.5096016,
      "provenance": "current_live_session",
      "confidence": 0.0,
      "current_game": null,
      "recent_run_context_facts": []
    },
    "social_state": {
      "recent_active_users": [],
      "recent_chat_count": 0,
      "last_raid": {},
      "last_cheer": {}
    },
    "promotion_profiles": [],
    "actions": {
      "attempts": [],
      "speech_requests": [],
      "model_calls": [
        {
          "key": "stream_response:v1:none",
          "method": "chat"
        }
      ],
      "research_calls": []
    },
    "receipts": [],
    "emitted_outputs": [
      {
        "event_id": "1786437793.6790042",
        "emitted": true,
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "reason": "",
        "text_digest": "",
        "text_present": false
      }
    ],
    "final_emission_results": [
      {
        "event_id": "1786437793.6790042",
        "emitted": true,
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "reason": "",
        "text_digest": "",
        "text_present": false
      }
    ],
    "database_watermarks": {
      "counts": {
        "chat_log": 1,
        "memory_facts": 0,
        "memory_chunks": 0,
        "stream_sessions": 1,
        "stream_chat_messages": 0,
        "stream_events": 1,
        "live_session_timeline": 5,
        "promotion_events": 0,
        "viewer_promotion_profiles": 0,
        "schema_migrations": 2,
        "conversations": 0,
        "open_threads": 0
      },
      "schema_migrations": [
        {
          "component": "cognitive_replay",
          "version": 1,
          "name": "replay_metadata",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "applied_at": "2026-08-11T08:43:13.502031+00:00"
        },
        {
          "component": "conversation_continuity",
          "version": 1,
          "name": "conversation_and_open_threads",
          "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
          "applied_at": "2026-08-11T08:43:13.615668+00:00"
        }
      ],
      "final_response_digest": "7e07c6a5322789f6",
      "final_response_present": true
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
    "title": null,
    "game": null,
    "category": null
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
    "last_resolution": {
      "consumed": false,
      "decision": "no_conversation",
      "reason": "no_compatible_active_conversation",
      "conversation_id": "",
      "reply_act": "UNKNOWN",
      "payload": {},
      "conversation": null,
      "latency_ms": 1.279400021303445
    },
    "legacy_pending_projection": {},
    "continuity_shadow_diff": {
      "legacy_result": false,
      "v2_result": false,
      "match": true,
      "difference_reason": ""
    },
    "shadow_metrics": {
      "total": 2,
      "matches": 2,
      "differences": 0,
      "match_rate": 1.0,
      "difference_reasons": {}
    },
    "performance": {
      "count": 2,
      "p50_ms": 1.29165,
      "p95_ms": 1.3039
    }
  },
  "open_threads": [],
  "memory": {
    "facts_count": 0,
    "chunks_count": 0
  },
  "beliefs": [],
  "game_state": {
    "game": "",
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
    "last_updated": 1786437793.5096016,
    "provenance": "current_live_session",
    "confidence": 0.0,
    "current_game": null,
    "recent_run_context_facts": []
  },
  "social_state": {
    "recent_active_users": [],
    "recent_chat_count": 0,
    "last_raid": {},
    "last_cheer": {}
  },
  "promotion_profiles": [],
  "actions": {
    "attempts": [],
    "speech_requests": [],
    "model_calls": [
      {
        "key": "stream_response:v1:none",
        "method": "chat"
      }
    ],
    "research_calls": []
  },
  "receipts": [],
  "emitted_outputs": [
    {
      "event_id": "1786437793.6790042",
      "emitted": true,
      "route": "local_owner_reply",
      "targets": [
        "local_ui"
      ],
      "reason": "",
      "text_digest": "",
      "text_present": false
    }
  ],
  "final_emission_results": [
    {
      "event_id": "1786437793.6790042",
      "emitted": true,
      "route": "local_owner_reply",
      "targets": [
        "local_ui"
      ],
      "reason": "",
      "text_digest": "",
      "text_present": false
    }
  ],
  "database_watermarks": {
    "counts": {
      "chat_log": 1,
      "memory_facts": 0,
      "memory_chunks": 0,
      "stream_sessions": 1,
      "stream_chat_messages": 0,
      "stream_events": 1,
      "live_session_timeline": 5,
      "promotion_events": 0,
      "viewer_promotion_profiles": 0,
      "schema_migrations": 2,
      "conversations": 0,
      "open_threads": 0
    },
    "schema_migrations": [
      {
        "component": "cognitive_replay",
        "version": 1,
        "name": "replay_metadata",
        "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
        "applied_at": "2026-08-11T08:43:13.502031+00:00"
      },
      {
        "component": "conversation_continuity",
        "version": 1,
        "name": "conversation_and_open_threads",
        "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
        "applied_at": "2026-08-11T08:43:13.615668+00:00"
      }
    ],
    "final_response_digest": "7e07c6a5322789f6",
    "final_response_present": true
  }
}
```

#### Restart evidence

```json
[]
```

### consolidation_format

- Status: **VERIFICATION_INCOMPLETE**
- Events: 5
- Restarts: 1
- Duration: 6.003092s
- Assertions passed/failed/skipped: 0/0/1


#### Checkpoint state

```json
{
  "start": {
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
      "title": null,
      "game": null,
      "category": null
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
    "beliefs": [],
    "game_state": {
      "game": "",
      "platform_version": "",
      "playthrough_type": "casual",
      "spoiler_policy": "spoiler_safe_hints",
      "current_location": null,
      "current_character": "",
      "party_members": [],
      "last_confirmed_progress": "",
      "current_objective": null,
      "challenge": "",
      "known_constraints": [],
      "last_updated": 1786437794.0242515,
      "provenance": "inferred",
      "confidence": 0.0,
      "current_game": null,
      "recent_run_context_facts": []
    },
    "social_state": {
      "recent_active_users": [],
      "recent_chat_count": 0,
      "last_raid": {},
      "last_cheer": {}
    },
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
        "schema_migrations": 2,
        "conversations": 0,
        "open_threads": 0
      },
      "schema_migrations": [
        {
          "component": "cognitive_replay",
          "version": 1,
          "name": "replay_metadata",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "applied_at": "2026-08-11T08:43:14.018660+00:00"
        },
        {
          "component": "conversation_continuity",
          "version": 1,
          "name": "conversation_and_open_threads",
          "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
          "applied_at": "2026-08-11T08:43:14.084294+00:00"
        }
      ],
      "final_response_digest": "",
      "final_response_present": false
    }
  },
  "chat": {
    "runtime": {
      "mode": "active",
      "hebe_sleeping": false,
      "is_running": false,
      "last_input_source": null,
      "last_intent": null,
      "last_firewall": {
        "source": "twitch_viewer",
        "authority": "viewer",
        "decision": "allow",
        "reason": "live_viewer_message"
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
      "title": null,
      "game": null,
      "category": null
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
    "beliefs": [],
    "game_state": {
      "game": "",
      "platform_version": "",
      "playthrough_type": "casual",
      "spoiler_policy": "spoiler_safe_hints",
      "current_location": null,
      "current_character": "",
      "party_members": [],
      "last_confirmed_progress": "",
      "current_objective": null,
      "challenge": "",
      "known_constraints": [],
      "last_updated": 1786437794.0242515,
      "provenance": "inferred",
      "confidence": 0.0,
      "current_game": null,
      "recent_run_context_facts": []
    },
    "social_state": {
      "recent_active_users": [
        "synthetic_chatter"
      ],
      "recent_chat_count": 1,
      "last_raid": {},
      "last_cheer": {}
    },
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
    "database_watermarks": {
      "counts": {
        "chat_log": 0,
        "memory_facts": 0,
        "memory_chunks": 0,
        "stream_sessions": 1,
        "stream_chat_messages": 1,
        "stream_events": 1,
        "live_session_timeline": 3,
        "promotion_events": 0,
        "viewer_promotion_profiles": 0,
        "schema_migrations": 2,
        "conversations": 0,
        "open_threads": 0
      },
      "schema_migrations": [
        {
          "component": "cognitive_replay",
          "version": 1,
          "name": "replay_metadata",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "applied_at": "2026-08-11T08:43:14.018660+00:00"
        },
        {
          "component": "conversation_continuity",
          "version": 1,
          "name": "conversation_and_open_threads",
          "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
          "applied_at": "2026-08-11T08:43:14.084294+00:00"
        }
      ],
      "final_response_digest": "",
      "final_response_present": false
    }
  },
  "end": {
    "runtime": {
      "mode": "active",
      "hebe_sleeping": false,
      "is_running": false,
      "last_input_source": null,
      "last_intent": null,
      "last_firewall": {
        "source": "twitch_viewer",
        "authority": "viewer",
        "decision": "allow",
        "reason": "live_viewer_message"
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
      "is_live": false,
      "live_status_known": true,
      "active_stream_session_id": null,
      "last_transition": "offline",
      "title": null,
      "game": null,
      "category": null
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
      "chunks_count": 1
    },
    "beliefs": [],
    "game_state": {
      "game": "",
      "platform_version": "",
      "playthrough_type": "casual",
      "spoiler_policy": "spoiler_safe_hints",
      "current_location": null,
      "current_character": "",
      "party_members": [],
      "last_confirmed_progress": "",
      "current_objective": null,
      "challenge": "",
      "known_constraints": [],
      "last_updated": 1786437794.0242515,
      "provenance": "inferred",
      "confidence": 0.0,
      "current_game": null,
      "recent_run_context_facts": []
    },
    "social_state": {
      "recent_active_users": [
        "synthetic_chatter"
      ],
      "recent_chat_count": 1,
      "last_raid": {},
      "last_cheer": {}
    },
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
    "database_watermarks": {
      "counts": {
        "chat_log": 0,
        "memory_facts": 0,
        "memory_chunks": 1,
        "stream_sessions": 1,
        "stream_chat_messages": 1,
        "stream_events": 2,
        "live_session_timeline": 4,
        "promotion_events": 0,
        "viewer_promotion_profiles": 0,
        "schema_migrations": 2,
        "conversations": 0,
        "open_threads": 0
      },
      "schema_migrations": [
        {
          "component": "cognitive_replay",
          "version": 1,
          "name": "replay_metadata",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "applied_at": "2026-08-11T08:43:14.018660+00:00"
        },
        {
          "component": "conversation_continuity",
          "version": 1,
          "name": "conversation_and_open_threads",
          "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
          "applied_at": "2026-08-11T08:43:14.084294+00:00"
        }
      ],
      "final_response_digest": "",
      "final_response_present": false
    }
  },
  "restart": {
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
      "enabled": false,
      "is_live": false,
      "live_status_known": false,
      "active_stream_session_id": null,
      "last_transition": null,
      "title": null,
      "game": null,
      "category": null
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
      "chunks_count": 1
    },
    "beliefs": [],
    "game_state": {
      "game": "",
      "platform_version": "",
      "playthrough_type": "casual",
      "spoiler_policy": "spoiler_safe_hints",
      "current_location": null,
      "current_character": "",
      "party_members": [],
      "last_confirmed_progress": "",
      "current_objective": null,
      "challenge": "",
      "known_constraints": [],
      "last_updated": 1786437799.7730515,
      "provenance": "inferred",
      "confidence": 0.0,
      "current_game": null,
      "recent_run_context_facts": []
    },
    "social_state": {
      "recent_active_users": [],
      "recent_chat_count": 0,
      "last_raid": {},
      "last_cheer": {}
    },
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
    "database_watermarks": {
      "counts": {
        "chat_log": 0,
        "memory_facts": 0,
        "memory_chunks": 1,
        "stream_sessions": 1,
        "stream_chat_messages": 1,
        "stream_events": 2,
        "live_session_timeline": 4,
        "promotion_events": 0,
        "viewer_promotion_profiles": 0,
        "schema_migrations": 2,
        "conversations": 0,
        "open_threads": 0
      },
      "schema_migrations": [
        {
          "component": "cognitive_replay",
          "version": 1,
          "name": "replay_metadata",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "applied_at": "2026-08-11T08:43:14.018660+00:00"
        },
        {
          "component": "conversation_continuity",
          "version": 1,
          "name": "conversation_and_open_threads",
          "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
          "applied_at": "2026-08-11T08:43:14.084294+00:00"
        }
      ],
      "final_response_digest": "",
      "final_response_present": false
    }
  },
  "maintenance": {
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
      "enabled": false,
      "is_live": false,
      "live_status_known": true,
      "active_stream_session_id": null,
      "last_transition": null,
      "title": null,
      "game": null,
      "category": null
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
      "chunks_count": 1
    },
    "beliefs": [],
    "game_state": {
      "game": "",
      "platform_version": "",
      "playthrough_type": "casual",
      "spoiler_policy": "spoiler_safe_hints",
      "current_location": null,
      "current_character": "",
      "party_members": [],
      "last_confirmed_progress": "",
      "current_objective": null,
      "challenge": "",
      "known_constraints": [],
      "last_updated": 1786437799.7730515,
      "provenance": "inferred",
      "confidence": 0.0,
      "current_game": null,
      "recent_run_context_facts": []
    },
    "social_state": {
      "recent_active_users": [],
      "recent_chat_count": 0,
      "last_raid": {},
      "last_cheer": {}
    },
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
    "database_watermarks": {
      "counts": {
        "chat_log": 0,
        "memory_facts": 0,
        "memory_chunks": 1,
        "stream_sessions": 1,
        "stream_chat_messages": 1,
        "stream_events": 2,
        "live_session_timeline": 5,
        "promotion_events": 0,
        "viewer_promotion_profiles": 0,
        "schema_migrations": 2,
        "conversations": 0,
        "open_threads": 0
      },
      "schema_migrations": [
        {
          "component": "cognitive_replay",
          "version": 1,
          "name": "replay_metadata",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "applied_at": "2026-08-11T08:43:14.018660+00:00"
        },
        {
          "component": "conversation_continuity",
          "version": 1,
          "name": "conversation_and_open_threads",
          "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
          "applied_at": "2026-08-11T08:43:14.084294+00:00"
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
    "enabled": false,
    "is_live": false,
    "live_status_known": true,
    "active_stream_session_id": null,
    "last_transition": null,
    "title": null,
    "game": null,
    "category": null
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
    "chunks_count": 1
  },
  "beliefs": [],
  "game_state": {
    "game": "",
    "platform_version": "",
    "playthrough_type": "casual",
    "spoiler_policy": "spoiler_safe_hints",
    "current_location": null,
    "current_character": "",
    "party_members": [],
    "last_confirmed_progress": "",
    "current_objective": null,
    "challenge": "",
    "known_constraints": [],
    "last_updated": 1786437799.7730515,
    "provenance": "inferred",
    "confidence": 0.0,
    "current_game": null,
    "recent_run_context_facts": []
  },
  "social_state": {
    "recent_active_users": [],
    "recent_chat_count": 0,
    "last_raid": {},
    "last_cheer": {}
  },
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
  "database_watermarks": {
    "counts": {
      "chat_log": 0,
      "memory_facts": 0,
      "memory_chunks": 1,
      "stream_sessions": 1,
      "stream_chat_messages": 1,
      "stream_events": 2,
      "live_session_timeline": 5,
      "promotion_events": 0,
      "viewer_promotion_profiles": 0,
      "schema_migrations": 2,
      "conversations": 0,
      "open_threads": 0
    },
    "schema_migrations": [
      {
        "component": "cognitive_replay",
        "version": 1,
        "name": "replay_metadata",
        "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
        "applied_at": "2026-08-11T08:43:14.018660+00:00"
      },
      {
        "component": "conversation_continuity",
        "version": 1,
        "name": "conversation_and_open_threads",
        "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
        "applied_at": "2026-08-11T08:43:14.084294+00:00"
      }
    ],
    "final_response_digest": "",
    "final_response_present": false
  }
}
```

#### Restart evidence

```json
[
  {
    "event_id": "restart",
    "old_engine_id": 2096479767248,
    "new_engine_id": 2096479606864,
    "old_engine_collected": true,
    "same_database": "C:\\Users\\Leo Nifelheim\\Documents\\Hebe\\hebe-ui\\artifacts\\cognitive-continuity-phase1\\final\\workspaces\\consolidation_format\\hebe-replay.sqlite3",
    "before_persisted_counts": {
      "chat_log": 0,
      "memory_facts": 0,
      "memory_chunks": 1,
      "stream_sessions": 1,
      "stream_chat_messages": 1,
      "stream_events": 2,
      "live_session_timeline": 4,
      "promotion_events": 0,
      "viewer_promotion_profiles": 0,
      "schema_migrations": 2,
      "conversations": 0,
      "open_threads": 0
    },
    "after_persisted_counts": {
      "chat_log": 0,
      "memory_facts": 0,
      "memory_chunks": 1,
      "stream_sessions": 1,
      "stream_chat_messages": 1,
      "stream_events": 2,
      "live_session_timeline": 4,
      "promotion_events": 0,
      "viewer_promotion_profiles": 0,
      "schema_migrations": 2,
      "conversations": 0,
      "open_threads": 0
    },
    "volatile_state_recreated": true
  }
]
```

### ffv_durable_run_format

- Status: **VERIFICATION_INCOMPLETE**
- Events: 7
- Restarts: 1
- Duration: 0.806105s
- Assertions passed/failed/skipped: 0/0/1


#### Checkpoint state

```json
{
  "start-1": {
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
      "title": "Crystal Roulette",
      "game": "Final Fantasy V",
      "category": "Final Fantasy V"
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
    "beliefs": [],
    "game_state": {
      "game": "Final Fantasy V",
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
      "last_updated": 1786464000.0,
      "provenance": "stream_context_sync",
      "confidence": 0.75,
      "current_game": "Final Fantasy V",
      "recent_run_context_facts": []
    },
    "social_state": {
      "recent_active_users": [],
      "recent_chat_count": 0,
      "last_raid": {},
      "last_cheer": {}
    },
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
        "schema_migrations": 2,
        "conversations": 0,
        "open_threads": 0
      },
      "schema_migrations": [
        {
          "component": "cognitive_replay",
          "version": 1,
          "name": "replay_metadata",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "applied_at": "2026-08-11T08:43:20.140479+00:00"
        },
        {
          "component": "conversation_continuity",
          "version": 1,
          "name": "conversation_and_open_threads",
          "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
          "applied_at": "2026-08-11T08:43:20.205670+00:00"
        }
      ],
      "final_response_digest": "",
      "final_response_present": false
    }
  },
  "roll": {
    "runtime": {
      "mode": "active",
      "hebe_sleeping": false,
      "is_running": false,
      "last_input_source": null,
      "last_intent": null,
      "last_firewall": {
        "source": "owner_stt_direct",
        "authority": "owner",
        "decision": "allow",
        "reason": "owner_direct"
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
      "title": "Crystal Roulette",
      "game": "Final Fantasy V",
      "category": "Final Fantasy V"
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
      "last_resolution": {
        "consumed": false,
        "decision": "no_conversation",
        "reason": "no_compatible_active_conversation",
        "conversation_id": "",
        "reply_act": "UNKNOWN",
        "payload": {},
        "conversation": null,
        "latency_ms": 1.4229000080376863
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
        "p50_ms": 1.4229,
        "p95_ms": 1.4229
      }
    },
    "open_threads": [],
    "memory": {
      "facts_count": 0,
      "chunks_count": 0
    },
    "beliefs": [],
    "game_state": {
      "game": "Final Fantasy V",
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
      "last_updated": 1786464000.0,
      "provenance": "stream_context_sync",
      "confidence": 0.75,
      "current_game": "Final Fantasy V",
      "recent_run_context_facts": []
    },
    "social_state": {
      "recent_active_users": [],
      "recent_chat_count": 0,
      "last_raid": {},
      "last_cheer": {}
    },
    "promotion_profiles": [],
    "actions": {
      "attempts": [],
      "speech_requests": [],
      "model_calls": [
        {
          "key": "stream_response:v1:none",
          "method": "chat"
        }
      ],
      "research_calls": []
    },
    "receipts": [],
    "emitted_outputs": [
      {
        "event_id": "1786437800.3125587",
        "emitted": true,
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "reason": "",
        "text_digest": "",
        "text_present": false
      }
    ],
    "final_emission_results": [
      {
        "event_id": "1786437800.3125587",
        "emitted": true,
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "reason": "",
        "text_digest": "",
        "text_present": false
      }
    ],
    "database_watermarks": {
      "counts": {
        "chat_log": 1,
        "memory_facts": 0,
        "memory_chunks": 0,
        "stream_sessions": 1,
        "stream_chat_messages": 0,
        "stream_events": 1,
        "live_session_timeline": 4,
        "promotion_events": 0,
        "viewer_promotion_profiles": 0,
        "schema_migrations": 2,
        "conversations": 0,
        "open_threads": 0
      },
      "schema_migrations": [
        {
          "component": "cognitive_replay",
          "version": 1,
          "name": "replay_metadata",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "applied_at": "2026-08-11T08:43:20.140479+00:00"
        },
        {
          "component": "conversation_continuity",
          "version": 1,
          "name": "conversation_and_open_threads",
          "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
          "applied_at": "2026-08-11T08:43:20.205670+00:00"
        }
      ],
      "final_response_digest": "7e07c6a5322789f6",
      "final_response_present": true
    }
  },
  "chat": {
    "runtime": {
      "mode": "active",
      "hebe_sleeping": false,
      "is_running": false,
      "last_input_source": null,
      "last_intent": null,
      "last_firewall": {
        "source": "twitch_viewer",
        "authority": "viewer",
        "decision": "allow",
        "reason": "live_viewer_message"
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
      "title": "Crystal Roulette",
      "game": "Final Fantasy V",
      "category": "Final Fantasy V"
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
      "last_resolution": {
        "consumed": false,
        "decision": "no_conversation",
        "reason": "no_compatible_active_conversation",
        "conversation_id": "",
        "reply_act": "UNKNOWN",
        "payload": {},
        "conversation": null,
        "latency_ms": 1.4229000080376863
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
        "p50_ms": 1.4229,
        "p95_ms": 1.4229
      }
    },
    "open_threads": [],
    "memory": {
      "facts_count": 0,
      "chunks_count": 0
    },
    "beliefs": [],
    "game_state": {
      "game": "Final Fantasy V",
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
      "last_updated": 1786464000.0,
      "provenance": "stream_context_sync",
      "confidence": 0.75,
      "current_game": "Final Fantasy V",
      "recent_run_context_facts": []
    },
    "social_state": {
      "recent_active_users": [
        "synthetic_viewer"
      ],
      "recent_chat_count": 1,
      "last_raid": {},
      "last_cheer": {}
    },
    "promotion_profiles": [],
    "actions": {
      "attempts": [],
      "speech_requests": [],
      "model_calls": [
        {
          "key": "stream_response:v1:none",
          "method": "chat"
        }
      ],
      "research_calls": []
    },
    "receipts": [],
    "emitted_outputs": [
      {
        "event_id": "1786437800.3125587",
        "emitted": true,
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "reason": "",
        "text_digest": "",
        "text_present": false
      }
    ],
    "final_emission_results": [
      {
        "event_id": "1786437800.3125587",
        "emitted": true,
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "reason": "",
        "text_digest": "",
        "text_present": false
      }
    ],
    "database_watermarks": {
      "counts": {
        "chat_log": 1,
        "memory_facts": 0,
        "memory_chunks": 0,
        "stream_sessions": 1,
        "stream_chat_messages": 1,
        "stream_events": 1,
        "live_session_timeline": 5,
        "promotion_events": 0,
        "viewer_promotion_profiles": 0,
        "schema_migrations": 2,
        "conversations": 0,
        "open_threads": 0
      },
      "schema_migrations": [
        {
          "component": "cognitive_replay",
          "version": 1,
          "name": "replay_metadata",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "applied_at": "2026-08-11T08:43:20.140479+00:00"
        },
        {
          "component": "conversation_continuity",
          "version": 1,
          "name": "conversation_and_open_threads",
          "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
          "applied_at": "2026-08-11T08:43:20.205670+00:00"
        }
      ],
      "final_response_digest": "",
      "final_response_present": false
    }
  },
  "reference": {
    "runtime": {
      "mode": "active",
      "hebe_sleeping": false,
      "is_running": false,
      "last_input_source": null,
      "last_intent": null,
      "last_firewall": {
        "source": "owner_stt_direct",
        "authority": "owner",
        "decision": "allow",
        "reason": "owner_direct"
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
      "title": "Crystal Roulette",
      "game": "Final Fantasy V",
      "category": "Final Fantasy V"
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
      "last_resolution": {
        "consumed": false,
        "decision": "no_conversation",
        "reason": "no_compatible_active_conversation",
        "conversation_id": "",
        "reply_act": "UNKNOWN",
        "payload": {},
        "conversation": null,
        "latency_ms": 1.4671999961137772
      },
      "legacy_pending_projection": {},
      "continuity_shadow_diff": {
        "legacy_result": false,
        "v2_result": false,
        "match": true,
        "difference_reason": ""
      },
      "shadow_metrics": {
        "total": 2,
        "matches": 2,
        "differences": 0,
        "match_rate": 1.0,
        "difference_reasons": {}
      },
      "performance": {
        "count": 2,
        "p50_ms": 1.44505,
        "p95_ms": 1.4672
      }
    },
    "open_threads": [],
    "memory": {
      "facts_count": 0,
      "chunks_count": 0
    },
    "beliefs": [],
    "game_state": {
      "game": "Final Fantasy V",
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
      "last_updated": 1786464000.0,
      "provenance": "stream_context_sync",
      "confidence": 0.75,
      "current_game": "Final Fantasy V",
      "recent_run_context_facts": []
    },
    "social_state": {
      "recent_active_users": [
        "synthetic_viewer"
      ],
      "recent_chat_count": 1,
      "last_raid": {},
      "last_cheer": {}
    },
    "promotion_profiles": [],
    "actions": {
      "attempts": [],
      "speech_requests": [],
      "model_calls": [
        {
          "key": "stream_response:v1:none",
          "method": "chat"
        },
        {
          "key": "stream_response:v1:none",
          "method": "chat"
        }
      ],
      "research_calls": []
    },
    "receipts": [],
    "emitted_outputs": [
      {
        "event_id": "1786437800.3125587",
        "emitted": true,
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "reason": "",
        "text_digest": "",
        "text_present": false
      },
      {
        "event_id": "1786437800.4196565",
        "emitted": true,
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "reason": "",
        "text_digest": "",
        "text_present": false
      }
    ],
    "final_emission_results": [
      {
        "event_id": "1786437800.3125587",
        "emitted": true,
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "reason": "",
        "text_digest": "",
        "text_present": false
      },
      {
        "event_id": "1786437800.4196565",
        "emitted": true,
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "reason": "",
        "text_digest": "",
        "text_present": false
      }
    ],
    "database_watermarks": {
      "counts": {
        "chat_log": 2,
        "memory_facts": 0,
        "memory_chunks": 0,
        "stream_sessions": 1,
        "stream_chat_messages": 1,
        "stream_events": 1,
        "live_session_timeline": 7,
        "promotion_events": 0,
        "viewer_promotion_profiles": 0,
        "schema_migrations": 2,
        "conversations": 0,
        "open_threads": 0
      },
      "schema_migrations": [
        {
          "component": "cognitive_replay",
          "version": 1,
          "name": "replay_metadata",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "applied_at": "2026-08-11T08:43:20.140479+00:00"
        },
        {
          "component": "conversation_continuity",
          "version": 1,
          "name": "conversation_and_open_threads",
          "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
          "applied_at": "2026-08-11T08:43:20.205670+00:00"
        }
      ],
      "final_response_digest": "7e07c6a5322789f6",
      "final_response_present": true
    }
  },
  "end-1": {
    "runtime": {
      "mode": "active",
      "hebe_sleeping": false,
      "is_running": false,
      "last_input_source": null,
      "last_intent": null,
      "last_firewall": {
        "source": "owner_stt_direct",
        "authority": "owner",
        "decision": "allow",
        "reason": "owner_direct"
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
      "is_live": false,
      "live_status_known": true,
      "active_stream_session_id": null,
      "last_transition": "offline",
      "title": "Crystal Roulette",
      "game": "Final Fantasy V",
      "category": "Final Fantasy V"
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
      "last_resolution": {
        "consumed": false,
        "decision": "no_conversation",
        "reason": "no_compatible_active_conversation",
        "conversation_id": "",
        "reply_act": "UNKNOWN",
        "payload": {},
        "conversation": null,
        "latency_ms": 1.4671999961137772
      },
      "legacy_pending_projection": {},
      "continuity_shadow_diff": {
        "legacy_result": false,
        "v2_result": false,
        "match": true,
        "difference_reason": ""
      },
      "shadow_metrics": {
        "total": 2,
        "matches": 2,
        "differences": 0,
        "match_rate": 1.0,
        "difference_reasons": {}
      },
      "performance": {
        "count": 2,
        "p50_ms": 1.44505,
        "p95_ms": 1.4672
      }
    },
    "open_threads": [],
    "memory": {
      "facts_count": 0,
      "chunks_count": 1
    },
    "beliefs": [],
    "game_state": {
      "game": "Final Fantasy V",
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
      "last_updated": 1786464000.0,
      "provenance": "stream_context_sync",
      "confidence": 0.75,
      "current_game": "Final Fantasy V",
      "recent_run_context_facts": []
    },
    "social_state": {
      "recent_active_users": [
        "synthetic_viewer"
      ],
      "recent_chat_count": 1,
      "last_raid": {},
      "last_cheer": {}
    },
    "promotion_profiles": [],
    "actions": {
      "attempts": [],
      "speech_requests": [],
      "model_calls": [
        {
          "key": "stream_response:v1:none",
          "method": "chat"
        },
        {
          "key": "stream_response:v1:none",
          "method": "chat"
        }
      ],
      "research_calls": []
    },
    "receipts": [],
    "emitted_outputs": [
      {
        "event_id": "1786437800.3125587",
        "emitted": true,
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "reason": "",
        "text_digest": "",
        "text_present": false
      },
      {
        "event_id": "1786437800.4196565",
        "emitted": true,
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "reason": "",
        "text_digest": "",
        "text_present": false
      }
    ],
    "final_emission_results": [
      {
        "event_id": "1786437800.3125587",
        "emitted": true,
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "reason": "",
        "text_digest": "",
        "text_present": false
      },
      {
        "event_id": "1786437800.4196565",
        "emitted": true,
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "reason": "",
        "text_digest": "",
        "text_present": false
      }
    ],
    "database_watermarks": {
      "counts": {
        "chat_log": 2,
        "memory_facts": 0,
        "memory_chunks": 1,
        "stream_sessions": 1,
        "stream_chat_messages": 1,
        "stream_events": 2,
        "live_session_timeline": 8,
        "promotion_events": 0,
        "viewer_promotion_profiles": 0,
        "schema_migrations": 2,
        "conversations": 0,
        "open_threads": 0
      },
      "schema_migrations": [
        {
          "component": "cognitive_replay",
          "version": 1,
          "name": "replay_metadata",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "applied_at": "2026-08-11T08:43:20.140479+00:00"
        },
        {
          "component": "conversation_continuity",
          "version": 1,
          "name": "conversation_and_open_threads",
          "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
          "applied_at": "2026-08-11T08:43:20.205670+00:00"
        }
      ],
      "final_response_digest": "7e07c6a5322789f6",
      "final_response_present": true
    }
  },
  "restart": {
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
      "enabled": false,
      "is_live": false,
      "live_status_known": false,
      "active_stream_session_id": null,
      "last_transition": null,
      "title": null,
      "game": null,
      "category": null
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
      "chunks_count": 1
    },
    "beliefs": [],
    "game_state": {
      "game": "",
      "platform_version": "",
      "playthrough_type": "casual",
      "spoiler_policy": "spoiler_safe_hints",
      "current_location": null,
      "current_character": "",
      "party_members": [],
      "last_confirmed_progress": "",
      "current_objective": null,
      "challenge": "",
      "known_constraints": [],
      "last_updated": 1786437800.6100397,
      "provenance": "inferred",
      "confidence": 0.0,
      "current_game": null,
      "recent_run_context_facts": []
    },
    "social_state": {
      "recent_active_users": [],
      "recent_chat_count": 0,
      "last_raid": {},
      "last_cheer": {}
    },
    "promotion_profiles": [],
    "actions": {
      "attempts": [],
      "speech_requests": [],
      "model_calls": [
        {
          "key": "stream_response:v1:none",
          "method": "chat"
        },
        {
          "key": "stream_response:v1:none",
          "method": "chat"
        }
      ],
      "research_calls": []
    },
    "receipts": [],
    "emitted_outputs": [
      {
        "event_id": "1786437800.3125587",
        "emitted": true,
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "reason": "",
        "text_digest": "",
        "text_present": false
      },
      {
        "event_id": "1786437800.4196565",
        "emitted": true,
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "reason": "",
        "text_digest": "",
        "text_present": false
      }
    ],
    "final_emission_results": [
      {
        "event_id": "1786437800.3125587",
        "emitted": true,
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "reason": "",
        "text_digest": "",
        "text_present": false
      },
      {
        "event_id": "1786437800.4196565",
        "emitted": true,
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "reason": "",
        "text_digest": "",
        "text_present": false
      }
    ],
    "database_watermarks": {
      "counts": {
        "chat_log": 2,
        "memory_facts": 0,
        "memory_chunks": 1,
        "stream_sessions": 1,
        "stream_chat_messages": 1,
        "stream_events": 2,
        "live_session_timeline": 8,
        "promotion_events": 0,
        "viewer_promotion_profiles": 0,
        "schema_migrations": 2,
        "conversations": 0,
        "open_threads": 0
      },
      "schema_migrations": [
        {
          "component": "cognitive_replay",
          "version": 1,
          "name": "replay_metadata",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "applied_at": "2026-08-11T08:43:20.140479+00:00"
        },
        {
          "component": "conversation_continuity",
          "version": 1,
          "name": "conversation_and_open_threads",
          "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
          "applied_at": "2026-08-11T08:43:20.205670+00:00"
        }
      ],
      "final_response_digest": "",
      "final_response_present": false
    }
  },
  "start-2": {
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
      "active_stream_session_id": 2,
      "last_transition": "online",
      "title": "Crystal Roulette continues",
      "game": "Final Fantasy V",
      "category": "Final Fantasy V"
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
      "chunks_count": 1
    },
    "beliefs": [],
    "game_state": {
      "game": "Final Fantasy V",
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
      "last_updated": 1786550400.0,
      "provenance": "stream_context_sync",
      "confidence": 0.75,
      "current_game": "Final Fantasy V",
      "recent_run_context_facts": []
    },
    "social_state": {
      "recent_active_users": [],
      "recent_chat_count": 0,
      "last_raid": {},
      "last_cheer": {}
    },
    "promotion_profiles": [],
    "actions": {
      "attempts": [],
      "speech_requests": [],
      "model_calls": [
        {
          "key": "stream_response:v1:none",
          "method": "chat"
        },
        {
          "key": "stream_response:v1:none",
          "method": "chat"
        }
      ],
      "research_calls": []
    },
    "receipts": [],
    "emitted_outputs": [
      {
        "event_id": "1786437800.3125587",
        "emitted": true,
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "reason": "",
        "text_digest": "",
        "text_present": false
      },
      {
        "event_id": "1786437800.4196565",
        "emitted": true,
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "reason": "",
        "text_digest": "",
        "text_present": false
      }
    ],
    "final_emission_results": [
      {
        "event_id": "1786437800.3125587",
        "emitted": true,
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "reason": "",
        "text_digest": "",
        "text_present": false
      },
      {
        "event_id": "1786437800.4196565",
        "emitted": true,
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "reason": "",
        "text_digest": "",
        "text_present": false
      }
    ],
    "database_watermarks": {
      "counts": {
        "chat_log": 2,
        "memory_facts": 0,
        "memory_chunks": 1,
        "stream_sessions": 2,
        "stream_chat_messages": 1,
        "stream_events": 3,
        "live_session_timeline": 10,
        "promotion_events": 0,
        "viewer_promotion_profiles": 0,
        "schema_migrations": 2,
        "conversations": 0,
        "open_threads": 0
      },
      "schema_migrations": [
        {
          "component": "cognitive_replay",
          "version": 1,
          "name": "replay_metadata",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "applied_at": "2026-08-11T08:43:20.140479+00:00"
        },
        {
          "component": "conversation_continuity",
          "version": 1,
          "name": "conversation_and_open_threads",
          "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
          "applied_at": "2026-08-11T08:43:20.205670+00:00"
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
    "active_stream_session_id": 2,
    "last_transition": "online",
    "title": "Crystal Roulette continues",
    "game": "Final Fantasy V",
    "category": "Final Fantasy V"
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
    "chunks_count": 1
  },
  "beliefs": [],
  "game_state": {
    "game": "Final Fantasy V",
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
    "last_updated": 1786550400.0,
    "provenance": "stream_context_sync",
    "confidence": 0.75,
    "current_game": "Final Fantasy V",
    "recent_run_context_facts": []
  },
  "social_state": {
    "recent_active_users": [],
    "recent_chat_count": 0,
    "last_raid": {},
    "last_cheer": {}
  },
  "promotion_profiles": [],
  "actions": {
    "attempts": [],
    "speech_requests": [],
    "model_calls": [
      {
        "key": "stream_response:v1:none",
        "method": "chat"
      },
      {
        "key": "stream_response:v1:none",
        "method": "chat"
      }
    ],
    "research_calls": []
  },
  "receipts": [],
  "emitted_outputs": [
    {
      "event_id": "1786437800.3125587",
      "emitted": true,
      "route": "local_owner_reply",
      "targets": [
        "local_ui"
      ],
      "reason": "",
      "text_digest": "",
      "text_present": false
    },
    {
      "event_id": "1786437800.4196565",
      "emitted": true,
      "route": "local_owner_reply",
      "targets": [
        "local_ui"
      ],
      "reason": "",
      "text_digest": "",
      "text_present": false
    }
  ],
  "final_emission_results": [
    {
      "event_id": "1786437800.3125587",
      "emitted": true,
      "route": "local_owner_reply",
      "targets": [
        "local_ui"
      ],
      "reason": "",
      "text_digest": "",
      "text_present": false
    },
    {
      "event_id": "1786437800.4196565",
      "emitted": true,
      "route": "local_owner_reply",
      "targets": [
        "local_ui"
      ],
      "reason": "",
      "text_digest": "",
      "text_present": false
    }
  ],
  "database_watermarks": {
    "counts": {
      "chat_log": 2,
      "memory_facts": 0,
      "memory_chunks": 1,
      "stream_sessions": 2,
      "stream_chat_messages": 1,
      "stream_events": 3,
      "live_session_timeline": 10,
      "promotion_events": 0,
      "viewer_promotion_profiles": 0,
      "schema_migrations": 2,
      "conversations": 0,
      "open_threads": 0
    },
    "schema_migrations": [
      {
        "component": "cognitive_replay",
        "version": 1,
        "name": "replay_metadata",
        "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
        "applied_at": "2026-08-11T08:43:20.140479+00:00"
      },
      {
        "component": "conversation_continuity",
        "version": 1,
        "name": "conversation_and_open_threads",
        "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
        "applied_at": "2026-08-11T08:43:20.205670+00:00"
      }
    ],
    "final_response_digest": "",
    "final_response_present": false
  }
}
```

#### Restart evidence

```json
[
  {
    "event_id": "restart",
    "old_engine_id": 2096479615824,
    "new_engine_id": 2096481065360,
    "old_engine_collected": true,
    "same_database": "C:\\Users\\Leo Nifelheim\\Documents\\Hebe\\hebe-ui\\artifacts\\cognitive-continuity-phase1\\final\\workspaces\\ffv_durable_run_format\\hebe-replay.sqlite3",
    "before_persisted_counts": {
      "chat_log": 2,
      "memory_facts": 0,
      "memory_chunks": 1,
      "stream_sessions": 1,
      "stream_chat_messages": 1,
      "stream_events": 2,
      "live_session_timeline": 8,
      "promotion_events": 0,
      "viewer_promotion_profiles": 0,
      "schema_migrations": 2,
      "conversations": 0,
      "open_threads": 0
    },
    "after_persisted_counts": {
      "chat_log": 2,
      "memory_facts": 0,
      "memory_chunks": 1,
      "stream_sessions": 1,
      "stream_chat_messages": 1,
      "stream_events": 2,
      "live_session_timeline": 8,
      "promotion_events": 0,
      "viewer_promotion_profiles": 0,
      "schema_migrations": 2,
      "conversations": 0,
      "open_threads": 0
    },
    "volatile_state_recreated": true
  }
]
```

### ivanxi_resub_promo_restart

- Status: **VERIFIED**
- Events: 7
- Restarts: 1
- Duration: 0.740286s
- Assertions passed/failed/skipped: 5/0/0


#### Checkpoint state

```json
{
  "stream-1-start": {
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
      "title": "[ENG/ESP] Crystal Roulette — Final Fantasy V",
      "game": "Final Fantasy V",
      "category": "Final Fantasy V"
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
    "beliefs": [],
    "game_state": {
      "game": "Final Fantasy V",
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
      "last_updated": 1786464000.0,
      "provenance": "stream_context_sync",
      "confidence": 0.75,
      "current_game": "Final Fantasy V",
      "recent_run_context_facts": []
    },
    "social_state": {
      "recent_active_users": [],
      "recent_chat_count": 0,
      "last_raid": {},
      "last_cheer": {}
    },
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
        "schema_migrations": 2,
        "conversations": 0,
        "open_threads": 0
      },
      "schema_migrations": [
        {
          "component": "cognitive_replay",
          "version": 1,
          "name": "replay_metadata",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "applied_at": "2026-08-11T08:43:21.032363+00:00"
        },
        {
          "component": "conversation_continuity",
          "version": 1,
          "name": "conversation_and_open_threads",
          "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
          "applied_at": "2026-08-11T08:43:21.097850+00:00"
        }
      ],
      "final_response_digest": "",
      "final_response_present": false
    }
  },
  "ivanxi-resub": {
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
      "title": "[ENG/ESP] Crystal Roulette — Final Fantasy V",
      "game": "Final Fantasy V",
      "category": "Final Fantasy V"
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
    "beliefs": [],
    "game_state": {
      "game": "Final Fantasy V",
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
      "last_updated": 1786464000.0,
      "provenance": "stream_context_sync",
      "confidence": 0.75,
      "current_game": "Final Fantasy V",
      "recent_run_context_facts": []
    },
    "social_state": {
      "recent_active_users": [],
      "recent_chat_count": 0,
      "last_raid": {},
      "last_cheer": {}
    },
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
        "schema_migrations": 2,
        "conversations": 0,
        "open_threads": 0
      },
      "schema_migrations": [
        {
          "component": "cognitive_replay",
          "version": 1,
          "name": "replay_metadata",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "applied_at": "2026-08-11T08:43:21.032363+00:00"
        },
        {
          "component": "conversation_continuity",
          "version": 1,
          "name": "conversation_and_open_threads",
          "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
          "applied_at": "2026-08-11T08:43:21.097850+00:00"
        }
      ],
      "final_response_digest": "",
      "final_response_present": false
    }
  },
  "ivanxi-chat": {
    "runtime": {
      "mode": "active",
      "hebe_sleeping": false,
      "is_running": false,
      "last_input_source": null,
      "last_intent": null,
      "last_firewall": {
        "source": "twitch_viewer",
        "authority": "viewer",
        "decision": "allow",
        "reason": "live_viewer_message"
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
      "title": "[ENG/ESP] Crystal Roulette — Final Fantasy V",
      "game": "Final Fantasy V",
      "category": "Final Fantasy V"
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
    "beliefs": [],
    "game_state": {
      "game": "Final Fantasy V",
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
      "last_updated": 1786464000.0,
      "provenance": "stream_context_sync",
      "confidence": 0.75,
      "current_game": "Final Fantasy V",
      "recent_run_context_facts": []
    },
    "social_state": {
      "recent_active_users": [
        "ivanxi_kun"
      ],
      "recent_chat_count": 1,
      "last_raid": {},
      "last_cheer": {}
    },
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
    "database_watermarks": {
      "counts": {
        "chat_log": 0,
        "memory_facts": 0,
        "memory_chunks": 0,
        "stream_sessions": 1,
        "stream_chat_messages": 1,
        "stream_events": 1,
        "live_session_timeline": 3,
        "promotion_events": 0,
        "viewer_promotion_profiles": 0,
        "schema_migrations": 2,
        "conversations": 0,
        "open_threads": 0
      },
      "schema_migrations": [
        {
          "component": "cognitive_replay",
          "version": 1,
          "name": "replay_metadata",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "applied_at": "2026-08-11T08:43:21.032363+00:00"
        },
        {
          "component": "conversation_continuity",
          "version": 1,
          "name": "conversation_and_open_threads",
          "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
          "applied_at": "2026-08-11T08:43:21.097850+00:00"
        }
      ],
      "final_response_digest": "",
      "final_response_present": false
    }
  },
  "owner-promo": {
    "runtime": {
      "mode": "active",
      "hebe_sleeping": false,
      "is_running": false,
      "last_input_source": null,
      "last_intent": null,
      "last_firewall": {
        "source": "owner_stt_direct",
        "authority": "owner",
        "decision": "allow",
        "reason": "owner_direct"
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
      "title": "[ENG/ESP] Crystal Roulette — Final Fantasy V",
      "game": "Final Fantasy V",
      "category": "Final Fantasy V"
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
      "last_resolution": {
        "consumed": false,
        "decision": "no_conversation",
        "reason": "no_compatible_active_conversation",
        "conversation_id": "",
        "reply_act": "UNKNOWN",
        "payload": {},
        "conversation": null,
        "latency_ms": 1.3291999930515885
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
        "p50_ms": 1.3292,
        "p95_ms": 1.3292
      }
    },
    "open_threads": [],
    "memory": {
      "facts_count": 0,
      "chunks_count": 0
    },
    "beliefs": [],
    "game_state": {
      "game": "Final Fantasy V",
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
      "last_updated": 1786464000.0,
      "provenance": "stream_context_sync",
      "confidence": 0.75,
      "current_game": "Final Fantasy V",
      "recent_run_context_facts": []
    },
    "social_state": {
      "recent_active_users": [
        "ivanxi_kun"
      ],
      "recent_chat_count": 1,
      "last_raid": {},
      "last_cheer": {}
    },
    "promotion_profiles": [
      {
        "twitch_user_id": "42",
        "current_login": "ivanxi_kun",
        "display_name": "ivanxi_kun",
        "auto_promo_mode": "first_message_each_stream",
        "created_by": "owner_command",
        "last_promoted_stream_id": "1",
        "owner_locked": 1,
        "active": 1
      }
    ],
    "actions": {
      "attempts": [
        {
          "operation": "twitch.shoutout",
          "payload": {
            "target": "ivanxi_kun",
            "command": "!so ivanxi_kun"
          },
          "outcome": {
            "success": true,
            "status": "sent",
            "message_id": "replay-so-1"
          }
        }
      ],
      "speech_requests": [],
      "model_calls": [],
      "research_calls": []
    },
    "receipts": [
      {
        "id": "promo_4706d2db511c500f94a3499dce27d6a1",
        "stream_session_id": "1",
        "source_event_id": "1786437801.2469096",
        "requested_by": "leo",
        "resolved_twitch_user_id": "42",
        "resolved_login": "ivanxi_kun",
        "trigger_type": "owner_learn_and_execute",
        "execution_status": "sent",
        "twitch_message_id": "replay-so-1",
        "failure_reason": ""
      }
    ],
    "emitted_outputs": [
      {
        "event_id": "1786437801.2469096",
        "emitted": true,
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "reason": "",
        "text_digest": "",
        "text_present": false
      }
    ],
    "final_emission_results": [
      {
        "event_id": "1786437801.2469096",
        "emitted": true,
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "reason": "",
        "text_digest": "",
        "text_present": false
      }
    ],
    "database_watermarks": {
      "counts": {
        "chat_log": 1,
        "memory_facts": 0,
        "memory_chunks": 0,
        "stream_sessions": 1,
        "stream_chat_messages": 1,
        "stream_events": 1,
        "live_session_timeline": 5,
        "promotion_events": 1,
        "viewer_promotion_profiles": 1,
        "schema_migrations": 2,
        "conversations": 0,
        "open_threads": 0
      },
      "schema_migrations": [
        {
          "component": "cognitive_replay",
          "version": 1,
          "name": "replay_metadata",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "applied_at": "2026-08-11T08:43:21.032363+00:00"
        },
        {
          "component": "conversation_continuity",
          "version": 1,
          "name": "conversation_and_open_threads",
          "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
          "applied_at": "2026-08-11T08:43:21.097850+00:00"
        }
      ],
      "final_response_digest": "",
      "final_response_present": false
    }
  },
  "stream-1-end": {
    "runtime": {
      "mode": "active",
      "hebe_sleeping": false,
      "is_running": false,
      "last_input_source": null,
      "last_intent": null,
      "last_firewall": {
        "source": "owner_stt_direct",
        "authority": "owner",
        "decision": "allow",
        "reason": "owner_direct"
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
      "is_live": false,
      "live_status_known": true,
      "active_stream_session_id": null,
      "last_transition": "offline",
      "title": "[ENG/ESP] Crystal Roulette — Final Fantasy V",
      "game": "Final Fantasy V",
      "category": "Final Fantasy V"
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
      "last_resolution": {
        "consumed": false,
        "decision": "no_conversation",
        "reason": "no_compatible_active_conversation",
        "conversation_id": "",
        "reply_act": "UNKNOWN",
        "payload": {},
        "conversation": null,
        "latency_ms": 1.3291999930515885
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
        "p50_ms": 1.3292,
        "p95_ms": 1.3292
      }
    },
    "open_threads": [],
    "memory": {
      "facts_count": 0,
      "chunks_count": 1
    },
    "beliefs": [],
    "game_state": {
      "game": "Final Fantasy V",
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
      "last_updated": 1786464000.0,
      "provenance": "stream_context_sync",
      "confidence": 0.75,
      "current_game": "Final Fantasy V",
      "recent_run_context_facts": []
    },
    "social_state": {
      "recent_active_users": [
        "ivanxi_kun"
      ],
      "recent_chat_count": 1,
      "last_raid": {},
      "last_cheer": {}
    },
    "promotion_profiles": [
      {
        "twitch_user_id": "42",
        "current_login": "ivanxi_kun",
        "display_name": "ivanxi_kun",
        "auto_promo_mode": "first_message_each_stream",
        "created_by": "owner_command",
        "last_promoted_stream_id": "1",
        "owner_locked": 1,
        "active": 1
      }
    ],
    "actions": {
      "attempts": [
        {
          "operation": "twitch.shoutout",
          "payload": {
            "target": "ivanxi_kun",
            "command": "!so ivanxi_kun"
          },
          "outcome": {
            "success": true,
            "status": "sent",
            "message_id": "replay-so-1"
          }
        }
      ],
      "speech_requests": [],
      "model_calls": [],
      "research_calls": []
    },
    "receipts": [
      {
        "id": "promo_4706d2db511c500f94a3499dce27d6a1",
        "stream_session_id": "1",
        "source_event_id": "1786437801.2469096",
        "requested_by": "leo",
        "resolved_twitch_user_id": "42",
        "resolved_login": "ivanxi_kun",
        "trigger_type": "owner_learn_and_execute",
        "execution_status": "sent",
        "twitch_message_id": "replay-so-1",
        "failure_reason": ""
      }
    ],
    "emitted_outputs": [
      {
        "event_id": "1786437801.2469096",
        "emitted": true,
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "reason": "",
        "text_digest": "",
        "text_present": false
      }
    ],
    "final_emission_results": [
      {
        "event_id": "1786437801.2469096",
        "emitted": true,
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "reason": "",
        "text_digest": "",
        "text_present": false
      }
    ],
    "database_watermarks": {
      "counts": {
        "chat_log": 1,
        "memory_facts": 0,
        "memory_chunks": 1,
        "stream_sessions": 1,
        "stream_chat_messages": 1,
        "stream_events": 2,
        "live_session_timeline": 6,
        "promotion_events": 1,
        "viewer_promotion_profiles": 1,
        "schema_migrations": 2,
        "conversations": 0,
        "open_threads": 0
      },
      "schema_migrations": [
        {
          "component": "cognitive_replay",
          "version": 1,
          "name": "replay_metadata",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "applied_at": "2026-08-11T08:43:21.032363+00:00"
        },
        {
          "component": "conversation_continuity",
          "version": 1,
          "name": "conversation_and_open_threads",
          "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
          "applied_at": "2026-08-11T08:43:21.097850+00:00"
        }
      ],
      "final_response_digest": "",
      "final_response_present": false
    }
  },
  "restart-1": {
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
      "enabled": false,
      "is_live": false,
      "live_status_known": false,
      "active_stream_session_id": null,
      "last_transition": null,
      "title": null,
      "game": null,
      "category": null
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
      "chunks_count": 1
    },
    "beliefs": [],
    "game_state": {
      "game": "",
      "platform_version": "",
      "playthrough_type": "casual",
      "spoiler_policy": "spoiler_safe_hints",
      "current_location": null,
      "current_character": "",
      "party_members": [],
      "last_confirmed_progress": "",
      "current_objective": null,
      "challenge": "",
      "known_constraints": [],
      "last_updated": 1786437801.450346,
      "provenance": "inferred",
      "confidence": 0.0,
      "current_game": null,
      "recent_run_context_facts": []
    },
    "social_state": {
      "recent_active_users": [],
      "recent_chat_count": 0,
      "last_raid": {},
      "last_cheer": {}
    },
    "promotion_profiles": [
      {
        "twitch_user_id": "42",
        "current_login": "ivanxi_kun",
        "display_name": "ivanxi_kun",
        "auto_promo_mode": "first_message_each_stream",
        "created_by": "owner_command",
        "last_promoted_stream_id": "1",
        "owner_locked": 1,
        "active": 1
      }
    ],
    "actions": {
      "attempts": [
        {
          "operation": "twitch.shoutout",
          "payload": {
            "target": "ivanxi_kun",
            "command": "!so ivanxi_kun"
          },
          "outcome": {
            "success": true,
            "status": "sent",
            "message_id": "replay-so-1"
          }
        }
      ],
      "speech_requests": [],
      "model_calls": [],
      "research_calls": []
    },
    "receipts": [
      {
        "id": "promo_4706d2db511c500f94a3499dce27d6a1",
        "stream_session_id": "1",
        "source_event_id": "1786437801.2469096",
        "requested_by": "leo",
        "resolved_twitch_user_id": "42",
        "resolved_login": "ivanxi_kun",
        "trigger_type": "owner_learn_and_execute",
        "execution_status": "sent",
        "twitch_message_id": "replay-so-1",
        "failure_reason": ""
      }
    ],
    "emitted_outputs": [
      {
        "event_id": "1786437801.2469096",
        "emitted": true,
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "reason": "",
        "text_digest": "",
        "text_present": false
      }
    ],
    "final_emission_results": [
      {
        "event_id": "1786437801.2469096",
        "emitted": true,
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "reason": "",
        "text_digest": "",
        "text_present": false
      }
    ],
    "database_watermarks": {
      "counts": {
        "chat_log": 1,
        "memory_facts": 0,
        "memory_chunks": 1,
        "stream_sessions": 1,
        "stream_chat_messages": 1,
        "stream_events": 2,
        "live_session_timeline": 6,
        "promotion_events": 1,
        "viewer_promotion_profiles": 1,
        "schema_migrations": 2,
        "conversations": 0,
        "open_threads": 0
      },
      "schema_migrations": [
        {
          "component": "cognitive_replay",
          "version": 1,
          "name": "replay_metadata",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "applied_at": "2026-08-11T08:43:21.032363+00:00"
        },
        {
          "component": "conversation_continuity",
          "version": 1,
          "name": "conversation_and_open_threads",
          "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
          "applied_at": "2026-08-11T08:43:21.097850+00:00"
        }
      ],
      "final_response_digest": "",
      "final_response_present": false
    }
  },
  "stream-2-start": {
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
      "active_stream_session_id": 2,
      "last_transition": "online",
      "title": "[ENG/ESP] Crystal Roulette continues — Final Fantasy V",
      "game": "Final Fantasy V",
      "category": "Final Fantasy V"
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
      "chunks_count": 1
    },
    "beliefs": [],
    "game_state": {
      "game": "Final Fantasy V",
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
      "last_updated": 1786550400.0,
      "provenance": "stream_context_sync",
      "confidence": 0.75,
      "current_game": "Final Fantasy V",
      "recent_run_context_facts": []
    },
    "social_state": {
      "recent_active_users": [],
      "recent_chat_count": 0,
      "last_raid": {},
      "last_cheer": {}
    },
    "promotion_profiles": [
      {
        "twitch_user_id": "42",
        "current_login": "ivanxi_kun",
        "display_name": "ivanxi_kun",
        "auto_promo_mode": "first_message_each_stream",
        "created_by": "owner_command",
        "last_promoted_stream_id": "1",
        "owner_locked": 1,
        "active": 1
      }
    ],
    "actions": {
      "attempts": [
        {
          "operation": "twitch.shoutout",
          "payload": {
            "target": "ivanxi_kun",
            "command": "!so ivanxi_kun"
          },
          "outcome": {
            "success": true,
            "status": "sent",
            "message_id": "replay-so-1"
          }
        }
      ],
      "speech_requests": [],
      "model_calls": [],
      "research_calls": []
    },
    "receipts": [
      {
        "id": "promo_4706d2db511c500f94a3499dce27d6a1",
        "stream_session_id": "1",
        "source_event_id": "1786437801.2469096",
        "requested_by": "leo",
        "resolved_twitch_user_id": "42",
        "resolved_login": "ivanxi_kun",
        "trigger_type": "owner_learn_and_execute",
        "execution_status": "sent",
        "twitch_message_id": "replay-so-1",
        "failure_reason": ""
      }
    ],
    "emitted_outputs": [
      {
        "event_id": "1786437801.2469096",
        "emitted": true,
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "reason": "",
        "text_digest": "",
        "text_present": false
      }
    ],
    "final_emission_results": [
      {
        "event_id": "1786437801.2469096",
        "emitted": true,
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "reason": "",
        "text_digest": "",
        "text_present": false
      }
    ],
    "database_watermarks": {
      "counts": {
        "chat_log": 1,
        "memory_facts": 0,
        "memory_chunks": 1,
        "stream_sessions": 2,
        "stream_chat_messages": 1,
        "stream_events": 3,
        "live_session_timeline": 8,
        "promotion_events": 1,
        "viewer_promotion_profiles": 1,
        "schema_migrations": 2,
        "conversations": 0,
        "open_threads": 0
      },
      "schema_migrations": [
        {
          "component": "cognitive_replay",
          "version": 1,
          "name": "replay_metadata",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "applied_at": "2026-08-11T08:43:21.032363+00:00"
        },
        {
          "component": "conversation_continuity",
          "version": 1,
          "name": "conversation_and_open_threads",
          "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
          "applied_at": "2026-08-11T08:43:21.097850+00:00"
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
    "active_stream_session_id": 2,
    "last_transition": "online",
    "title": "[ENG/ESP] Crystal Roulette continues — Final Fantasy V",
    "game": "Final Fantasy V",
    "category": "Final Fantasy V"
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
    "chunks_count": 1
  },
  "beliefs": [],
  "game_state": {
    "game": "Final Fantasy V",
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
    "last_updated": 1786550400.0,
    "provenance": "stream_context_sync",
    "confidence": 0.75,
    "current_game": "Final Fantasy V",
    "recent_run_context_facts": []
  },
  "social_state": {
    "recent_active_users": [],
    "recent_chat_count": 0,
    "last_raid": {},
    "last_cheer": {}
  },
  "promotion_profiles": [
    {
      "twitch_user_id": "42",
      "current_login": "ivanxi_kun",
      "display_name": "ivanxi_kun",
      "auto_promo_mode": "first_message_each_stream",
      "created_by": "owner_command",
      "last_promoted_stream_id": "1",
      "owner_locked": 1,
      "active": 1
    }
  ],
  "actions": {
    "attempts": [
      {
        "operation": "twitch.shoutout",
        "payload": {
          "target": "ivanxi_kun",
          "command": "!so ivanxi_kun"
        },
        "outcome": {
          "success": true,
          "status": "sent",
          "message_id": "replay-so-1"
        }
      }
    ],
    "speech_requests": [],
    "model_calls": [],
    "research_calls": []
  },
  "receipts": [
    {
      "id": "promo_4706d2db511c500f94a3499dce27d6a1",
      "stream_session_id": "1",
      "source_event_id": "1786437801.2469096",
      "requested_by": "leo",
      "resolved_twitch_user_id": "42",
      "resolved_login": "ivanxi_kun",
      "trigger_type": "owner_learn_and_execute",
      "execution_status": "sent",
      "twitch_message_id": "replay-so-1",
      "failure_reason": ""
    }
  ],
  "emitted_outputs": [
    {
      "event_id": "1786437801.2469096",
      "emitted": true,
      "route": "local_owner_reply",
      "targets": [
        "local_ui"
      ],
      "reason": "",
      "text_digest": "",
      "text_present": false
    }
  ],
  "final_emission_results": [
    {
      "event_id": "1786437801.2469096",
      "emitted": true,
      "route": "local_owner_reply",
      "targets": [
        "local_ui"
      ],
      "reason": "",
      "text_digest": "",
      "text_present": false
    }
  ],
  "database_watermarks": {
    "counts": {
      "chat_log": 1,
      "memory_facts": 0,
      "memory_chunks": 1,
      "stream_sessions": 2,
      "stream_chat_messages": 1,
      "stream_events": 3,
      "live_session_timeline": 8,
      "promotion_events": 1,
      "viewer_promotion_profiles": 1,
      "schema_migrations": 2,
      "conversations": 0,
      "open_threads": 0
    },
    "schema_migrations": [
      {
        "component": "cognitive_replay",
        "version": 1,
        "name": "replay_metadata",
        "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
        "applied_at": "2026-08-11T08:43:21.032363+00:00"
      },
      {
        "component": "conversation_continuity",
        "version": 1,
        "name": "conversation_and_open_threads",
        "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
        "applied_at": "2026-08-11T08:43:21.097850+00:00"
      }
    ],
    "final_response_digest": "",
    "final_response_present": false
  }
}
```

#### Restart evidence

```json
[
  {
    "event_id": "restart-1",
    "old_engine_id": 2096479948944,
    "new_engine_id": 2096479859024,
    "old_engine_collected": true,
    "same_database": "C:\\Users\\Leo Nifelheim\\Documents\\Hebe\\hebe-ui\\artifacts\\cognitive-continuity-phase1\\final\\workspaces\\ivanxi_resub_promo_restart\\hebe-replay.sqlite3",
    "before_persisted_counts": {
      "chat_log": 1,
      "memory_facts": 0,
      "memory_chunks": 1,
      "stream_sessions": 1,
      "stream_chat_messages": 1,
      "stream_events": 2,
      "live_session_timeline": 6,
      "promotion_events": 1,
      "viewer_promotion_profiles": 1,
      "schema_migrations": 2,
      "conversations": 0,
      "open_threads": 0
    },
    "after_persisted_counts": {
      "chat_log": 1,
      "memory_facts": 0,
      "memory_chunks": 1,
      "stream_sessions": 1,
      "stream_chat_messages": 1,
      "stream_events": 2,
      "live_session_timeline": 6,
      "promotion_events": 1,
      "viewer_promotion_profiles": 1,
      "schema_migrations": 2,
      "conversations": 0,
      "open_threads": 0
    },
    "volatile_state_recreated": true
  }
]
```

### owner_correction_format

- Status: **VERIFICATION_INCOMPLETE**
- Events: 6
- Restarts: 1
- Duration: 6.30366s
- Assertions passed/failed/skipped: 0/0/1


#### Checkpoint state

```json
{
  "start": {
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
      "title": null,
      "game": "Final Fantasy V",
      "category": "Final Fantasy V"
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
    "beliefs": [],
    "game_state": {
      "game": "Final Fantasy V",
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
      "last_updated": 1786464000.0,
      "provenance": "stream_context_sync",
      "confidence": 0.75,
      "current_game": "Final Fantasy V",
      "recent_run_context_facts": []
    },
    "social_state": {
      "recent_active_users": [],
      "recent_chat_count": 0,
      "last_raid": {},
      "last_cheer": {}
    },
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
        "schema_migrations": 2,
        "conversations": 0,
        "open_threads": 0
      },
      "schema_migrations": [
        {
          "component": "cognitive_replay",
          "version": 1,
          "name": "replay_metadata",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "applied_at": "2026-08-11T08:43:21.843413+00:00"
        },
        {
          "component": "conversation_continuity",
          "version": 1,
          "name": "conversation_and_open_threads",
          "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
          "applied_at": "2026-08-11T08:43:21.928483+00:00"
        }
      ],
      "final_response_digest": "",
      "final_response_present": false
    }
  },
  "third": {
    "runtime": {
      "mode": "active",
      "hebe_sleeping": false,
      "is_running": false,
      "last_input_source": null,
      "last_intent": null,
      "last_firewall": {
        "source": "owner_stt_direct",
        "authority": "owner",
        "decision": "allow",
        "reason": "owner_direct"
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
      "title": null,
      "game": "Final Fantasy V",
      "category": "Final Fantasy V"
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
      "last_resolution": {
        "consumed": false,
        "decision": "no_conversation",
        "reason": "no_compatible_active_conversation",
        "conversation_id": "",
        "reply_act": "UNKNOWN",
        "payload": {},
        "conversation": null,
        "latency_ms": 1.6226000152528286
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
        "p50_ms": 1.6226,
        "p95_ms": 1.6226
      }
    },
    "open_threads": [],
    "memory": {
      "facts_count": 0,
      "chunks_count": 0
    },
    "beliefs": [],
    "game_state": {
      "game": "Final Fantasy V",
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
      "last_updated": 1786464000.0,
      "provenance": "stream_context_sync",
      "confidence": 0.75,
      "current_game": "Final Fantasy V",
      "recent_run_context_facts": []
    },
    "social_state": {
      "recent_active_users": [],
      "recent_chat_count": 0,
      "last_raid": {},
      "last_cheer": {}
    },
    "promotion_profiles": [],
    "actions": {
      "attempts": [],
      "speech_requests": [],
      "model_calls": [
        {
          "key": "stream_response:v1:none",
          "method": "chat"
        }
      ],
      "research_calls": []
    },
    "receipts": [],
    "emitted_outputs": [
      {
        "event_id": "1786437802.0719497",
        "emitted": true,
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "reason": "",
        "text_digest": "",
        "text_present": false
      }
    ],
    "final_emission_results": [
      {
        "event_id": "1786437802.0719497",
        "emitted": true,
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "reason": "",
        "text_digest": "",
        "text_present": false
      }
    ],
    "database_watermarks": {
      "counts": {
        "chat_log": 1,
        "memory_facts": 0,
        "memory_chunks": 0,
        "stream_sessions": 1,
        "stream_chat_messages": 0,
        "stream_events": 1,
        "live_session_timeline": 4,
        "promotion_events": 0,
        "viewer_promotion_profiles": 0,
        "schema_migrations": 2,
        "conversations": 0,
        "open_threads": 0
      },
      "schema_migrations": [
        {
          "component": "cognitive_replay",
          "version": 1,
          "name": "replay_metadata",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "applied_at": "2026-08-11T08:43:21.843413+00:00"
        },
        {
          "component": "conversation_continuity",
          "version": 1,
          "name": "conversation_and_open_threads",
          "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
          "applied_at": "2026-08-11T08:43:21.928483+00:00"
        }
      ],
      "final_response_digest": "7e07c6a5322789f6",
      "final_response_present": true
    }
  },
  "correction": {
    "runtime": {
      "mode": "active",
      "hebe_sleeping": false,
      "is_running": false,
      "last_input_source": null,
      "last_intent": null,
      "last_firewall": {
        "source": "owner_stt_direct",
        "authority": "owner",
        "decision": "allow",
        "reason": "owner_direct"
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
      "title": null,
      "game": "Final Fantasy V",
      "category": "Final Fantasy V"
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
      "last_resolution": {
        "consumed": false,
        "decision": "no_conversation",
        "reason": "no_compatible_active_conversation",
        "conversation_id": "",
        "reply_act": "UNKNOWN",
        "payload": {},
        "conversation": null,
        "latency_ms": 1.5742999967187643
      },
      "legacy_pending_projection": {},
      "continuity_shadow_diff": {
        "legacy_result": false,
        "v2_result": false,
        "match": true,
        "difference_reason": ""
      },
      "shadow_metrics": {
        "total": 2,
        "matches": 2,
        "differences": 0,
        "match_rate": 1.0,
        "difference_reasons": {}
      },
      "performance": {
        "count": 2,
        "p50_ms": 1.59845,
        "p95_ms": 1.6226
      }
    },
    "open_threads": [],
    "memory": {
      "facts_count": 0,
      "chunks_count": 0
    },
    "beliefs": [],
    "game_state": {
      "game": "Final Fantasy V",
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
      "last_updated": 1786464000.0,
      "provenance": "stream_context_sync",
      "confidence": 0.75,
      "current_game": "Final Fantasy V",
      "recent_run_context_facts": []
    },
    "social_state": {
      "recent_active_users": [],
      "recent_chat_count": 0,
      "last_raid": {},
      "last_cheer": {}
    },
    "promotion_profiles": [],
    "actions": {
      "attempts": [],
      "speech_requests": [],
      "model_calls": [
        {
          "key": "stream_response:v1:none",
          "method": "chat"
        },
        {
          "key": "stream_response:v1:none",
          "method": "chat"
        }
      ],
      "research_calls": []
    },
    "receipts": [],
    "emitted_outputs": [
      {
        "event_id": "1786437802.0719497",
        "emitted": true,
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "reason": "",
        "text_digest": "",
        "text_present": false
      }
    ],
    "final_emission_results": [
      {
        "event_id": "1786437802.0719497",
        "emitted": true,
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "reason": "",
        "text_digest": "",
        "text_present": false
      }
    ],
    "database_watermarks": {
      "counts": {
        "chat_log": 2,
        "memory_facts": 0,
        "memory_chunks": 0,
        "stream_sessions": 1,
        "stream_chat_messages": 0,
        "stream_events": 1,
        "live_session_timeline": 5,
        "promotion_events": 0,
        "viewer_promotion_profiles": 0,
        "schema_migrations": 2,
        "conversations": 0,
        "open_threads": 0
      },
      "schema_migrations": [
        {
          "component": "cognitive_replay",
          "version": 1,
          "name": "replay_metadata",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "applied_at": "2026-08-11T08:43:21.843413+00:00"
        },
        {
          "component": "conversation_continuity",
          "version": 1,
          "name": "conversation_and_open_threads",
          "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
          "applied_at": "2026-08-11T08:43:21.928483+00:00"
        }
      ],
      "final_response_digest": "7e07c6a5322789f6",
      "final_response_present": true
    }
  },
  "end": {
    "runtime": {
      "mode": "active",
      "hebe_sleeping": false,
      "is_running": false,
      "last_input_source": null,
      "last_intent": null,
      "last_firewall": {
        "source": "owner_stt_direct",
        "authority": "owner",
        "decision": "allow",
        "reason": "owner_direct"
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
      "is_live": false,
      "live_status_known": true,
      "active_stream_session_id": null,
      "last_transition": "offline",
      "title": null,
      "game": "Final Fantasy V",
      "category": "Final Fantasy V"
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
      "last_resolution": {
        "consumed": false,
        "decision": "no_conversation",
        "reason": "no_compatible_active_conversation",
        "conversation_id": "",
        "reply_act": "UNKNOWN",
        "payload": {},
        "conversation": null,
        "latency_ms": 1.5742999967187643
      },
      "legacy_pending_projection": {},
      "continuity_shadow_diff": {
        "legacy_result": false,
        "v2_result": false,
        "match": true,
        "difference_reason": ""
      },
      "shadow_metrics": {
        "total": 2,
        "matches": 2,
        "differences": 0,
        "match_rate": 1.0,
        "difference_reasons": {}
      },
      "performance": {
        "count": 2,
        "p50_ms": 1.59845,
        "p95_ms": 1.6226
      }
    },
    "open_threads": [],
    "memory": {
      "facts_count": 0,
      "chunks_count": 1
    },
    "beliefs": [],
    "game_state": {
      "game": "Final Fantasy V",
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
      "last_updated": 1786464000.0,
      "provenance": "stream_context_sync",
      "confidence": 0.75,
      "current_game": "Final Fantasy V",
      "recent_run_context_facts": []
    },
    "social_state": {
      "recent_active_users": [],
      "recent_chat_count": 0,
      "last_raid": {},
      "last_cheer": {}
    },
    "promotion_profiles": [],
    "actions": {
      "attempts": [],
      "speech_requests": [],
      "model_calls": [
        {
          "key": "stream_response:v1:none",
          "method": "chat"
        },
        {
          "key": "stream_response:v1:none",
          "method": "chat"
        }
      ],
      "research_calls": []
    },
    "receipts": [],
    "emitted_outputs": [
      {
        "event_id": "1786437802.0719497",
        "emitted": true,
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "reason": "",
        "text_digest": "",
        "text_present": false
      }
    ],
    "final_emission_results": [
      {
        "event_id": "1786437802.0719497",
        "emitted": true,
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "reason": "",
        "text_digest": "",
        "text_present": false
      }
    ],
    "database_watermarks": {
      "counts": {
        "chat_log": 2,
        "memory_facts": 0,
        "memory_chunks": 1,
        "stream_sessions": 1,
        "stream_chat_messages": 0,
        "stream_events": 2,
        "live_session_timeline": 6,
        "promotion_events": 0,
        "viewer_promotion_profiles": 0,
        "schema_migrations": 2,
        "conversations": 0,
        "open_threads": 0
      },
      "schema_migrations": [
        {
          "component": "cognitive_replay",
          "version": 1,
          "name": "replay_metadata",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "applied_at": "2026-08-11T08:43:21.843413+00:00"
        },
        {
          "component": "conversation_continuity",
          "version": 1,
          "name": "conversation_and_open_threads",
          "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
          "applied_at": "2026-08-11T08:43:21.928483+00:00"
        }
      ],
      "final_response_digest": "7e07c6a5322789f6",
      "final_response_present": true
    }
  },
  "restart": {
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
      "enabled": false,
      "is_live": false,
      "live_status_known": false,
      "active_stream_session_id": null,
      "last_transition": null,
      "title": null,
      "game": null,
      "category": null
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
      "chunks_count": 1
    },
    "beliefs": [],
    "game_state": {
      "game": "",
      "platform_version": "",
      "playthrough_type": "casual",
      "spoiler_policy": "spoiler_safe_hints",
      "current_location": null,
      "current_character": "",
      "party_members": [],
      "last_confirmed_progress": "",
      "current_objective": null,
      "challenge": "",
      "known_constraints": [],
      "last_updated": 1786437807.8244133,
      "provenance": "inferred",
      "confidence": 0.0,
      "current_game": null,
      "recent_run_context_facts": []
    },
    "social_state": {
      "recent_active_users": [],
      "recent_chat_count": 0,
      "last_raid": {},
      "last_cheer": {}
    },
    "promotion_profiles": [],
    "actions": {
      "attempts": [],
      "speech_requests": [],
      "model_calls": [
        {
          "key": "stream_response:v1:none",
          "method": "chat"
        },
        {
          "key": "stream_response:v1:none",
          "method": "chat"
        }
      ],
      "research_calls": []
    },
    "receipts": [],
    "emitted_outputs": [
      {
        "event_id": "1786437802.0719497",
        "emitted": true,
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "reason": "",
        "text_digest": "",
        "text_present": false
      }
    ],
    "final_emission_results": [
      {
        "event_id": "1786437802.0719497",
        "emitted": true,
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "reason": "",
        "text_digest": "",
        "text_present": false
      }
    ],
    "database_watermarks": {
      "counts": {
        "chat_log": 2,
        "memory_facts": 0,
        "memory_chunks": 1,
        "stream_sessions": 1,
        "stream_chat_messages": 0,
        "stream_events": 2,
        "live_session_timeline": 6,
        "promotion_events": 0,
        "viewer_promotion_profiles": 0,
        "schema_migrations": 2,
        "conversations": 0,
        "open_threads": 0
      },
      "schema_migrations": [
        {
          "component": "cognitive_replay",
          "version": 1,
          "name": "replay_metadata",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "applied_at": "2026-08-11T08:43:21.843413+00:00"
        },
        {
          "component": "conversation_continuity",
          "version": 1,
          "name": "conversation_and_open_threads",
          "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
          "applied_at": "2026-08-11T08:43:21.928483+00:00"
        }
      ],
      "final_response_digest": "",
      "final_response_present": false
    }
  },
  "later": {
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
      "active_stream_session_id": 2,
      "last_transition": "online",
      "title": null,
      "game": "Final Fantasy V",
      "category": "Final Fantasy V"
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
      "chunks_count": 1
    },
    "beliefs": [],
    "game_state": {
      "game": "Final Fantasy V",
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
      "last_updated": 1786550400.0,
      "provenance": "stream_context_sync",
      "confidence": 0.75,
      "current_game": "Final Fantasy V",
      "recent_run_context_facts": []
    },
    "social_state": {
      "recent_active_users": [],
      "recent_chat_count": 0,
      "last_raid": {},
      "last_cheer": {}
    },
    "promotion_profiles": [],
    "actions": {
      "attempts": [],
      "speech_requests": [],
      "model_calls": [
        {
          "key": "stream_response:v1:none",
          "method": "chat"
        },
        {
          "key": "stream_response:v1:none",
          "method": "chat"
        }
      ],
      "research_calls": []
    },
    "receipts": [],
    "emitted_outputs": [
      {
        "event_id": "1786437802.0719497",
        "emitted": true,
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "reason": "",
        "text_digest": "",
        "text_present": false
      }
    ],
    "final_emission_results": [
      {
        "event_id": "1786437802.0719497",
        "emitted": true,
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "reason": "",
        "text_digest": "",
        "text_present": false
      }
    ],
    "database_watermarks": {
      "counts": {
        "chat_log": 2,
        "memory_facts": 0,
        "memory_chunks": 1,
        "stream_sessions": 2,
        "stream_chat_messages": 0,
        "stream_events": 3,
        "live_session_timeline": 8,
        "promotion_events": 0,
        "viewer_promotion_profiles": 0,
        "schema_migrations": 2,
        "conversations": 0,
        "open_threads": 0
      },
      "schema_migrations": [
        {
          "component": "cognitive_replay",
          "version": 1,
          "name": "replay_metadata",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "applied_at": "2026-08-11T08:43:21.843413+00:00"
        },
        {
          "component": "conversation_continuity",
          "version": 1,
          "name": "conversation_and_open_threads",
          "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
          "applied_at": "2026-08-11T08:43:21.928483+00:00"
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
    "active_stream_session_id": 2,
    "last_transition": "online",
    "title": null,
    "game": "Final Fantasy V",
    "category": "Final Fantasy V"
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
    "chunks_count": 1
  },
  "beliefs": [],
  "game_state": {
    "game": "Final Fantasy V",
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
    "last_updated": 1786550400.0,
    "provenance": "stream_context_sync",
    "confidence": 0.75,
    "current_game": "Final Fantasy V",
    "recent_run_context_facts": []
  },
  "social_state": {
    "recent_active_users": [],
    "recent_chat_count": 0,
    "last_raid": {},
    "last_cheer": {}
  },
  "promotion_profiles": [],
  "actions": {
    "attempts": [],
    "speech_requests": [],
    "model_calls": [
      {
        "key": "stream_response:v1:none",
        "method": "chat"
      },
      {
        "key": "stream_response:v1:none",
        "method": "chat"
      }
    ],
    "research_calls": []
  },
  "receipts": [],
  "emitted_outputs": [
    {
      "event_id": "1786437802.0719497",
      "emitted": true,
      "route": "local_owner_reply",
      "targets": [
        "local_ui"
      ],
      "reason": "",
      "text_digest": "",
      "text_present": false
    }
  ],
  "final_emission_results": [
    {
      "event_id": "1786437802.0719497",
      "emitted": true,
      "route": "local_owner_reply",
      "targets": [
        "local_ui"
      ],
      "reason": "",
      "text_digest": "",
      "text_present": false
    }
  ],
  "database_watermarks": {
    "counts": {
      "chat_log": 2,
      "memory_facts": 0,
      "memory_chunks": 1,
      "stream_sessions": 2,
      "stream_chat_messages": 0,
      "stream_events": 3,
      "live_session_timeline": 8,
      "promotion_events": 0,
      "viewer_promotion_profiles": 0,
      "schema_migrations": 2,
      "conversations": 0,
      "open_threads": 0
    },
    "schema_migrations": [
      {
        "component": "cognitive_replay",
        "version": 1,
        "name": "replay_metadata",
        "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
        "applied_at": "2026-08-11T08:43:21.843413+00:00"
      },
      {
        "component": "conversation_continuity",
        "version": 1,
        "name": "conversation_and_open_threads",
        "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
        "applied_at": "2026-08-11T08:43:21.928483+00:00"
      }
    ],
    "final_response_digest": "",
    "final_response_present": false
  }
}
```

#### Restart evidence

```json
[
  {
    "event_id": "restart",
    "old_engine_id": 2096479817872,
    "new_engine_id": 2096480060432,
    "old_engine_collected": true,
    "same_database": "C:\\Users\\Leo Nifelheim\\Documents\\Hebe\\hebe-ui\\artifacts\\cognitive-continuity-phase1\\final\\workspaces\\owner_correction_format\\hebe-replay.sqlite3",
    "before_persisted_counts": {
      "chat_log": 2,
      "memory_facts": 0,
      "memory_chunks": 1,
      "stream_sessions": 1,
      "stream_chat_messages": 0,
      "stream_events": 2,
      "live_session_timeline": 6,
      "promotion_events": 0,
      "viewer_promotion_profiles": 0,
      "schema_migrations": 2,
      "conversations": 0,
      "open_threads": 0
    },
    "after_persisted_counts": {
      "chat_log": 2,
      "memory_facts": 0,
      "memory_chunks": 1,
      "stream_sessions": 1,
      "stream_chat_messages": 0,
      "stream_events": 2,
      "live_session_timeline": 6,
      "promotion_events": 0,
      "viewer_promotion_profiles": 0,
      "schema_migrations": 2,
      "conversations": 0,
      "open_threads": 0
    },
    "volatile_state_recreated": true
  }
]
```

### raid_transition_foundation

- Status: **VERIFIED**
- Events: 3
- Restarts: 0
- Duration: 5.855996s
- Assertions passed/failed/skipped: 2/0/0


#### Checkpoint state

```json
{
  "start": {
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
      "title": null,
      "game": null,
      "category": null
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
    "beliefs": [],
    "game_state": {
      "game": "",
      "platform_version": "",
      "playthrough_type": "casual",
      "spoiler_policy": "spoiler_safe_hints",
      "current_location": null,
      "current_character": "",
      "party_members": [],
      "last_confirmed_progress": "",
      "current_objective": null,
      "challenge": "",
      "known_constraints": [],
      "last_updated": 1786437808.2151096,
      "provenance": "inferred",
      "confidence": 0.0,
      "current_game": null,
      "recent_run_context_facts": []
    },
    "social_state": {
      "recent_active_users": [],
      "recent_chat_count": 0,
      "last_raid": {},
      "last_cheer": {}
    },
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
        "schema_migrations": 2,
        "conversations": 0,
        "open_threads": 0
      },
      "schema_migrations": [
        {
          "component": "cognitive_replay",
          "version": 1,
          "name": "replay_metadata",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "applied_at": "2026-08-11T08:43:28.208504+00:00"
        },
        {
          "component": "conversation_continuity",
          "version": 1,
          "name": "conversation_and_open_threads",
          "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
          "applied_at": "2026-08-11T08:43:28.277977+00:00"
        }
      ],
      "final_response_digest": "",
      "final_response_present": false
    }
  },
  "raid": {
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
      "title": null,
      "game": null,
      "category": null
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
    "beliefs": [],
    "game_state": {
      "game": "",
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
      "last_updated": 1786437808.2151096,
      "provenance": "current_live_session",
      "confidence": 0.0,
      "current_game": null,
      "recent_run_context_facts": []
    },
    "social_state": {
      "recent_active_users": [],
      "recent_chat_count": 0,
      "last_raid": {
        "display_name": "SyntheticRaider",
        "user_login": "synthetic_raider",
        "viewer_count": 12,
        "source": "eventsub",
        "ts": 1786464005.0
      },
      "last_cheer": {}
    },
    "promotion_profiles": [],
    "actions": {
      "attempts": [
        {
          "operation": "twitch.send_message",
          "payload": {
            "text": "Gracias por la raid, SyntheticRaider."
          },
          "outcome": {
            "success": true,
            "status": "sent",
            "message_id": "raid-thanks-1"
          }
        }
      ],
      "speech_requests": [],
      "model_calls": [
        {
          "key": "stream_response:v1:none",
          "method": "chat"
        }
      ],
      "research_calls": []
    },
    "receipts": [],
    "emitted_outputs": [
      {
        "event_id": "raid",
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
        "event_id": "raid",
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
    "database_watermarks": {
      "counts": {
        "chat_log": 0,
        "memory_facts": 0,
        "memory_chunks": 0,
        "stream_sessions": 1,
        "stream_chat_messages": 0,
        "stream_events": 2,
        "live_session_timeline": 3,
        "promotion_events": 0,
        "viewer_promotion_profiles": 0,
        "schema_migrations": 2,
        "conversations": 0,
        "open_threads": 0
      },
      "schema_migrations": [
        {
          "component": "cognitive_replay",
          "version": 1,
          "name": "replay_metadata",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "applied_at": "2026-08-11T08:43:28.208504+00:00"
        },
        {
          "component": "conversation_continuity",
          "version": 1,
          "name": "conversation_and_open_threads",
          "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
          "applied_at": "2026-08-11T08:43:28.277977+00:00"
        }
      ],
      "final_response_digest": "",
      "final_response_present": false
    }
  },
  "end": {
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
      "is_live": false,
      "live_status_known": true,
      "active_stream_session_id": null,
      "last_transition": "offline",
      "title": null,
      "game": null,
      "category": null
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
    "beliefs": [],
    "game_state": {
      "game": "",
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
      "last_updated": 1786437808.2151096,
      "provenance": "current_live_session",
      "confidence": 0.0,
      "current_game": null,
      "recent_run_context_facts": []
    },
    "social_state": {
      "recent_active_users": [],
      "recent_chat_count": 0,
      "last_raid": {
        "display_name": "SyntheticRaider",
        "user_login": "synthetic_raider",
        "viewer_count": 12,
        "source": "eventsub",
        "ts": 1786464005.0
      },
      "last_cheer": {}
    },
    "promotion_profiles": [],
    "actions": {
      "attempts": [
        {
          "operation": "twitch.send_message",
          "payload": {
            "text": "Gracias por la raid, SyntheticRaider."
          },
          "outcome": {
            "success": true,
            "status": "sent",
            "message_id": "raid-thanks-1"
          }
        }
      ],
      "speech_requests": [],
      "model_calls": [
        {
          "key": "stream_response:v1:none",
          "method": "chat"
        }
      ],
      "research_calls": []
    },
    "receipts": [],
    "emitted_outputs": [
      {
        "event_id": "raid",
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
        "event_id": "raid",
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
    "database_watermarks": {
      "counts": {
        "chat_log": 0,
        "memory_facts": 0,
        "memory_chunks": 0,
        "stream_sessions": 1,
        "stream_chat_messages": 0,
        "stream_events": 3,
        "live_session_timeline": 4,
        "promotion_events": 0,
        "viewer_promotion_profiles": 0,
        "schema_migrations": 2,
        "conversations": 0,
        "open_threads": 0
      },
      "schema_migrations": [
        {
          "component": "cognitive_replay",
          "version": 1,
          "name": "replay_metadata",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "applied_at": "2026-08-11T08:43:28.208504+00:00"
        },
        {
          "component": "conversation_continuity",
          "version": 1,
          "name": "conversation_and_open_threads",
          "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
          "applied_at": "2026-08-11T08:43:28.277977+00:00"
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
    "is_live": false,
    "live_status_known": true,
    "active_stream_session_id": null,
    "last_transition": "offline",
    "title": null,
    "game": null,
    "category": null
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
  "beliefs": [],
  "game_state": {
    "game": "",
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
    "last_updated": 1786437808.2151096,
    "provenance": "current_live_session",
    "confidence": 0.0,
    "current_game": null,
    "recent_run_context_facts": []
  },
  "social_state": {
    "recent_active_users": [],
    "recent_chat_count": 0,
    "last_raid": {
      "display_name": "SyntheticRaider",
      "user_login": "synthetic_raider",
      "viewer_count": 12,
      "source": "eventsub",
      "ts": 1786464005.0
    },
    "last_cheer": {}
  },
  "promotion_profiles": [],
  "actions": {
    "attempts": [
      {
        "operation": "twitch.send_message",
        "payload": {
          "text": "Gracias por la raid, SyntheticRaider."
        },
        "outcome": {
          "success": true,
          "status": "sent",
          "message_id": "raid-thanks-1"
        }
      }
    ],
    "speech_requests": [],
    "model_calls": [
      {
        "key": "stream_response:v1:none",
        "method": "chat"
      }
    ],
    "research_calls": []
  },
  "receipts": [],
  "emitted_outputs": [
    {
      "event_id": "raid",
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
      "event_id": "raid",
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
  "database_watermarks": {
    "counts": {
      "chat_log": 0,
      "memory_facts": 0,
      "memory_chunks": 0,
      "stream_sessions": 1,
      "stream_chat_messages": 0,
      "stream_events": 3,
      "live_session_timeline": 4,
      "promotion_events": 0,
      "viewer_promotion_profiles": 0,
      "schema_migrations": 2,
      "conversations": 0,
      "open_threads": 0
    },
    "schema_migrations": [
      {
        "component": "cognitive_replay",
        "version": 1,
        "name": "replay_metadata",
        "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
        "applied_at": "2026-08-11T08:43:28.208504+00:00"
      },
      {
        "component": "conversation_continuity",
        "version": 1,
        "name": "conversation_and_open_threads",
        "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
        "applied_at": "2026-08-11T08:43:28.277977+00:00"
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

### receipt_truth

- Status: **VERIFIED**
- Events: 7
- Restarts: 0
- Duration: 0.679983s
- Assertions passed/failed/skipped: 4/0/0


#### Checkpoint state

```json
{
  "start": {
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
      "title": null,
      "game": null,
      "category": null
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
    "beliefs": [],
    "game_state": {
      "game": "",
      "platform_version": "",
      "playthrough_type": "casual",
      "spoiler_policy": "spoiler_safe_hints",
      "current_location": null,
      "current_character": "",
      "party_members": [],
      "last_confirmed_progress": "",
      "current_objective": null,
      "challenge": "",
      "known_constraints": [],
      "last_updated": 1786437814.1742978,
      "provenance": "inferred",
      "confidence": 0.0,
      "current_game": null,
      "recent_run_context_facts": []
    },
    "social_state": {
      "recent_active_users": [],
      "recent_chat_count": 0,
      "last_raid": {},
      "last_cheer": {}
    },
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
        "schema_migrations": 2,
        "conversations": 0,
        "open_threads": 0
      },
      "schema_migrations": [
        {
          "component": "cognitive_replay",
          "version": 1,
          "name": "replay_metadata",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "applied_at": "2026-08-11T08:43:34.167927+00:00"
        },
        {
          "component": "conversation_continuity",
          "version": 1,
          "name": "conversation_and_open_threads",
          "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
          "applied_at": "2026-08-11T08:43:34.237646+00:00"
        }
      ],
      "final_response_digest": "",
      "final_response_present": false
    }
  },
  "alice-chat": {
    "runtime": {
      "mode": "active",
      "hebe_sleeping": false,
      "is_running": false,
      "last_input_source": null,
      "last_intent": null,
      "last_firewall": {
        "source": "twitch_viewer",
        "authority": "viewer",
        "decision": "allow",
        "reason": "live_viewer_message"
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
      "title": null,
      "game": null,
      "category": null
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
    "beliefs": [],
    "game_state": {
      "game": "",
      "platform_version": "",
      "playthrough_type": "casual",
      "spoiler_policy": "spoiler_safe_hints",
      "current_location": null,
      "current_character": "",
      "party_members": [],
      "last_confirmed_progress": "",
      "current_objective": null,
      "challenge": "",
      "known_constraints": [],
      "last_updated": 1786437814.1742978,
      "provenance": "inferred",
      "confidence": 0.0,
      "current_game": null,
      "recent_run_context_facts": []
    },
    "social_state": {
      "recent_active_users": [
        "alice_test"
      ],
      "recent_chat_count": 1,
      "last_raid": {},
      "last_cheer": {}
    },
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
    "database_watermarks": {
      "counts": {
        "chat_log": 0,
        "memory_facts": 0,
        "memory_chunks": 0,
        "stream_sessions": 1,
        "stream_chat_messages": 1,
        "stream_events": 1,
        "live_session_timeline": 3,
        "promotion_events": 0,
        "viewer_promotion_profiles": 0,
        "schema_migrations": 2,
        "conversations": 0,
        "open_threads": 0
      },
      "schema_migrations": [
        {
          "component": "cognitive_replay",
          "version": 1,
          "name": "replay_metadata",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "applied_at": "2026-08-11T08:43:34.167927+00:00"
        },
        {
          "component": "conversation_continuity",
          "version": 1,
          "name": "conversation_and_open_threads",
          "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
          "applied_at": "2026-08-11T08:43:34.237646+00:00"
        }
      ],
      "final_response_digest": "",
      "final_response_present": false
    }
  },
  "bob-chat": {
    "runtime": {
      "mode": "active",
      "hebe_sleeping": false,
      "is_running": false,
      "last_input_source": null,
      "last_intent": null,
      "last_firewall": {
        "source": "twitch_viewer",
        "authority": "viewer",
        "decision": "allow",
        "reason": "live_viewer_message"
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
      "title": null,
      "game": null,
      "category": null
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
    "beliefs": [],
    "game_state": {
      "game": "",
      "platform_version": "",
      "playthrough_type": "casual",
      "spoiler_policy": "spoiler_safe_hints",
      "current_location": null,
      "current_character": "",
      "party_members": [],
      "last_confirmed_progress": "",
      "current_objective": null,
      "challenge": "",
      "known_constraints": [],
      "last_updated": 1786437814.1742978,
      "provenance": "inferred",
      "confidence": 0.0,
      "current_game": null,
      "recent_run_context_facts": []
    },
    "social_state": {
      "recent_active_users": [
        "alice_test",
        "bob_test"
      ],
      "recent_chat_count": 2,
      "last_raid": {},
      "last_cheer": {}
    },
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
    "database_watermarks": {
      "counts": {
        "chat_log": 0,
        "memory_facts": 0,
        "memory_chunks": 0,
        "stream_sessions": 1,
        "stream_chat_messages": 2,
        "stream_events": 1,
        "live_session_timeline": 4,
        "promotion_events": 0,
        "viewer_promotion_profiles": 0,
        "schema_migrations": 2,
        "conversations": 0,
        "open_threads": 0
      },
      "schema_migrations": [
        {
          "component": "cognitive_replay",
          "version": 1,
          "name": "replay_metadata",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "applied_at": "2026-08-11T08:43:34.167927+00:00"
        },
        {
          "component": "conversation_continuity",
          "version": 1,
          "name": "conversation_and_open_threads",
          "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
          "applied_at": "2026-08-11T08:43:34.237646+00:00"
        }
      ],
      "final_response_digest": "",
      "final_response_present": false
    }
  },
  "carol-chat": {
    "runtime": {
      "mode": "active",
      "hebe_sleeping": false,
      "is_running": false,
      "last_input_source": null,
      "last_intent": null,
      "last_firewall": {
        "source": "twitch_viewer",
        "authority": "viewer",
        "decision": "allow",
        "reason": "live_viewer_message"
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
      "title": null,
      "game": null,
      "category": null
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
    "beliefs": [],
    "game_state": {
      "game": "",
      "platform_version": "",
      "playthrough_type": "casual",
      "spoiler_policy": "spoiler_safe_hints",
      "current_location": null,
      "current_character": "",
      "party_members": [],
      "last_confirmed_progress": "",
      "current_objective": null,
      "challenge": "",
      "known_constraints": [],
      "last_updated": 1786437814.1742978,
      "provenance": "inferred",
      "confidence": 0.0,
      "current_game": null,
      "recent_run_context_facts": []
    },
    "social_state": {
      "recent_active_users": [
        "alice_test",
        "bob_test",
        "carol_test"
      ],
      "recent_chat_count": 3,
      "last_raid": {},
      "last_cheer": {}
    },
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
    "database_watermarks": {
      "counts": {
        "chat_log": 0,
        "memory_facts": 0,
        "memory_chunks": 0,
        "stream_sessions": 1,
        "stream_chat_messages": 3,
        "stream_events": 1,
        "live_session_timeline": 5,
        "promotion_events": 0,
        "viewer_promotion_profiles": 0,
        "schema_migrations": 2,
        "conversations": 0,
        "open_threads": 0
      },
      "schema_migrations": [
        {
          "component": "cognitive_replay",
          "version": 1,
          "name": "replay_metadata",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "applied_at": "2026-08-11T08:43:34.167927+00:00"
        },
        {
          "component": "conversation_continuity",
          "version": 1,
          "name": "conversation_and_open_threads",
          "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
          "applied_at": "2026-08-11T08:43:34.237646+00:00"
        }
      ],
      "final_response_digest": "",
      "final_response_present": false
    }
  },
  "success": {
    "runtime": {
      "mode": "active",
      "hebe_sleeping": false,
      "is_running": false,
      "last_input_source": null,
      "last_intent": null,
      "last_firewall": {
        "source": "owner_stt_direct",
        "authority": "owner",
        "decision": "allow",
        "reason": "owner_direct"
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
      "title": null,
      "game": null,
      "category": null
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
      "last_resolution": {
        "consumed": false,
        "decision": "no_conversation",
        "reason": "no_compatible_active_conversation",
        "conversation_id": "",
        "reply_act": "UNKNOWN",
        "payload": {},
        "conversation": null,
        "latency_ms": 1.4336000313051045
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
        "p50_ms": 1.4336,
        "p95_ms": 1.4336
      }
    },
    "open_threads": [],
    "memory": {
      "facts_count": 0,
      "chunks_count": 0
    },
    "beliefs": [],
    "game_state": {
      "game": "",
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
      "last_updated": 1786437814.1742978,
      "provenance": "current_live_session",
      "confidence": 0.0,
      "current_game": null,
      "recent_run_context_facts": []
    },
    "social_state": {
      "recent_active_users": [
        "alice_test",
        "bob_test",
        "carol_test"
      ],
      "recent_chat_count": 3,
      "last_raid": {},
      "last_cheer": {}
    },
    "promotion_profiles": [
      {
        "twitch_user_id": "501",
        "current_login": "alice_test",
        "display_name": "alice_test",
        "auto_promo_mode": "first_message_each_stream",
        "created_by": "owner_command",
        "last_promoted_stream_id": "1",
        "owner_locked": 1,
        "active": 1
      }
    ],
    "actions": {
      "attempts": [
        {
          "operation": "twitch.shoutout",
          "payload": {
            "target": "alice_test",
            "command": "!so alice_test"
          },
          "outcome": {
            "success": true,
            "status": "sent",
            "message_id": "receipt-success"
          }
        }
      ],
      "speech_requests": [],
      "model_calls": [],
      "research_calls": []
    },
    "receipts": [
      {
        "id": "promo_9957df32356355528825eedeb574e7bb",
        "stream_session_id": "1",
        "source_event_id": "1786437814.3985791",
        "requested_by": "leo",
        "resolved_twitch_user_id": "501",
        "resolved_login": "alice_test",
        "trigger_type": "owner_learn_and_execute",
        "execution_status": "sent",
        "twitch_message_id": "receipt-success",
        "failure_reason": ""
      }
    ],
    "emitted_outputs": [
      {
        "event_id": "1786437814.3985791",
        "emitted": true,
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "reason": "",
        "text_digest": "",
        "text_present": false
      }
    ],
    "final_emission_results": [
      {
        "event_id": "1786437814.3985791",
        "emitted": true,
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "reason": "",
        "text_digest": "",
        "text_present": false
      }
    ],
    "database_watermarks": {
      "counts": {
        "chat_log": 1,
        "memory_facts": 0,
        "memory_chunks": 0,
        "stream_sessions": 1,
        "stream_chat_messages": 3,
        "stream_events": 1,
        "live_session_timeline": 7,
        "promotion_events": 1,
        "viewer_promotion_profiles": 1,
        "schema_migrations": 2,
        "conversations": 0,
        "open_threads": 0
      },
      "schema_migrations": [
        {
          "component": "cognitive_replay",
          "version": 1,
          "name": "replay_metadata",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "applied_at": "2026-08-11T08:43:34.167927+00:00"
        },
        {
          "component": "conversation_continuity",
          "version": 1,
          "name": "conversation_and_open_threads",
          "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
          "applied_at": "2026-08-11T08:43:34.237646+00:00"
        }
      ],
      "final_response_digest": "",
      "final_response_present": false
    }
  },
  "failure": {
    "runtime": {
      "mode": "active",
      "hebe_sleeping": false,
      "is_running": false,
      "last_input_source": null,
      "last_intent": null,
      "last_firewall": {
        "source": "owner_stt_direct",
        "authority": "owner",
        "decision": "allow",
        "reason": "owner_direct"
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
      "title": null,
      "game": null,
      "category": null
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
      "last_resolution": {
        "consumed": false,
        "decision": "no_conversation",
        "reason": "no_compatible_active_conversation",
        "conversation_id": "",
        "reply_act": "UNKNOWN",
        "payload": {},
        "conversation": null,
        "latency_ms": 1.421599998138845
      },
      "legacy_pending_projection": {},
      "continuity_shadow_diff": {
        "legacy_result": false,
        "v2_result": false,
        "match": true,
        "difference_reason": ""
      },
      "shadow_metrics": {
        "total": 2,
        "matches": 2,
        "differences": 0,
        "match_rate": 1.0,
        "difference_reasons": {}
      },
      "performance": {
        "count": 2,
        "p50_ms": 1.4276,
        "p95_ms": 1.4336
      }
    },
    "open_threads": [],
    "memory": {
      "facts_count": 0,
      "chunks_count": 0
    },
    "beliefs": [],
    "game_state": {
      "game": "",
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
      "last_updated": 1786437814.1742978,
      "provenance": "current_live_session",
      "confidence": 0.0,
      "current_game": null,
      "recent_run_context_facts": []
    },
    "social_state": {
      "recent_active_users": [
        "alice_test",
        "bob_test",
        "carol_test"
      ],
      "recent_chat_count": 3,
      "last_raid": {},
      "last_cheer": {}
    },
    "promotion_profiles": [
      {
        "twitch_user_id": "501",
        "current_login": "alice_test",
        "display_name": "alice_test",
        "auto_promo_mode": "first_message_each_stream",
        "created_by": "owner_command",
        "last_promoted_stream_id": "1",
        "owner_locked": 1,
        "active": 1
      }
    ],
    "actions": {
      "attempts": [
        {
          "operation": "twitch.shoutout",
          "payload": {
            "target": "alice_test",
            "command": "!so alice_test"
          },
          "outcome": {
            "success": true,
            "status": "sent",
            "message_id": "receipt-success"
          }
        },
        {
          "operation": "twitch.shoutout",
          "payload": {
            "target": "bob_test",
            "command": "!so bob_test"
          },
          "outcome": {
            "success": false,
            "status": "failed",
            "reason": "synthetic_failure"
          }
        }
      ],
      "speech_requests": [],
      "model_calls": [],
      "research_calls": []
    },
    "receipts": [
      {
        "id": "promo_9957df32356355528825eedeb574e7bb",
        "stream_session_id": "1",
        "source_event_id": "1786437814.3985791",
        "requested_by": "leo",
        "resolved_twitch_user_id": "501",
        "resolved_login": "alice_test",
        "trigger_type": "owner_learn_and_execute",
        "execution_status": "sent",
        "twitch_message_id": "receipt-success",
        "failure_reason": ""
      },
      {
        "id": "promo_50152700fa2556f585f71dd7baab9482",
        "stream_session_id": "1",
        "source_event_id": "1786437814.4874983",
        "requested_by": "leo",
        "resolved_twitch_user_id": "502",
        "resolved_login": "bob_test",
        "trigger_type": "owner_learn_and_execute",
        "execution_status": "failed",
        "twitch_message_id": "",
        "failure_reason": "send_failed: RuntimeError: Twitch shoutout command returned false"
      }
    ],
    "emitted_outputs": [
      {
        "event_id": "1786437814.3985791",
        "emitted": true,
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "reason": "",
        "text_digest": "",
        "text_present": false
      },
      {
        "event_id": "1786437814.4874983",
        "emitted": true,
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "reason": "",
        "text_digest": "",
        "text_present": false
      }
    ],
    "final_emission_results": [
      {
        "event_id": "1786437814.3985791",
        "emitted": true,
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "reason": "",
        "text_digest": "",
        "text_present": false
      },
      {
        "event_id": "1786437814.4874983",
        "emitted": true,
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "reason": "",
        "text_digest": "",
        "text_present": false
      }
    ],
    "database_watermarks": {
      "counts": {
        "chat_log": 2,
        "memory_facts": 0,
        "memory_chunks": 0,
        "stream_sessions": 1,
        "stream_chat_messages": 3,
        "stream_events": 1,
        "live_session_timeline": 9,
        "promotion_events": 2,
        "viewer_promotion_profiles": 1,
        "schema_migrations": 2,
        "conversations": 0,
        "open_threads": 0
      },
      "schema_migrations": [
        {
          "component": "cognitive_replay",
          "version": 1,
          "name": "replay_metadata",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "applied_at": "2026-08-11T08:43:34.167927+00:00"
        },
        {
          "component": "conversation_continuity",
          "version": 1,
          "name": "conversation_and_open_threads",
          "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
          "applied_at": "2026-08-11T08:43:34.237646+00:00"
        }
      ],
      "final_response_digest": "",
      "final_response_present": false
    }
  },
  "timeout": {
    "runtime": {
      "mode": "active",
      "hebe_sleeping": false,
      "is_running": false,
      "last_input_source": null,
      "last_intent": null,
      "last_firewall": {
        "source": "owner_stt_direct",
        "authority": "owner",
        "decision": "allow",
        "reason": "owner_direct"
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
      "title": null,
      "game": null,
      "category": null
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
      "last_resolution": {
        "consumed": false,
        "decision": "no_conversation",
        "reason": "no_compatible_active_conversation",
        "conversation_id": "",
        "reply_act": "UNKNOWN",
        "payload": {},
        "conversation": null,
        "latency_ms": 1.305700046941638
      },
      "legacy_pending_projection": {},
      "continuity_shadow_diff": {
        "legacy_result": false,
        "v2_result": false,
        "match": true,
        "difference_reason": ""
      },
      "shadow_metrics": {
        "total": 3,
        "matches": 3,
        "differences": 0,
        "match_rate": 1.0,
        "difference_reasons": {}
      },
      "performance": {
        "count": 3,
        "p50_ms": 1.4216,
        "p95_ms": 1.4336
      }
    },
    "open_threads": [],
    "memory": {
      "facts_count": 0,
      "chunks_count": 0
    },
    "beliefs": [],
    "game_state": {
      "game": "",
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
      "last_updated": 1786437814.1742978,
      "provenance": "current_live_session",
      "confidence": 0.0,
      "current_game": null,
      "recent_run_context_facts": []
    },
    "social_state": {
      "recent_active_users": [
        "alice_test",
        "bob_test",
        "carol_test"
      ],
      "recent_chat_count": 3,
      "last_raid": {},
      "last_cheer": {}
    },
    "promotion_profiles": [
      {
        "twitch_user_id": "501",
        "current_login": "alice_test",
        "display_name": "alice_test",
        "auto_promo_mode": "first_message_each_stream",
        "created_by": "owner_command",
        "last_promoted_stream_id": "1",
        "owner_locked": 1,
        "active": 1
      }
    ],
    "actions": {
      "attempts": [
        {
          "operation": "twitch.shoutout",
          "payload": {
            "target": "alice_test",
            "command": "!so alice_test"
          },
          "outcome": {
            "success": true,
            "status": "sent",
            "message_id": "receipt-success"
          }
        },
        {
          "operation": "twitch.shoutout",
          "payload": {
            "target": "bob_test",
            "command": "!so bob_test"
          },
          "outcome": {
            "success": false,
            "status": "failed",
            "reason": "synthetic_failure"
          }
        },
        {
          "operation": "twitch.shoutout",
          "payload": {
            "target": "carol_test",
            "command": "!so carol_test"
          },
          "outcome": {
            "success": false,
            "status": "timeout",
            "reason": "synthetic_timeout"
          }
        }
      ],
      "speech_requests": [],
      "model_calls": [],
      "research_calls": []
    },
    "receipts": [
      {
        "id": "promo_9957df32356355528825eedeb574e7bb",
        "stream_session_id": "1",
        "source_event_id": "1786437814.3985791",
        "requested_by": "leo",
        "resolved_twitch_user_id": "501",
        "resolved_login": "alice_test",
        "trigger_type": "owner_learn_and_execute",
        "execution_status": "sent",
        "twitch_message_id": "receipt-success",
        "failure_reason": ""
      },
      {
        "id": "promo_50152700fa2556f585f71dd7baab9482",
        "stream_session_id": "1",
        "source_event_id": "1786437814.4874983",
        "requested_by": "leo",
        "resolved_twitch_user_id": "502",
        "resolved_login": "bob_test",
        "trigger_type": "owner_learn_and_execute",
        "execution_status": "failed",
        "twitch_message_id": "",
        "failure_reason": "send_failed: RuntimeError: Twitch shoutout command returned false"
      },
      {
        "id": "promo_91c44fbd1d025a0aafdcce244be1fd04",
        "stream_session_id": "1",
        "source_event_id": "1786437814.5754154",
        "requested_by": "leo",
        "resolved_twitch_user_id": "503",
        "resolved_login": "carol_test",
        "trigger_type": "owner_learn_and_execute",
        "execution_status": "failed",
        "twitch_message_id": "",
        "failure_reason": "send_failed: TimeoutError: synthetic_timeout"
      }
    ],
    "emitted_outputs": [
      {
        "event_id": "1786437814.3985791",
        "emitted": true,
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "reason": "",
        "text_digest": "",
        "text_present": false
      },
      {
        "event_id": "1786437814.4874983",
        "emitted": true,
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "reason": "",
        "text_digest": "",
        "text_present": false
      },
      {
        "event_id": "1786437814.5754154",
        "emitted": true,
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "reason": "",
        "text_digest": "",
        "text_present": false
      }
    ],
    "final_emission_results": [
      {
        "event_id": "1786437814.3985791",
        "emitted": true,
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "reason": "",
        "text_digest": "",
        "text_present": false
      },
      {
        "event_id": "1786437814.4874983",
        "emitted": true,
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "reason": "",
        "text_digest": "",
        "text_present": false
      },
      {
        "event_id": "1786437814.5754154",
        "emitted": true,
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "reason": "",
        "text_digest": "",
        "text_present": false
      }
    ],
    "database_watermarks": {
      "counts": {
        "chat_log": 3,
        "memory_facts": 0,
        "memory_chunks": 0,
        "stream_sessions": 1,
        "stream_chat_messages": 3,
        "stream_events": 1,
        "live_session_timeline": 11,
        "promotion_events": 3,
        "viewer_promotion_profiles": 1,
        "schema_migrations": 2,
        "conversations": 0,
        "open_threads": 0
      },
      "schema_migrations": [
        {
          "component": "cognitive_replay",
          "version": 1,
          "name": "replay_metadata",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "applied_at": "2026-08-11T08:43:34.167927+00:00"
        },
        {
          "component": "conversation_continuity",
          "version": 1,
          "name": "conversation_and_open_threads",
          "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
          "applied_at": "2026-08-11T08:43:34.237646+00:00"
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
      "source": "owner_stt_direct",
      "authority": "owner",
      "decision": "allow",
      "reason": "owner_direct"
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
    "title": null,
    "game": null,
    "category": null
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
    "last_resolution": {
      "consumed": false,
      "decision": "no_conversation",
      "reason": "no_compatible_active_conversation",
      "conversation_id": "",
      "reply_act": "UNKNOWN",
      "payload": {},
      "conversation": null,
      "latency_ms": 1.305700046941638
    },
    "legacy_pending_projection": {},
    "continuity_shadow_diff": {
      "legacy_result": false,
      "v2_result": false,
      "match": true,
      "difference_reason": ""
    },
    "shadow_metrics": {
      "total": 3,
      "matches": 3,
      "differences": 0,
      "match_rate": 1.0,
      "difference_reasons": {}
    },
    "performance": {
      "count": 3,
      "p50_ms": 1.4216,
      "p95_ms": 1.4336
    }
  },
  "open_threads": [],
  "memory": {
    "facts_count": 0,
    "chunks_count": 0
  },
  "beliefs": [],
  "game_state": {
    "game": "",
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
    "last_updated": 1786437814.1742978,
    "provenance": "current_live_session",
    "confidence": 0.0,
    "current_game": null,
    "recent_run_context_facts": []
  },
  "social_state": {
    "recent_active_users": [
      "alice_test",
      "bob_test",
      "carol_test"
    ],
    "recent_chat_count": 3,
    "last_raid": {},
    "last_cheer": {}
  },
  "promotion_profiles": [
    {
      "twitch_user_id": "501",
      "current_login": "alice_test",
      "display_name": "alice_test",
      "auto_promo_mode": "first_message_each_stream",
      "created_by": "owner_command",
      "last_promoted_stream_id": "1",
      "owner_locked": 1,
      "active": 1
    }
  ],
  "actions": {
    "attempts": [
      {
        "operation": "twitch.shoutout",
        "payload": {
          "target": "alice_test",
          "command": "!so alice_test"
        },
        "outcome": {
          "success": true,
          "status": "sent",
          "message_id": "receipt-success"
        }
      },
      {
        "operation": "twitch.shoutout",
        "payload": {
          "target": "bob_test",
          "command": "!so bob_test"
        },
        "outcome": {
          "success": false,
          "status": "failed",
          "reason": "synthetic_failure"
        }
      },
      {
        "operation": "twitch.shoutout",
        "payload": {
          "target": "carol_test",
          "command": "!so carol_test"
        },
        "outcome": {
          "success": false,
          "status": "timeout",
          "reason": "synthetic_timeout"
        }
      }
    ],
    "speech_requests": [],
    "model_calls": [],
    "research_calls": []
  },
  "receipts": [
    {
      "id": "promo_9957df32356355528825eedeb574e7bb",
      "stream_session_id": "1",
      "source_event_id": "1786437814.3985791",
      "requested_by": "leo",
      "resolved_twitch_user_id": "501",
      "resolved_login": "alice_test",
      "trigger_type": "owner_learn_and_execute",
      "execution_status": "sent",
      "twitch_message_id": "receipt-success",
      "failure_reason": ""
    },
    {
      "id": "promo_50152700fa2556f585f71dd7baab9482",
      "stream_session_id": "1",
      "source_event_id": "1786437814.4874983",
      "requested_by": "leo",
      "resolved_twitch_user_id": "502",
      "resolved_login": "bob_test",
      "trigger_type": "owner_learn_and_execute",
      "execution_status": "failed",
      "twitch_message_id": "",
      "failure_reason": "send_failed: RuntimeError: Twitch shoutout command returned false"
    },
    {
      "id": "promo_91c44fbd1d025a0aafdcce244be1fd04",
      "stream_session_id": "1",
      "source_event_id": "1786437814.5754154",
      "requested_by": "leo",
      "resolved_twitch_user_id": "503",
      "resolved_login": "carol_test",
      "trigger_type": "owner_learn_and_execute",
      "execution_status": "failed",
      "twitch_message_id": "",
      "failure_reason": "send_failed: TimeoutError: synthetic_timeout"
    }
  ],
  "emitted_outputs": [
    {
      "event_id": "1786437814.3985791",
      "emitted": true,
      "route": "local_owner_reply",
      "targets": [
        "local_ui"
      ],
      "reason": "",
      "text_digest": "",
      "text_present": false
    },
    {
      "event_id": "1786437814.4874983",
      "emitted": true,
      "route": "local_owner_reply",
      "targets": [
        "local_ui"
      ],
      "reason": "",
      "text_digest": "",
      "text_present": false
    },
    {
      "event_id": "1786437814.5754154",
      "emitted": true,
      "route": "local_owner_reply",
      "targets": [
        "local_ui"
      ],
      "reason": "",
      "text_digest": "",
      "text_present": false
    }
  ],
  "final_emission_results": [
    {
      "event_id": "1786437814.3985791",
      "emitted": true,
      "route": "local_owner_reply",
      "targets": [
        "local_ui"
      ],
      "reason": "",
      "text_digest": "",
      "text_present": false
    },
    {
      "event_id": "1786437814.4874983",
      "emitted": true,
      "route": "local_owner_reply",
      "targets": [
        "local_ui"
      ],
      "reason": "",
      "text_digest": "",
      "text_present": false
    },
    {
      "event_id": "1786437814.5754154",
      "emitted": true,
      "route": "local_owner_reply",
      "targets": [
        "local_ui"
      ],
      "reason": "",
      "text_digest": "",
      "text_present": false
    }
  ],
  "database_watermarks": {
    "counts": {
      "chat_log": 3,
      "memory_facts": 0,
      "memory_chunks": 0,
      "stream_sessions": 1,
      "stream_chat_messages": 3,
      "stream_events": 1,
      "live_session_timeline": 11,
      "promotion_events": 3,
      "viewer_promotion_profiles": 1,
      "schema_migrations": 2,
      "conversations": 0,
      "open_threads": 0
    },
    "schema_migrations": [
      {
        "component": "cognitive_replay",
        "version": 1,
        "name": "replay_metadata",
        "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
        "applied_at": "2026-08-11T08:43:34.167927+00:00"
      },
      {
        "component": "conversation_continuity",
        "version": 1,
        "name": "conversation_and_open_threads",
        "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
        "applied_at": "2026-08-11T08:43:34.237646+00:00"
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

### research_fixture_foundation

- Status: **VERIFIED**
- Events: 1
- Restarts: 0
- Duration: 0.234604s
- Assertions passed/failed/skipped: 2/0/0


#### Checkpoint state

```json
{
  "research": {
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
      "enabled": false,
      "is_live": false,
      "live_status_known": false,
      "active_stream_session_id": null,
      "last_transition": null,
      "title": null,
      "game": null,
      "category": null
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
    "beliefs": [],
    "game_state": {
      "game": "",
      "platform_version": "",
      "playthrough_type": "casual",
      "spoiler_policy": "spoiler_safe_hints",
      "current_location": null,
      "current_character": "",
      "party_members": [],
      "last_confirmed_progress": "",
      "current_objective": null,
      "challenge": "",
      "known_constraints": [],
      "last_updated": 1786437814.8887353,
      "provenance": "inferred",
      "confidence": 0.0,
      "current_game": null,
      "recent_run_context_facts": []
    },
    "social_state": {
      "recent_active_users": [],
      "recent_chat_count": 0,
      "last_raid": {},
      "last_cheer": {}
    },
    "promotion_profiles": [],
    "actions": {
      "attempts": [],
      "speech_requests": [],
      "model_calls": [],
      "research_calls": [
        {
          "key": "Final Fantasy V core systems and spoiler-free premise spoiler-safe no future story information",
          "query": "Final Fantasy V core systems and spoiler-free premise spoiler-safe no future story information",
          "constraints": "{\"entity\": \"core systems and spoiler-free premise\", \"expected_fact_type\": \"general_mechanics\", \"spoiler_limit\": \"strict\", \"strict_first_playthrough\": true}"
        }
      ]
    },
    "receipts": [],
    "emitted_outputs": [],
    "final_emission_results": [],
    "database_watermarks": {
      "counts": {
        "chat_log": 0,
        "memory_facts": 0,
        "memory_chunks": 0,
        "stream_sessions": 0,
        "stream_chat_messages": 0,
        "stream_events": 0,
        "live_session_timeline": 0,
        "promotion_events": 0,
        "viewer_promotion_profiles": 0,
        "schema_migrations": 2,
        "conversations": 0,
        "open_threads": 0
      },
      "schema_migrations": [
        {
          "component": "cognitive_replay",
          "version": 1,
          "name": "replay_metadata",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "applied_at": "2026-08-11T08:43:34.883619+00:00"
        },
        {
          "component": "conversation_continuity",
          "version": 1,
          "name": "conversation_and_open_threads",
          "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
          "applied_at": "2026-08-11T08:43:34.947337+00:00"
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
    "enabled": false,
    "is_live": false,
    "live_status_known": false,
    "active_stream_session_id": null,
    "last_transition": null,
    "title": null,
    "game": null,
    "category": null
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
  "beliefs": [],
  "game_state": {
    "game": "",
    "platform_version": "",
    "playthrough_type": "casual",
    "spoiler_policy": "spoiler_safe_hints",
    "current_location": null,
    "current_character": "",
    "party_members": [],
    "last_confirmed_progress": "",
    "current_objective": null,
    "challenge": "",
    "known_constraints": [],
    "last_updated": 1786437814.8887353,
    "provenance": "inferred",
    "confidence": 0.0,
    "current_game": null,
    "recent_run_context_facts": []
  },
  "social_state": {
    "recent_active_users": [],
    "recent_chat_count": 0,
    "last_raid": {},
    "last_cheer": {}
  },
  "promotion_profiles": [],
  "actions": {
    "attempts": [],
    "speech_requests": [],
    "model_calls": [],
    "research_calls": [
      {
        "key": "Final Fantasy V core systems and spoiler-free premise spoiler-safe no future story information",
        "query": "Final Fantasy V core systems and spoiler-free premise spoiler-safe no future story information",
        "constraints": "{\"entity\": \"core systems and spoiler-free premise\", \"expected_fact_type\": \"general_mechanics\", \"spoiler_limit\": \"strict\", \"strict_first_playthrough\": true}"
      }
    ]
  },
  "receipts": [],
  "emitted_outputs": [],
  "final_emission_results": [],
  "database_watermarks": {
    "counts": {
      "chat_log": 0,
      "memory_facts": 0,
      "memory_chunks": 0,
      "stream_sessions": 0,
      "stream_chat_messages": 0,
      "stream_events": 0,
      "live_session_timeline": 0,
      "promotion_events": 0,
      "viewer_promotion_profiles": 0,
      "schema_migrations": 2,
      "conversations": 0,
      "open_threads": 0
    },
    "schema_migrations": [
      {
        "component": "cognitive_replay",
        "version": 1,
        "name": "replay_metadata",
        "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
        "applied_at": "2026-08-11T08:43:34.883619+00:00"
      },
      {
        "component": "conversation_continuity",
        "version": 1,
        "name": "conversation_and_open_threads",
        "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
        "applied_at": "2026-08-11T08:43:34.947337+00:00"
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

### temporal_social_thread_format

- Status: **VERIFICATION_INCOMPLETE**
- Events: 7
- Restarts: 1
- Duration: 6.212487s
- Assertions passed/failed/skipped: 0/0/1


#### Checkpoint state

```json
{
  "start-1": {
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
      "title": null,
      "game": null,
      "category": null
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
    "beliefs": [],
    "game_state": {
      "game": "",
      "platform_version": "",
      "playthrough_type": "casual",
      "spoiler_policy": "spoiler_safe_hints",
      "current_location": null,
      "current_character": "",
      "party_members": [],
      "last_confirmed_progress": "",
      "current_objective": null,
      "challenge": "",
      "known_constraints": [],
      "last_updated": 1786437815.2385707,
      "provenance": "inferred",
      "confidence": 0.0,
      "current_game": null,
      "recent_run_context_facts": []
    },
    "social_state": {
      "recent_active_users": [],
      "recent_chat_count": 0,
      "last_raid": {},
      "last_cheer": {}
    },
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
        "schema_migrations": 2,
        "conversations": 0,
        "open_threads": 0
      },
      "schema_migrations": [
        {
          "component": "cognitive_replay",
          "version": 1,
          "name": "replay_metadata",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "applied_at": "2026-08-11T08:43:35.231477+00:00"
        },
        {
          "component": "conversation_continuity",
          "version": 1,
          "name": "conversation_and_open_threads",
          "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
          "applied_at": "2026-08-11T08:43:35.330380+00:00"
        }
      ],
      "final_response_digest": "",
      "final_response_present": false
    }
  },
  "ill": {
    "runtime": {
      "mode": "active",
      "hebe_sleeping": false,
      "is_running": false,
      "last_input_source": null,
      "last_intent": null,
      "last_firewall": {
        "source": "twitch_viewer",
        "authority": "viewer",
        "decision": "allow",
        "reason": "live_viewer_message"
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
      "title": null,
      "game": null,
      "category": null
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
    "beliefs": [],
    "game_state": {
      "game": "",
      "platform_version": "",
      "playthrough_type": "casual",
      "spoiler_policy": "spoiler_safe_hints",
      "current_location": null,
      "current_character": "",
      "party_members": [],
      "last_confirmed_progress": "",
      "current_objective": null,
      "challenge": "",
      "known_constraints": [],
      "last_updated": 1786437815.2385707,
      "provenance": "inferred",
      "confidence": 0.0,
      "current_game": null,
      "recent_run_context_facts": []
    },
    "social_state": {
      "recent_active_users": [
        "synthetic_student"
      ],
      "recent_chat_count": 1,
      "last_raid": {},
      "last_cheer": {}
    },
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
    "database_watermarks": {
      "counts": {
        "chat_log": 0,
        "memory_facts": 0,
        "memory_chunks": 0,
        "stream_sessions": 1,
        "stream_chat_messages": 1,
        "stream_events": 1,
        "live_session_timeline": 3,
        "promotion_events": 0,
        "viewer_promotion_profiles": 0,
        "schema_migrations": 2,
        "conversations": 0,
        "open_threads": 0
      },
      "schema_migrations": [
        {
          "component": "cognitive_replay",
          "version": 1,
          "name": "replay_metadata",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "applied_at": "2026-08-11T08:43:35.231477+00:00"
        },
        {
          "component": "conversation_continuity",
          "version": 1,
          "name": "conversation_and_open_threads",
          "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
          "applied_at": "2026-08-11T08:43:35.330380+00:00"
        }
      ],
      "final_response_digest": "",
      "final_response_present": false
    }
  },
  "end-1": {
    "runtime": {
      "mode": "active",
      "hebe_sleeping": false,
      "is_running": false,
      "last_input_source": null,
      "last_intent": null,
      "last_firewall": {
        "source": "twitch_viewer",
        "authority": "viewer",
        "decision": "allow",
        "reason": "live_viewer_message"
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
      "is_live": false,
      "live_status_known": true,
      "active_stream_session_id": null,
      "last_transition": "offline",
      "title": null,
      "game": null,
      "category": null
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
      "chunks_count": 1
    },
    "beliefs": [],
    "game_state": {
      "game": "",
      "platform_version": "",
      "playthrough_type": "casual",
      "spoiler_policy": "spoiler_safe_hints",
      "current_location": null,
      "current_character": "",
      "party_members": [],
      "last_confirmed_progress": "",
      "current_objective": null,
      "challenge": "",
      "known_constraints": [],
      "last_updated": 1786437815.2385707,
      "provenance": "inferred",
      "confidence": 0.0,
      "current_game": null,
      "recent_run_context_facts": []
    },
    "social_state": {
      "recent_active_users": [
        "synthetic_student"
      ],
      "recent_chat_count": 1,
      "last_raid": {},
      "last_cheer": {}
    },
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
    "database_watermarks": {
      "counts": {
        "chat_log": 0,
        "memory_facts": 0,
        "memory_chunks": 1,
        "stream_sessions": 1,
        "stream_chat_messages": 1,
        "stream_events": 2,
        "live_session_timeline": 4,
        "promotion_events": 0,
        "viewer_promotion_profiles": 0,
        "schema_migrations": 2,
        "conversations": 0,
        "open_threads": 0
      },
      "schema_migrations": [
        {
          "component": "cognitive_replay",
          "version": 1,
          "name": "replay_metadata",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "applied_at": "2026-08-11T08:43:35.231477+00:00"
        },
        {
          "component": "conversation_continuity",
          "version": 1,
          "name": "conversation_and_open_threads",
          "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
          "applied_at": "2026-08-11T08:43:35.330380+00:00"
        }
      ],
      "final_response_digest": "",
      "final_response_present": false
    }
  },
  "restart": {
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
      "enabled": false,
      "is_live": false,
      "live_status_known": false,
      "active_stream_session_id": null,
      "last_transition": null,
      "title": null,
      "game": null,
      "category": null
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
      "chunks_count": 1
    },
    "beliefs": [],
    "game_state": {
      "game": "",
      "platform_version": "",
      "playthrough_type": "casual",
      "spoiler_policy": "spoiler_safe_hints",
      "current_location": null,
      "current_character": "",
      "party_members": [],
      "last_confirmed_progress": "",
      "current_objective": null,
      "challenge": "",
      "known_constraints": [],
      "last_updated": 1786437821.0447893,
      "provenance": "inferred",
      "confidence": 0.0,
      "current_game": null,
      "recent_run_context_facts": []
    },
    "social_state": {
      "recent_active_users": [],
      "recent_chat_count": 0,
      "last_raid": {},
      "last_cheer": {}
    },
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
    "database_watermarks": {
      "counts": {
        "chat_log": 0,
        "memory_facts": 0,
        "memory_chunks": 1,
        "stream_sessions": 1,
        "stream_chat_messages": 1,
        "stream_events": 2,
        "live_session_timeline": 4,
        "promotion_events": 0,
        "viewer_promotion_profiles": 0,
        "schema_migrations": 2,
        "conversations": 0,
        "open_threads": 0
      },
      "schema_migrations": [
        {
          "component": "cognitive_replay",
          "version": 1,
          "name": "replay_metadata",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "applied_at": "2026-08-11T08:43:35.231477+00:00"
        },
        {
          "component": "conversation_continuity",
          "version": 1,
          "name": "conversation_and_open_threads",
          "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
          "applied_at": "2026-08-11T08:43:35.330380+00:00"
        }
      ],
      "final_response_digest": "",
      "final_response_present": false
    }
  },
  "start-2": {
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
      "active_stream_session_id": 2,
      "last_transition": "online",
      "title": null,
      "game": null,
      "category": null
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
      "chunks_count": 1
    },
    "beliefs": [],
    "game_state": {
      "game": "",
      "platform_version": "",
      "playthrough_type": "casual",
      "spoiler_policy": "spoiler_safe_hints",
      "current_location": null,
      "current_character": "",
      "party_members": [],
      "last_confirmed_progress": "",
      "current_objective": null,
      "challenge": "",
      "known_constraints": [],
      "last_updated": 1786437821.0447893,
      "provenance": "inferred",
      "confidence": 0.0,
      "current_game": null,
      "recent_run_context_facts": []
    },
    "social_state": {
      "recent_active_users": [],
      "recent_chat_count": 0,
      "last_raid": {},
      "last_cheer": {}
    },
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
    "database_watermarks": {
      "counts": {
        "chat_log": 0,
        "memory_facts": 0,
        "memory_chunks": 1,
        "stream_sessions": 2,
        "stream_chat_messages": 1,
        "stream_events": 3,
        "live_session_timeline": 6,
        "promotion_events": 0,
        "viewer_promotion_profiles": 0,
        "schema_migrations": 2,
        "conversations": 0,
        "open_threads": 0
      },
      "schema_migrations": [
        {
          "component": "cognitive_replay",
          "version": 1,
          "name": "replay_metadata",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "applied_at": "2026-08-11T08:43:35.231477+00:00"
        },
        {
          "component": "conversation_continuity",
          "version": 1,
          "name": "conversation_and_open_threads",
          "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
          "applied_at": "2026-08-11T08:43:35.330380+00:00"
        }
      ],
      "final_response_digest": "",
      "final_response_present": false
    }
  },
  "return": {
    "runtime": {
      "mode": "active",
      "hebe_sleeping": false,
      "is_running": false,
      "last_input_source": null,
      "last_intent": null,
      "last_firewall": {
        "source": "twitch_viewer",
        "authority": "viewer",
        "decision": "allow",
        "reason": "live_viewer_message"
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
      "active_stream_session_id": 2,
      "last_transition": "online",
      "title": null,
      "game": null,
      "category": null
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
      "chunks_count": 1
    },
    "beliefs": [],
    "game_state": {
      "game": "",
      "platform_version": "",
      "playthrough_type": "casual",
      "spoiler_policy": "spoiler_safe_hints",
      "current_location": null,
      "current_character": "",
      "party_members": [],
      "last_confirmed_progress": "",
      "current_objective": null,
      "challenge": "",
      "known_constraints": [],
      "last_updated": 1786437821.0447893,
      "provenance": "inferred",
      "confidence": 0.0,
      "current_game": null,
      "recent_run_context_facts": []
    },
    "social_state": {
      "recent_active_users": [
        "synthetic_student"
      ],
      "recent_chat_count": 1,
      "last_raid": {},
      "last_cheer": {}
    },
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
    "database_watermarks": {
      "counts": {
        "chat_log": 0,
        "memory_facts": 0,
        "memory_chunks": 1,
        "stream_sessions": 2,
        "stream_chat_messages": 2,
        "stream_events": 3,
        "live_session_timeline": 7,
        "promotion_events": 0,
        "viewer_promotion_profiles": 0,
        "schema_migrations": 2,
        "conversations": 0,
        "open_threads": 0
      },
      "schema_migrations": [
        {
          "component": "cognitive_replay",
          "version": 1,
          "name": "replay_metadata",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "applied_at": "2026-08-11T08:43:35.231477+00:00"
        },
        {
          "component": "conversation_continuity",
          "version": 1,
          "name": "conversation_and_open_threads",
          "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
          "applied_at": "2026-08-11T08:43:35.330380+00:00"
        }
      ],
      "final_response_digest": "",
      "final_response_present": false
    }
  },
  "expire": {
    "runtime": {
      "mode": "active",
      "hebe_sleeping": false,
      "is_running": false,
      "last_input_source": null,
      "last_intent": null,
      "last_firewall": {
        "source": "twitch_viewer",
        "authority": "viewer",
        "decision": "allow",
        "reason": "live_viewer_message"
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
      "active_stream_session_id": 2,
      "last_transition": "online",
      "title": null,
      "game": null,
      "category": null
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
      "chunks_count": 2
    },
    "beliefs": [],
    "game_state": {
      "game": "",
      "platform_version": "",
      "playthrough_type": "casual",
      "spoiler_policy": "spoiler_safe_hints",
      "current_location": null,
      "current_character": "",
      "party_members": [],
      "last_confirmed_progress": "",
      "current_objective": null,
      "challenge": "",
      "known_constraints": [],
      "last_updated": 1786437821.0447893,
      "provenance": "inferred",
      "confidence": 0.0,
      "current_game": null,
      "recent_run_context_facts": []
    },
    "social_state": {
      "recent_active_users": [
        "synthetic_student"
      ],
      "recent_chat_count": 1,
      "last_raid": {},
      "last_cheer": {}
    },
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
    "database_watermarks": {
      "counts": {
        "chat_log": 0,
        "memory_facts": 0,
        "memory_chunks": 2,
        "stream_sessions": 2,
        "stream_chat_messages": 2,
        "stream_events": 3,
        "live_session_timeline": 8,
        "promotion_events": 0,
        "viewer_promotion_profiles": 0,
        "schema_migrations": 2,
        "conversations": 0,
        "open_threads": 0
      },
      "schema_migrations": [
        {
          "component": "cognitive_replay",
          "version": 1,
          "name": "replay_metadata",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "applied_at": "2026-08-11T08:43:35.231477+00:00"
        },
        {
          "component": "conversation_continuity",
          "version": 1,
          "name": "conversation_and_open_threads",
          "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
          "applied_at": "2026-08-11T08:43:35.330380+00:00"
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
      "source": "twitch_viewer",
      "authority": "viewer",
      "decision": "allow",
      "reason": "live_viewer_message"
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
    "active_stream_session_id": 2,
    "last_transition": "online",
    "title": null,
    "game": null,
    "category": null
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
    "chunks_count": 2
  },
  "beliefs": [],
  "game_state": {
    "game": "",
    "platform_version": "",
    "playthrough_type": "casual",
    "spoiler_policy": "spoiler_safe_hints",
    "current_location": null,
    "current_character": "",
    "party_members": [],
    "last_confirmed_progress": "",
    "current_objective": null,
    "challenge": "",
    "known_constraints": [],
    "last_updated": 1786437821.0447893,
    "provenance": "inferred",
    "confidence": 0.0,
    "current_game": null,
    "recent_run_context_facts": []
  },
  "social_state": {
    "recent_active_users": [
      "synthetic_student"
    ],
    "recent_chat_count": 1,
    "last_raid": {},
    "last_cheer": {}
  },
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
  "database_watermarks": {
    "counts": {
      "chat_log": 0,
      "memory_facts": 0,
      "memory_chunks": 2,
      "stream_sessions": 2,
      "stream_chat_messages": 2,
      "stream_events": 3,
      "live_session_timeline": 8,
      "promotion_events": 0,
      "viewer_promotion_profiles": 0,
      "schema_migrations": 2,
      "conversations": 0,
      "open_threads": 0
    },
    "schema_migrations": [
      {
        "component": "cognitive_replay",
        "version": 1,
        "name": "replay_metadata",
        "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
        "applied_at": "2026-08-11T08:43:35.231477+00:00"
      },
      {
        "component": "conversation_continuity",
        "version": 1,
        "name": "conversation_and_open_threads",
        "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
        "applied_at": "2026-08-11T08:43:35.330380+00:00"
      }
    ],
    "final_response_digest": "",
    "final_response_present": false
  }
}
```

#### Restart evidence

```json
[
  {
    "event_id": "restart",
    "old_engine_id": 2096480115600,
    "new_engine_id": 2096479566992,
    "old_engine_collected": true,
    "same_database": "C:\\Users\\Leo Nifelheim\\Documents\\Hebe\\hebe-ui\\artifacts\\cognitive-continuity-phase1\\final\\workspaces\\temporal_social_thread_format\\hebe-replay.sqlite3",
    "before_persisted_counts": {
      "chat_log": 0,
      "memory_facts": 0,
      "memory_chunks": 1,
      "stream_sessions": 1,
      "stream_chat_messages": 1,
      "stream_events": 2,
      "live_session_timeline": 4,
      "promotion_events": 0,
      "viewer_promotion_profiles": 0,
      "schema_migrations": 2,
      "conversations": 0,
      "open_threads": 0
    },
    "after_persisted_counts": {
      "chat_log": 0,
      "memory_facts": 0,
      "memory_chunks": 1,
      "stream_sessions": 1,
      "stream_chat_messages": 1,
      "stream_events": 2,
      "live_session_timeline": 4,
      "promotion_events": 0,
      "viewer_promotion_profiles": 0,
      "schema_migrations": 2,
      "conversations": 0,
      "open_threads": 0
    },
    "volatile_state_recreated": true
  }
]
```

### phase1_a_ivanxi_wake_free

- Status: **VERIFIED**
- Events: 8
- Restarts: 1
- Duration: 0.799154s
- Assertions passed/failed/skipped: 7/0/0


#### Checkpoint state

```json
{
  "start": {
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
      "title": "FFV",
      "game": "Final Fantasy V",
      "category": "Final Fantasy V"
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
    "beliefs": [],
    "game_state": {
      "game": "Final Fantasy V",
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
      "last_updated": 1786464000.0,
      "provenance": "stream_context_sync",
      "confidence": 0.75,
      "current_game": "Final Fantasy V",
      "recent_run_context_facts": []
    },
    "social_state": {
      "recent_active_users": [],
      "recent_chat_count": 0,
      "last_raid": {},
      "last_cheer": {}
    },
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
        "schema_migrations": 2,
        "conversations": 0,
        "open_threads": 0
      },
      "schema_migrations": [
        {
          "component": "cognitive_replay",
          "version": 1,
          "name": "replay_metadata",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "applied_at": "2026-08-11T08:43:41.462632+00:00"
        },
        {
          "component": "conversation_continuity",
          "version": 1,
          "name": "conversation_and_open_threads",
          "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
          "applied_at": "2026-08-11T08:43:41.531488+00:00"
        }
      ],
      "final_response_digest": "",
      "final_response_present": false
    }
  },
  "resub": {
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
      "title": "FFV",
      "game": "Final Fantasy V",
      "category": "Final Fantasy V"
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
    "beliefs": [],
    "game_state": {
      "game": "Final Fantasy V",
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
      "last_updated": 1786464000.0,
      "provenance": "stream_context_sync",
      "confidence": 0.75,
      "current_game": "Final Fantasy V",
      "recent_run_context_facts": []
    },
    "social_state": {
      "recent_active_users": [],
      "recent_chat_count": 0,
      "last_raid": {},
      "last_cheer": {}
    },
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
        "schema_migrations": 2,
        "conversations": 0,
        "open_threads": 0
      },
      "schema_migrations": [
        {
          "component": "cognitive_replay",
          "version": 1,
          "name": "replay_metadata",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "applied_at": "2026-08-11T08:43:41.462632+00:00"
        },
        {
          "component": "conversation_continuity",
          "version": 1,
          "name": "conversation_and_open_threads",
          "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
          "applied_at": "2026-08-11T08:43:41.531488+00:00"
        }
      ],
      "final_response_digest": "",
      "final_response_present": false
    }
  },
  "chat": {
    "runtime": {
      "mode": "active",
      "hebe_sleeping": false,
      "is_running": false,
      "last_input_source": null,
      "last_intent": null,
      "last_firewall": {
        "source": "twitch_viewer",
        "authority": "viewer",
        "decision": "allow",
        "reason": "live_viewer_message"
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
      "title": "FFV",
      "game": "Final Fantasy V",
      "category": "Final Fantasy V"
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
    "beliefs": [],
    "game_state": {
      "game": "Final Fantasy V",
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
      "last_updated": 1786464000.0,
      "provenance": "stream_context_sync",
      "confidence": 0.75,
      "current_game": "Final Fantasy V",
      "recent_run_context_facts": []
    },
    "social_state": {
      "recent_active_users": [
        "ivanxi_kun"
      ],
      "recent_chat_count": 1,
      "last_raid": {},
      "last_cheer": {}
    },
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
    "database_watermarks": {
      "counts": {
        "chat_log": 0,
        "memory_facts": 0,
        "memory_chunks": 0,
        "stream_sessions": 1,
        "stream_chat_messages": 1,
        "stream_events": 1,
        "live_session_timeline": 3,
        "promotion_events": 0,
        "viewer_promotion_profiles": 0,
        "schema_migrations": 2,
        "conversations": 0,
        "open_threads": 0
      },
      "schema_migrations": [
        {
          "component": "cognitive_replay",
          "version": 1,
          "name": "replay_metadata",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "applied_at": "2026-08-11T08:43:41.462632+00:00"
        },
        {
          "component": "conversation_continuity",
          "version": 1,
          "name": "conversation_and_open_threads",
          "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
          "applied_at": "2026-08-11T08:43:41.531488+00:00"
        }
      ],
      "final_response_digest": "",
      "final_response_present": false
    }
  },
  "ask": {
    "runtime": {
      "mode": "active",
      "hebe_sleeping": false,
      "is_running": false,
      "last_input_source": null,
      "last_intent": null,
      "last_firewall": {
        "source": "owner_stt_direct",
        "authority": "owner",
        "decision": "allow",
        "reason": "owner_direct"
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
      "title": "FFV",
      "game": "Final Fantasy V",
      "category": "Final Fantasy V"
    },
    "current_scene": {},
    "pending": {
      "clarification": {
        "id": "promotion_fa5896fdd3fb5c56bfb4ac98f53dcc74",
        "pending_id": "promotion_fa5896fdd3fb5c56bfb4ac98f53dcc74",
        "kind": "promotion_target_clarification",
        "expected_reply_type": "twitch_username_or_viewer_alias",
        "authority": "owner",
        "authority_required": "owner",
        "allowed_sources": [
          "stt_voice",
          "ui"
        ],
        "capability_needed": "twitch.promotion",
        "opened_by_event_id": "",
        "opened_by_speech_act": "clarification_question",
        "explicit_question_asked": true,
        "can_accept_no_wake_followup": true,
        "can_accept_emote_followup": false,
        "ttl_seconds": 60.0,
        "created_at": 1786464003.0,
        "expires_at": 1786464063.0,
        "max_attempts": 1,
        "attempts": 0,
        "status": "active",
        "compatible_intents": [
          "promotion_target_answer"
        ],
        "incompatible_intents": [
          "stream_monologue",
          "low_confidence_target"
        ],
        "target_raw": "ivanxi",
        "candidates": [
          "ivanxi_kun"
        ],
        "reason": "medium_confidence",
        "fallback_text": "Creo que me has pedido un SO, pero necesito confirmación.",
        "starts_after_tts_end": 0.0,
        "capture_window_seconds": 12,
        "owner_voice_only": true,
        "wake_not_required": true,
        "minimum_target_confidence": 0.78,
        "actual_tts_completion_time": 0.0,
        "buffered_answers": [],
        "conversation_context": "owner_live_control",
        "conversation_context_id": "1",
        "conversation_id": "conv_f0595c65f1345ef48baa1d860777968c"
      },
      "conversation_turn": null
    },
    "conversation": {
      "active": {
        "id": "conv_f0595c65f1345ef48baa1d860777968c",
        "context_kind": "owner_live_control",
        "context_id": "1",
        "attention_state": "HANDED_OFF",
        "turn_owner": "leo",
        "expected_reply_type": "yes_no",
        "topic": "promotion_target_clarification",
        "origin_event_id": "promotion_fa5896fdd3fb5c56bfb4ac98f53dcc74",
        "last_event_id": "promotion_fa5896fdd3fb5c56bfb4ac98f53dcc74",
        "opened_at": 1786464003.0,
        "last_turn_at": 1786464003.0,
        "expires_at": 1786464063.0,
        "status": "WAITING_ON_LEO",
        "closure_reason": "",
        "version": 1,
        "participants": [
          "leo",
          "hebe"
        ],
        "domain_payload": {
          "legacy_pending_id": "promotion_fa5896fdd3fb5c56bfb4ac98f53dcc74",
          "legacy_kind": "promotion_target_clarification",
          "pending": {
            "id": "promotion_fa5896fdd3fb5c56bfb4ac98f53dcc74",
            "pending_id": "promotion_fa5896fdd3fb5c56bfb4ac98f53dcc74",
            "kind": "promotion_target_clarification",
            "expected_reply_type": "twitch_username_or_viewer_alias",
            "authority": "owner",
            "authority_required": "owner",
            "allowed_sources": [
              "stt_voice",
              "ui"
            ],
            "capability_needed": "twitch.promotion",
            "opened_by_event_id": "",
            "opened_by_speech_act": "clarification_question",
            "explicit_question_asked": true,
            "can_accept_no_wake_followup": true,
            "can_accept_emote_followup": false,
            "ttl_seconds": 60.0,
            "created_at": 1786464003.0,
            "expires_at": 1786464063.0,
            "max_attempts": 1,
            "attempts": 0,
            "status": "active",
            "compatible_intents": [
              "promotion_target_answer"
            ],
            "incompatible_intents": [
              "stream_monologue",
              "low_confidence_target"
            ],
            "target_raw": "ivanxi",
            "candidates": [
              "ivanxi_kun"
            ],
            "reason": "medium_confidence",
            "fallback_text": "Creo que me has pedido un SO, pero necesito confirmación.",
            "starts_after_tts_end": 0.0,
            "capture_window_seconds": 12,
            "owner_voice_only": true,
            "wake_not_required": true,
            "minimum_target_confidence": 0.78,
            "actual_tts_completion_time": 0.0,
            "buffered_answers": [],
            "conversation_context": "owner_live_control",
            "conversation_context_id": "1"
          }
        },
        "consumed_event_ids": []
      },
      "latest": {
        "id": "conv_f0595c65f1345ef48baa1d860777968c",
        "context_kind": "owner_live_control",
        "context_id": "1",
        "attention_state": "HANDED_OFF",
        "turn_owner": "leo",
        "expected_reply_type": "yes_no",
        "topic": "promotion_target_clarification",
        "origin_event_id": "promotion_fa5896fdd3fb5c56bfb4ac98f53dcc74",
        "last_event_id": "promotion_fa5896fdd3fb5c56bfb4ac98f53dcc74",
        "opened_at": 1786464003.0,
        "last_turn_at": 1786464003.0,
        "expires_at": 1786464063.0,
        "status": "WAITING_ON_LEO",
        "closure_reason": "",
        "version": 1,
        "participants": [
          "leo",
          "hebe"
        ],
        "domain_payload": {
          "legacy_pending_id": "promotion_fa5896fdd3fb5c56bfb4ac98f53dcc74",
          "legacy_kind": "promotion_target_clarification",
          "pending": {
            "id": "promotion_fa5896fdd3fb5c56bfb4ac98f53dcc74",
            "pending_id": "promotion_fa5896fdd3fb5c56bfb4ac98f53dcc74",
            "kind": "promotion_target_clarification",
            "expected_reply_type": "twitch_username_or_viewer_alias",
            "authority": "owner",
            "authority_required": "owner",
            "allowed_sources": [
              "stt_voice",
              "ui"
            ],
            "capability_needed": "twitch.promotion",
            "opened_by_event_id": "",
            "opened_by_speech_act": "clarification_question",
            "explicit_question_asked": true,
            "can_accept_no_wake_followup": true,
            "can_accept_emote_followup": false,
            "ttl_seconds": 60.0,
            "created_at": 1786464003.0,
            "expires_at": 1786464063.0,
            "max_attempts": 1,
            "attempts": 0,
            "status": "active",
            "compatible_intents": [
              "promotion_target_answer"
            ],
            "incompatible_intents": [
              "stream_monologue",
              "low_confidence_target"
            ],
            "target_raw": "ivanxi",
            "candidates": [
              "ivanxi_kun"
            ],
            "reason": "medium_confidence",
            "fallback_text": "Creo que me has pedido un SO, pero necesito confirmación.",
            "starts_after_tts_end": 0.0,
            "capture_window_seconds": 12,
            "owner_voice_only": true,
            "wake_not_required": true,
            "minimum_target_confidence": 0.78,
            "actual_tts_completion_time": 0.0,
            "buffered_answers": [],
            "conversation_context": "owner_live_control",
            "conversation_context_id": "1"
          }
        },
        "consumed_event_ids": []
      },
      "all": [
        {
          "id": "conv_f0595c65f1345ef48baa1d860777968c",
          "context_kind": "owner_live_control",
          "context_id": "1",
          "attention_state": "HANDED_OFF",
          "turn_owner": "leo",
          "expected_reply_type": "yes_no",
          "topic": "promotion_target_clarification",
          "origin_event_id": "promotion_fa5896fdd3fb5c56bfb4ac98f53dcc74",
          "last_event_id": "promotion_fa5896fdd3fb5c56bfb4ac98f53dcc74",
          "opened_at": 1786464003.0,
          "last_turn_at": 1786464003.0,
          "expires_at": 1786464063.0,
          "status": "WAITING_ON_LEO",
          "closure_reason": "",
          "version": 1,
          "participants": [
            "leo",
            "hebe"
          ],
          "domain_payload": {
            "legacy_pending_id": "promotion_fa5896fdd3fb5c56bfb4ac98f53dcc74",
            "legacy_kind": "promotion_target_clarification",
            "pending": {
              "id": "promotion_fa5896fdd3fb5c56bfb4ac98f53dcc74",
              "pending_id": "promotion_fa5896fdd3fb5c56bfb4ac98f53dcc74",
              "kind": "promotion_target_clarification",
              "expected_reply_type": "twitch_username_or_viewer_alias",
              "authority": "owner",
              "authority_required": "owner",
              "allowed_sources": [
                "stt_voice",
                "ui"
              ],
              "capability_needed": "twitch.promotion",
              "opened_by_event_id": "",
              "opened_by_speech_act": "clarification_question",
              "explicit_question_asked": true,
              "can_accept_no_wake_followup": true,
              "can_accept_emote_followup": false,
              "ttl_seconds": 60.0,
              "created_at": 1786464003.0,
              "expires_at": 1786464063.0,
              "max_attempts": 1,
              "attempts": 0,
              "status": "active",
              "compatible_intents": [
                "promotion_target_answer"
              ],
              "incompatible_intents": [
                "stream_monologue",
                "low_confidence_target"
              ],
              "target_raw": "ivanxi",
              "candidates": [
                "ivanxi_kun"
              ],
              "reason": "medium_confidence",
              "fallback_text": "Creo que me has pedido un SO, pero necesito confirmación.",
              "starts_after_tts_end": 0.0,
              "capture_window_seconds": 12,
              "owner_voice_only": true,
              "wake_not_required": true,
              "minimum_target_confidence": 0.78,
              "actual_tts_completion_time": 0.0,
              "buffered_answers": [],
              "conversation_context": "owner_live_control",
              "conversation_context_id": "1"
            }
          },
          "consumed_event_ids": []
        }
      ],
      "last_resolution": {
        "consumed": false,
        "decision": "no_conversation",
        "reason": "no_compatible_active_conversation",
        "conversation_id": "",
        "reply_act": "UNKNOWN",
        "payload": {},
        "conversation": null,
        "latency_ms": 1.5475000254809856
      },
      "legacy_pending_projection": {
        "direction": "legacy_to_v2",
        "pending_id": "promotion_fa5896fdd3fb5c56bfb4ac98f53dcc74",
        "conversation_id": "conv_f0595c65f1345ef48baa1d860777968c"
      },
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
        "p50_ms": 1.5475,
        "p95_ms": 1.5475
      }
    },
    "open_threads": [
      {
        "id": "thread_087c9cb7e13851e3ace88273f89a75fd",
        "thread_type": "clarification",
        "scope_kind": "owner_live_control",
        "scope_id": "1",
        "subject_ref": "conv_f0595c65f1345ef48baa1d860777968c",
        "summary": "Unresolved promotion_target_clarification clarification",
        "origin_event_id": "promotion_fa5896fdd3fb5c56bfb4ac98f53dcc74",
        "latest_event_id": "promotion_fa5896fdd3fb5c56bfb4ac98f53dcc74",
        "status": "WAITING_ON_LEO",
        "priority": 50,
        "created_at": 1786464003.0,
        "relevance_until": 1786464063.0,
        "valid_until": 1786464063.0,
        "resolved_at": 0.0,
        "resolution_event_id": "",
        "sensitivity": "normal",
        "version": 1,
        "participant_ids": [
          "leo",
          "hebe"
        ]
      }
    ],
    "memory": {
      "facts_count": 0,
      "chunks_count": 0
    },
    "beliefs": [],
    "game_state": {
      "game": "Final Fantasy V",
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
      "last_updated": 1786464000.0,
      "provenance": "stream_context_sync",
      "confidence": 0.75,
      "current_game": "Final Fantasy V",
      "recent_run_context_facts": []
    },
    "social_state": {
      "recent_active_users": [
        "ivanxi_kun"
      ],
      "recent_chat_count": 1,
      "last_raid": {},
      "last_cheer": {}
    },
    "promotion_profiles": [],
    "actions": {
      "attempts": [],
      "speech_requests": [],
      "model_calls": [],
      "research_calls": []
    },
    "receipts": [],
    "emitted_outputs": [
      {
        "event_id": "1786437821.6677",
        "emitted": true,
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "reason": "",
        "text_digest": "",
        "text_present": false
      }
    ],
    "final_emission_results": [
      {
        "event_id": "1786437821.6677",
        "emitted": true,
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "reason": "",
        "text_digest": "",
        "text_present": false
      }
    ],
    "database_watermarks": {
      "counts": {
        "chat_log": 1,
        "memory_facts": 0,
        "memory_chunks": 0,
        "stream_sessions": 1,
        "stream_chat_messages": 1,
        "stream_events": 1,
        "live_session_timeline": 5,
        "promotion_events": 0,
        "viewer_promotion_profiles": 0,
        "schema_migrations": 2,
        "conversations": 1,
        "open_threads": 1
      },
      "schema_migrations": [
        {
          "component": "cognitive_replay",
          "version": 1,
          "name": "replay_metadata",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "applied_at": "2026-08-11T08:43:41.462632+00:00"
        },
        {
          "component": "conversation_continuity",
          "version": 1,
          "name": "conversation_and_open_threads",
          "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
          "applied_at": "2026-08-11T08:43:41.531488+00:00"
        }
      ],
      "final_response_digest": "",
      "final_response_present": false
    }
  },
  "yes": {
    "runtime": {
      "mode": "active",
      "hebe_sleeping": false,
      "is_running": false,
      "last_input_source": null,
      "last_intent": null,
      "last_firewall": {
        "source": "owner_stt_followup",
        "authority": "owner",
        "decision": "allow",
        "reason": "owner_related_followup"
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
      "title": "FFV",
      "game": "Final Fantasy V",
      "category": "Final Fantasy V"
    },
    "current_scene": {},
    "pending": {
      "clarification": null,
      "conversation_turn": null
    },
    "conversation": {
      "active": {},
      "latest": {
        "id": "conv_f0595c65f1345ef48baa1d860777968c",
        "context_kind": "owner_live_control",
        "context_id": "1",
        "attention_state": "RELEASED",
        "turn_owner": "leo",
        "expected_reply_type": "yes_no",
        "topic": "promotion_target_clarification",
        "origin_event_id": "promotion_fa5896fdd3fb5c56bfb4ac98f53dcc74",
        "last_event_id": "yes",
        "opened_at": 1786464003.0,
        "last_turn_at": 1786464004.0,
        "expires_at": 1786464063.0,
        "status": "CLOSED",
        "closure_reason": "reply_consumed",
        "version": 2,
        "participants": [
          "leo",
          "hebe"
        ],
        "domain_payload": {
          "legacy_pending_id": "promotion_fa5896fdd3fb5c56bfb4ac98f53dcc74",
          "legacy_kind": "promotion_target_clarification",
          "pending": {
            "id": "promotion_fa5896fdd3fb5c56bfb4ac98f53dcc74",
            "pending_id": "promotion_fa5896fdd3fb5c56bfb4ac98f53dcc74",
            "kind": "promotion_target_clarification",
            "expected_reply_type": "twitch_username_or_viewer_alias",
            "authority": "owner",
            "authority_required": "owner",
            "allowed_sources": [
              "stt_voice",
              "ui"
            ],
            "capability_needed": "twitch.promotion",
            "opened_by_event_id": "",
            "opened_by_speech_act": "clarification_question",
            "explicit_question_asked": true,
            "can_accept_no_wake_followup": true,
            "can_accept_emote_followup": false,
            "ttl_seconds": 60.0,
            "created_at": 1786464003.0,
            "expires_at": 1786464063.0,
            "max_attempts": 1,
            "attempts": 0,
            "status": "active",
            "compatible_intents": [
              "promotion_target_answer"
            ],
            "incompatible_intents": [
              "stream_monologue",
              "low_confidence_target"
            ],
            "target_raw": "ivanxi",
            "candidates": [
              "ivanxi_kun"
            ],
            "reason": "medium_confidence",
            "fallback_text": "Creo que me has pedido un SO, pero necesito confirmación.",
            "starts_after_tts_end": 0.0,
            "capture_window_seconds": 12,
            "owner_voice_only": true,
            "wake_not_required": true,
            "minimum_target_confidence": 0.78,
            "actual_tts_completion_time": 0.0,
            "buffered_answers": [],
            "conversation_context": "owner_live_control",
            "conversation_context_id": "1"
          }
        },
        "consumed_event_ids": [
          "yes"
        ]
      },
      "all": [
        {
          "id": "conv_f0595c65f1345ef48baa1d860777968c",
          "context_kind": "owner_live_control",
          "context_id": "1",
          "attention_state": "RELEASED",
          "turn_owner": "leo",
          "expected_reply_type": "yes_no",
          "topic": "promotion_target_clarification",
          "origin_event_id": "promotion_fa5896fdd3fb5c56bfb4ac98f53dcc74",
          "last_event_id": "yes",
          "opened_at": 1786464003.0,
          "last_turn_at": 1786464004.0,
          "expires_at": 1786464063.0,
          "status": "CLOSED",
          "closure_reason": "reply_consumed",
          "version": 2,
          "participants": [
            "leo",
            "hebe"
          ],
          "domain_payload": {
            "legacy_pending_id": "promotion_fa5896fdd3fb5c56bfb4ac98f53dcc74",
            "legacy_kind": "promotion_target_clarification",
            "pending": {
              "id": "promotion_fa5896fdd3fb5c56bfb4ac98f53dcc74",
              "pending_id": "promotion_fa5896fdd3fb5c56bfb4ac98f53dcc74",
              "kind": "promotion_target_clarification",
              "expected_reply_type": "twitch_username_or_viewer_alias",
              "authority": "owner",
              "authority_required": "owner",
              "allowed_sources": [
                "stt_voice",
                "ui"
              ],
              "capability_needed": "twitch.promotion",
              "opened_by_event_id": "",
              "opened_by_speech_act": "clarification_question",
              "explicit_question_asked": true,
              "can_accept_no_wake_followup": true,
              "can_accept_emote_followup": false,
              "ttl_seconds": 60.0,
              "created_at": 1786464003.0,
              "expires_at": 1786464063.0,
              "max_attempts": 1,
              "attempts": 0,
              "status": "active",
              "compatible_intents": [
                "promotion_target_answer"
              ],
              "incompatible_intents": [
                "stream_monologue",
                "low_confidence_target"
              ],
              "target_raw": "ivanxi",
              "candidates": [
                "ivanxi_kun"
              ],
              "reason": "medium_confidence",
              "fallback_text": "Creo que me has pedido un SO, pero necesito confirmación.",
              "starts_after_tts_end": 0.0,
              "capture_window_seconds": 12,
              "owner_voice_only": true,
              "wake_not_required": true,
              "minimum_target_confidence": 0.78,
              "actual_tts_completion_time": 0.0,
              "buffered_answers": [],
              "conversation_context": "owner_live_control",
              "conversation_context_id": "1"
            }
          },
          "consumed_event_ids": [
            "yes"
          ]
        }
      ],
      "last_resolution": {
        "consumed": true,
        "decision": "compatible_reply",
        "reason": "deterministic_affirm",
        "conversation_id": "conv_f0595c65f1345ef48baa1d860777968c",
        "reply_act": "AFFIRM",
        "payload": {
          "value": true,
          "domain": {
            "legacy_pending_id": "promotion_fa5896fdd3fb5c56bfb4ac98f53dcc74",
            "legacy_kind": "promotion_target_clarification",
            "pending": {
              "id": "promotion_fa5896fdd3fb5c56bfb4ac98f53dcc74",
              "pending_id": "promotion_fa5896fdd3fb5c56bfb4ac98f53dcc74",
              "kind": "promotion_target_clarification",
              "expected_reply_type": "twitch_username_or_viewer_alias",
              "authority": "owner",
              "authority_required": "owner",
              "allowed_sources": [
                "stt_voice",
                "ui"
              ],
              "capability_needed": "twitch.promotion",
              "opened_by_event_id": "",
              "opened_by_speech_act": "clarification_question",
              "explicit_question_asked": true,
              "can_accept_no_wake_followup": true,
              "can_accept_emote_followup": false,
              "ttl_seconds": 60.0,
              "created_at": 1786464003.0,
              "expires_at": 1786464063.0,
              "max_attempts": 1,
              "attempts": 0,
              "status": "active",
              "compatible_intents": [
                "promotion_target_answer"
              ],
              "incompatible_intents": [
                "stream_monologue",
                "low_confidence_target"
              ],
              "target_raw": "ivanxi",
              "candidates": [
                "ivanxi_kun"
              ],
              "reason": "medium_confidence",
              "fallback_text": "Creo que me has pedido un SO, pero necesito confirmación.",
              "starts_after_tts_end": 0.0,
              "capture_window_seconds": 12,
              "owner_voice_only": true,
              "wake_not_required": true,
              "minimum_target_confidence": 0.78,
              "actual_tts_completion_time": 0.0,
              "buffered_answers": [],
              "conversation_context": "owner_live_control",
              "conversation_context_id": "1"
            }
          },
          "expected_reply_type": "yes_no"
        },
        "conversation": {
          "id": "conv_f0595c65f1345ef48baa1d860777968c",
          "context_kind": "owner_live_control",
          "context_id": "1",
          "participants": [
            "leo",
            "hebe"
          ],
          "attention_state": "RELEASED",
          "turn_owner": "leo",
          "expected_reply": {
            "type": "yes_no",
            "allowed_sources": [
              "owner_stt"
            ],
            "allowed_participant": "leo",
            "semantic_constraints": {
              "allow_deictic": true,
              "min_words": 1,
              "max_words": 40
            },
            "candidate_refs": [
              "ivanxi_kun"
            ],
            "expires_at": 1786464063.0,
            "consume_policy": "once"
          },
          "topic": "promotion_target_clarification",
          "origin_event_id": "promotion_fa5896fdd3fb5c56bfb4ac98f53dcc74",
          "last_event_id": "yes",
          "opened_at": 1786464003.0,
          "last_turn_at": 1786464004.0,
          "expires_at": 1786464063.0,
          "status": "CLOSED",
          "closure_reason": "reply_consumed",
          "version": 2,
          "domain_payload": {
            "legacy_pending_id": "promotion_fa5896fdd3fb5c56bfb4ac98f53dcc74",
            "legacy_kind": "promotion_target_clarification",
            "pending": {
              "id": "promotion_fa5896fdd3fb5c56bfb4ac98f53dcc74",
              "pending_id": "promotion_fa5896fdd3fb5c56bfb4ac98f53dcc74",
              "kind": "promotion_target_clarification",
              "expected_reply_type": "twitch_username_or_viewer_alias",
              "authority": "owner",
              "authority_required": "owner",
              "allowed_sources": [
                "stt_voice",
                "ui"
              ],
              "capability_needed": "twitch.promotion",
              "opened_by_event_id": "",
              "opened_by_speech_act": "clarification_question",
              "explicit_question_asked": true,
              "can_accept_no_wake_followup": true,
              "can_accept_emote_followup": false,
              "ttl_seconds": 60.0,
              "created_at": 1786464003.0,
              "expires_at": 1786464063.0,
              "max_attempts": 1,
              "attempts": 0,
              "status": "active",
              "compatible_intents": [
                "promotion_target_answer"
              ],
              "incompatible_intents": [
                "stream_monologue",
                "low_confidence_target"
              ],
              "target_raw": "ivanxi",
              "candidates": [
                "ivanxi_kun"
              ],
              "reason": "medium_confidence",
              "fallback_text": "Creo que me has pedido un SO, pero necesito confirmación.",
              "starts_after_tts_end": 0.0,
              "capture_window_seconds": 12,
              "owner_voice_only": true,
              "wake_not_required": true,
              "minimum_target_confidence": 0.78,
              "actual_tts_completion_time": 0.0,
              "buffered_answers": [],
              "conversation_context": "owner_live_control",
              "conversation_context_id": "1"
            }
          },
          "consumed_event_ids": [
            "yes"
          ]
        },
        "latency_ms": 12.200800003483891
      },
      "legacy_pending_projection": {
        "direction": "legacy_to_v2",
        "pending_id": "promotion_fa5896fdd3fb5c56bfb4ac98f53dcc74",
        "conversation_id": "conv_f0595c65f1345ef48baa1d860777968c"
      },
      "continuity_shadow_diff": {
        "legacy_result": true,
        "v2_result": true,
        "match": true,
        "difference_reason": ""
      },
      "shadow_metrics": {
        "total": 2,
        "matches": 2,
        "differences": 0,
        "match_rate": 1.0,
        "difference_reasons": {}
      },
      "performance": {
        "count": 2,
        "p50_ms": 6.87415,
        "p95_ms": 12.2008
      }
    },
    "open_threads": [
      {
        "id": "thread_087c9cb7e13851e3ace88273f89a75fd",
        "thread_type": "clarification",
        "scope_kind": "owner_live_control",
        "scope_id": "1",
        "subject_ref": "conv_f0595c65f1345ef48baa1d860777968c",
        "summary": "Unresolved promotion_target_clarification clarification",
        "origin_event_id": "promotion_fa5896fdd3fb5c56bfb4ac98f53dcc74",
        "latest_event_id": "yes",
        "status": "RESOLVED",
        "priority": 50,
        "created_at": 1786464003.0,
        "relevance_until": 1786464063.0,
        "valid_until": 1786464063.0,
        "resolved_at": 1786464004.0,
        "resolution_event_id": "yes",
        "sensitivity": "normal",
        "version": 2,
        "participant_ids": [
          "leo",
          "hebe"
        ]
      }
    ],
    "memory": {
      "facts_count": 0,
      "chunks_count": 0
    },
    "beliefs": [],
    "game_state": {
      "game": "Final Fantasy V",
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
      "last_updated": 1786464000.0,
      "provenance": "stream_context_sync",
      "confidence": 0.75,
      "current_game": "Final Fantasy V",
      "recent_run_context_facts": []
    },
    "social_state": {
      "recent_active_users": [
        "ivanxi_kun"
      ],
      "recent_chat_count": 1,
      "last_raid": {},
      "last_cheer": {}
    },
    "promotion_profiles": [
      {
        "twitch_user_id": "42",
        "current_login": "ivanxi_kun",
        "display_name": "ivanxi_kun",
        "auto_promo_mode": "first_message_each_stream",
        "created_by": "owner_command",
        "last_promoted_stream_id": "1",
        "owner_locked": 1,
        "active": 1
      }
    ],
    "actions": {
      "attempts": [
        {
          "operation": "twitch.shoutout",
          "payload": {
            "target": "ivanxi_kun",
            "command": "!so ivanxi_kun"
          },
          "outcome": {
            "success": true,
            "status": "sent",
            "message_id": "phase1-so-1"
          }
        }
      ],
      "speech_requests": [],
      "model_calls": [],
      "research_calls": []
    },
    "receipts": [
      {
        "id": "promo_87952bc70e925d048386d071f1132bd3",
        "stream_session_id": "1",
        "source_event_id": "promotion_78c8333bdce95645ac5614a33480a439",
        "requested_by": "leo",
        "resolved_twitch_user_id": "42",
        "resolved_login": "ivanxi_kun",
        "trigger_type": "owner_learn_and_execute",
        "execution_status": "sent",
        "twitch_message_id": "phase1-so-1",
        "failure_reason": ""
      }
    ],
    "emitted_outputs": [
      {
        "event_id": "1786437821.6677",
        "emitted": true,
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "reason": "",
        "text_digest": "",
        "text_present": false
      },
      {
        "event_id": "1786437821.7453382",
        "emitted": true,
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "reason": "",
        "text_digest": "",
        "text_present": false
      }
    ],
    "final_emission_results": [
      {
        "event_id": "1786437821.6677",
        "emitted": true,
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "reason": "",
        "text_digest": "",
        "text_present": false
      },
      {
        "event_id": "1786437821.7453382",
        "emitted": true,
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "reason": "",
        "text_digest": "",
        "text_present": false
      }
    ],
    "database_watermarks": {
      "counts": {
        "chat_log": 2,
        "memory_facts": 0,
        "memory_chunks": 0,
        "stream_sessions": 1,
        "stream_chat_messages": 1,
        "stream_events": 1,
        "live_session_timeline": 7,
        "promotion_events": 1,
        "viewer_promotion_profiles": 1,
        "schema_migrations": 2,
        "conversations": 1,
        "open_threads": 1
      },
      "schema_migrations": [
        {
          "component": "cognitive_replay",
          "version": 1,
          "name": "replay_metadata",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "applied_at": "2026-08-11T08:43:41.462632+00:00"
        },
        {
          "component": "conversation_continuity",
          "version": 1,
          "name": "conversation_and_open_threads",
          "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
          "applied_at": "2026-08-11T08:43:41.531488+00:00"
        }
      ],
      "final_response_digest": "",
      "final_response_present": false
    }
  },
  "end": {
    "runtime": {
      "mode": "active",
      "hebe_sleeping": false,
      "is_running": false,
      "last_input_source": null,
      "last_intent": null,
      "last_firewall": {
        "source": "owner_stt_followup",
        "authority": "owner",
        "decision": "allow",
        "reason": "owner_related_followup"
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
      "is_live": false,
      "live_status_known": true,
      "active_stream_session_id": null,
      "last_transition": "offline",
      "title": "FFV",
      "game": "Final Fantasy V",
      "category": "Final Fantasy V"
    },
    "current_scene": {},
    "pending": {
      "clarification": null,
      "conversation_turn": null
    },
    "conversation": {
      "active": {},
      "latest": {
        "id": "conv_f0595c65f1345ef48baa1d860777968c",
        "context_kind": "owner_live_control",
        "context_id": "1",
        "attention_state": "RELEASED",
        "turn_owner": "leo",
        "expected_reply_type": "yes_no",
        "topic": "promotion_target_clarification",
        "origin_event_id": "promotion_fa5896fdd3fb5c56bfb4ac98f53dcc74",
        "last_event_id": "yes",
        "opened_at": 1786464003.0,
        "last_turn_at": 1786464004.0,
        "expires_at": 1786464063.0,
        "status": "CLOSED",
        "closure_reason": "reply_consumed",
        "version": 2,
        "participants": [
          "leo",
          "hebe"
        ],
        "domain_payload": {
          "legacy_pending_id": "promotion_fa5896fdd3fb5c56bfb4ac98f53dcc74",
          "legacy_kind": "promotion_target_clarification",
          "pending": {
            "id": "promotion_fa5896fdd3fb5c56bfb4ac98f53dcc74",
            "pending_id": "promotion_fa5896fdd3fb5c56bfb4ac98f53dcc74",
            "kind": "promotion_target_clarification",
            "expected_reply_type": "twitch_username_or_viewer_alias",
            "authority": "owner",
            "authority_required": "owner",
            "allowed_sources": [
              "stt_voice",
              "ui"
            ],
            "capability_needed": "twitch.promotion",
            "opened_by_event_id": "",
            "opened_by_speech_act": "clarification_question",
            "explicit_question_asked": true,
            "can_accept_no_wake_followup": true,
            "can_accept_emote_followup": false,
            "ttl_seconds": 60.0,
            "created_at": 1786464003.0,
            "expires_at": 1786464063.0,
            "max_attempts": 1,
            "attempts": 0,
            "status": "active",
            "compatible_intents": [
              "promotion_target_answer"
            ],
            "incompatible_intents": [
              "stream_monologue",
              "low_confidence_target"
            ],
            "target_raw": "ivanxi",
            "candidates": [
              "ivanxi_kun"
            ],
            "reason": "medium_confidence",
            "fallback_text": "Creo que me has pedido un SO, pero necesito confirmación.",
            "starts_after_tts_end": 0.0,
            "capture_window_seconds": 12,
            "owner_voice_only": true,
            "wake_not_required": true,
            "minimum_target_confidence": 0.78,
            "actual_tts_completion_time": 0.0,
            "buffered_answers": [],
            "conversation_context": "owner_live_control",
            "conversation_context_id": "1"
          }
        },
        "consumed_event_ids": [
          "yes"
        ]
      },
      "all": [
        {
          "id": "conv_f0595c65f1345ef48baa1d860777968c",
          "context_kind": "owner_live_control",
          "context_id": "1",
          "attention_state": "RELEASED",
          "turn_owner": "leo",
          "expected_reply_type": "yes_no",
          "topic": "promotion_target_clarification",
          "origin_event_id": "promotion_fa5896fdd3fb5c56bfb4ac98f53dcc74",
          "last_event_id": "yes",
          "opened_at": 1786464003.0,
          "last_turn_at": 1786464004.0,
          "expires_at": 1786464063.0,
          "status": "CLOSED",
          "closure_reason": "reply_consumed",
          "version": 2,
          "participants": [
            "leo",
            "hebe"
          ],
          "domain_payload": {
            "legacy_pending_id": "promotion_fa5896fdd3fb5c56bfb4ac98f53dcc74",
            "legacy_kind": "promotion_target_clarification",
            "pending": {
              "id": "promotion_fa5896fdd3fb5c56bfb4ac98f53dcc74",
              "pending_id": "promotion_fa5896fdd3fb5c56bfb4ac98f53dcc74",
              "kind": "promotion_target_clarification",
              "expected_reply_type": "twitch_username_or_viewer_alias",
              "authority": "owner",
              "authority_required": "owner",
              "allowed_sources": [
                "stt_voice",
                "ui"
              ],
              "capability_needed": "twitch.promotion",
              "opened_by_event_id": "",
              "opened_by_speech_act": "clarification_question",
              "explicit_question_asked": true,
              "can_accept_no_wake_followup": true,
              "can_accept_emote_followup": false,
              "ttl_seconds": 60.0,
              "created_at": 1786464003.0,
              "expires_at": 1786464063.0,
              "max_attempts": 1,
              "attempts": 0,
              "status": "active",
              "compatible_intents": [
                "promotion_target_answer"
              ],
              "incompatible_intents": [
                "stream_monologue",
                "low_confidence_target"
              ],
              "target_raw": "ivanxi",
              "candidates": [
                "ivanxi_kun"
              ],
              "reason": "medium_confidence",
              "fallback_text": "Creo que me has pedido un SO, pero necesito confirmación.",
              "starts_after_tts_end": 0.0,
              "capture_window_seconds": 12,
              "owner_voice_only": true,
              "wake_not_required": true,
              "minimum_target_confidence": 0.78,
              "actual_tts_completion_time": 0.0,
              "buffered_answers": [],
              "conversation_context": "owner_live_control",
              "conversation_context_id": "1"
            }
          },
          "consumed_event_ids": [
            "yes"
          ]
        }
      ],
      "last_resolution": {
        "consumed": true,
        "decision": "compatible_reply",
        "reason": "deterministic_affirm",
        "conversation_id": "conv_f0595c65f1345ef48baa1d860777968c",
        "reply_act": "AFFIRM",
        "payload": {
          "value": true,
          "domain": {
            "legacy_pending_id": "promotion_fa5896fdd3fb5c56bfb4ac98f53dcc74",
            "legacy_kind": "promotion_target_clarification",
            "pending": {
              "id": "promotion_fa5896fdd3fb5c56bfb4ac98f53dcc74",
              "pending_id": "promotion_fa5896fdd3fb5c56bfb4ac98f53dcc74",
              "kind": "promotion_target_clarification",
              "expected_reply_type": "twitch_username_or_viewer_alias",
              "authority": "owner",
              "authority_required": "owner",
              "allowed_sources": [
                "stt_voice",
                "ui"
              ],
              "capability_needed": "twitch.promotion",
              "opened_by_event_id": "",
              "opened_by_speech_act": "clarification_question",
              "explicit_question_asked": true,
              "can_accept_no_wake_followup": true,
              "can_accept_emote_followup": false,
              "ttl_seconds": 60.0,
              "created_at": 1786464003.0,
              "expires_at": 1786464063.0,
              "max_attempts": 1,
              "attempts": 0,
              "status": "active",
              "compatible_intents": [
                "promotion_target_answer"
              ],
              "incompatible_intents": [
                "stream_monologue",
                "low_confidence_target"
              ],
              "target_raw": "ivanxi",
              "candidates": [
                "ivanxi_kun"
              ],
              "reason": "medium_confidence",
              "fallback_text": "Creo que me has pedido un SO, pero necesito confirmación.",
              "starts_after_tts_end": 0.0,
              "capture_window_seconds": 12,
              "owner_voice_only": true,
              "wake_not_required": true,
              "minimum_target_confidence": 0.78,
              "actual_tts_completion_time": 0.0,
              "buffered_answers": [],
              "conversation_context": "owner_live_control",
              "conversation_context_id": "1"
            }
          },
          "expected_reply_type": "yes_no"
        },
        "conversation": {
          "id": "conv_f0595c65f1345ef48baa1d860777968c",
          "context_kind": "owner_live_control",
          "context_id": "1",
          "participants": [
            "leo",
            "hebe"
          ],
          "attention_state": "RELEASED",
          "turn_owner": "leo",
          "expected_reply": {
            "type": "yes_no",
            "allowed_sources": [
              "owner_stt"
            ],
            "allowed_participant": "leo",
            "semantic_constraints": {
              "allow_deictic": true,
              "min_words": 1,
              "max_words": 40
            },
            "candidate_refs": [
              "ivanxi_kun"
            ],
            "expires_at": 1786464063.0,
            "consume_policy": "once"
          },
          "topic": "promotion_target_clarification",
          "origin_event_id": "promotion_fa5896fdd3fb5c56bfb4ac98f53dcc74",
          "last_event_id": "yes",
          "opened_at": 1786464003.0,
          "last_turn_at": 1786464004.0,
          "expires_at": 1786464063.0,
          "status": "CLOSED",
          "closure_reason": "reply_consumed",
          "version": 2,
          "domain_payload": {
            "legacy_pending_id": "promotion_fa5896fdd3fb5c56bfb4ac98f53dcc74",
            "legacy_kind": "promotion_target_clarification",
            "pending": {
              "id": "promotion_fa5896fdd3fb5c56bfb4ac98f53dcc74",
              "pending_id": "promotion_fa5896fdd3fb5c56bfb4ac98f53dcc74",
              "kind": "promotion_target_clarification",
              "expected_reply_type": "twitch_username_or_viewer_alias",
              "authority": "owner",
              "authority_required": "owner",
              "allowed_sources": [
                "stt_voice",
                "ui"
              ],
              "capability_needed": "twitch.promotion",
              "opened_by_event_id": "",
              "opened_by_speech_act": "clarification_question",
              "explicit_question_asked": true,
              "can_accept_no_wake_followup": true,
              "can_accept_emote_followup": false,
              "ttl_seconds": 60.0,
              "created_at": 1786464003.0,
              "expires_at": 1786464063.0,
              "max_attempts": 1,
              "attempts": 0,
              "status": "active",
              "compatible_intents": [
                "promotion_target_answer"
              ],
              "incompatible_intents": [
                "stream_monologue",
                "low_confidence_target"
              ],
              "target_raw": "ivanxi",
              "candidates": [
                "ivanxi_kun"
              ],
              "reason": "medium_confidence",
              "fallback_text": "Creo que me has pedido un SO, pero necesito confirmación.",
              "starts_after_tts_end": 0.0,
              "capture_window_seconds": 12,
              "owner_voice_only": true,
              "wake_not_required": true,
              "minimum_target_confidence": 0.78,
              "actual_tts_completion_time": 0.0,
              "buffered_answers": [],
              "conversation_context": "owner_live_control",
              "conversation_context_id": "1"
            }
          },
          "consumed_event_ids": [
            "yes"
          ]
        },
        "latency_ms": 12.200800003483891
      },
      "legacy_pending_projection": {
        "direction": "legacy_to_v2",
        "pending_id": "promotion_fa5896fdd3fb5c56bfb4ac98f53dcc74",
        "conversation_id": "conv_f0595c65f1345ef48baa1d860777968c"
      },
      "continuity_shadow_diff": {
        "legacy_result": true,
        "v2_result": true,
        "match": true,
        "difference_reason": ""
      },
      "shadow_metrics": {
        "total": 2,
        "matches": 2,
        "differences": 0,
        "match_rate": 1.0,
        "difference_reasons": {}
      },
      "performance": {
        "count": 2,
        "p50_ms": 6.87415,
        "p95_ms": 12.2008
      }
    },
    "open_threads": [
      {
        "id": "thread_087c9cb7e13851e3ace88273f89a75fd",
        "thread_type": "clarification",
        "scope_kind": "owner_live_control",
        "scope_id": "1",
        "subject_ref": "conv_f0595c65f1345ef48baa1d860777968c",
        "summary": "Unresolved promotion_target_clarification clarification",
        "origin_event_id": "promotion_fa5896fdd3fb5c56bfb4ac98f53dcc74",
        "latest_event_id": "yes",
        "status": "RESOLVED",
        "priority": 50,
        "created_at": 1786464003.0,
        "relevance_until": 1786464063.0,
        "valid_until": 1786464063.0,
        "resolved_at": 1786464004.0,
        "resolution_event_id": "yes",
        "sensitivity": "normal",
        "version": 2,
        "participant_ids": [
          "leo",
          "hebe"
        ]
      }
    ],
    "memory": {
      "facts_count": 0,
      "chunks_count": 0
    },
    "beliefs": [],
    "game_state": {
      "game": "Final Fantasy V",
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
      "last_updated": 1786464000.0,
      "provenance": "stream_context_sync",
      "confidence": 0.75,
      "current_game": "Final Fantasy V",
      "recent_run_context_facts": []
    },
    "social_state": {
      "recent_active_users": [
        "ivanxi_kun"
      ],
      "recent_chat_count": 1,
      "last_raid": {},
      "last_cheer": {}
    },
    "promotion_profiles": [
      {
        "twitch_user_id": "42",
        "current_login": "ivanxi_kun",
        "display_name": "ivanxi_kun",
        "auto_promo_mode": "first_message_each_stream",
        "created_by": "owner_command",
        "last_promoted_stream_id": "1",
        "owner_locked": 1,
        "active": 1
      }
    ],
    "actions": {
      "attempts": [
        {
          "operation": "twitch.shoutout",
          "payload": {
            "target": "ivanxi_kun",
            "command": "!so ivanxi_kun"
          },
          "outcome": {
            "success": true,
            "status": "sent",
            "message_id": "phase1-so-1"
          }
        }
      ],
      "speech_requests": [],
      "model_calls": [],
      "research_calls": []
    },
    "receipts": [
      {
        "id": "promo_87952bc70e925d048386d071f1132bd3",
        "stream_session_id": "1",
        "source_event_id": "promotion_78c8333bdce95645ac5614a33480a439",
        "requested_by": "leo",
        "resolved_twitch_user_id": "42",
        "resolved_login": "ivanxi_kun",
        "trigger_type": "owner_learn_and_execute",
        "execution_status": "sent",
        "twitch_message_id": "phase1-so-1",
        "failure_reason": ""
      }
    ],
    "emitted_outputs": [
      {
        "event_id": "1786437821.6677",
        "emitted": true,
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "reason": "",
        "text_digest": "",
        "text_present": false
      },
      {
        "event_id": "1786437821.7453382",
        "emitted": true,
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "reason": "",
        "text_digest": "",
        "text_present": false
      }
    ],
    "final_emission_results": [
      {
        "event_id": "1786437821.6677",
        "emitted": true,
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "reason": "",
        "text_digest": "",
        "text_present": false
      },
      {
        "event_id": "1786437821.7453382",
        "emitted": true,
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "reason": "",
        "text_digest": "",
        "text_present": false
      }
    ],
    "database_watermarks": {
      "counts": {
        "chat_log": 2,
        "memory_facts": 0,
        "memory_chunks": 0,
        "stream_sessions": 1,
        "stream_chat_messages": 1,
        "stream_events": 2,
        "live_session_timeline": 8,
        "promotion_events": 1,
        "viewer_promotion_profiles": 1,
        "schema_migrations": 2,
        "conversations": 1,
        "open_threads": 1
      },
      "schema_migrations": [
        {
          "component": "cognitive_replay",
          "version": 1,
          "name": "replay_metadata",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "applied_at": "2026-08-11T08:43:41.462632+00:00"
        },
        {
          "component": "conversation_continuity",
          "version": 1,
          "name": "conversation_and_open_threads",
          "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
          "applied_at": "2026-08-11T08:43:41.531488+00:00"
        }
      ],
      "final_response_digest": "",
      "final_response_present": false
    }
  },
  "restart": {
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
      "enabled": false,
      "is_live": false,
      "live_status_known": false,
      "active_stream_session_id": null,
      "last_transition": null,
      "title": null,
      "game": null,
      "category": null
    },
    "current_scene": {},
    "pending": {
      "clarification": null,
      "conversation_turn": null
    },
    "conversation": {
      "active": {},
      "latest": {
        "id": "conv_f0595c65f1345ef48baa1d860777968c",
        "context_kind": "owner_live_control",
        "context_id": "1",
        "attention_state": "RELEASED",
        "turn_owner": "leo",
        "expected_reply_type": "yes_no",
        "topic": "promotion_target_clarification",
        "origin_event_id": "promotion_fa5896fdd3fb5c56bfb4ac98f53dcc74",
        "last_event_id": "yes",
        "opened_at": 1786464003.0,
        "last_turn_at": 1786464004.0,
        "expires_at": 1786464063.0,
        "status": "CLOSED",
        "closure_reason": "reply_consumed",
        "version": 2,
        "participants": [
          "leo",
          "hebe"
        ],
        "domain_payload": {
          "legacy_pending_id": "promotion_fa5896fdd3fb5c56bfb4ac98f53dcc74",
          "legacy_kind": "promotion_target_clarification",
          "pending": {
            "id": "promotion_fa5896fdd3fb5c56bfb4ac98f53dcc74",
            "pending_id": "promotion_fa5896fdd3fb5c56bfb4ac98f53dcc74",
            "kind": "promotion_target_clarification",
            "expected_reply_type": "twitch_username_or_viewer_alias",
            "authority": "owner",
            "authority_required": "owner",
            "allowed_sources": [
              "stt_voice",
              "ui"
            ],
            "capability_needed": "twitch.promotion",
            "opened_by_event_id": "",
            "opened_by_speech_act": "clarification_question",
            "explicit_question_asked": true,
            "can_accept_no_wake_followup": true,
            "can_accept_emote_followup": false,
            "ttl_seconds": 60.0,
            "created_at": 1786464003.0,
            "expires_at": 1786464063.0,
            "max_attempts": 1,
            "attempts": 0,
            "status": "active",
            "compatible_intents": [
              "promotion_target_answer"
            ],
            "incompatible_intents": [
              "stream_monologue",
              "low_confidence_target"
            ],
            "target_raw": "ivanxi",
            "candidates": [
              "ivanxi_kun"
            ],
            "reason": "medium_confidence",
            "fallback_text": "Creo que me has pedido un SO, pero necesito confirmación.",
            "starts_after_tts_end": 0.0,
            "capture_window_seconds": 12,
            "owner_voice_only": true,
            "wake_not_required": true,
            "minimum_target_confidence": 0.78,
            "actual_tts_completion_time": 0.0,
            "buffered_answers": [],
            "conversation_context": "owner_live_control",
            "conversation_context_id": "1"
          }
        },
        "consumed_event_ids": [
          "yes"
        ]
      },
      "all": [
        {
          "id": "conv_f0595c65f1345ef48baa1d860777968c",
          "context_kind": "owner_live_control",
          "context_id": "1",
          "attention_state": "RELEASED",
          "turn_owner": "leo",
          "expected_reply_type": "yes_no",
          "topic": "promotion_target_clarification",
          "origin_event_id": "promotion_fa5896fdd3fb5c56bfb4ac98f53dcc74",
          "last_event_id": "yes",
          "opened_at": 1786464003.0,
          "last_turn_at": 1786464004.0,
          "expires_at": 1786464063.0,
          "status": "CLOSED",
          "closure_reason": "reply_consumed",
          "version": 2,
          "participants": [
            "leo",
            "hebe"
          ],
          "domain_payload": {
            "legacy_pending_id": "promotion_fa5896fdd3fb5c56bfb4ac98f53dcc74",
            "legacy_kind": "promotion_target_clarification",
            "pending": {
              "id": "promotion_fa5896fdd3fb5c56bfb4ac98f53dcc74",
              "pending_id": "promotion_fa5896fdd3fb5c56bfb4ac98f53dcc74",
              "kind": "promotion_target_clarification",
              "expected_reply_type": "twitch_username_or_viewer_alias",
              "authority": "owner",
              "authority_required": "owner",
              "allowed_sources": [
                "stt_voice",
                "ui"
              ],
              "capability_needed": "twitch.promotion",
              "opened_by_event_id": "",
              "opened_by_speech_act": "clarification_question",
              "explicit_question_asked": true,
              "can_accept_no_wake_followup": true,
              "can_accept_emote_followup": false,
              "ttl_seconds": 60.0,
              "created_at": 1786464003.0,
              "expires_at": 1786464063.0,
              "max_attempts": 1,
              "attempts": 0,
              "status": "active",
              "compatible_intents": [
                "promotion_target_answer"
              ],
              "incompatible_intents": [
                "stream_monologue",
                "low_confidence_target"
              ],
              "target_raw": "ivanxi",
              "candidates": [
                "ivanxi_kun"
              ],
              "reason": "medium_confidence",
              "fallback_text": "Creo que me has pedido un SO, pero necesito confirmación.",
              "starts_after_tts_end": 0.0,
              "capture_window_seconds": 12,
              "owner_voice_only": true,
              "wake_not_required": true,
              "minimum_target_confidence": 0.78,
              "actual_tts_completion_time": 0.0,
              "buffered_answers": [],
              "conversation_context": "owner_live_control",
              "conversation_context_id": "1"
            }
          },
          "consumed_event_ids": [
            "yes"
          ]
        }
      ],
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
    "open_threads": [
      {
        "id": "thread_087c9cb7e13851e3ace88273f89a75fd",
        "thread_type": "clarification",
        "scope_kind": "owner_live_control",
        "scope_id": "1",
        "subject_ref": "conv_f0595c65f1345ef48baa1d860777968c",
        "summary": "Unresolved promotion_target_clarification clarification",
        "origin_event_id": "promotion_fa5896fdd3fb5c56bfb4ac98f53dcc74",
        "latest_event_id": "yes",
        "status": "RESOLVED",
        "priority": 50,
        "created_at": 1786464003.0,
        "relevance_until": 1786464063.0,
        "valid_until": 1786464063.0,
        "resolved_at": 1786464004.0,
        "resolution_event_id": "yes",
        "sensitivity": "normal",
        "version": 2,
        "participant_ids": [
          "leo",
          "hebe"
        ]
      }
    ],
    "memory": {
      "facts_count": 0,
      "chunks_count": 0
    },
    "beliefs": [],
    "game_state": {
      "game": "",
      "platform_version": "",
      "playthrough_type": "casual",
      "spoiler_policy": "spoiler_safe_hints",
      "current_location": null,
      "current_character": "",
      "party_members": [],
      "last_confirmed_progress": "",
      "current_objective": null,
      "challenge": "",
      "known_constraints": [],
      "last_updated": 1786437821.948097,
      "provenance": "inferred",
      "confidence": 0.0,
      "current_game": null,
      "recent_run_context_facts": []
    },
    "social_state": {
      "recent_active_users": [],
      "recent_chat_count": 0,
      "last_raid": {},
      "last_cheer": {}
    },
    "promotion_profiles": [
      {
        "twitch_user_id": "42",
        "current_login": "ivanxi_kun",
        "display_name": "ivanxi_kun",
        "auto_promo_mode": "first_message_each_stream",
        "created_by": "owner_command",
        "last_promoted_stream_id": "1",
        "owner_locked": 1,
        "active": 1
      }
    ],
    "actions": {
      "attempts": [
        {
          "operation": "twitch.shoutout",
          "payload": {
            "target": "ivanxi_kun",
            "command": "!so ivanxi_kun"
          },
          "outcome": {
            "success": true,
            "status": "sent",
            "message_id": "phase1-so-1"
          }
        }
      ],
      "speech_requests": [],
      "model_calls": [],
      "research_calls": []
    },
    "receipts": [
      {
        "id": "promo_87952bc70e925d048386d071f1132bd3",
        "stream_session_id": "1",
        "source_event_id": "promotion_78c8333bdce95645ac5614a33480a439",
        "requested_by": "leo",
        "resolved_twitch_user_id": "42",
        "resolved_login": "ivanxi_kun",
        "trigger_type": "owner_learn_and_execute",
        "execution_status": "sent",
        "twitch_message_id": "phase1-so-1",
        "failure_reason": ""
      }
    ],
    "emitted_outputs": [
      {
        "event_id": "1786437821.6677",
        "emitted": true,
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "reason": "",
        "text_digest": "",
        "text_present": false
      },
      {
        "event_id": "1786437821.7453382",
        "emitted": true,
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "reason": "",
        "text_digest": "",
        "text_present": false
      }
    ],
    "final_emission_results": [
      {
        "event_id": "1786437821.6677",
        "emitted": true,
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "reason": "",
        "text_digest": "",
        "text_present": false
      },
      {
        "event_id": "1786437821.7453382",
        "emitted": true,
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "reason": "",
        "text_digest": "",
        "text_present": false
      }
    ],
    "database_watermarks": {
      "counts": {
        "chat_log": 2,
        "memory_facts": 0,
        "memory_chunks": 0,
        "stream_sessions": 1,
        "stream_chat_messages": 1,
        "stream_events": 2,
        "live_session_timeline": 8,
        "promotion_events": 1,
        "viewer_promotion_profiles": 1,
        "schema_migrations": 2,
        "conversations": 1,
        "open_threads": 1
      },
      "schema_migrations": [
        {
          "component": "cognitive_replay",
          "version": 1,
          "name": "replay_metadata",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "applied_at": "2026-08-11T08:43:41.462632+00:00"
        },
        {
          "component": "conversation_continuity",
          "version": 1,
          "name": "conversation_and_open_threads",
          "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
          "applied_at": "2026-08-11T08:43:41.531488+00:00"
        }
      ],
      "final_response_digest": "",
      "final_response_present": false
    }
  },
  "later": {
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
      "active_stream_session_id": 2,
      "last_transition": "online",
      "title": "FFV later",
      "game": "Final Fantasy V",
      "category": "Final Fantasy V"
    },
    "current_scene": {},
    "pending": {
      "clarification": null,
      "conversation_turn": null
    },
    "conversation": {
      "active": {},
      "latest": {
        "id": "conv_f0595c65f1345ef48baa1d860777968c",
        "context_kind": "owner_live_control",
        "context_id": "1",
        "attention_state": "RELEASED",
        "turn_owner": "leo",
        "expected_reply_type": "yes_no",
        "topic": "promotion_target_clarification",
        "origin_event_id": "promotion_fa5896fdd3fb5c56bfb4ac98f53dcc74",
        "last_event_id": "yes",
        "opened_at": 1786464003.0,
        "last_turn_at": 1786464004.0,
        "expires_at": 1786464063.0,
        "status": "CLOSED",
        "closure_reason": "reply_consumed",
        "version": 2,
        "participants": [
          "leo",
          "hebe"
        ],
        "domain_payload": {
          "legacy_pending_id": "promotion_fa5896fdd3fb5c56bfb4ac98f53dcc74",
          "legacy_kind": "promotion_target_clarification",
          "pending": {
            "id": "promotion_fa5896fdd3fb5c56bfb4ac98f53dcc74",
            "pending_id": "promotion_fa5896fdd3fb5c56bfb4ac98f53dcc74",
            "kind": "promotion_target_clarification",
            "expected_reply_type": "twitch_username_or_viewer_alias",
            "authority": "owner",
            "authority_required": "owner",
            "allowed_sources": [
              "stt_voice",
              "ui"
            ],
            "capability_needed": "twitch.promotion",
            "opened_by_event_id": "",
            "opened_by_speech_act": "clarification_question",
            "explicit_question_asked": true,
            "can_accept_no_wake_followup": true,
            "can_accept_emote_followup": false,
            "ttl_seconds": 60.0,
            "created_at": 1786464003.0,
            "expires_at": 1786464063.0,
            "max_attempts": 1,
            "attempts": 0,
            "status": "active",
            "compatible_intents": [
              "promotion_target_answer"
            ],
            "incompatible_intents": [
              "stream_monologue",
              "low_confidence_target"
            ],
            "target_raw": "ivanxi",
            "candidates": [
              "ivanxi_kun"
            ],
            "reason": "medium_confidence",
            "fallback_text": "Creo que me has pedido un SO, pero necesito confirmación.",
            "starts_after_tts_end": 0.0,
            "capture_window_seconds": 12,
            "owner_voice_only": true,
            "wake_not_required": true,
            "minimum_target_confidence": 0.78,
            "actual_tts_completion_time": 0.0,
            "buffered_answers": [],
            "conversation_context": "owner_live_control",
            "conversation_context_id": "1"
          }
        },
        "consumed_event_ids": [
          "yes"
        ]
      },
      "all": [
        {
          "id": "conv_f0595c65f1345ef48baa1d860777968c",
          "context_kind": "owner_live_control",
          "context_id": "1",
          "attention_state": "RELEASED",
          "turn_owner": "leo",
          "expected_reply_type": "yes_no",
          "topic": "promotion_target_clarification",
          "origin_event_id": "promotion_fa5896fdd3fb5c56bfb4ac98f53dcc74",
          "last_event_id": "yes",
          "opened_at": 1786464003.0,
          "last_turn_at": 1786464004.0,
          "expires_at": 1786464063.0,
          "status": "CLOSED",
          "closure_reason": "reply_consumed",
          "version": 2,
          "participants": [
            "leo",
            "hebe"
          ],
          "domain_payload": {
            "legacy_pending_id": "promotion_fa5896fdd3fb5c56bfb4ac98f53dcc74",
            "legacy_kind": "promotion_target_clarification",
            "pending": {
              "id": "promotion_fa5896fdd3fb5c56bfb4ac98f53dcc74",
              "pending_id": "promotion_fa5896fdd3fb5c56bfb4ac98f53dcc74",
              "kind": "promotion_target_clarification",
              "expected_reply_type": "twitch_username_or_viewer_alias",
              "authority": "owner",
              "authority_required": "owner",
              "allowed_sources": [
                "stt_voice",
                "ui"
              ],
              "capability_needed": "twitch.promotion",
              "opened_by_event_id": "",
              "opened_by_speech_act": "clarification_question",
              "explicit_question_asked": true,
              "can_accept_no_wake_followup": true,
              "can_accept_emote_followup": false,
              "ttl_seconds": 60.0,
              "created_at": 1786464003.0,
              "expires_at": 1786464063.0,
              "max_attempts": 1,
              "attempts": 0,
              "status": "active",
              "compatible_intents": [
                "promotion_target_answer"
              ],
              "incompatible_intents": [
                "stream_monologue",
                "low_confidence_target"
              ],
              "target_raw": "ivanxi",
              "candidates": [
                "ivanxi_kun"
              ],
              "reason": "medium_confidence",
              "fallback_text": "Creo que me has pedido un SO, pero necesito confirmación.",
              "starts_after_tts_end": 0.0,
              "capture_window_seconds": 12,
              "owner_voice_only": true,
              "wake_not_required": true,
              "minimum_target_confidence": 0.78,
              "actual_tts_completion_time": 0.0,
              "buffered_answers": [],
              "conversation_context": "owner_live_control",
              "conversation_context_id": "1"
            }
          },
          "consumed_event_ids": [
            "yes"
          ]
        }
      ],
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
    "open_threads": [
      {
        "id": "thread_087c9cb7e13851e3ace88273f89a75fd",
        "thread_type": "clarification",
        "scope_kind": "owner_live_control",
        "scope_id": "1",
        "subject_ref": "conv_f0595c65f1345ef48baa1d860777968c",
        "summary": "Unresolved promotion_target_clarification clarification",
        "origin_event_id": "promotion_fa5896fdd3fb5c56bfb4ac98f53dcc74",
        "latest_event_id": "yes",
        "status": "RESOLVED",
        "priority": 50,
        "created_at": 1786464003.0,
        "relevance_until": 1786464063.0,
        "valid_until": 1786464063.0,
        "resolved_at": 1786464004.0,
        "resolution_event_id": "yes",
        "sensitivity": "normal",
        "version": 2,
        "participant_ids": [
          "leo",
          "hebe"
        ]
      }
    ],
    "memory": {
      "facts_count": 0,
      "chunks_count": 0
    },
    "beliefs": [],
    "game_state": {
      "game": "Final Fantasy V",
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
      "last_updated": 1786550400.0,
      "provenance": "stream_context_sync",
      "confidence": 0.75,
      "current_game": "Final Fantasy V",
      "recent_run_context_facts": []
    },
    "social_state": {
      "recent_active_users": [],
      "recent_chat_count": 0,
      "last_raid": {},
      "last_cheer": {}
    },
    "promotion_profiles": [
      {
        "twitch_user_id": "42",
        "current_login": "ivanxi_kun",
        "display_name": "ivanxi_kun",
        "auto_promo_mode": "first_message_each_stream",
        "created_by": "owner_command",
        "last_promoted_stream_id": "1",
        "owner_locked": 1,
        "active": 1
      }
    ],
    "actions": {
      "attempts": [
        {
          "operation": "twitch.shoutout",
          "payload": {
            "target": "ivanxi_kun",
            "command": "!so ivanxi_kun"
          },
          "outcome": {
            "success": true,
            "status": "sent",
            "message_id": "phase1-so-1"
          }
        }
      ],
      "speech_requests": [],
      "model_calls": [],
      "research_calls": []
    },
    "receipts": [
      {
        "id": "promo_87952bc70e925d048386d071f1132bd3",
        "stream_session_id": "1",
        "source_event_id": "promotion_78c8333bdce95645ac5614a33480a439",
        "requested_by": "leo",
        "resolved_twitch_user_id": "42",
        "resolved_login": "ivanxi_kun",
        "trigger_type": "owner_learn_and_execute",
        "execution_status": "sent",
        "twitch_message_id": "phase1-so-1",
        "failure_reason": ""
      }
    ],
    "emitted_outputs": [
      {
        "event_id": "1786437821.6677",
        "emitted": true,
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "reason": "",
        "text_digest": "",
        "text_present": false
      },
      {
        "event_id": "1786437821.7453382",
        "emitted": true,
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "reason": "",
        "text_digest": "",
        "text_present": false
      }
    ],
    "final_emission_results": [
      {
        "event_id": "1786437821.6677",
        "emitted": true,
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "reason": "",
        "text_digest": "",
        "text_present": false
      },
      {
        "event_id": "1786437821.7453382",
        "emitted": true,
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "reason": "",
        "text_digest": "",
        "text_present": false
      }
    ],
    "database_watermarks": {
      "counts": {
        "chat_log": 2,
        "memory_facts": 0,
        "memory_chunks": 0,
        "stream_sessions": 2,
        "stream_chat_messages": 1,
        "stream_events": 3,
        "live_session_timeline": 10,
        "promotion_events": 1,
        "viewer_promotion_profiles": 1,
        "schema_migrations": 2,
        "conversations": 1,
        "open_threads": 1
      },
      "schema_migrations": [
        {
          "component": "cognitive_replay",
          "version": 1,
          "name": "replay_metadata",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "applied_at": "2026-08-11T08:43:41.462632+00:00"
        },
        {
          "component": "conversation_continuity",
          "version": 1,
          "name": "conversation_and_open_threads",
          "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
          "applied_at": "2026-08-11T08:43:41.531488+00:00"
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
    "active_stream_session_id": 2,
    "last_transition": "online",
    "title": "FFV later",
    "game": "Final Fantasy V",
    "category": "Final Fantasy V"
  },
  "current_scene": {},
  "pending": {
    "clarification": null,
    "conversation_turn": null
  },
  "conversation": {
    "active": {},
    "latest": {
      "id": "conv_f0595c65f1345ef48baa1d860777968c",
      "context_kind": "owner_live_control",
      "context_id": "1",
      "attention_state": "RELEASED",
      "turn_owner": "leo",
      "expected_reply_type": "yes_no",
      "topic": "promotion_target_clarification",
      "origin_event_id": "promotion_fa5896fdd3fb5c56bfb4ac98f53dcc74",
      "last_event_id": "yes",
      "opened_at": 1786464003.0,
      "last_turn_at": 1786464004.0,
      "expires_at": 1786464063.0,
      "status": "CLOSED",
      "closure_reason": "reply_consumed",
      "version": 2,
      "participants": [
        "leo",
        "hebe"
      ],
      "domain_payload": {
        "legacy_pending_id": "promotion_fa5896fdd3fb5c56bfb4ac98f53dcc74",
        "legacy_kind": "promotion_target_clarification",
        "pending": {
          "id": "promotion_fa5896fdd3fb5c56bfb4ac98f53dcc74",
          "pending_id": "promotion_fa5896fdd3fb5c56bfb4ac98f53dcc74",
          "kind": "promotion_target_clarification",
          "expected_reply_type": "twitch_username_or_viewer_alias",
          "authority": "owner",
          "authority_required": "owner",
          "allowed_sources": [
            "stt_voice",
            "ui"
          ],
          "capability_needed": "twitch.promotion",
          "opened_by_event_id": "",
          "opened_by_speech_act": "clarification_question",
          "explicit_question_asked": true,
          "can_accept_no_wake_followup": true,
          "can_accept_emote_followup": false,
          "ttl_seconds": 60.0,
          "created_at": 1786464003.0,
          "expires_at": 1786464063.0,
          "max_attempts": 1,
          "attempts": 0,
          "status": "active",
          "compatible_intents": [
            "promotion_target_answer"
          ],
          "incompatible_intents": [
            "stream_monologue",
            "low_confidence_target"
          ],
          "target_raw": "ivanxi",
          "candidates": [
            "ivanxi_kun"
          ],
          "reason": "medium_confidence",
          "fallback_text": "Creo que me has pedido un SO, pero necesito confirmación.",
          "starts_after_tts_end": 0.0,
          "capture_window_seconds": 12,
          "owner_voice_only": true,
          "wake_not_required": true,
          "minimum_target_confidence": 0.78,
          "actual_tts_completion_time": 0.0,
          "buffered_answers": [],
          "conversation_context": "owner_live_control",
          "conversation_context_id": "1"
        }
      },
      "consumed_event_ids": [
        "yes"
      ]
    },
    "all": [
      {
        "id": "conv_f0595c65f1345ef48baa1d860777968c",
        "context_kind": "owner_live_control",
        "context_id": "1",
        "attention_state": "RELEASED",
        "turn_owner": "leo",
        "expected_reply_type": "yes_no",
        "topic": "promotion_target_clarification",
        "origin_event_id": "promotion_fa5896fdd3fb5c56bfb4ac98f53dcc74",
        "last_event_id": "yes",
        "opened_at": 1786464003.0,
        "last_turn_at": 1786464004.0,
        "expires_at": 1786464063.0,
        "status": "CLOSED",
        "closure_reason": "reply_consumed",
        "version": 2,
        "participants": [
          "leo",
          "hebe"
        ],
        "domain_payload": {
          "legacy_pending_id": "promotion_fa5896fdd3fb5c56bfb4ac98f53dcc74",
          "legacy_kind": "promotion_target_clarification",
          "pending": {
            "id": "promotion_fa5896fdd3fb5c56bfb4ac98f53dcc74",
            "pending_id": "promotion_fa5896fdd3fb5c56bfb4ac98f53dcc74",
            "kind": "promotion_target_clarification",
            "expected_reply_type": "twitch_username_or_viewer_alias",
            "authority": "owner",
            "authority_required": "owner",
            "allowed_sources": [
              "stt_voice",
              "ui"
            ],
            "capability_needed": "twitch.promotion",
            "opened_by_event_id": "",
            "opened_by_speech_act": "clarification_question",
            "explicit_question_asked": true,
            "can_accept_no_wake_followup": true,
            "can_accept_emote_followup": false,
            "ttl_seconds": 60.0,
            "created_at": 1786464003.0,
            "expires_at": 1786464063.0,
            "max_attempts": 1,
            "attempts": 0,
            "status": "active",
            "compatible_intents": [
              "promotion_target_answer"
            ],
            "incompatible_intents": [
              "stream_monologue",
              "low_confidence_target"
            ],
            "target_raw": "ivanxi",
            "candidates": [
              "ivanxi_kun"
            ],
            "reason": "medium_confidence",
            "fallback_text": "Creo que me has pedido un SO, pero necesito confirmación.",
            "starts_after_tts_end": 0.0,
            "capture_window_seconds": 12,
            "owner_voice_only": true,
            "wake_not_required": true,
            "minimum_target_confidence": 0.78,
            "actual_tts_completion_time": 0.0,
            "buffered_answers": [],
            "conversation_context": "owner_live_control",
            "conversation_context_id": "1"
          }
        },
        "consumed_event_ids": [
          "yes"
        ]
      }
    ],
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
  "open_threads": [
    {
      "id": "thread_087c9cb7e13851e3ace88273f89a75fd",
      "thread_type": "clarification",
      "scope_kind": "owner_live_control",
      "scope_id": "1",
      "subject_ref": "conv_f0595c65f1345ef48baa1d860777968c",
      "summary": "Unresolved promotion_target_clarification clarification",
      "origin_event_id": "promotion_fa5896fdd3fb5c56bfb4ac98f53dcc74",
      "latest_event_id": "yes",
      "status": "RESOLVED",
      "priority": 50,
      "created_at": 1786464003.0,
      "relevance_until": 1786464063.0,
      "valid_until": 1786464063.0,
      "resolved_at": 1786464004.0,
      "resolution_event_id": "yes",
      "sensitivity": "normal",
      "version": 2,
      "participant_ids": [
        "leo",
        "hebe"
      ]
    }
  ],
  "memory": {
    "facts_count": 0,
    "chunks_count": 0
  },
  "beliefs": [],
  "game_state": {
    "game": "Final Fantasy V",
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
    "last_updated": 1786550400.0,
    "provenance": "stream_context_sync",
    "confidence": 0.75,
    "current_game": "Final Fantasy V",
    "recent_run_context_facts": []
  },
  "social_state": {
    "recent_active_users": [],
    "recent_chat_count": 0,
    "last_raid": {},
    "last_cheer": {}
  },
  "promotion_profiles": [
    {
      "twitch_user_id": "42",
      "current_login": "ivanxi_kun",
      "display_name": "ivanxi_kun",
      "auto_promo_mode": "first_message_each_stream",
      "created_by": "owner_command",
      "last_promoted_stream_id": "1",
      "owner_locked": 1,
      "active": 1
    }
  ],
  "actions": {
    "attempts": [
      {
        "operation": "twitch.shoutout",
        "payload": {
          "target": "ivanxi_kun",
          "command": "!so ivanxi_kun"
        },
        "outcome": {
          "success": true,
          "status": "sent",
          "message_id": "phase1-so-1"
        }
      }
    ],
    "speech_requests": [],
    "model_calls": [],
    "research_calls": []
  },
  "receipts": [
    {
      "id": "promo_87952bc70e925d048386d071f1132bd3",
      "stream_session_id": "1",
      "source_event_id": "promotion_78c8333bdce95645ac5614a33480a439",
      "requested_by": "leo",
      "resolved_twitch_user_id": "42",
      "resolved_login": "ivanxi_kun",
      "trigger_type": "owner_learn_and_execute",
      "execution_status": "sent",
      "twitch_message_id": "phase1-so-1",
      "failure_reason": ""
    }
  ],
  "emitted_outputs": [
    {
      "event_id": "1786437821.6677",
      "emitted": true,
      "route": "local_owner_reply",
      "targets": [
        "local_ui"
      ],
      "reason": "",
      "text_digest": "",
      "text_present": false
    },
    {
      "event_id": "1786437821.7453382",
      "emitted": true,
      "route": "local_owner_reply",
      "targets": [
        "local_ui"
      ],
      "reason": "",
      "text_digest": "",
      "text_present": false
    }
  ],
  "final_emission_results": [
    {
      "event_id": "1786437821.6677",
      "emitted": true,
      "route": "local_owner_reply",
      "targets": [
        "local_ui"
      ],
      "reason": "",
      "text_digest": "",
      "text_present": false
    },
    {
      "event_id": "1786437821.7453382",
      "emitted": true,
      "route": "local_owner_reply",
      "targets": [
        "local_ui"
      ],
      "reason": "",
      "text_digest": "",
      "text_present": false
    }
  ],
  "database_watermarks": {
    "counts": {
      "chat_log": 2,
      "memory_facts": 0,
      "memory_chunks": 0,
      "stream_sessions": 2,
      "stream_chat_messages": 1,
      "stream_events": 3,
      "live_session_timeline": 10,
      "promotion_events": 1,
      "viewer_promotion_profiles": 1,
      "schema_migrations": 2,
      "conversations": 1,
      "open_threads": 1
    },
    "schema_migrations": [
      {
        "component": "cognitive_replay",
        "version": 1,
        "name": "replay_metadata",
        "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
        "applied_at": "2026-08-11T08:43:41.462632+00:00"
      },
      {
        "component": "conversation_continuity",
        "version": 1,
        "name": "conversation_and_open_threads",
        "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
        "applied_at": "2026-08-11T08:43:41.531488+00:00"
      }
    ],
    "final_response_digest": "",
    "final_response_present": false
  }
}
```

#### Restart evidence

```json
[
  {
    "event_id": "restart",
    "old_engine_id": 2096480225360,
    "new_engine_id": 2096480712720,
    "old_engine_collected": true,
    "same_database": "C:\\Users\\Leo Nifelheim\\Documents\\Hebe\\hebe-ui\\artifacts\\cognitive-continuity-phase1\\final\\workspaces\\phase1_a_ivanxi_wake_free\\hebe-replay.sqlite3",
    "before_persisted_counts": {
      "chat_log": 2,
      "memory_facts": 0,
      "memory_chunks": 0,
      "stream_sessions": 1,
      "stream_chat_messages": 1,
      "stream_events": 2,
      "live_session_timeline": 8,
      "promotion_events": 1,
      "viewer_promotion_profiles": 1,
      "schema_migrations": 2,
      "conversations": 1,
      "open_threads": 1
    },
    "after_persisted_counts": {
      "chat_log": 2,
      "memory_facts": 0,
      "memory_chunks": 0,
      "stream_sessions": 1,
      "stream_chat_messages": 1,
      "stream_events": 2,
      "live_session_timeline": 8,
      "promotion_events": 1,
      "viewer_promotion_profiles": 1,
      "schema_migrations": 2,
      "conversations": 1,
      "open_threads": 1
    },
    "volatile_state_recreated": true
  }
]
```

### phase1_b_ambient_owner

- Status: **VERIFIED**
- Events: 3
- Restarts: 0
- Duration: 0.280726s
- Assertions passed/failed/skipped: 4/0/0


#### Checkpoint state

```json
{
  "open": {
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
      "enabled": false,
      "is_live": false,
      "live_status_known": false,
      "active_stream_session_id": null,
      "last_transition": null,
      "title": null,
      "game": null,
      "category": null
    },
    "current_scene": {},
    "pending": {
      "clarification": null,
      "conversation_turn": null
    },
    "conversation": {
      "active": {
        "id": "conv_57e0257075e85bdf966e84eca7592b19",
        "context_kind": "owner_local",
        "context_id": "leo_local",
        "attention_state": "HANDED_OFF",
        "turn_owner": "leo",
        "expected_reply_type": "yes_no",
        "topic": "replay_handoff",
        "origin_event_id": "open",
        "last_event_id": "open",
        "opened_at": 1000.0,
        "last_turn_at": 1000.0,
        "expires_at": 1060.0,
        "status": "WAITING_ON_LEO",
        "closure_reason": "",
        "version": 1,
        "participants": [
          "leo",
          "hebe"
        ],
        "domain_payload": {},
        "consumed_event_ids": []
      },
      "latest": {
        "id": "conv_57e0257075e85bdf966e84eca7592b19",
        "context_kind": "owner_local",
        "context_id": "leo_local",
        "attention_state": "HANDED_OFF",
        "turn_owner": "leo",
        "expected_reply_type": "yes_no",
        "topic": "replay_handoff",
        "origin_event_id": "open",
        "last_event_id": "open",
        "opened_at": 1000.0,
        "last_turn_at": 1000.0,
        "expires_at": 1060.0,
        "status": "WAITING_ON_LEO",
        "closure_reason": "",
        "version": 1,
        "participants": [
          "leo",
          "hebe"
        ],
        "domain_payload": {},
        "consumed_event_ids": []
      },
      "all": [
        {
          "id": "conv_57e0257075e85bdf966e84eca7592b19",
          "context_kind": "owner_local",
          "context_id": "leo_local",
          "attention_state": "HANDED_OFF",
          "turn_owner": "leo",
          "expected_reply_type": "yes_no",
          "topic": "replay_handoff",
          "origin_event_id": "open",
          "last_event_id": "open",
          "opened_at": 1000.0,
          "last_turn_at": 1000.0,
          "expires_at": 1060.0,
          "status": "WAITING_ON_LEO",
          "closure_reason": "",
          "version": 1,
          "participants": [
            "leo",
            "hebe"
          ],
          "domain_payload": {},
          "consumed_event_ids": []
        }
      ],
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
    "open_threads": [
      {
        "id": "thread_40d82b7fa2af54ed96832df318165993",
        "thread_type": "clarification",
        "scope_kind": "owner_local",
        "scope_id": "leo_local",
        "subject_ref": "conv_57e0257075e85bdf966e84eca7592b19",
        "summary": "Unresolved replay_handoff clarification",
        "origin_event_id": "open",
        "latest_event_id": "open",
        "status": "WAITING_ON_LEO",
        "priority": 50,
        "created_at": 1000.0,
        "relevance_until": 1060.0,
        "valid_until": 1060.0,
        "resolved_at": 0.0,
        "resolution_event_id": "",
        "sensitivity": "normal",
        "version": 1,
        "participant_ids": [
          "leo",
          "hebe"
        ]
      }
    ],
    "memory": {
      "facts_count": 0,
      "chunks_count": 0
    },
    "beliefs": [],
    "game_state": {
      "game": "",
      "platform_version": "",
      "playthrough_type": "casual",
      "spoiler_policy": "spoiler_safe_hints",
      "current_location": null,
      "current_character": "",
      "party_members": [],
      "last_confirmed_progress": "",
      "current_objective": null,
      "challenge": "",
      "known_constraints": [],
      "last_updated": 1786437822.3551676,
      "provenance": "inferred",
      "confidence": 0.0,
      "current_game": null,
      "recent_run_context_facts": []
    },
    "social_state": {
      "recent_active_users": [],
      "recent_chat_count": 0,
      "last_raid": {},
      "last_cheer": {}
    },
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
    "database_watermarks": {
      "counts": {
        "chat_log": 0,
        "memory_facts": 0,
        "memory_chunks": 0,
        "stream_sessions": 0,
        "stream_chat_messages": 0,
        "stream_events": 0,
        "live_session_timeline": 0,
        "promotion_events": 0,
        "viewer_promotion_profiles": 0,
        "schema_migrations": 2,
        "conversations": 1,
        "open_threads": 1
      },
      "schema_migrations": [
        {
          "component": "cognitive_replay",
          "version": 1,
          "name": "replay_metadata",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "applied_at": "2026-08-11T08:43:42.350123+00:00"
        },
        {
          "component": "conversation_continuity",
          "version": 1,
          "name": "conversation_and_open_threads",
          "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
          "applied_at": "2026-08-11T08:43:42.415717+00:00"
        }
      ],
      "final_response_digest": "",
      "final_response_present": false
    }
  },
  "ambient": {
    "runtime": {
      "mode": "active",
      "hebe_sleeping": false,
      "is_running": false,
      "last_input_source": null,
      "last_intent": null,
      "last_firewall": {
        "source": "ambient_stt",
        "authority": "ambient",
        "decision": "ignore",
        "reason": "offline_stream"
      },
      "last_policy": {
        "source": null,
        "authority": null,
        "decision": null,
        "reason": null
      }
    },
    "stream_session": {
      "enabled": false,
      "is_live": false,
      "live_status_known": false,
      "active_stream_session_id": null,
      "last_transition": null,
      "title": null,
      "game": null,
      "category": null
    },
    "current_scene": {},
    "pending": {
      "clarification": null,
      "conversation_turn": null
    },
    "conversation": {
      "active": {
        "id": "conv_57e0257075e85bdf966e84eca7592b19",
        "context_kind": "owner_local",
        "context_id": "leo_local",
        "attention_state": "HANDED_OFF",
        "turn_owner": "leo",
        "expected_reply_type": "yes_no",
        "topic": "replay_handoff",
        "origin_event_id": "open",
        "last_event_id": "open",
        "opened_at": 1000.0,
        "last_turn_at": 1000.0,
        "expires_at": 1060.0,
        "status": "WAITING_ON_LEO",
        "closure_reason": "",
        "version": 1,
        "participants": [
          "leo",
          "hebe"
        ],
        "domain_payload": {},
        "consumed_event_ids": []
      },
      "latest": {
        "id": "conv_57e0257075e85bdf966e84eca7592b19",
        "context_kind": "owner_local",
        "context_id": "leo_local",
        "attention_state": "HANDED_OFF",
        "turn_owner": "leo",
        "expected_reply_type": "yes_no",
        "topic": "replay_handoff",
        "origin_event_id": "open",
        "last_event_id": "open",
        "opened_at": 1000.0,
        "last_turn_at": 1000.0,
        "expires_at": 1060.0,
        "status": "WAITING_ON_LEO",
        "closure_reason": "",
        "version": 1,
        "participants": [
          "leo",
          "hebe"
        ],
        "domain_payload": {},
        "consumed_event_ids": []
      },
      "all": [
        {
          "id": "conv_57e0257075e85bdf966e84eca7592b19",
          "context_kind": "owner_local",
          "context_id": "leo_local",
          "attention_state": "HANDED_OFF",
          "turn_owner": "leo",
          "expected_reply_type": "yes_no",
          "topic": "replay_handoff",
          "origin_event_id": "open",
          "last_event_id": "open",
          "opened_at": 1000.0,
          "last_turn_at": 1000.0,
          "expires_at": 1060.0,
          "status": "WAITING_ON_LEO",
          "closure_reason": "",
          "version": 1,
          "participants": [
            "leo",
            "hebe"
          ],
          "domain_payload": {},
          "consumed_event_ids": []
        }
      ],
      "last_resolution": {
        "consumed": false,
        "decision": "reject",
        "reason": "participant_mismatch",
        "conversation_id": "conv_57e0257075e85bdf966e84eca7592b19",
        "reply_act": "UNKNOWN",
        "payload": {},
        "conversation": {
          "id": "conv_57e0257075e85bdf966e84eca7592b19",
          "context_kind": "owner_local",
          "context_id": "leo_local",
          "participants": [
            "leo",
            "hebe"
          ],
          "attention_state": "HANDED_OFF",
          "turn_owner": "leo",
          "expected_reply": {
            "type": "yes_no",
            "allowed_sources": [
              "owner_stt"
            ],
            "allowed_participant": "leo",
            "semantic_constraints": {},
            "candidate_refs": [],
            "expires_at": 1060.0,
            "consume_policy": "once"
          },
          "topic": "replay_handoff",
          "origin_event_id": "open",
          "last_event_id": "open",
          "opened_at": 1000.0,
          "last_turn_at": 1000.0,
          "expires_at": 1060.0,
          "status": "WAITING_ON_LEO",
          "closure_reason": "",
          "version": 1,
          "domain_payload": {},
          "consumed_event_ids": []
        },
        "latency_ms": 1.459200051613152
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
        "p50_ms": 1.4592,
        "p95_ms": 1.4592
      }
    },
    "open_threads": [
      {
        "id": "thread_40d82b7fa2af54ed96832df318165993",
        "thread_type": "clarification",
        "scope_kind": "owner_local",
        "scope_id": "leo_local",
        "subject_ref": "conv_57e0257075e85bdf966e84eca7592b19",
        "summary": "Unresolved replay_handoff clarification",
        "origin_event_id": "open",
        "latest_event_id": "open",
        "status": "WAITING_ON_LEO",
        "priority": 50,
        "created_at": 1000.0,
        "relevance_until": 1060.0,
        "valid_until": 1060.0,
        "resolved_at": 0.0,
        "resolution_event_id": "",
        "sensitivity": "normal",
        "version": 1,
        "participant_ids": [
          "leo",
          "hebe"
        ]
      }
    ],
    "memory": {
      "facts_count": 0,
      "chunks_count": 0
    },
    "beliefs": [],
    "game_state": {
      "game": "",
      "platform_version": "",
      "playthrough_type": "casual",
      "spoiler_policy": "spoiler_safe_hints",
      "current_location": null,
      "current_character": "",
      "party_members": [],
      "last_confirmed_progress": "",
      "current_objective": null,
      "challenge": "",
      "known_constraints": [],
      "last_updated": 1786437822.3551676,
      "provenance": "inferred",
      "confidence": 0.0,
      "current_game": null,
      "recent_run_context_facts": []
    },
    "social_state": {
      "recent_active_users": [],
      "recent_chat_count": 0,
      "last_raid": {},
      "last_cheer": {}
    },
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
    "database_watermarks": {
      "counts": {
        "chat_log": 0,
        "memory_facts": 0,
        "memory_chunks": 0,
        "stream_sessions": 0,
        "stream_chat_messages": 0,
        "stream_events": 0,
        "live_session_timeline": 0,
        "promotion_events": 0,
        "viewer_promotion_profiles": 0,
        "schema_migrations": 2,
        "conversations": 1,
        "open_threads": 1
      },
      "schema_migrations": [
        {
          "component": "cognitive_replay",
          "version": 1,
          "name": "replay_metadata",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "applied_at": "2026-08-11T08:43:42.350123+00:00"
        },
        {
          "component": "conversation_continuity",
          "version": 1,
          "name": "conversation_and_open_threads",
          "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
          "applied_at": "2026-08-11T08:43:42.415717+00:00"
        }
      ],
      "final_response_digest": "",
      "final_response_present": false
    }
  },
  "owner": {
    "runtime": {
      "mode": "active",
      "hebe_sleeping": false,
      "is_running": false,
      "last_input_source": null,
      "last_intent": null,
      "last_firewall": {
        "source": "ambient_stt",
        "authority": "ambient",
        "decision": "ignore",
        "reason": "offline_stream"
      },
      "last_policy": {
        "source": null,
        "authority": null,
        "decision": null,
        "reason": null
      }
    },
    "stream_session": {
      "enabled": false,
      "is_live": false,
      "live_status_known": false,
      "active_stream_session_id": null,
      "last_transition": null,
      "title": null,
      "game": null,
      "category": null
    },
    "current_scene": {},
    "pending": {
      "clarification": null,
      "conversation_turn": null
    },
    "conversation": {
      "active": {},
      "latest": {
        "id": "conv_57e0257075e85bdf966e84eca7592b19",
        "context_kind": "owner_local",
        "context_id": "leo_local",
        "attention_state": "RELEASED",
        "turn_owner": "leo",
        "expected_reply_type": "yes_no",
        "topic": "replay_handoff",
        "origin_event_id": "open",
        "last_event_id": "owner",
        "opened_at": 1000.0,
        "last_turn_at": 1002.0,
        "expires_at": 1060.0,
        "status": "CLOSED",
        "closure_reason": "reply_consumed",
        "version": 2,
        "participants": [
          "leo",
          "hebe"
        ],
        "domain_payload": {},
        "consumed_event_ids": [
          "owner"
        ]
      },
      "all": [
        {
          "id": "conv_57e0257075e85bdf966e84eca7592b19",
          "context_kind": "owner_local",
          "context_id": "leo_local",
          "attention_state": "RELEASED",
          "turn_owner": "leo",
          "expected_reply_type": "yes_no",
          "topic": "replay_handoff",
          "origin_event_id": "open",
          "last_event_id": "owner",
          "opened_at": 1000.0,
          "last_turn_at": 1002.0,
          "expires_at": 1060.0,
          "status": "CLOSED",
          "closure_reason": "reply_consumed",
          "version": 2,
          "participants": [
            "leo",
            "hebe"
          ],
          "domain_payload": {},
          "consumed_event_ids": [
            "owner"
          ]
        }
      ],
      "last_resolution": {
        "consumed": true,
        "decision": "compatible_reply",
        "reason": "deterministic_affirm",
        "conversation_id": "conv_57e0257075e85bdf966e84eca7592b19",
        "reply_act": "AFFIRM",
        "payload": {
          "value": true,
          "domain": {},
          "expected_reply_type": "yes_no"
        },
        "conversation": {
          "id": "conv_57e0257075e85bdf966e84eca7592b19",
          "context_kind": "owner_local",
          "context_id": "leo_local",
          "participants": [
            "leo",
            "hebe"
          ],
          "attention_state": "RELEASED",
          "turn_owner": "leo",
          "expected_reply": {
            "type": "yes_no",
            "allowed_sources": [
              "owner_stt"
            ],
            "allowed_participant": "leo",
            "semantic_constraints": {},
            "candidate_refs": [],
            "expires_at": 1060.0,
            "consume_policy": "once"
          },
          "topic": "replay_handoff",
          "origin_event_id": "open",
          "last_event_id": "owner",
          "opened_at": 1000.0,
          "last_turn_at": 1002.0,
          "expires_at": 1060.0,
          "status": "CLOSED",
          "closure_reason": "reply_consumed",
          "version": 2,
          "domain_payload": {},
          "consumed_event_ids": [
            "owner"
          ]
        },
        "latency_ms": 8.948499977122992
      },
      "legacy_pending_projection": {},
      "continuity_shadow_diff": {
        "legacy_result": false,
        "v2_result": true,
        "match": false,
        "difference_reason": "deterministic_affirm"
      },
      "shadow_metrics": {
        "total": 2,
        "matches": 1,
        "differences": 1,
        "match_rate": 0.5,
        "difference_reasons": {
          "deterministic_affirm": 1
        }
      },
      "performance": {
        "count": 2,
        "p50_ms": 5.20385,
        "p95_ms": 8.9485
      }
    },
    "open_threads": [
      {
        "id": "thread_40d82b7fa2af54ed96832df318165993",
        "thread_type": "clarification",
        "scope_kind": "owner_local",
        "scope_id": "leo_local",
        "subject_ref": "conv_57e0257075e85bdf966e84eca7592b19",
        "summary": "Unresolved replay_handoff clarification",
        "origin_event_id": "open",
        "latest_event_id": "owner",
        "status": "RESOLVED",
        "priority": 50,
        "created_at": 1000.0,
        "relevance_until": 1060.0,
        "valid_until": 1060.0,
        "resolved_at": 1002.0,
        "resolution_event_id": "owner",
        "sensitivity": "normal",
        "version": 2,
        "participant_ids": [
          "leo",
          "hebe"
        ]
      }
    ],
    "memory": {
      "facts_count": 0,
      "chunks_count": 0
    },
    "beliefs": [],
    "game_state": {
      "game": "",
      "platform_version": "",
      "playthrough_type": "casual",
      "spoiler_policy": "spoiler_safe_hints",
      "current_location": null,
      "current_character": "",
      "party_members": [],
      "last_confirmed_progress": "",
      "current_objective": null,
      "challenge": "",
      "known_constraints": [],
      "last_updated": 1786437822.3551676,
      "provenance": "inferred",
      "confidence": 0.0,
      "current_game": null,
      "recent_run_context_facts": []
    },
    "social_state": {
      "recent_active_users": [],
      "recent_chat_count": 0,
      "last_raid": {},
      "last_cheer": {}
    },
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
    "database_watermarks": {
      "counts": {
        "chat_log": 0,
        "memory_facts": 0,
        "memory_chunks": 0,
        "stream_sessions": 0,
        "stream_chat_messages": 0,
        "stream_events": 0,
        "live_session_timeline": 0,
        "promotion_events": 0,
        "viewer_promotion_profiles": 0,
        "schema_migrations": 2,
        "conversations": 1,
        "open_threads": 1
      },
      "schema_migrations": [
        {
          "component": "cognitive_replay",
          "version": 1,
          "name": "replay_metadata",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "applied_at": "2026-08-11T08:43:42.350123+00:00"
        },
        {
          "component": "conversation_continuity",
          "version": 1,
          "name": "conversation_and_open_threads",
          "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
          "applied_at": "2026-08-11T08:43:42.415717+00:00"
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
      "source": "ambient_stt",
      "authority": "ambient",
      "decision": "ignore",
      "reason": "offline_stream"
    },
    "last_policy": {
      "source": null,
      "authority": null,
      "decision": null,
      "reason": null
    }
  },
  "stream_session": {
    "enabled": false,
    "is_live": false,
    "live_status_known": false,
    "active_stream_session_id": null,
    "last_transition": null,
    "title": null,
    "game": null,
    "category": null
  },
  "current_scene": {},
  "pending": {
    "clarification": null,
    "conversation_turn": null
  },
  "conversation": {
    "active": {},
    "latest": {
      "id": "conv_57e0257075e85bdf966e84eca7592b19",
      "context_kind": "owner_local",
      "context_id": "leo_local",
      "attention_state": "RELEASED",
      "turn_owner": "leo",
      "expected_reply_type": "yes_no",
      "topic": "replay_handoff",
      "origin_event_id": "open",
      "last_event_id": "owner",
      "opened_at": 1000.0,
      "last_turn_at": 1002.0,
      "expires_at": 1060.0,
      "status": "CLOSED",
      "closure_reason": "reply_consumed",
      "version": 2,
      "participants": [
        "leo",
        "hebe"
      ],
      "domain_payload": {},
      "consumed_event_ids": [
        "owner"
      ]
    },
    "all": [
      {
        "id": "conv_57e0257075e85bdf966e84eca7592b19",
        "context_kind": "owner_local",
        "context_id": "leo_local",
        "attention_state": "RELEASED",
        "turn_owner": "leo",
        "expected_reply_type": "yes_no",
        "topic": "replay_handoff",
        "origin_event_id": "open",
        "last_event_id": "owner",
        "opened_at": 1000.0,
        "last_turn_at": 1002.0,
        "expires_at": 1060.0,
        "status": "CLOSED",
        "closure_reason": "reply_consumed",
        "version": 2,
        "participants": [
          "leo",
          "hebe"
        ],
        "domain_payload": {},
        "consumed_event_ids": [
          "owner"
        ]
      }
    ],
    "last_resolution": {
      "consumed": true,
      "decision": "compatible_reply",
      "reason": "deterministic_affirm",
      "conversation_id": "conv_57e0257075e85bdf966e84eca7592b19",
      "reply_act": "AFFIRM",
      "payload": {
        "value": true,
        "domain": {},
        "expected_reply_type": "yes_no"
      },
      "conversation": {
        "id": "conv_57e0257075e85bdf966e84eca7592b19",
        "context_kind": "owner_local",
        "context_id": "leo_local",
        "participants": [
          "leo",
          "hebe"
        ],
        "attention_state": "RELEASED",
        "turn_owner": "leo",
        "expected_reply": {
          "type": "yes_no",
          "allowed_sources": [
            "owner_stt"
          ],
          "allowed_participant": "leo",
          "semantic_constraints": {},
          "candidate_refs": [],
          "expires_at": 1060.0,
          "consume_policy": "once"
        },
        "topic": "replay_handoff",
        "origin_event_id": "open",
        "last_event_id": "owner",
        "opened_at": 1000.0,
        "last_turn_at": 1002.0,
        "expires_at": 1060.0,
        "status": "CLOSED",
        "closure_reason": "reply_consumed",
        "version": 2,
        "domain_payload": {},
        "consumed_event_ids": [
          "owner"
        ]
      },
      "latency_ms": 8.948499977122992
    },
    "legacy_pending_projection": {},
    "continuity_shadow_diff": {
      "legacy_result": false,
      "v2_result": true,
      "match": false,
      "difference_reason": "deterministic_affirm"
    },
    "shadow_metrics": {
      "total": 2,
      "matches": 1,
      "differences": 1,
      "match_rate": 0.5,
      "difference_reasons": {
        "deterministic_affirm": 1
      }
    },
    "performance": {
      "count": 2,
      "p50_ms": 5.20385,
      "p95_ms": 8.9485
    }
  },
  "open_threads": [
    {
      "id": "thread_40d82b7fa2af54ed96832df318165993",
      "thread_type": "clarification",
      "scope_kind": "owner_local",
      "scope_id": "leo_local",
      "subject_ref": "conv_57e0257075e85bdf966e84eca7592b19",
      "summary": "Unresolved replay_handoff clarification",
      "origin_event_id": "open",
      "latest_event_id": "owner",
      "status": "RESOLVED",
      "priority": 50,
      "created_at": 1000.0,
      "relevance_until": 1060.0,
      "valid_until": 1060.0,
      "resolved_at": 1002.0,
      "resolution_event_id": "owner",
      "sensitivity": "normal",
      "version": 2,
      "participant_ids": [
        "leo",
        "hebe"
      ]
    }
  ],
  "memory": {
    "facts_count": 0,
    "chunks_count": 0
  },
  "beliefs": [],
  "game_state": {
    "game": "",
    "platform_version": "",
    "playthrough_type": "casual",
    "spoiler_policy": "spoiler_safe_hints",
    "current_location": null,
    "current_character": "",
    "party_members": [],
    "last_confirmed_progress": "",
    "current_objective": null,
    "challenge": "",
    "known_constraints": [],
    "last_updated": 1786437822.3551676,
    "provenance": "inferred",
    "confidence": 0.0,
    "current_game": null,
    "recent_run_context_facts": []
  },
  "social_state": {
    "recent_active_users": [],
    "recent_chat_count": 0,
    "last_raid": {},
    "last_cheer": {}
  },
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
  "database_watermarks": {
    "counts": {
      "chat_log": 0,
      "memory_facts": 0,
      "memory_chunks": 0,
      "stream_sessions": 0,
      "stream_chat_messages": 0,
      "stream_events": 0,
      "live_session_timeline": 0,
      "promotion_events": 0,
      "viewer_promotion_profiles": 0,
      "schema_migrations": 2,
      "conversations": 1,
      "open_threads": 1
    },
    "schema_migrations": [
      {
        "component": "cognitive_replay",
        "version": 1,
        "name": "replay_metadata",
        "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
        "applied_at": "2026-08-11T08:43:42.350123+00:00"
      },
      {
        "component": "conversation_continuity",
        "version": 1,
        "name": "conversation_and_open_threads",
        "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
        "applied_at": "2026-08-11T08:43:42.415717+00:00"
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

### phase1_c_expired

- Status: **VERIFIED**
- Events: 3
- Restarts: 0
- Duration: 0.278039s
- Assertions passed/failed/skipped: 2/0/0


#### Checkpoint state

```json
{
  "open": {
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
      "enabled": false,
      "is_live": false,
      "live_status_known": false,
      "active_stream_session_id": null,
      "last_transition": null,
      "title": null,
      "game": null,
      "category": null
    },
    "current_scene": {},
    "pending": {
      "clarification": null,
      "conversation_turn": null
    },
    "conversation": {
      "active": {
        "id": "conv_5add27b4469a5395b06047e7328dad18",
        "context_kind": "owner_local",
        "context_id": "leo_local",
        "attention_state": "HANDED_OFF",
        "turn_owner": "leo",
        "expected_reply_type": "yes_no",
        "topic": "replay_handoff",
        "origin_event_id": "open",
        "last_event_id": "open",
        "opened_at": 1000.0,
        "last_turn_at": 1000.0,
        "expires_at": 1005.0,
        "status": "WAITING_ON_LEO",
        "closure_reason": "",
        "version": 1,
        "participants": [
          "leo",
          "hebe"
        ],
        "domain_payload": {},
        "consumed_event_ids": []
      },
      "latest": {
        "id": "conv_5add27b4469a5395b06047e7328dad18",
        "context_kind": "owner_local",
        "context_id": "leo_local",
        "attention_state": "HANDED_OFF",
        "turn_owner": "leo",
        "expected_reply_type": "yes_no",
        "topic": "replay_handoff",
        "origin_event_id": "open",
        "last_event_id": "open",
        "opened_at": 1000.0,
        "last_turn_at": 1000.0,
        "expires_at": 1005.0,
        "status": "WAITING_ON_LEO",
        "closure_reason": "",
        "version": 1,
        "participants": [
          "leo",
          "hebe"
        ],
        "domain_payload": {},
        "consumed_event_ids": []
      },
      "all": [
        {
          "id": "conv_5add27b4469a5395b06047e7328dad18",
          "context_kind": "owner_local",
          "context_id": "leo_local",
          "attention_state": "HANDED_OFF",
          "turn_owner": "leo",
          "expected_reply_type": "yes_no",
          "topic": "replay_handoff",
          "origin_event_id": "open",
          "last_event_id": "open",
          "opened_at": 1000.0,
          "last_turn_at": 1000.0,
          "expires_at": 1005.0,
          "status": "WAITING_ON_LEO",
          "closure_reason": "",
          "version": 1,
          "participants": [
            "leo",
            "hebe"
          ],
          "domain_payload": {},
          "consumed_event_ids": []
        }
      ],
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
    "open_threads": [
      {
        "id": "thread_f263e49836685b3f9c7441004d215f75",
        "thread_type": "clarification",
        "scope_kind": "owner_local",
        "scope_id": "leo_local",
        "subject_ref": "conv_5add27b4469a5395b06047e7328dad18",
        "summary": "Unresolved replay_handoff clarification",
        "origin_event_id": "open",
        "latest_event_id": "open",
        "status": "WAITING_ON_LEO",
        "priority": 50,
        "created_at": 1000.0,
        "relevance_until": 1005.0,
        "valid_until": 1005.0,
        "resolved_at": 0.0,
        "resolution_event_id": "",
        "sensitivity": "normal",
        "version": 1,
        "participant_ids": [
          "leo",
          "hebe"
        ]
      }
    ],
    "memory": {
      "facts_count": 0,
      "chunks_count": 0
    },
    "beliefs": [],
    "game_state": {
      "game": "",
      "platform_version": "",
      "playthrough_type": "casual",
      "spoiler_policy": "spoiler_safe_hints",
      "current_location": null,
      "current_character": "",
      "party_members": [],
      "last_confirmed_progress": "",
      "current_objective": null,
      "challenge": "",
      "known_constraints": [],
      "last_updated": 1786437822.703677,
      "provenance": "inferred",
      "confidence": 0.0,
      "current_game": null,
      "recent_run_context_facts": []
    },
    "social_state": {
      "recent_active_users": [],
      "recent_chat_count": 0,
      "last_raid": {},
      "last_cheer": {}
    },
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
    "database_watermarks": {
      "counts": {
        "chat_log": 0,
        "memory_facts": 0,
        "memory_chunks": 0,
        "stream_sessions": 0,
        "stream_chat_messages": 0,
        "stream_events": 0,
        "live_session_timeline": 0,
        "promotion_events": 0,
        "viewer_promotion_profiles": 0,
        "schema_migrations": 2,
        "conversations": 1,
        "open_threads": 1
      },
      "schema_migrations": [
        {
          "component": "cognitive_replay",
          "version": 1,
          "name": "replay_metadata",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "applied_at": "2026-08-11T08:43:42.699121+00:00"
        },
        {
          "component": "conversation_continuity",
          "version": 1,
          "name": "conversation_and_open_threads",
          "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
          "applied_at": "2026-08-11T08:43:42.765257+00:00"
        }
      ],
      "final_response_digest": "",
      "final_response_present": false
    }
  },
  "advance": {
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
      "enabled": false,
      "is_live": false,
      "live_status_known": true,
      "active_stream_session_id": null,
      "last_transition": null,
      "title": null,
      "game": null,
      "category": null
    },
    "current_scene": {},
    "pending": {
      "clarification": null,
      "conversation_turn": null
    },
    "conversation": {
      "active": {},
      "latest": {
        "id": "conv_5add27b4469a5395b06047e7328dad18",
        "context_kind": "owner_local",
        "context_id": "leo_local",
        "attention_state": "RELEASED",
        "turn_owner": "leo",
        "expected_reply_type": "yes_no",
        "topic": "replay_handoff",
        "origin_event_id": "open",
        "last_event_id": "open",
        "opened_at": 1000.0,
        "last_turn_at": 1000.0,
        "expires_at": 1005.0,
        "status": "EXPIRED",
        "closure_reason": "ttl",
        "version": 2,
        "participants": [
          "leo",
          "hebe"
        ],
        "domain_payload": {},
        "consumed_event_ids": []
      },
      "all": [
        {
          "id": "conv_5add27b4469a5395b06047e7328dad18",
          "context_kind": "owner_local",
          "context_id": "leo_local",
          "attention_state": "RELEASED",
          "turn_owner": "leo",
          "expected_reply_type": "yes_no",
          "topic": "replay_handoff",
          "origin_event_id": "open",
          "last_event_id": "open",
          "opened_at": 1000.0,
          "last_turn_at": 1000.0,
          "expires_at": 1005.0,
          "status": "EXPIRED",
          "closure_reason": "ttl",
          "version": 2,
          "participants": [
            "leo",
            "hebe"
          ],
          "domain_payload": {},
          "consumed_event_ids": []
        }
      ],
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
    "open_threads": [
      {
        "id": "thread_f263e49836685b3f9c7441004d215f75",
        "thread_type": "clarification",
        "scope_kind": "owner_local",
        "scope_id": "leo_local",
        "subject_ref": "conv_5add27b4469a5395b06047e7328dad18",
        "summary": "Unresolved replay_handoff clarification",
        "origin_event_id": "open",
        "latest_event_id": "ttl",
        "status": "EXPIRED",
        "priority": 50,
        "created_at": 1000.0,
        "relevance_until": 1005.0,
        "valid_until": 1005.0,
        "resolved_at": 1011.0,
        "resolution_event_id": "ttl",
        "sensitivity": "normal",
        "version": 2,
        "participant_ids": [
          "leo",
          "hebe"
        ]
      }
    ],
    "memory": {
      "facts_count": 0,
      "chunks_count": 0
    },
    "beliefs": [],
    "game_state": {
      "game": "",
      "platform_version": "",
      "playthrough_type": "casual",
      "spoiler_policy": "spoiler_safe_hints",
      "current_location": null,
      "current_character": "",
      "party_members": [],
      "last_confirmed_progress": "",
      "current_objective": null,
      "challenge": "",
      "known_constraints": [],
      "last_updated": 1786437822.703677,
      "provenance": "inferred",
      "confidence": 0.0,
      "current_game": null,
      "recent_run_context_facts": []
    },
    "social_state": {
      "recent_active_users": [],
      "recent_chat_count": 0,
      "last_raid": {},
      "last_cheer": {}
    },
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
    "database_watermarks": {
      "counts": {
        "chat_log": 0,
        "memory_facts": 0,
        "memory_chunks": 0,
        "stream_sessions": 0,
        "stream_chat_messages": 0,
        "stream_events": 0,
        "live_session_timeline": 1,
        "promotion_events": 0,
        "viewer_promotion_profiles": 0,
        "schema_migrations": 2,
        "conversations": 1,
        "open_threads": 1
      },
      "schema_migrations": [
        {
          "component": "cognitive_replay",
          "version": 1,
          "name": "replay_metadata",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "applied_at": "2026-08-11T08:43:42.699121+00:00"
        },
        {
          "component": "conversation_continuity",
          "version": 1,
          "name": "conversation_and_open_threads",
          "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
          "applied_at": "2026-08-11T08:43:42.765257+00:00"
        }
      ],
      "final_response_digest": "",
      "final_response_present": false
    }
  },
  "late": {
    "runtime": {
      "mode": "active",
      "hebe_sleeping": false,
      "is_running": false,
      "last_input_source": null,
      "last_intent": null,
      "last_firewall": {
        "source": "ambient_stt",
        "authority": "ambient",
        "decision": "ignore",
        "reason": "offline_stream"
      },
      "last_policy": {
        "source": null,
        "authority": null,
        "decision": null,
        "reason": null
      }
    },
    "stream_session": {
      "enabled": false,
      "is_live": false,
      "live_status_known": true,
      "active_stream_session_id": null,
      "last_transition": null,
      "title": null,
      "game": null,
      "category": null
    },
    "current_scene": {},
    "pending": {
      "clarification": null,
      "conversation_turn": null
    },
    "conversation": {
      "active": {},
      "latest": {
        "id": "conv_5add27b4469a5395b06047e7328dad18",
        "context_kind": "owner_local",
        "context_id": "leo_local",
        "attention_state": "RELEASED",
        "turn_owner": "leo",
        "expected_reply_type": "yes_no",
        "topic": "replay_handoff",
        "origin_event_id": "open",
        "last_event_id": "open",
        "opened_at": 1000.0,
        "last_turn_at": 1000.0,
        "expires_at": 1005.0,
        "status": "EXPIRED",
        "closure_reason": "ttl",
        "version": 2,
        "participants": [
          "leo",
          "hebe"
        ],
        "domain_payload": {},
        "consumed_event_ids": []
      },
      "all": [
        {
          "id": "conv_5add27b4469a5395b06047e7328dad18",
          "context_kind": "owner_local",
          "context_id": "leo_local",
          "attention_state": "RELEASED",
          "turn_owner": "leo",
          "expected_reply_type": "yes_no",
          "topic": "replay_handoff",
          "origin_event_id": "open",
          "last_event_id": "open",
          "opened_at": 1000.0,
          "last_turn_at": 1000.0,
          "expires_at": 1005.0,
          "status": "EXPIRED",
          "closure_reason": "ttl",
          "version": 2,
          "participants": [
            "leo",
            "hebe"
          ],
          "domain_payload": {},
          "consumed_event_ids": []
        }
      ],
      "last_resolution": {
        "consumed": false,
        "decision": "no_conversation",
        "reason": "no_compatible_active_conversation",
        "conversation_id": "",
        "reply_act": "UNKNOWN",
        "payload": {},
        "conversation": null,
        "latency_ms": 1.4462000108323991
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
        "p50_ms": 1.4462,
        "p95_ms": 1.4462
      }
    },
    "open_threads": [
      {
        "id": "thread_f263e49836685b3f9c7441004d215f75",
        "thread_type": "clarification",
        "scope_kind": "owner_local",
        "scope_id": "leo_local",
        "subject_ref": "conv_5add27b4469a5395b06047e7328dad18",
        "summary": "Unresolved replay_handoff clarification",
        "origin_event_id": "open",
        "latest_event_id": "ttl",
        "status": "EXPIRED",
        "priority": 50,
        "created_at": 1000.0,
        "relevance_until": 1005.0,
        "valid_until": 1005.0,
        "resolved_at": 1011.0,
        "resolution_event_id": "ttl",
        "sensitivity": "normal",
        "version": 2,
        "participant_ids": [
          "leo",
          "hebe"
        ]
      }
    ],
    "memory": {
      "facts_count": 0,
      "chunks_count": 0
    },
    "beliefs": [],
    "game_state": {
      "game": "",
      "platform_version": "",
      "playthrough_type": "casual",
      "spoiler_policy": "spoiler_safe_hints",
      "current_location": null,
      "current_character": "",
      "party_members": [],
      "last_confirmed_progress": "",
      "current_objective": null,
      "challenge": "",
      "known_constraints": [],
      "last_updated": 1786437822.703677,
      "provenance": "inferred",
      "confidence": 0.0,
      "current_game": null,
      "recent_run_context_facts": []
    },
    "social_state": {
      "recent_active_users": [],
      "recent_chat_count": 0,
      "last_raid": {},
      "last_cheer": {}
    },
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
    "database_watermarks": {
      "counts": {
        "chat_log": 0,
        "memory_facts": 0,
        "memory_chunks": 0,
        "stream_sessions": 0,
        "stream_chat_messages": 0,
        "stream_events": 0,
        "live_session_timeline": 1,
        "promotion_events": 0,
        "viewer_promotion_profiles": 0,
        "schema_migrations": 2,
        "conversations": 1,
        "open_threads": 1
      },
      "schema_migrations": [
        {
          "component": "cognitive_replay",
          "version": 1,
          "name": "replay_metadata",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "applied_at": "2026-08-11T08:43:42.699121+00:00"
        },
        {
          "component": "conversation_continuity",
          "version": 1,
          "name": "conversation_and_open_threads",
          "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
          "applied_at": "2026-08-11T08:43:42.765257+00:00"
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
      "source": "ambient_stt",
      "authority": "ambient",
      "decision": "ignore",
      "reason": "offline_stream"
    },
    "last_policy": {
      "source": null,
      "authority": null,
      "decision": null,
      "reason": null
    }
  },
  "stream_session": {
    "enabled": false,
    "is_live": false,
    "live_status_known": true,
    "active_stream_session_id": null,
    "last_transition": null,
    "title": null,
    "game": null,
    "category": null
  },
  "current_scene": {},
  "pending": {
    "clarification": null,
    "conversation_turn": null
  },
  "conversation": {
    "active": {},
    "latest": {
      "id": "conv_5add27b4469a5395b06047e7328dad18",
      "context_kind": "owner_local",
      "context_id": "leo_local",
      "attention_state": "RELEASED",
      "turn_owner": "leo",
      "expected_reply_type": "yes_no",
      "topic": "replay_handoff",
      "origin_event_id": "open",
      "last_event_id": "open",
      "opened_at": 1000.0,
      "last_turn_at": 1000.0,
      "expires_at": 1005.0,
      "status": "EXPIRED",
      "closure_reason": "ttl",
      "version": 2,
      "participants": [
        "leo",
        "hebe"
      ],
      "domain_payload": {},
      "consumed_event_ids": []
    },
    "all": [
      {
        "id": "conv_5add27b4469a5395b06047e7328dad18",
        "context_kind": "owner_local",
        "context_id": "leo_local",
        "attention_state": "RELEASED",
        "turn_owner": "leo",
        "expected_reply_type": "yes_no",
        "topic": "replay_handoff",
        "origin_event_id": "open",
        "last_event_id": "open",
        "opened_at": 1000.0,
        "last_turn_at": 1000.0,
        "expires_at": 1005.0,
        "status": "EXPIRED",
        "closure_reason": "ttl",
        "version": 2,
        "participants": [
          "leo",
          "hebe"
        ],
        "domain_payload": {},
        "consumed_event_ids": []
      }
    ],
    "last_resolution": {
      "consumed": false,
      "decision": "no_conversation",
      "reason": "no_compatible_active_conversation",
      "conversation_id": "",
      "reply_act": "UNKNOWN",
      "payload": {},
      "conversation": null,
      "latency_ms": 1.4462000108323991
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
      "p50_ms": 1.4462,
      "p95_ms": 1.4462
    }
  },
  "open_threads": [
    {
      "id": "thread_f263e49836685b3f9c7441004d215f75",
      "thread_type": "clarification",
      "scope_kind": "owner_local",
      "scope_id": "leo_local",
      "subject_ref": "conv_5add27b4469a5395b06047e7328dad18",
      "summary": "Unresolved replay_handoff clarification",
      "origin_event_id": "open",
      "latest_event_id": "ttl",
      "status": "EXPIRED",
      "priority": 50,
      "created_at": 1000.0,
      "relevance_until": 1005.0,
      "valid_until": 1005.0,
      "resolved_at": 1011.0,
      "resolution_event_id": "ttl",
      "sensitivity": "normal",
      "version": 2,
      "participant_ids": [
        "leo",
        "hebe"
      ]
    }
  ],
  "memory": {
    "facts_count": 0,
    "chunks_count": 0
  },
  "beliefs": [],
  "game_state": {
    "game": "",
    "platform_version": "",
    "playthrough_type": "casual",
    "spoiler_policy": "spoiler_safe_hints",
    "current_location": null,
    "current_character": "",
    "party_members": [],
    "last_confirmed_progress": "",
    "current_objective": null,
    "challenge": "",
    "known_constraints": [],
    "last_updated": 1786437822.703677,
    "provenance": "inferred",
    "confidence": 0.0,
    "current_game": null,
    "recent_run_context_facts": []
  },
  "social_state": {
    "recent_active_users": [],
    "recent_chat_count": 0,
    "last_raid": {},
    "last_cheer": {}
  },
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
  "database_watermarks": {
    "counts": {
      "chat_log": 0,
      "memory_facts": 0,
      "memory_chunks": 0,
      "stream_sessions": 0,
      "stream_chat_messages": 0,
      "stream_events": 0,
      "live_session_timeline": 1,
      "promotion_events": 0,
      "viewer_promotion_profiles": 0,
      "schema_migrations": 2,
      "conversations": 1,
      "open_threads": 1
    },
    "schema_migrations": [
      {
        "component": "cognitive_replay",
        "version": 1,
        "name": "replay_metadata",
        "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
        "applied_at": "2026-08-11T08:43:42.699121+00:00"
      },
      {
        "component": "conversation_continuity",
        "version": 1,
        "name": "conversation_and_open_threads",
        "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
        "applied_at": "2026-08-11T08:43:42.765257+00:00"
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

### phase1_d_interruption

- Status: **VERIFIED**
- Events: 3
- Restarts: 0
- Duration: 0.328851s
- Assertions passed/failed/skipped: 3/0/0


#### Checkpoint state

```json
{
  "open": {
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
      "enabled": false,
      "is_live": false,
      "live_status_known": false,
      "active_stream_session_id": null,
      "last_transition": null,
      "title": null,
      "game": null,
      "category": null
    },
    "current_scene": {},
    "pending": {
      "clarification": null,
      "conversation_turn": null
    },
    "conversation": {
      "active": {
        "id": "conv_c804b3917cef5c71bfb7ca2b5b899232",
        "context_kind": "owner_local",
        "context_id": "leo_local",
        "attention_state": "HANDED_OFF",
        "turn_owner": "leo",
        "expected_reply_type": "yes_no",
        "topic": "replay_handoff",
        "origin_event_id": "open",
        "last_event_id": "open",
        "opened_at": 1000.0,
        "last_turn_at": 1000.0,
        "expires_at": 1060.0,
        "status": "WAITING_ON_LEO",
        "closure_reason": "",
        "version": 1,
        "participants": [
          "leo",
          "hebe"
        ],
        "domain_payload": {},
        "consumed_event_ids": []
      },
      "latest": {
        "id": "conv_c804b3917cef5c71bfb7ca2b5b899232",
        "context_kind": "owner_local",
        "context_id": "leo_local",
        "attention_state": "HANDED_OFF",
        "turn_owner": "leo",
        "expected_reply_type": "yes_no",
        "topic": "replay_handoff",
        "origin_event_id": "open",
        "last_event_id": "open",
        "opened_at": 1000.0,
        "last_turn_at": 1000.0,
        "expires_at": 1060.0,
        "status": "WAITING_ON_LEO",
        "closure_reason": "",
        "version": 1,
        "participants": [
          "leo",
          "hebe"
        ],
        "domain_payload": {},
        "consumed_event_ids": []
      },
      "all": [
        {
          "id": "conv_c804b3917cef5c71bfb7ca2b5b899232",
          "context_kind": "owner_local",
          "context_id": "leo_local",
          "attention_state": "HANDED_OFF",
          "turn_owner": "leo",
          "expected_reply_type": "yes_no",
          "topic": "replay_handoff",
          "origin_event_id": "open",
          "last_event_id": "open",
          "opened_at": 1000.0,
          "last_turn_at": 1000.0,
          "expires_at": 1060.0,
          "status": "WAITING_ON_LEO",
          "closure_reason": "",
          "version": 1,
          "participants": [
            "leo",
            "hebe"
          ],
          "domain_payload": {},
          "consumed_event_ids": []
        }
      ],
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
    "open_threads": [
      {
        "id": "thread_6434d5c72b7e5401860d33fdb4e0b9bc",
        "thread_type": "clarification",
        "scope_kind": "owner_local",
        "scope_id": "leo_local",
        "subject_ref": "conv_c804b3917cef5c71bfb7ca2b5b899232",
        "summary": "Unresolved replay_handoff clarification",
        "origin_event_id": "open",
        "latest_event_id": "open",
        "status": "WAITING_ON_LEO",
        "priority": 50,
        "created_at": 1000.0,
        "relevance_until": 1060.0,
        "valid_until": 1060.0,
        "resolved_at": 0.0,
        "resolution_event_id": "",
        "sensitivity": "normal",
        "version": 1,
        "participant_ids": [
          "leo",
          "hebe"
        ]
      }
    ],
    "memory": {
      "facts_count": 0,
      "chunks_count": 0
    },
    "beliefs": [],
    "game_state": {
      "game": "",
      "platform_version": "",
      "playthrough_type": "casual",
      "spoiler_policy": "spoiler_safe_hints",
      "current_location": null,
      "current_character": "",
      "party_members": [],
      "last_confirmed_progress": "",
      "current_objective": null,
      "challenge": "",
      "known_constraints": [],
      "last_updated": 1786437823.0581493,
      "provenance": "inferred",
      "confidence": 0.0,
      "current_game": null,
      "recent_run_context_facts": []
    },
    "social_state": {
      "recent_active_users": [],
      "recent_chat_count": 0,
      "last_raid": {},
      "last_cheer": {}
    },
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
    "database_watermarks": {
      "counts": {
        "chat_log": 0,
        "memory_facts": 0,
        "memory_chunks": 0,
        "stream_sessions": 0,
        "stream_chat_messages": 0,
        "stream_events": 0,
        "live_session_timeline": 0,
        "promotion_events": 0,
        "viewer_promotion_profiles": 0,
        "schema_migrations": 2,
        "conversations": 1,
        "open_threads": 1
      },
      "schema_migrations": [
        {
          "component": "cognitive_replay",
          "version": 1,
          "name": "replay_metadata",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "applied_at": "2026-08-11T08:43:43.051464+00:00"
        },
        {
          "component": "conversation_continuity",
          "version": 1,
          "name": "conversation_and_open_threads",
          "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
          "applied_at": "2026-08-11T08:43:43.120091+00:00"
        }
      ],
      "final_response_digest": "",
      "final_response_present": false
    }
  },
  "new": {
    "runtime": {
      "mode": "active",
      "hebe_sleeping": false,
      "is_running": false,
      "last_input_source": null,
      "last_intent": null,
      "last_firewall": {
        "source": "owner_stt_direct",
        "authority": "owner",
        "decision": "allow",
        "reason": "owner_direct"
      },
      "last_policy": {
        "source": null,
        "authority": null,
        "decision": null,
        "reason": null
      }
    },
    "stream_session": {
      "enabled": false,
      "is_live": false,
      "live_status_known": false,
      "active_stream_session_id": null,
      "last_transition": null,
      "title": null,
      "game": null,
      "category": null
    },
    "current_scene": {},
    "pending": {
      "clarification": null,
      "conversation_turn": null
    },
    "conversation": {
      "active": {},
      "latest": {
        "id": "conv_c804b3917cef5c71bfb7ca2b5b899232",
        "context_kind": "owner_local",
        "context_id": "leo_local",
        "attention_state": "RELEASED",
        "turn_owner": "leo",
        "expected_reply_type": "yes_no",
        "topic": "replay_handoff",
        "origin_event_id": "open",
        "last_event_id": "new",
        "opened_at": 1000.0,
        "last_turn_at": 1001.0,
        "expires_at": 1060.0,
        "status": "INTERRUPTED",
        "closure_reason": "new_owner_command_interrupted",
        "version": 2,
        "participants": [
          "leo",
          "hebe"
        ],
        "domain_payload": {},
        "consumed_event_ids": []
      },
      "all": [
        {
          "id": "conv_c804b3917cef5c71bfb7ca2b5b899232",
          "context_kind": "owner_local",
          "context_id": "leo_local",
          "attention_state": "RELEASED",
          "turn_owner": "leo",
          "expected_reply_type": "yes_no",
          "topic": "replay_handoff",
          "origin_event_id": "open",
          "last_event_id": "new",
          "opened_at": 1000.0,
          "last_turn_at": 1001.0,
          "expires_at": 1060.0,
          "status": "INTERRUPTED",
          "closure_reason": "new_owner_command_interrupted",
          "version": 2,
          "participants": [
            "leo",
            "hebe"
          ],
          "domain_payload": {},
          "consumed_event_ids": []
        }
      ],
      "last_resolution": {
        "consumed": false,
        "decision": "interrupt",
        "reason": "new_owner_command_interrupted",
        "conversation_id": "conv_c804b3917cef5c71bfb7ca2b5b899232",
        "reply_act": "UNKNOWN",
        "payload": {},
        "conversation": {
          "id": "conv_c804b3917cef5c71bfb7ca2b5b899232",
          "context_kind": "owner_local",
          "context_id": "leo_local",
          "participants": [
            "leo",
            "hebe"
          ],
          "attention_state": "RELEASED",
          "turn_owner": "leo",
          "expected_reply": {
            "type": "yes_no",
            "allowed_sources": [
              "owner_stt"
            ],
            "allowed_participant": "leo",
            "semantic_constraints": {},
            "candidate_refs": [],
            "expires_at": 1060.0,
            "consume_policy": "once"
          },
          "topic": "replay_handoff",
          "origin_event_id": "open",
          "last_event_id": "new",
          "opened_at": 1000.0,
          "last_turn_at": 1001.0,
          "expires_at": 1060.0,
          "status": "INTERRUPTED",
          "closure_reason": "new_owner_command_interrupted",
          "version": 2,
          "domain_payload": {},
          "consumed_event_ids": []
        },
        "latency_ms": 8.667499991133809
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
        "p50_ms": 8.6675,
        "p95_ms": 8.6675
      }
    },
    "open_threads": [
      {
        "id": "thread_6434d5c72b7e5401860d33fdb4e0b9bc",
        "thread_type": "clarification",
        "scope_kind": "owner_local",
        "scope_id": "leo_local",
        "subject_ref": "conv_c804b3917cef5c71bfb7ca2b5b899232",
        "summary": "Unresolved replay_handoff clarification",
        "origin_event_id": "open",
        "latest_event_id": "new",
        "status": "ARCHIVED",
        "priority": 50,
        "created_at": 1000.0,
        "relevance_until": 1060.0,
        "valid_until": 1060.0,
        "resolved_at": 1001.0,
        "resolution_event_id": "new",
        "sensitivity": "normal",
        "version": 2,
        "participant_ids": [
          "leo",
          "hebe"
        ]
      }
    ],
    "memory": {
      "facts_count": 0,
      "chunks_count": 0
    },
    "beliefs": [],
    "game_state": {
      "game": "",
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
      "last_updated": 1786437823.0581493,
      "provenance": "current_live_session",
      "confidence": 0.0,
      "current_game": null,
      "recent_run_context_facts": []
    },
    "social_state": {
      "recent_active_users": [],
      "recent_chat_count": 0,
      "last_raid": {},
      "last_cheer": {}
    },
    "promotion_profiles": [],
    "actions": {
      "attempts": [
        {
          "operation": "desktop.open_app",
          "payload": {
            "app_id": "{'app_id': 'obs', 'display_name': 'OBS Studio', 'aliases': ['obs', 'obs studio'], 'executable_path': 'C:\\\\Program Files\\\\obs-studio\\\\bin\\\\64bit\\\\obs64.exe', 'working_directory': None, 'launch_args': [], 'enabled': True, 'requires_confirmation': False, 'created_at': '2026-08-11T08:43:43.181895+00:00', 'updated_at': '2026-08-11T08:43:43.181895+00:00', 'source': 'builtin', 'name': 'OBS Studio', 'command': 'C:\\\\Program Files\\\\obs-studio\\\\bin\\\\64bit\\\\obs64.exe', 'process_name': 'obs64.exe', 'window_title': 'OBS Studio'}"
          },
          "outcome": {
            "success": false,
            "status": "unconfigured",
            "reason": "external_outcome_missing"
          }
        }
      ],
      "speech_requests": [],
      "model_calls": [
        {
          "key": "stream_response:v1:none",
          "method": "chat"
        }
      ],
      "research_calls": []
    },
    "receipts": [],
    "emitted_outputs": [
      {
        "event_id": "1786437823.1373138",
        "emitted": true,
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "reason": "",
        "text_digest": "",
        "text_present": false
      }
    ],
    "final_emission_results": [
      {
        "event_id": "1786437823.1373138",
        "emitted": true,
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "reason": "",
        "text_digest": "",
        "text_present": false
      }
    ],
    "database_watermarks": {
      "counts": {
        "chat_log": 1,
        "memory_facts": 0,
        "memory_chunks": 0,
        "stream_sessions": 0,
        "stream_chat_messages": 0,
        "stream_events": 0,
        "live_session_timeline": 2,
        "promotion_events": 0,
        "viewer_promotion_profiles": 0,
        "schema_migrations": 2,
        "conversations": 1,
        "open_threads": 1
      },
      "schema_migrations": [
        {
          "component": "cognitive_replay",
          "version": 1,
          "name": "replay_metadata",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "applied_at": "2026-08-11T08:43:43.051464+00:00"
        },
        {
          "component": "conversation_continuity",
          "version": 1,
          "name": "conversation_and_open_threads",
          "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
          "applied_at": "2026-08-11T08:43:43.120091+00:00"
        }
      ],
      "final_response_digest": "7e07c6a5322789f6",
      "final_response_present": true
    }
  },
  "later": {
    "runtime": {
      "mode": "active",
      "hebe_sleeping": false,
      "is_running": false,
      "last_input_source": null,
      "last_intent": null,
      "last_firewall": {
        "source": "ambient_stt",
        "authority": "ambient",
        "decision": "ignore",
        "reason": "offline_stream"
      },
      "last_policy": {
        "source": null,
        "authority": null,
        "decision": null,
        "reason": null
      }
    },
    "stream_session": {
      "enabled": false,
      "is_live": false,
      "live_status_known": false,
      "active_stream_session_id": null,
      "last_transition": null,
      "title": null,
      "game": null,
      "category": null
    },
    "current_scene": {},
    "pending": {
      "clarification": null,
      "conversation_turn": null
    },
    "conversation": {
      "active": {},
      "latest": {
        "id": "conv_c804b3917cef5c71bfb7ca2b5b899232",
        "context_kind": "owner_local",
        "context_id": "leo_local",
        "attention_state": "RELEASED",
        "turn_owner": "leo",
        "expected_reply_type": "yes_no",
        "topic": "replay_handoff",
        "origin_event_id": "open",
        "last_event_id": "new",
        "opened_at": 1000.0,
        "last_turn_at": 1001.0,
        "expires_at": 1060.0,
        "status": "INTERRUPTED",
        "closure_reason": "new_owner_command_interrupted",
        "version": 2,
        "participants": [
          "leo",
          "hebe"
        ],
        "domain_payload": {},
        "consumed_event_ids": []
      },
      "all": [
        {
          "id": "conv_c804b3917cef5c71bfb7ca2b5b899232",
          "context_kind": "owner_local",
          "context_id": "leo_local",
          "attention_state": "RELEASED",
          "turn_owner": "leo",
          "expected_reply_type": "yes_no",
          "topic": "replay_handoff",
          "origin_event_id": "open",
          "last_event_id": "new",
          "opened_at": 1000.0,
          "last_turn_at": 1001.0,
          "expires_at": 1060.0,
          "status": "INTERRUPTED",
          "closure_reason": "new_owner_command_interrupted",
          "version": 2,
          "participants": [
            "leo",
            "hebe"
          ],
          "domain_payload": {},
          "consumed_event_ids": []
        }
      ],
      "last_resolution": {
        "consumed": false,
        "decision": "no_conversation",
        "reason": "no_compatible_active_conversation",
        "conversation_id": "",
        "reply_act": "UNKNOWN",
        "payload": {},
        "conversation": null,
        "latency_ms": 1.4452000032179058
      },
      "legacy_pending_projection": {},
      "continuity_shadow_diff": {
        "legacy_result": false,
        "v2_result": false,
        "match": true,
        "difference_reason": ""
      },
      "shadow_metrics": {
        "total": 2,
        "matches": 2,
        "differences": 0,
        "match_rate": 1.0,
        "difference_reasons": {}
      },
      "performance": {
        "count": 2,
        "p50_ms": 5.05635,
        "p95_ms": 8.6675
      }
    },
    "open_threads": [
      {
        "id": "thread_6434d5c72b7e5401860d33fdb4e0b9bc",
        "thread_type": "clarification",
        "scope_kind": "owner_local",
        "scope_id": "leo_local",
        "subject_ref": "conv_c804b3917cef5c71bfb7ca2b5b899232",
        "summary": "Unresolved replay_handoff clarification",
        "origin_event_id": "open",
        "latest_event_id": "new",
        "status": "ARCHIVED",
        "priority": 50,
        "created_at": 1000.0,
        "relevance_until": 1060.0,
        "valid_until": 1060.0,
        "resolved_at": 1001.0,
        "resolution_event_id": "new",
        "sensitivity": "normal",
        "version": 2,
        "participant_ids": [
          "leo",
          "hebe"
        ]
      }
    ],
    "memory": {
      "facts_count": 0,
      "chunks_count": 0
    },
    "beliefs": [],
    "game_state": {
      "game": "",
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
      "last_updated": 1786437823.0581493,
      "provenance": "current_live_session",
      "confidence": 0.0,
      "current_game": null,
      "recent_run_context_facts": []
    },
    "social_state": {
      "recent_active_users": [],
      "recent_chat_count": 0,
      "last_raid": {},
      "last_cheer": {}
    },
    "promotion_profiles": [],
    "actions": {
      "attempts": [
        {
          "operation": "desktop.open_app",
          "payload": {
            "app_id": "{'app_id': 'obs', 'display_name': 'OBS Studio', 'aliases': ['obs', 'obs studio'], 'executable_path': 'C:\\\\Program Files\\\\obs-studio\\\\bin\\\\64bit\\\\obs64.exe', 'working_directory': None, 'launch_args': [], 'enabled': True, 'requires_confirmation': False, 'created_at': '2026-08-11T08:43:43.181895+00:00', 'updated_at': '2026-08-11T08:43:43.181895+00:00', 'source': 'builtin', 'name': 'OBS Studio', 'command': 'C:\\\\Program Files\\\\obs-studio\\\\bin\\\\64bit\\\\obs64.exe', 'process_name': 'obs64.exe', 'window_title': 'OBS Studio'}"
          },
          "outcome": {
            "success": false,
            "status": "unconfigured",
            "reason": "external_outcome_missing"
          }
        }
      ],
      "speech_requests": [],
      "model_calls": [
        {
          "key": "stream_response:v1:none",
          "method": "chat"
        }
      ],
      "research_calls": []
    },
    "receipts": [],
    "emitted_outputs": [
      {
        "event_id": "1786437823.1373138",
        "emitted": true,
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "reason": "",
        "text_digest": "",
        "text_present": false
      }
    ],
    "final_emission_results": [
      {
        "event_id": "1786437823.1373138",
        "emitted": true,
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "reason": "",
        "text_digest": "",
        "text_present": false
      }
    ],
    "database_watermarks": {
      "counts": {
        "chat_log": 1,
        "memory_facts": 0,
        "memory_chunks": 0,
        "stream_sessions": 0,
        "stream_chat_messages": 0,
        "stream_events": 0,
        "live_session_timeline": 2,
        "promotion_events": 0,
        "viewer_promotion_profiles": 0,
        "schema_migrations": 2,
        "conversations": 1,
        "open_threads": 1
      },
      "schema_migrations": [
        {
          "component": "cognitive_replay",
          "version": 1,
          "name": "replay_metadata",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "applied_at": "2026-08-11T08:43:43.051464+00:00"
        },
        {
          "component": "conversation_continuity",
          "version": 1,
          "name": "conversation_and_open_threads",
          "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
          "applied_at": "2026-08-11T08:43:43.120091+00:00"
        }
      ],
      "final_response_digest": "7e07c6a5322789f6",
      "final_response_present": true
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
      "source": "ambient_stt",
      "authority": "ambient",
      "decision": "ignore",
      "reason": "offline_stream"
    },
    "last_policy": {
      "source": null,
      "authority": null,
      "decision": null,
      "reason": null
    }
  },
  "stream_session": {
    "enabled": false,
    "is_live": false,
    "live_status_known": false,
    "active_stream_session_id": null,
    "last_transition": null,
    "title": null,
    "game": null,
    "category": null
  },
  "current_scene": {},
  "pending": {
    "clarification": null,
    "conversation_turn": null
  },
  "conversation": {
    "active": {},
    "latest": {
      "id": "conv_c804b3917cef5c71bfb7ca2b5b899232",
      "context_kind": "owner_local",
      "context_id": "leo_local",
      "attention_state": "RELEASED",
      "turn_owner": "leo",
      "expected_reply_type": "yes_no",
      "topic": "replay_handoff",
      "origin_event_id": "open",
      "last_event_id": "new",
      "opened_at": 1000.0,
      "last_turn_at": 1001.0,
      "expires_at": 1060.0,
      "status": "INTERRUPTED",
      "closure_reason": "new_owner_command_interrupted",
      "version": 2,
      "participants": [
        "leo",
        "hebe"
      ],
      "domain_payload": {},
      "consumed_event_ids": []
    },
    "all": [
      {
        "id": "conv_c804b3917cef5c71bfb7ca2b5b899232",
        "context_kind": "owner_local",
        "context_id": "leo_local",
        "attention_state": "RELEASED",
        "turn_owner": "leo",
        "expected_reply_type": "yes_no",
        "topic": "replay_handoff",
        "origin_event_id": "open",
        "last_event_id": "new",
        "opened_at": 1000.0,
        "last_turn_at": 1001.0,
        "expires_at": 1060.0,
        "status": "INTERRUPTED",
        "closure_reason": "new_owner_command_interrupted",
        "version": 2,
        "participants": [
          "leo",
          "hebe"
        ],
        "domain_payload": {},
        "consumed_event_ids": []
      }
    ],
    "last_resolution": {
      "consumed": false,
      "decision": "no_conversation",
      "reason": "no_compatible_active_conversation",
      "conversation_id": "",
      "reply_act": "UNKNOWN",
      "payload": {},
      "conversation": null,
      "latency_ms": 1.4452000032179058
    },
    "legacy_pending_projection": {},
    "continuity_shadow_diff": {
      "legacy_result": false,
      "v2_result": false,
      "match": true,
      "difference_reason": ""
    },
    "shadow_metrics": {
      "total": 2,
      "matches": 2,
      "differences": 0,
      "match_rate": 1.0,
      "difference_reasons": {}
    },
    "performance": {
      "count": 2,
      "p50_ms": 5.05635,
      "p95_ms": 8.6675
    }
  },
  "open_threads": [
    {
      "id": "thread_6434d5c72b7e5401860d33fdb4e0b9bc",
      "thread_type": "clarification",
      "scope_kind": "owner_local",
      "scope_id": "leo_local",
      "subject_ref": "conv_c804b3917cef5c71bfb7ca2b5b899232",
      "summary": "Unresolved replay_handoff clarification",
      "origin_event_id": "open",
      "latest_event_id": "new",
      "status": "ARCHIVED",
      "priority": 50,
      "created_at": 1000.0,
      "relevance_until": 1060.0,
      "valid_until": 1060.0,
      "resolved_at": 1001.0,
      "resolution_event_id": "new",
      "sensitivity": "normal",
      "version": 2,
      "participant_ids": [
        "leo",
        "hebe"
      ]
    }
  ],
  "memory": {
    "facts_count": 0,
    "chunks_count": 0
  },
  "beliefs": [],
  "game_state": {
    "game": "",
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
    "last_updated": 1786437823.0581493,
    "provenance": "current_live_session",
    "confidence": 0.0,
    "current_game": null,
    "recent_run_context_facts": []
  },
  "social_state": {
    "recent_active_users": [],
    "recent_chat_count": 0,
    "last_raid": {},
    "last_cheer": {}
  },
  "promotion_profiles": [],
  "actions": {
    "attempts": [
      {
        "operation": "desktop.open_app",
        "payload": {
          "app_id": "{'app_id': 'obs', 'display_name': 'OBS Studio', 'aliases': ['obs', 'obs studio'], 'executable_path': 'C:\\\\Program Files\\\\obs-studio\\\\bin\\\\64bit\\\\obs64.exe', 'working_directory': None, 'launch_args': [], 'enabled': True, 'requires_confirmation': False, 'created_at': '2026-08-11T08:43:43.181895+00:00', 'updated_at': '2026-08-11T08:43:43.181895+00:00', 'source': 'builtin', 'name': 'OBS Studio', 'command': 'C:\\\\Program Files\\\\obs-studio\\\\bin\\\\64bit\\\\obs64.exe', 'process_name': 'obs64.exe', 'window_title': 'OBS Studio'}"
        },
        "outcome": {
          "success": false,
          "status": "unconfigured",
          "reason": "external_outcome_missing"
        }
      }
    ],
    "speech_requests": [],
    "model_calls": [
      {
        "key": "stream_response:v1:none",
        "method": "chat"
      }
    ],
    "research_calls": []
  },
  "receipts": [],
  "emitted_outputs": [
    {
      "event_id": "1786437823.1373138",
      "emitted": true,
      "route": "local_owner_reply",
      "targets": [
        "local_ui"
      ],
      "reason": "",
      "text_digest": "",
      "text_present": false
    }
  ],
  "final_emission_results": [
    {
      "event_id": "1786437823.1373138",
      "emitted": true,
      "route": "local_owner_reply",
      "targets": [
        "local_ui"
      ],
      "reason": "",
      "text_digest": "",
      "text_present": false
    }
  ],
  "database_watermarks": {
    "counts": {
      "chat_log": 1,
      "memory_facts": 0,
      "memory_chunks": 0,
      "stream_sessions": 0,
      "stream_chat_messages": 0,
      "stream_events": 0,
      "live_session_timeline": 2,
      "promotion_events": 0,
      "viewer_promotion_profiles": 0,
      "schema_migrations": 2,
      "conversations": 1,
      "open_threads": 1
    },
    "schema_migrations": [
      {
        "component": "cognitive_replay",
        "version": 1,
        "name": "replay_metadata",
        "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
        "applied_at": "2026-08-11T08:43:43.051464+00:00"
      },
      {
        "component": "conversation_continuity",
        "version": 1,
        "name": "conversation_and_open_threads",
        "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
        "applied_at": "2026-08-11T08:43:43.120091+00:00"
      }
    ],
    "final_response_digest": "7e07c6a5322789f6",
    "final_response_present": true
  }
}
```

#### Restart evidence

```json
[]
```

### phase1_e_entity_selection

- Status: **VERIFIED**
- Events: 2
- Restarts: 0
- Duration: 0.29551s
- Assertions passed/failed/skipped: 3/0/0


#### Checkpoint state

```json
{
  "open": {
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
      "enabled": false,
      "is_live": false,
      "live_status_known": false,
      "active_stream_session_id": null,
      "last_transition": null,
      "title": null,
      "game": null,
      "category": null
    },
    "current_scene": {},
    "pending": {
      "clarification": null,
      "conversation_turn": null
    },
    "conversation": {
      "active": {
        "id": "conv_7330e36271425811816a8a2818f4d648",
        "context_kind": "owner_local",
        "context_id": "leo_local",
        "attention_state": "HANDED_OFF",
        "turn_owner": "leo",
        "expected_reply_type": "entity_selection",
        "topic": "replay_handoff",
        "origin_event_id": "open",
        "last_event_id": "open",
        "opened_at": 1000.0,
        "last_turn_at": 1000.0,
        "expires_at": 1060.0,
        "status": "WAITING_ON_LEO",
        "closure_reason": "",
        "version": 1,
        "participants": [
          "leo",
          "hebe"
        ],
        "domain_payload": {},
        "consumed_event_ids": []
      },
      "latest": {
        "id": "conv_7330e36271425811816a8a2818f4d648",
        "context_kind": "owner_local",
        "context_id": "leo_local",
        "attention_state": "HANDED_OFF",
        "turn_owner": "leo",
        "expected_reply_type": "entity_selection",
        "topic": "replay_handoff",
        "origin_event_id": "open",
        "last_event_id": "open",
        "opened_at": 1000.0,
        "last_turn_at": 1000.0,
        "expires_at": 1060.0,
        "status": "WAITING_ON_LEO",
        "closure_reason": "",
        "version": 1,
        "participants": [
          "leo",
          "hebe"
        ],
        "domain_payload": {},
        "consumed_event_ids": []
      },
      "all": [
        {
          "id": "conv_7330e36271425811816a8a2818f4d648",
          "context_kind": "owner_local",
          "context_id": "leo_local",
          "attention_state": "HANDED_OFF",
          "turn_owner": "leo",
          "expected_reply_type": "entity_selection",
          "topic": "replay_handoff",
          "origin_event_id": "open",
          "last_event_id": "open",
          "opened_at": 1000.0,
          "last_turn_at": 1000.0,
          "expires_at": 1060.0,
          "status": "WAITING_ON_LEO",
          "closure_reason": "",
          "version": 1,
          "participants": [
            "leo",
            "hebe"
          ],
          "domain_payload": {},
          "consumed_event_ids": []
        }
      ],
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
    "open_threads": [
      {
        "id": "thread_63a77c6345785adeaec6b8c928cd7cb7",
        "thread_type": "clarification",
        "scope_kind": "owner_local",
        "scope_id": "leo_local",
        "subject_ref": "conv_7330e36271425811816a8a2818f4d648",
        "summary": "Unresolved replay_handoff clarification",
        "origin_event_id": "open",
        "latest_event_id": "open",
        "status": "WAITING_ON_LEO",
        "priority": 50,
        "created_at": 1000.0,
        "relevance_until": 1060.0,
        "valid_until": 1060.0,
        "resolved_at": 0.0,
        "resolution_event_id": "",
        "sensitivity": "normal",
        "version": 1,
        "participant_ids": [
          "leo",
          "hebe"
        ]
      }
    ],
    "memory": {
      "facts_count": 0,
      "chunks_count": 0
    },
    "beliefs": [],
    "game_state": {
      "game": "",
      "platform_version": "",
      "playthrough_type": "casual",
      "spoiler_policy": "spoiler_safe_hints",
      "current_location": null,
      "current_character": "",
      "party_members": [],
      "last_confirmed_progress": "",
      "current_objective": null,
      "challenge": "",
      "known_constraints": [],
      "last_updated": 1786437823.482709,
      "provenance": "inferred",
      "confidence": 0.0,
      "current_game": null,
      "recent_run_context_facts": []
    },
    "social_state": {
      "recent_active_users": [],
      "recent_chat_count": 0,
      "last_raid": {},
      "last_cheer": {}
    },
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
    "database_watermarks": {
      "counts": {
        "chat_log": 0,
        "memory_facts": 0,
        "memory_chunks": 0,
        "stream_sessions": 0,
        "stream_chat_messages": 0,
        "stream_events": 0,
        "live_session_timeline": 0,
        "promotion_events": 0,
        "viewer_promotion_profiles": 0,
        "schema_migrations": 2,
        "conversations": 1,
        "open_threads": 1
      },
      "schema_migrations": [
        {
          "component": "cognitive_replay",
          "version": 1,
          "name": "replay_metadata",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "applied_at": "2026-08-11T08:43:43.476168+00:00"
        },
        {
          "component": "conversation_continuity",
          "version": 1,
          "name": "conversation_and_open_threads",
          "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
          "applied_at": "2026-08-11T08:43:43.553023+00:00"
        }
      ],
      "final_response_digest": "",
      "final_response_present": false
    }
  },
  "select": {
    "runtime": {
      "mode": "active",
      "hebe_sleeping": false,
      "is_running": false,
      "last_input_source": null,
      "last_intent": null,
      "last_firewall": {
        "source": "ambient_stt",
        "authority": "ambient",
        "decision": "ignore",
        "reason": "offline_stream"
      },
      "last_policy": {
        "source": null,
        "authority": null,
        "decision": null,
        "reason": null
      }
    },
    "stream_session": {
      "enabled": false,
      "is_live": false,
      "live_status_known": false,
      "active_stream_session_id": null,
      "last_transition": null,
      "title": null,
      "game": null,
      "category": null
    },
    "current_scene": {},
    "pending": {
      "clarification": null,
      "conversation_turn": null
    },
    "conversation": {
      "active": {},
      "latest": {
        "id": "conv_7330e36271425811816a8a2818f4d648",
        "context_kind": "owner_local",
        "context_id": "leo_local",
        "attention_state": "RELEASED",
        "turn_owner": "leo",
        "expected_reply_type": "entity_selection",
        "topic": "replay_handoff",
        "origin_event_id": "open",
        "last_event_id": "select",
        "opened_at": 1000.0,
        "last_turn_at": 1001.0,
        "expires_at": 1060.0,
        "status": "CLOSED",
        "closure_reason": "reply_consumed",
        "version": 2,
        "participants": [
          "leo",
          "hebe"
        ],
        "domain_payload": {},
        "consumed_event_ids": [
          "select"
        ]
      },
      "all": [
        {
          "id": "conv_7330e36271425811816a8a2818f4d648",
          "context_kind": "owner_local",
          "context_id": "leo_local",
          "attention_state": "RELEASED",
          "turn_owner": "leo",
          "expected_reply_type": "entity_selection",
          "topic": "replay_handoff",
          "origin_event_id": "open",
          "last_event_id": "select",
          "opened_at": 1000.0,
          "last_turn_at": 1001.0,
          "expires_at": 1060.0,
          "status": "CLOSED",
          "closure_reason": "reply_consumed",
          "version": 2,
          "participants": [
            "leo",
            "hebe"
          ],
          "domain_payload": {},
          "consumed_event_ids": [
            "select"
          ]
        }
      ],
      "last_resolution": {
        "consumed": true,
        "decision": "compatible_reply",
        "reason": "ordinal_selection",
        "conversation_id": "conv_7330e36271425811816a8a2818f4d648",
        "reply_act": "SELECT",
        "payload": {
          "index": 1,
          "candidate": "ivanxi",
          "domain": {},
          "expected_reply_type": "entity_selection"
        },
        "conversation": {
          "id": "conv_7330e36271425811816a8a2818f4d648",
          "context_kind": "owner_local",
          "context_id": "leo_local",
          "participants": [
            "leo",
            "hebe"
          ],
          "attention_state": "RELEASED",
          "turn_owner": "leo",
          "expected_reply": {
            "type": "entity_selection",
            "allowed_sources": [
              "owner_stt"
            ],
            "allowed_participant": "leo",
            "semantic_constraints": {},
            "candidate_refs": [
              "ivan",
              "ivanxi"
            ],
            "expires_at": 1060.0,
            "consume_policy": "once"
          },
          "topic": "replay_handoff",
          "origin_event_id": "open",
          "last_event_id": "select",
          "opened_at": 1000.0,
          "last_turn_at": 1001.0,
          "expires_at": 1060.0,
          "status": "CLOSED",
          "closure_reason": "reply_consumed",
          "version": 2,
          "domain_payload": {},
          "consumed_event_ids": [
            "select"
          ]
        },
        "latency_ms": 9.844999993219972
      },
      "legacy_pending_projection": {},
      "continuity_shadow_diff": {
        "legacy_result": false,
        "v2_result": true,
        "match": false,
        "difference_reason": "ordinal_selection"
      },
      "shadow_metrics": {
        "total": 1,
        "matches": 0,
        "differences": 1,
        "match_rate": 0.0,
        "difference_reasons": {
          "ordinal_selection": 1
        }
      },
      "performance": {
        "count": 1,
        "p50_ms": 9.845,
        "p95_ms": 9.845
      }
    },
    "open_threads": [
      {
        "id": "thread_63a77c6345785adeaec6b8c928cd7cb7",
        "thread_type": "clarification",
        "scope_kind": "owner_local",
        "scope_id": "leo_local",
        "subject_ref": "conv_7330e36271425811816a8a2818f4d648",
        "summary": "Unresolved replay_handoff clarification",
        "origin_event_id": "open",
        "latest_event_id": "select",
        "status": "RESOLVED",
        "priority": 50,
        "created_at": 1000.0,
        "relevance_until": 1060.0,
        "valid_until": 1060.0,
        "resolved_at": 1001.0,
        "resolution_event_id": "select",
        "sensitivity": "normal",
        "version": 2,
        "participant_ids": [
          "leo",
          "hebe"
        ]
      }
    ],
    "memory": {
      "facts_count": 0,
      "chunks_count": 0
    },
    "beliefs": [],
    "game_state": {
      "game": "",
      "platform_version": "",
      "playthrough_type": "casual",
      "spoiler_policy": "spoiler_safe_hints",
      "current_location": null,
      "current_character": "",
      "party_members": [],
      "last_confirmed_progress": "",
      "current_objective": null,
      "challenge": "",
      "known_constraints": [],
      "last_updated": 1786437823.482709,
      "provenance": "inferred",
      "confidence": 0.0,
      "current_game": null,
      "recent_run_context_facts": []
    },
    "social_state": {
      "recent_active_users": [],
      "recent_chat_count": 0,
      "last_raid": {},
      "last_cheer": {}
    },
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
    "database_watermarks": {
      "counts": {
        "chat_log": 0,
        "memory_facts": 0,
        "memory_chunks": 0,
        "stream_sessions": 0,
        "stream_chat_messages": 0,
        "stream_events": 0,
        "live_session_timeline": 0,
        "promotion_events": 0,
        "viewer_promotion_profiles": 0,
        "schema_migrations": 2,
        "conversations": 1,
        "open_threads": 1
      },
      "schema_migrations": [
        {
          "component": "cognitive_replay",
          "version": 1,
          "name": "replay_metadata",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "applied_at": "2026-08-11T08:43:43.476168+00:00"
        },
        {
          "component": "conversation_continuity",
          "version": 1,
          "name": "conversation_and_open_threads",
          "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
          "applied_at": "2026-08-11T08:43:43.553023+00:00"
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
      "source": "ambient_stt",
      "authority": "ambient",
      "decision": "ignore",
      "reason": "offline_stream"
    },
    "last_policy": {
      "source": null,
      "authority": null,
      "decision": null,
      "reason": null
    }
  },
  "stream_session": {
    "enabled": false,
    "is_live": false,
    "live_status_known": false,
    "active_stream_session_id": null,
    "last_transition": null,
    "title": null,
    "game": null,
    "category": null
  },
  "current_scene": {},
  "pending": {
    "clarification": null,
    "conversation_turn": null
  },
  "conversation": {
    "active": {},
    "latest": {
      "id": "conv_7330e36271425811816a8a2818f4d648",
      "context_kind": "owner_local",
      "context_id": "leo_local",
      "attention_state": "RELEASED",
      "turn_owner": "leo",
      "expected_reply_type": "entity_selection",
      "topic": "replay_handoff",
      "origin_event_id": "open",
      "last_event_id": "select",
      "opened_at": 1000.0,
      "last_turn_at": 1001.0,
      "expires_at": 1060.0,
      "status": "CLOSED",
      "closure_reason": "reply_consumed",
      "version": 2,
      "participants": [
        "leo",
        "hebe"
      ],
      "domain_payload": {},
      "consumed_event_ids": [
        "select"
      ]
    },
    "all": [
      {
        "id": "conv_7330e36271425811816a8a2818f4d648",
        "context_kind": "owner_local",
        "context_id": "leo_local",
        "attention_state": "RELEASED",
        "turn_owner": "leo",
        "expected_reply_type": "entity_selection",
        "topic": "replay_handoff",
        "origin_event_id": "open",
        "last_event_id": "select",
        "opened_at": 1000.0,
        "last_turn_at": 1001.0,
        "expires_at": 1060.0,
        "status": "CLOSED",
        "closure_reason": "reply_consumed",
        "version": 2,
        "participants": [
          "leo",
          "hebe"
        ],
        "domain_payload": {},
        "consumed_event_ids": [
          "select"
        ]
      }
    ],
    "last_resolution": {
      "consumed": true,
      "decision": "compatible_reply",
      "reason": "ordinal_selection",
      "conversation_id": "conv_7330e36271425811816a8a2818f4d648",
      "reply_act": "SELECT",
      "payload": {
        "index": 1,
        "candidate": "ivanxi",
        "domain": {},
        "expected_reply_type": "entity_selection"
      },
      "conversation": {
        "id": "conv_7330e36271425811816a8a2818f4d648",
        "context_kind": "owner_local",
        "context_id": "leo_local",
        "participants": [
          "leo",
          "hebe"
        ],
        "attention_state": "RELEASED",
        "turn_owner": "leo",
        "expected_reply": {
          "type": "entity_selection",
          "allowed_sources": [
            "owner_stt"
          ],
          "allowed_participant": "leo",
          "semantic_constraints": {},
          "candidate_refs": [
            "ivan",
            "ivanxi"
          ],
          "expires_at": 1060.0,
          "consume_policy": "once"
        },
        "topic": "replay_handoff",
        "origin_event_id": "open",
        "last_event_id": "select",
        "opened_at": 1000.0,
        "last_turn_at": 1001.0,
        "expires_at": 1060.0,
        "status": "CLOSED",
        "closure_reason": "reply_consumed",
        "version": 2,
        "domain_payload": {},
        "consumed_event_ids": [
          "select"
        ]
      },
      "latency_ms": 9.844999993219972
    },
    "legacy_pending_projection": {},
    "continuity_shadow_diff": {
      "legacy_result": false,
      "v2_result": true,
      "match": false,
      "difference_reason": "ordinal_selection"
    },
    "shadow_metrics": {
      "total": 1,
      "matches": 0,
      "differences": 1,
      "match_rate": 0.0,
      "difference_reasons": {
        "ordinal_selection": 1
      }
    },
    "performance": {
      "count": 1,
      "p50_ms": 9.845,
      "p95_ms": 9.845
    }
  },
  "open_threads": [
    {
      "id": "thread_63a77c6345785adeaec6b8c928cd7cb7",
      "thread_type": "clarification",
      "scope_kind": "owner_local",
      "scope_id": "leo_local",
      "subject_ref": "conv_7330e36271425811816a8a2818f4d648",
      "summary": "Unresolved replay_handoff clarification",
      "origin_event_id": "open",
      "latest_event_id": "select",
      "status": "RESOLVED",
      "priority": 50,
      "created_at": 1000.0,
      "relevance_until": 1060.0,
      "valid_until": 1060.0,
      "resolved_at": 1001.0,
      "resolution_event_id": "select",
      "sensitivity": "normal",
      "version": 2,
      "participant_ids": [
        "leo",
        "hebe"
      ]
    }
  ],
  "memory": {
    "facts_count": 0,
    "chunks_count": 0
  },
  "beliefs": [],
  "game_state": {
    "game": "",
    "platform_version": "",
    "playthrough_type": "casual",
    "spoiler_policy": "spoiler_safe_hints",
    "current_location": null,
    "current_character": "",
    "party_members": [],
    "last_confirmed_progress": "",
    "current_objective": null,
    "challenge": "",
    "known_constraints": [],
    "last_updated": 1786437823.482709,
    "provenance": "inferred",
    "confidence": 0.0,
    "current_game": null,
    "recent_run_context_facts": []
  },
  "social_state": {
    "recent_active_users": [],
    "recent_chat_count": 0,
    "last_raid": {},
    "last_cheer": {}
  },
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
  "database_watermarks": {
    "counts": {
      "chat_log": 0,
      "memory_facts": 0,
      "memory_chunks": 0,
      "stream_sessions": 0,
      "stream_chat_messages": 0,
      "stream_events": 0,
      "live_session_timeline": 0,
      "promotion_events": 0,
      "viewer_promotion_profiles": 0,
      "schema_migrations": 2,
      "conversations": 1,
      "open_threads": 1
    },
    "schema_migrations": [
      {
        "component": "cognitive_replay",
        "version": 1,
        "name": "replay_metadata",
        "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
        "applied_at": "2026-08-11T08:43:43.476168+00:00"
      },
      {
        "component": "conversation_continuity",
        "version": 1,
        "name": "conversation_and_open_threads",
        "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
        "applied_at": "2026-08-11T08:43:43.553023+00:00"
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

### phase1_f_wrong_context

- Status: **VERIFIED**
- Events: 2
- Restarts: 0
- Duration: 0.29627s
- Assertions passed/failed/skipped: 2/0/0


#### Checkpoint state

```json
{
  "open": {
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
      "enabled": false,
      "is_live": false,
      "live_status_known": false,
      "active_stream_session_id": null,
      "last_transition": null,
      "title": null,
      "game": null,
      "category": null
    },
    "current_scene": {},
    "pending": {
      "clarification": null,
      "conversation_turn": null
    },
    "conversation": {
      "active": {
        "id": "conv_64ac3812aebc567db08bb74f75c6b875",
        "context_kind": "private_ui",
        "context_id": "leo_ui",
        "attention_state": "HANDED_OFF",
        "turn_owner": "leo",
        "expected_reply_type": "yes_no",
        "topic": "replay_handoff",
        "origin_event_id": "open",
        "last_event_id": "open",
        "opened_at": 1000.0,
        "last_turn_at": 1000.0,
        "expires_at": 1060.0,
        "status": "WAITING_ON_LEO",
        "closure_reason": "",
        "version": 1,
        "participants": [
          "leo",
          "hebe"
        ],
        "domain_payload": {},
        "consumed_event_ids": []
      },
      "latest": {
        "id": "conv_64ac3812aebc567db08bb74f75c6b875",
        "context_kind": "private_ui",
        "context_id": "leo_ui",
        "attention_state": "HANDED_OFF",
        "turn_owner": "leo",
        "expected_reply_type": "yes_no",
        "topic": "replay_handoff",
        "origin_event_id": "open",
        "last_event_id": "open",
        "opened_at": 1000.0,
        "last_turn_at": 1000.0,
        "expires_at": 1060.0,
        "status": "WAITING_ON_LEO",
        "closure_reason": "",
        "version": 1,
        "participants": [
          "leo",
          "hebe"
        ],
        "domain_payload": {},
        "consumed_event_ids": []
      },
      "all": [
        {
          "id": "conv_64ac3812aebc567db08bb74f75c6b875",
          "context_kind": "private_ui",
          "context_id": "leo_ui",
          "attention_state": "HANDED_OFF",
          "turn_owner": "leo",
          "expected_reply_type": "yes_no",
          "topic": "replay_handoff",
          "origin_event_id": "open",
          "last_event_id": "open",
          "opened_at": 1000.0,
          "last_turn_at": 1000.0,
          "expires_at": 1060.0,
          "status": "WAITING_ON_LEO",
          "closure_reason": "",
          "version": 1,
          "participants": [
            "leo",
            "hebe"
          ],
          "domain_payload": {},
          "consumed_event_ids": []
        }
      ],
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
    "open_threads": [
      {
        "id": "thread_11ca60b6832055f29ee4a8840af5e120",
        "thread_type": "clarification",
        "scope_kind": "private_ui",
        "scope_id": "leo_ui",
        "subject_ref": "conv_64ac3812aebc567db08bb74f75c6b875",
        "summary": "Unresolved replay_handoff clarification",
        "origin_event_id": "open",
        "latest_event_id": "open",
        "status": "WAITING_ON_LEO",
        "priority": 50,
        "created_at": 1000.0,
        "relevance_until": 1060.0,
        "valid_until": 1060.0,
        "resolved_at": 0.0,
        "resolution_event_id": "",
        "sensitivity": "normal",
        "version": 1,
        "participant_ids": [
          "leo",
          "hebe"
        ]
      }
    ],
    "memory": {
      "facts_count": 0,
      "chunks_count": 0
    },
    "beliefs": [],
    "game_state": {
      "game": "",
      "platform_version": "",
      "playthrough_type": "casual",
      "spoiler_policy": "spoiler_safe_hints",
      "current_location": null,
      "current_character": "",
      "party_members": [],
      "last_confirmed_progress": "",
      "current_objective": null,
      "challenge": "",
      "known_constraints": [],
      "last_updated": 1786437823.8522012,
      "provenance": "inferred",
      "confidence": 0.0,
      "current_game": null,
      "recent_run_context_facts": []
    },
    "social_state": {
      "recent_active_users": [],
      "recent_chat_count": 0,
      "last_raid": {},
      "last_cheer": {}
    },
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
    "database_watermarks": {
      "counts": {
        "chat_log": 0,
        "memory_facts": 0,
        "memory_chunks": 0,
        "stream_sessions": 0,
        "stream_chat_messages": 0,
        "stream_events": 0,
        "live_session_timeline": 0,
        "promotion_events": 0,
        "viewer_promotion_profiles": 0,
        "schema_migrations": 2,
        "conversations": 1,
        "open_threads": 1
      },
      "schema_migrations": [
        {
          "component": "cognitive_replay",
          "version": 1,
          "name": "replay_metadata",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "applied_at": "2026-08-11T08:43:43.844013+00:00"
        },
        {
          "component": "conversation_continuity",
          "version": 1,
          "name": "conversation_and_open_threads",
          "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
          "applied_at": "2026-08-11T08:43:43.922920+00:00"
        }
      ],
      "final_response_digest": "",
      "final_response_present": false
    }
  },
  "wrong": {
    "runtime": {
      "mode": "active",
      "hebe_sleeping": false,
      "is_running": false,
      "last_input_source": null,
      "last_intent": null,
      "last_firewall": {
        "source": "ambient_stt",
        "authority": "ambient",
        "decision": "ignore",
        "reason": "offline_stream"
      },
      "last_policy": {
        "source": null,
        "authority": null,
        "decision": null,
        "reason": null
      }
    },
    "stream_session": {
      "enabled": false,
      "is_live": false,
      "live_status_known": false,
      "active_stream_session_id": null,
      "last_transition": null,
      "title": null,
      "game": null,
      "category": null
    },
    "current_scene": {},
    "pending": {
      "clarification": null,
      "conversation_turn": null
    },
    "conversation": {
      "active": {
        "id": "conv_64ac3812aebc567db08bb74f75c6b875",
        "context_kind": "private_ui",
        "context_id": "leo_ui",
        "attention_state": "HANDED_OFF",
        "turn_owner": "leo",
        "expected_reply_type": "yes_no",
        "topic": "replay_handoff",
        "origin_event_id": "open",
        "last_event_id": "open",
        "opened_at": 1000.0,
        "last_turn_at": 1000.0,
        "expires_at": 1060.0,
        "status": "WAITING_ON_LEO",
        "closure_reason": "",
        "version": 1,
        "participants": [
          "leo",
          "hebe"
        ],
        "domain_payload": {},
        "consumed_event_ids": []
      },
      "latest": {
        "id": "conv_64ac3812aebc567db08bb74f75c6b875",
        "context_kind": "private_ui",
        "context_id": "leo_ui",
        "attention_state": "HANDED_OFF",
        "turn_owner": "leo",
        "expected_reply_type": "yes_no",
        "topic": "replay_handoff",
        "origin_event_id": "open",
        "last_event_id": "open",
        "opened_at": 1000.0,
        "last_turn_at": 1000.0,
        "expires_at": 1060.0,
        "status": "WAITING_ON_LEO",
        "closure_reason": "",
        "version": 1,
        "participants": [
          "leo",
          "hebe"
        ],
        "domain_payload": {},
        "consumed_event_ids": []
      },
      "all": [
        {
          "id": "conv_64ac3812aebc567db08bb74f75c6b875",
          "context_kind": "private_ui",
          "context_id": "leo_ui",
          "attention_state": "HANDED_OFF",
          "turn_owner": "leo",
          "expected_reply_type": "yes_no",
          "topic": "replay_handoff",
          "origin_event_id": "open",
          "last_event_id": "open",
          "opened_at": 1000.0,
          "last_turn_at": 1000.0,
          "expires_at": 1060.0,
          "status": "WAITING_ON_LEO",
          "closure_reason": "",
          "version": 1,
          "participants": [
            "leo",
            "hebe"
          ],
          "domain_payload": {},
          "consumed_event_ids": []
        }
      ],
      "last_resolution": {
        "consumed": false,
        "decision": "no_conversation",
        "reason": "no_compatible_active_conversation",
        "conversation_id": "",
        "reply_act": "UNKNOWN",
        "payload": {},
        "conversation": null,
        "latency_ms": 1.3158000074326992
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
        "p50_ms": 1.3158,
        "p95_ms": 1.3158
      }
    },
    "open_threads": [
      {
        "id": "thread_11ca60b6832055f29ee4a8840af5e120",
        "thread_type": "clarification",
        "scope_kind": "private_ui",
        "scope_id": "leo_ui",
        "subject_ref": "conv_64ac3812aebc567db08bb74f75c6b875",
        "summary": "Unresolved replay_handoff clarification",
        "origin_event_id": "open",
        "latest_event_id": "open",
        "status": "WAITING_ON_LEO",
        "priority": 50,
        "created_at": 1000.0,
        "relevance_until": 1060.0,
        "valid_until": 1060.0,
        "resolved_at": 0.0,
        "resolution_event_id": "",
        "sensitivity": "normal",
        "version": 1,
        "participant_ids": [
          "leo",
          "hebe"
        ]
      }
    ],
    "memory": {
      "facts_count": 0,
      "chunks_count": 0
    },
    "beliefs": [],
    "game_state": {
      "game": "",
      "platform_version": "",
      "playthrough_type": "casual",
      "spoiler_policy": "spoiler_safe_hints",
      "current_location": null,
      "current_character": "",
      "party_members": [],
      "last_confirmed_progress": "",
      "current_objective": null,
      "challenge": "",
      "known_constraints": [],
      "last_updated": 1786437823.8522012,
      "provenance": "inferred",
      "confidence": 0.0,
      "current_game": null,
      "recent_run_context_facts": []
    },
    "social_state": {
      "recent_active_users": [],
      "recent_chat_count": 0,
      "last_raid": {},
      "last_cheer": {}
    },
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
    "database_watermarks": {
      "counts": {
        "chat_log": 0,
        "memory_facts": 0,
        "memory_chunks": 0,
        "stream_sessions": 0,
        "stream_chat_messages": 0,
        "stream_events": 0,
        "live_session_timeline": 0,
        "promotion_events": 0,
        "viewer_promotion_profiles": 0,
        "schema_migrations": 2,
        "conversations": 1,
        "open_threads": 1
      },
      "schema_migrations": [
        {
          "component": "cognitive_replay",
          "version": 1,
          "name": "replay_metadata",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "applied_at": "2026-08-11T08:43:43.844013+00:00"
        },
        {
          "component": "conversation_continuity",
          "version": 1,
          "name": "conversation_and_open_threads",
          "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
          "applied_at": "2026-08-11T08:43:43.922920+00:00"
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
      "source": "ambient_stt",
      "authority": "ambient",
      "decision": "ignore",
      "reason": "offline_stream"
    },
    "last_policy": {
      "source": null,
      "authority": null,
      "decision": null,
      "reason": null
    }
  },
  "stream_session": {
    "enabled": false,
    "is_live": false,
    "live_status_known": false,
    "active_stream_session_id": null,
    "last_transition": null,
    "title": null,
    "game": null,
    "category": null
  },
  "current_scene": {},
  "pending": {
    "clarification": null,
    "conversation_turn": null
  },
  "conversation": {
    "active": {
      "id": "conv_64ac3812aebc567db08bb74f75c6b875",
      "context_kind": "private_ui",
      "context_id": "leo_ui",
      "attention_state": "HANDED_OFF",
      "turn_owner": "leo",
      "expected_reply_type": "yes_no",
      "topic": "replay_handoff",
      "origin_event_id": "open",
      "last_event_id": "open",
      "opened_at": 1000.0,
      "last_turn_at": 1000.0,
      "expires_at": 1060.0,
      "status": "WAITING_ON_LEO",
      "closure_reason": "",
      "version": 1,
      "participants": [
        "leo",
        "hebe"
      ],
      "domain_payload": {},
      "consumed_event_ids": []
    },
    "latest": {
      "id": "conv_64ac3812aebc567db08bb74f75c6b875",
      "context_kind": "private_ui",
      "context_id": "leo_ui",
      "attention_state": "HANDED_OFF",
      "turn_owner": "leo",
      "expected_reply_type": "yes_no",
      "topic": "replay_handoff",
      "origin_event_id": "open",
      "last_event_id": "open",
      "opened_at": 1000.0,
      "last_turn_at": 1000.0,
      "expires_at": 1060.0,
      "status": "WAITING_ON_LEO",
      "closure_reason": "",
      "version": 1,
      "participants": [
        "leo",
        "hebe"
      ],
      "domain_payload": {},
      "consumed_event_ids": []
    },
    "all": [
      {
        "id": "conv_64ac3812aebc567db08bb74f75c6b875",
        "context_kind": "private_ui",
        "context_id": "leo_ui",
        "attention_state": "HANDED_OFF",
        "turn_owner": "leo",
        "expected_reply_type": "yes_no",
        "topic": "replay_handoff",
        "origin_event_id": "open",
        "last_event_id": "open",
        "opened_at": 1000.0,
        "last_turn_at": 1000.0,
        "expires_at": 1060.0,
        "status": "WAITING_ON_LEO",
        "closure_reason": "",
        "version": 1,
        "participants": [
          "leo",
          "hebe"
        ],
        "domain_payload": {},
        "consumed_event_ids": []
      }
    ],
    "last_resolution": {
      "consumed": false,
      "decision": "no_conversation",
      "reason": "no_compatible_active_conversation",
      "conversation_id": "",
      "reply_act": "UNKNOWN",
      "payload": {},
      "conversation": null,
      "latency_ms": 1.3158000074326992
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
      "p50_ms": 1.3158,
      "p95_ms": 1.3158
    }
  },
  "open_threads": [
    {
      "id": "thread_11ca60b6832055f29ee4a8840af5e120",
      "thread_type": "clarification",
      "scope_kind": "private_ui",
      "scope_id": "leo_ui",
      "subject_ref": "conv_64ac3812aebc567db08bb74f75c6b875",
      "summary": "Unresolved replay_handoff clarification",
      "origin_event_id": "open",
      "latest_event_id": "open",
      "status": "WAITING_ON_LEO",
      "priority": 50,
      "created_at": 1000.0,
      "relevance_until": 1060.0,
      "valid_until": 1060.0,
      "resolved_at": 0.0,
      "resolution_event_id": "",
      "sensitivity": "normal",
      "version": 1,
      "participant_ids": [
        "leo",
        "hebe"
      ]
    }
  ],
  "memory": {
    "facts_count": 0,
    "chunks_count": 0
  },
  "beliefs": [],
  "game_state": {
    "game": "",
    "platform_version": "",
    "playthrough_type": "casual",
    "spoiler_policy": "spoiler_safe_hints",
    "current_location": null,
    "current_character": "",
    "party_members": [],
    "last_confirmed_progress": "",
    "current_objective": null,
    "challenge": "",
    "known_constraints": [],
    "last_updated": 1786437823.8522012,
    "provenance": "inferred",
    "confidence": 0.0,
    "current_game": null,
    "recent_run_context_facts": []
  },
  "social_state": {
    "recent_active_users": [],
    "recent_chat_count": 0,
    "last_raid": {},
    "last_cheer": {}
  },
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
  "database_watermarks": {
    "counts": {
      "chat_log": 0,
      "memory_facts": 0,
      "memory_chunks": 0,
      "stream_sessions": 0,
      "stream_chat_messages": 0,
      "stream_events": 0,
      "live_session_timeline": 0,
      "promotion_events": 0,
      "viewer_promotion_profiles": 0,
      "schema_migrations": 2,
      "conversations": 1,
      "open_threads": 1
    },
    "schema_migrations": [
      {
        "component": "cognitive_replay",
        "version": 1,
        "name": "replay_metadata",
        "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
        "applied_at": "2026-08-11T08:43:43.844013+00:00"
      },
      {
        "component": "conversation_continuity",
        "version": 1,
        "name": "conversation_and_open_threads",
        "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
        "applied_at": "2026-08-11T08:43:43.922920+00:00"
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

### phase1_g_duplicate

- Status: **VERIFIED**
- Events: 3
- Restarts: 0
- Duration: 0.314026s
- Assertions passed/failed/skipped: 3/0/0


#### Checkpoint state

```json
{
  "open": {
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
      "enabled": false,
      "is_live": false,
      "live_status_known": false,
      "active_stream_session_id": null,
      "last_transition": null,
      "title": null,
      "game": null,
      "category": null
    },
    "current_scene": {},
    "pending": {
      "clarification": null,
      "conversation_turn": null
    },
    "conversation": {
      "active": {
        "id": "conv_09ed48e1319c5176b7337a1cd5bae634",
        "context_kind": "owner_local",
        "context_id": "leo_local",
        "attention_state": "HANDED_OFF",
        "turn_owner": "leo",
        "expected_reply_type": "yes_no",
        "topic": "replay_handoff",
        "origin_event_id": "open",
        "last_event_id": "open",
        "opened_at": 1000.0,
        "last_turn_at": 1000.0,
        "expires_at": 1060.0,
        "status": "WAITING_ON_LEO",
        "closure_reason": "",
        "version": 1,
        "participants": [
          "leo",
          "hebe"
        ],
        "domain_payload": {},
        "consumed_event_ids": []
      },
      "latest": {
        "id": "conv_09ed48e1319c5176b7337a1cd5bae634",
        "context_kind": "owner_local",
        "context_id": "leo_local",
        "attention_state": "HANDED_OFF",
        "turn_owner": "leo",
        "expected_reply_type": "yes_no",
        "topic": "replay_handoff",
        "origin_event_id": "open",
        "last_event_id": "open",
        "opened_at": 1000.0,
        "last_turn_at": 1000.0,
        "expires_at": 1060.0,
        "status": "WAITING_ON_LEO",
        "closure_reason": "",
        "version": 1,
        "participants": [
          "leo",
          "hebe"
        ],
        "domain_payload": {},
        "consumed_event_ids": []
      },
      "all": [
        {
          "id": "conv_09ed48e1319c5176b7337a1cd5bae634",
          "context_kind": "owner_local",
          "context_id": "leo_local",
          "attention_state": "HANDED_OFF",
          "turn_owner": "leo",
          "expected_reply_type": "yes_no",
          "topic": "replay_handoff",
          "origin_event_id": "open",
          "last_event_id": "open",
          "opened_at": 1000.0,
          "last_turn_at": 1000.0,
          "expires_at": 1060.0,
          "status": "WAITING_ON_LEO",
          "closure_reason": "",
          "version": 1,
          "participants": [
            "leo",
            "hebe"
          ],
          "domain_payload": {},
          "consumed_event_ids": []
        }
      ],
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
    "open_threads": [
      {
        "id": "thread_657ac5abbfc45c27a8bb7b3d804ed26d",
        "thread_type": "clarification",
        "scope_kind": "owner_local",
        "scope_id": "leo_local",
        "subject_ref": "conv_09ed48e1319c5176b7337a1cd5bae634",
        "summary": "Unresolved replay_handoff clarification",
        "origin_event_id": "open",
        "latest_event_id": "open",
        "status": "WAITING_ON_LEO",
        "priority": 50,
        "created_at": 1000.0,
        "relevance_until": 1060.0,
        "valid_until": 1060.0,
        "resolved_at": 0.0,
        "resolution_event_id": "",
        "sensitivity": "normal",
        "version": 1,
        "participant_ids": [
          "leo",
          "hebe"
        ]
      }
    ],
    "memory": {
      "facts_count": 0,
      "chunks_count": 0
    },
    "beliefs": [],
    "game_state": {
      "game": "",
      "platform_version": "",
      "playthrough_type": "casual",
      "spoiler_policy": "spoiler_safe_hints",
      "current_location": null,
      "current_character": "",
      "party_members": [],
      "last_confirmed_progress": "",
      "current_objective": null,
      "challenge": "",
      "known_constraints": [],
      "last_updated": 1786437824.222492,
      "provenance": "inferred",
      "confidence": 0.0,
      "current_game": null,
      "recent_run_context_facts": []
    },
    "social_state": {
      "recent_active_users": [],
      "recent_chat_count": 0,
      "last_raid": {},
      "last_cheer": {}
    },
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
    "database_watermarks": {
      "counts": {
        "chat_log": 0,
        "memory_facts": 0,
        "memory_chunks": 0,
        "stream_sessions": 0,
        "stream_chat_messages": 0,
        "stream_events": 0,
        "live_session_timeline": 0,
        "promotion_events": 0,
        "viewer_promotion_profiles": 0,
        "schema_migrations": 2,
        "conversations": 1,
        "open_threads": 1
      },
      "schema_migrations": [
        {
          "component": "cognitive_replay",
          "version": 1,
          "name": "replay_metadata",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "applied_at": "2026-08-11T08:43:44.212972+00:00"
        },
        {
          "component": "conversation_continuity",
          "version": 1,
          "name": "conversation_and_open_threads",
          "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
          "applied_at": "2026-08-11T08:43:44.296748+00:00"
        }
      ],
      "final_response_digest": "",
      "final_response_present": false
    }
  },
  "first": {
    "runtime": {
      "mode": "active",
      "hebe_sleeping": false,
      "is_running": false,
      "last_input_source": null,
      "last_intent": null,
      "last_firewall": {
        "source": "ambient_stt",
        "authority": "ambient",
        "decision": "ignore",
        "reason": "offline_stream"
      },
      "last_policy": {
        "source": null,
        "authority": null,
        "decision": null,
        "reason": null
      }
    },
    "stream_session": {
      "enabled": false,
      "is_live": false,
      "live_status_known": false,
      "active_stream_session_id": null,
      "last_transition": null,
      "title": null,
      "game": null,
      "category": null
    },
    "current_scene": {},
    "pending": {
      "clarification": null,
      "conversation_turn": null
    },
    "conversation": {
      "active": {},
      "latest": {
        "id": "conv_09ed48e1319c5176b7337a1cd5bae634",
        "context_kind": "owner_local",
        "context_id": "leo_local",
        "attention_state": "RELEASED",
        "turn_owner": "leo",
        "expected_reply_type": "yes_no",
        "topic": "replay_handoff",
        "origin_event_id": "open",
        "last_event_id": "same-reply",
        "opened_at": 1000.0,
        "last_turn_at": 1001.0,
        "expires_at": 1060.0,
        "status": "CLOSED",
        "closure_reason": "reply_consumed",
        "version": 2,
        "participants": [
          "leo",
          "hebe"
        ],
        "domain_payload": {},
        "consumed_event_ids": [
          "same-reply"
        ]
      },
      "all": [
        {
          "id": "conv_09ed48e1319c5176b7337a1cd5bae634",
          "context_kind": "owner_local",
          "context_id": "leo_local",
          "attention_state": "RELEASED",
          "turn_owner": "leo",
          "expected_reply_type": "yes_no",
          "topic": "replay_handoff",
          "origin_event_id": "open",
          "last_event_id": "same-reply",
          "opened_at": 1000.0,
          "last_turn_at": 1001.0,
          "expires_at": 1060.0,
          "status": "CLOSED",
          "closure_reason": "reply_consumed",
          "version": 2,
          "participants": [
            "leo",
            "hebe"
          ],
          "domain_payload": {},
          "consumed_event_ids": [
            "same-reply"
          ]
        }
      ],
      "last_resolution": {
        "consumed": true,
        "decision": "compatible_reply",
        "reason": "deterministic_affirm",
        "conversation_id": "conv_09ed48e1319c5176b7337a1cd5bae634",
        "reply_act": "AFFIRM",
        "payload": {
          "value": true,
          "domain": {},
          "expected_reply_type": "yes_no"
        },
        "conversation": {
          "id": "conv_09ed48e1319c5176b7337a1cd5bae634",
          "context_kind": "owner_local",
          "context_id": "leo_local",
          "participants": [
            "leo",
            "hebe"
          ],
          "attention_state": "RELEASED",
          "turn_owner": "leo",
          "expected_reply": {
            "type": "yes_no",
            "allowed_sources": [
              "owner_stt"
            ],
            "allowed_participant": "leo",
            "semantic_constraints": {},
            "candidate_refs": [],
            "expires_at": 1060.0,
            "consume_policy": "once"
          },
          "topic": "replay_handoff",
          "origin_event_id": "open",
          "last_event_id": "same-reply",
          "opened_at": 1000.0,
          "last_turn_at": 1001.0,
          "expires_at": 1060.0,
          "status": "CLOSED",
          "closure_reason": "reply_consumed",
          "version": 2,
          "domain_payload": {},
          "consumed_event_ids": [
            "same-reply"
          ]
        },
        "latency_ms": 10.251899948343635
      },
      "legacy_pending_projection": {},
      "continuity_shadow_diff": {
        "legacy_result": false,
        "v2_result": true,
        "match": false,
        "difference_reason": "deterministic_affirm"
      },
      "shadow_metrics": {
        "total": 1,
        "matches": 0,
        "differences": 1,
        "match_rate": 0.0,
        "difference_reasons": {
          "deterministic_affirm": 1
        }
      },
      "performance": {
        "count": 1,
        "p50_ms": 10.2519,
        "p95_ms": 10.2519
      }
    },
    "open_threads": [
      {
        "id": "thread_657ac5abbfc45c27a8bb7b3d804ed26d",
        "thread_type": "clarification",
        "scope_kind": "owner_local",
        "scope_id": "leo_local",
        "subject_ref": "conv_09ed48e1319c5176b7337a1cd5bae634",
        "summary": "Unresolved replay_handoff clarification",
        "origin_event_id": "open",
        "latest_event_id": "same-reply",
        "status": "RESOLVED",
        "priority": 50,
        "created_at": 1000.0,
        "relevance_until": 1060.0,
        "valid_until": 1060.0,
        "resolved_at": 1001.0,
        "resolution_event_id": "same-reply",
        "sensitivity": "normal",
        "version": 2,
        "participant_ids": [
          "leo",
          "hebe"
        ]
      }
    ],
    "memory": {
      "facts_count": 0,
      "chunks_count": 0
    },
    "beliefs": [],
    "game_state": {
      "game": "",
      "platform_version": "",
      "playthrough_type": "casual",
      "spoiler_policy": "spoiler_safe_hints",
      "current_location": null,
      "current_character": "",
      "party_members": [],
      "last_confirmed_progress": "",
      "current_objective": null,
      "challenge": "",
      "known_constraints": [],
      "last_updated": 1786437824.222492,
      "provenance": "inferred",
      "confidence": 0.0,
      "current_game": null,
      "recent_run_context_facts": []
    },
    "social_state": {
      "recent_active_users": [],
      "recent_chat_count": 0,
      "last_raid": {},
      "last_cheer": {}
    },
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
    "database_watermarks": {
      "counts": {
        "chat_log": 0,
        "memory_facts": 0,
        "memory_chunks": 0,
        "stream_sessions": 0,
        "stream_chat_messages": 0,
        "stream_events": 0,
        "live_session_timeline": 0,
        "promotion_events": 0,
        "viewer_promotion_profiles": 0,
        "schema_migrations": 2,
        "conversations": 1,
        "open_threads": 1
      },
      "schema_migrations": [
        {
          "component": "cognitive_replay",
          "version": 1,
          "name": "replay_metadata",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "applied_at": "2026-08-11T08:43:44.212972+00:00"
        },
        {
          "component": "conversation_continuity",
          "version": 1,
          "name": "conversation_and_open_threads",
          "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
          "applied_at": "2026-08-11T08:43:44.296748+00:00"
        }
      ],
      "final_response_digest": "",
      "final_response_present": false
    }
  },
  "duplicate": {
    "runtime": {
      "mode": "active",
      "hebe_sleeping": false,
      "is_running": false,
      "last_input_source": null,
      "last_intent": null,
      "last_firewall": {
        "source": "ambient_stt",
        "authority": "ambient",
        "decision": "ignore",
        "reason": "offline_stream"
      },
      "last_policy": {
        "source": null,
        "authority": null,
        "decision": null,
        "reason": null
      }
    },
    "stream_session": {
      "enabled": false,
      "is_live": false,
      "live_status_known": false,
      "active_stream_session_id": null,
      "last_transition": null,
      "title": null,
      "game": null,
      "category": null
    },
    "current_scene": {},
    "pending": {
      "clarification": null,
      "conversation_turn": null
    },
    "conversation": {
      "active": {},
      "latest": {
        "id": "conv_09ed48e1319c5176b7337a1cd5bae634",
        "context_kind": "owner_local",
        "context_id": "leo_local",
        "attention_state": "RELEASED",
        "turn_owner": "leo",
        "expected_reply_type": "yes_no",
        "topic": "replay_handoff",
        "origin_event_id": "open",
        "last_event_id": "same-reply",
        "opened_at": 1000.0,
        "last_turn_at": 1001.0,
        "expires_at": 1060.0,
        "status": "CLOSED",
        "closure_reason": "reply_consumed",
        "version": 2,
        "participants": [
          "leo",
          "hebe"
        ],
        "domain_payload": {},
        "consumed_event_ids": [
          "same-reply"
        ]
      },
      "all": [
        {
          "id": "conv_09ed48e1319c5176b7337a1cd5bae634",
          "context_kind": "owner_local",
          "context_id": "leo_local",
          "attention_state": "RELEASED",
          "turn_owner": "leo",
          "expected_reply_type": "yes_no",
          "topic": "replay_handoff",
          "origin_event_id": "open",
          "last_event_id": "same-reply",
          "opened_at": 1000.0,
          "last_turn_at": 1001.0,
          "expires_at": 1060.0,
          "status": "CLOSED",
          "closure_reason": "reply_consumed",
          "version": 2,
          "participants": [
            "leo",
            "hebe"
          ],
          "domain_payload": {},
          "consumed_event_ids": [
            "same-reply"
          ]
        }
      ],
      "last_resolution": {
        "consumed": true,
        "decision": "compatible_reply",
        "reason": "deterministic_affirm",
        "conversation_id": "conv_09ed48e1319c5176b7337a1cd5bae634",
        "reply_act": "AFFIRM",
        "payload": {
          "value": true,
          "domain": {},
          "expected_reply_type": "yes_no"
        },
        "conversation": {
          "id": "conv_09ed48e1319c5176b7337a1cd5bae634",
          "context_kind": "owner_local",
          "context_id": "leo_local",
          "participants": [
            "leo",
            "hebe"
          ],
          "attention_state": "RELEASED",
          "turn_owner": "leo",
          "expected_reply": {
            "type": "yes_no",
            "allowed_sources": [
              "owner_stt"
            ],
            "allowed_participant": "leo",
            "semantic_constraints": {},
            "candidate_refs": [],
            "expires_at": 1060.0,
            "consume_policy": "once"
          },
          "topic": "replay_handoff",
          "origin_event_id": "open",
          "last_event_id": "same-reply",
          "opened_at": 1000.0,
          "last_turn_at": 1001.0,
          "expires_at": 1060.0,
          "status": "CLOSED",
          "closure_reason": "reply_consumed",
          "version": 2,
          "domain_payload": {},
          "consumed_event_ids": [
            "same-reply"
          ]
        },
        "latency_ms": 10.251899948343635
      },
      "legacy_pending_projection": {},
      "continuity_shadow_diff": {
        "legacy_result": false,
        "v2_result": true,
        "match": false,
        "difference_reason": "deterministic_affirm"
      },
      "shadow_metrics": {
        "total": 1,
        "matches": 0,
        "differences": 1,
        "match_rate": 0.0,
        "difference_reasons": {
          "deterministic_affirm": 1
        }
      },
      "performance": {
        "count": 1,
        "p50_ms": 10.2519,
        "p95_ms": 10.2519
      }
    },
    "open_threads": [
      {
        "id": "thread_657ac5abbfc45c27a8bb7b3d804ed26d",
        "thread_type": "clarification",
        "scope_kind": "owner_local",
        "scope_id": "leo_local",
        "subject_ref": "conv_09ed48e1319c5176b7337a1cd5bae634",
        "summary": "Unresolved replay_handoff clarification",
        "origin_event_id": "open",
        "latest_event_id": "same-reply",
        "status": "RESOLVED",
        "priority": 50,
        "created_at": 1000.0,
        "relevance_until": 1060.0,
        "valid_until": 1060.0,
        "resolved_at": 1001.0,
        "resolution_event_id": "same-reply",
        "sensitivity": "normal",
        "version": 2,
        "participant_ids": [
          "leo",
          "hebe"
        ]
      }
    ],
    "memory": {
      "facts_count": 0,
      "chunks_count": 0
    },
    "beliefs": [],
    "game_state": {
      "game": "",
      "platform_version": "",
      "playthrough_type": "casual",
      "spoiler_policy": "spoiler_safe_hints",
      "current_location": null,
      "current_character": "",
      "party_members": [],
      "last_confirmed_progress": "",
      "current_objective": null,
      "challenge": "",
      "known_constraints": [],
      "last_updated": 1786437824.222492,
      "provenance": "inferred",
      "confidence": 0.0,
      "current_game": null,
      "recent_run_context_facts": []
    },
    "social_state": {
      "recent_active_users": [],
      "recent_chat_count": 0,
      "last_raid": {},
      "last_cheer": {}
    },
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
    "database_watermarks": {
      "counts": {
        "chat_log": 0,
        "memory_facts": 0,
        "memory_chunks": 0,
        "stream_sessions": 0,
        "stream_chat_messages": 0,
        "stream_events": 0,
        "live_session_timeline": 0,
        "promotion_events": 0,
        "viewer_promotion_profiles": 0,
        "schema_migrations": 2,
        "conversations": 1,
        "open_threads": 1
      },
      "schema_migrations": [
        {
          "component": "cognitive_replay",
          "version": 1,
          "name": "replay_metadata",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "applied_at": "2026-08-11T08:43:44.212972+00:00"
        },
        {
          "component": "conversation_continuity",
          "version": 1,
          "name": "conversation_and_open_threads",
          "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
          "applied_at": "2026-08-11T08:43:44.296748+00:00"
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
      "source": "ambient_stt",
      "authority": "ambient",
      "decision": "ignore",
      "reason": "offline_stream"
    },
    "last_policy": {
      "source": null,
      "authority": null,
      "decision": null,
      "reason": null
    }
  },
  "stream_session": {
    "enabled": false,
    "is_live": false,
    "live_status_known": false,
    "active_stream_session_id": null,
    "last_transition": null,
    "title": null,
    "game": null,
    "category": null
  },
  "current_scene": {},
  "pending": {
    "clarification": null,
    "conversation_turn": null
  },
  "conversation": {
    "active": {},
    "latest": {
      "id": "conv_09ed48e1319c5176b7337a1cd5bae634",
      "context_kind": "owner_local",
      "context_id": "leo_local",
      "attention_state": "RELEASED",
      "turn_owner": "leo",
      "expected_reply_type": "yes_no",
      "topic": "replay_handoff",
      "origin_event_id": "open",
      "last_event_id": "same-reply",
      "opened_at": 1000.0,
      "last_turn_at": 1001.0,
      "expires_at": 1060.0,
      "status": "CLOSED",
      "closure_reason": "reply_consumed",
      "version": 2,
      "participants": [
        "leo",
        "hebe"
      ],
      "domain_payload": {},
      "consumed_event_ids": [
        "same-reply"
      ]
    },
    "all": [
      {
        "id": "conv_09ed48e1319c5176b7337a1cd5bae634",
        "context_kind": "owner_local",
        "context_id": "leo_local",
        "attention_state": "RELEASED",
        "turn_owner": "leo",
        "expected_reply_type": "yes_no",
        "topic": "replay_handoff",
        "origin_event_id": "open",
        "last_event_id": "same-reply",
        "opened_at": 1000.0,
        "last_turn_at": 1001.0,
        "expires_at": 1060.0,
        "status": "CLOSED",
        "closure_reason": "reply_consumed",
        "version": 2,
        "participants": [
          "leo",
          "hebe"
        ],
        "domain_payload": {},
        "consumed_event_ids": [
          "same-reply"
        ]
      }
    ],
    "last_resolution": {
      "consumed": true,
      "decision": "compatible_reply",
      "reason": "deterministic_affirm",
      "conversation_id": "conv_09ed48e1319c5176b7337a1cd5bae634",
      "reply_act": "AFFIRM",
      "payload": {
        "value": true,
        "domain": {},
        "expected_reply_type": "yes_no"
      },
      "conversation": {
        "id": "conv_09ed48e1319c5176b7337a1cd5bae634",
        "context_kind": "owner_local",
        "context_id": "leo_local",
        "participants": [
          "leo",
          "hebe"
        ],
        "attention_state": "RELEASED",
        "turn_owner": "leo",
        "expected_reply": {
          "type": "yes_no",
          "allowed_sources": [
            "owner_stt"
          ],
          "allowed_participant": "leo",
          "semantic_constraints": {},
          "candidate_refs": [],
          "expires_at": 1060.0,
          "consume_policy": "once"
        },
        "topic": "replay_handoff",
        "origin_event_id": "open",
        "last_event_id": "same-reply",
        "opened_at": 1000.0,
        "last_turn_at": 1001.0,
        "expires_at": 1060.0,
        "status": "CLOSED",
        "closure_reason": "reply_consumed",
        "version": 2,
        "domain_payload": {},
        "consumed_event_ids": [
          "same-reply"
        ]
      },
      "latency_ms": 10.251899948343635
    },
    "legacy_pending_projection": {},
    "continuity_shadow_diff": {
      "legacy_result": false,
      "v2_result": true,
      "match": false,
      "difference_reason": "deterministic_affirm"
    },
    "shadow_metrics": {
      "total": 1,
      "matches": 0,
      "differences": 1,
      "match_rate": 0.0,
      "difference_reasons": {
        "deterministic_affirm": 1
      }
    },
    "performance": {
      "count": 1,
      "p50_ms": 10.2519,
      "p95_ms": 10.2519
    }
  },
  "open_threads": [
    {
      "id": "thread_657ac5abbfc45c27a8bb7b3d804ed26d",
      "thread_type": "clarification",
      "scope_kind": "owner_local",
      "scope_id": "leo_local",
      "subject_ref": "conv_09ed48e1319c5176b7337a1cd5bae634",
      "summary": "Unresolved replay_handoff clarification",
      "origin_event_id": "open",
      "latest_event_id": "same-reply",
      "status": "RESOLVED",
      "priority": 50,
      "created_at": 1000.0,
      "relevance_until": 1060.0,
      "valid_until": 1060.0,
      "resolved_at": 1001.0,
      "resolution_event_id": "same-reply",
      "sensitivity": "normal",
      "version": 2,
      "participant_ids": [
        "leo",
        "hebe"
      ]
    }
  ],
  "memory": {
    "facts_count": 0,
    "chunks_count": 0
  },
  "beliefs": [],
  "game_state": {
    "game": "",
    "platform_version": "",
    "playthrough_type": "casual",
    "spoiler_policy": "spoiler_safe_hints",
    "current_location": null,
    "current_character": "",
    "party_members": [],
    "last_confirmed_progress": "",
    "current_objective": null,
    "challenge": "",
    "known_constraints": [],
    "last_updated": 1786437824.222492,
    "provenance": "inferred",
    "confidence": 0.0,
    "current_game": null,
    "recent_run_context_facts": []
  },
  "social_state": {
    "recent_active_users": [],
    "recent_chat_count": 0,
    "last_raid": {},
    "last_cheer": {}
  },
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
  "database_watermarks": {
    "counts": {
      "chat_log": 0,
      "memory_facts": 0,
      "memory_chunks": 0,
      "stream_sessions": 0,
      "stream_chat_messages": 0,
      "stream_events": 0,
      "live_session_timeline": 0,
      "promotion_events": 0,
      "viewer_promotion_profiles": 0,
      "schema_migrations": 2,
      "conversations": 1,
      "open_threads": 1
    },
    "schema_migrations": [
      {
        "component": "cognitive_replay",
        "version": 1,
        "name": "replay_metadata",
        "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
        "applied_at": "2026-08-11T08:43:44.212972+00:00"
      },
      {
        "component": "conversation_continuity",
        "version": 1,
        "name": "conversation_and_open_threads",
        "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
        "applied_at": "2026-08-11T08:43:44.296748+00:00"
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

### phase1_h_restart_stale

- Status: **VERIFIED**
- Events: 3
- Restarts: 1
- Duration: 0.420593s
- Assertions passed/failed/skipped: 3/0/0


#### Checkpoint state

```json
{
  "open": {
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
      "enabled": false,
      "is_live": false,
      "live_status_known": false,
      "active_stream_session_id": null,
      "last_transition": null,
      "title": null,
      "game": null,
      "category": null
    },
    "current_scene": {},
    "pending": {
      "clarification": null,
      "conversation_turn": null
    },
    "conversation": {
      "active": {
        "id": "conv_d35db56c3bdf59faa1125861e60a4875",
        "context_kind": "owner_local",
        "context_id": "leo_local",
        "attention_state": "HANDED_OFF",
        "turn_owner": "leo",
        "expected_reply_type": "yes_no",
        "topic": "replay_handoff",
        "origin_event_id": "open",
        "last_event_id": "open",
        "opened_at": 1000.0,
        "last_turn_at": 1000.0,
        "expires_at": 1060.0,
        "status": "WAITING_ON_LEO",
        "closure_reason": "",
        "version": 1,
        "participants": [
          "leo",
          "hebe"
        ],
        "domain_payload": {},
        "consumed_event_ids": []
      },
      "latest": {
        "id": "conv_d35db56c3bdf59faa1125861e60a4875",
        "context_kind": "owner_local",
        "context_id": "leo_local",
        "attention_state": "HANDED_OFF",
        "turn_owner": "leo",
        "expected_reply_type": "yes_no",
        "topic": "replay_handoff",
        "origin_event_id": "open",
        "last_event_id": "open",
        "opened_at": 1000.0,
        "last_turn_at": 1000.0,
        "expires_at": 1060.0,
        "status": "WAITING_ON_LEO",
        "closure_reason": "",
        "version": 1,
        "participants": [
          "leo",
          "hebe"
        ],
        "domain_payload": {},
        "consumed_event_ids": []
      },
      "all": [
        {
          "id": "conv_d35db56c3bdf59faa1125861e60a4875",
          "context_kind": "owner_local",
          "context_id": "leo_local",
          "attention_state": "HANDED_OFF",
          "turn_owner": "leo",
          "expected_reply_type": "yes_no",
          "topic": "replay_handoff",
          "origin_event_id": "open",
          "last_event_id": "open",
          "opened_at": 1000.0,
          "last_turn_at": 1000.0,
          "expires_at": 1060.0,
          "status": "WAITING_ON_LEO",
          "closure_reason": "",
          "version": 1,
          "participants": [
            "leo",
            "hebe"
          ],
          "domain_payload": {},
          "consumed_event_ids": []
        }
      ],
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
    "open_threads": [
      {
        "id": "thread_44cc61231e8b530093accff83a7368a5",
        "thread_type": "clarification",
        "scope_kind": "owner_local",
        "scope_id": "leo_local",
        "subject_ref": "conv_d35db56c3bdf59faa1125861e60a4875",
        "summary": "Unresolved replay_handoff clarification",
        "origin_event_id": "open",
        "latest_event_id": "open",
        "status": "WAITING_ON_LEO",
        "priority": 50,
        "created_at": 1000.0,
        "relevance_until": 1060.0,
        "valid_until": 1060.0,
        "resolved_at": 0.0,
        "resolution_event_id": "",
        "sensitivity": "normal",
        "version": 1,
        "participant_ids": [
          "leo",
          "hebe"
        ]
      }
    ],
    "memory": {
      "facts_count": 0,
      "chunks_count": 0
    },
    "beliefs": [],
    "game_state": {
      "game": "",
      "platform_version": "",
      "playthrough_type": "casual",
      "spoiler_policy": "spoiler_safe_hints",
      "current_location": null,
      "current_character": "",
      "party_members": [],
      "last_confirmed_progress": "",
      "current_objective": null,
      "challenge": "",
      "known_constraints": [],
      "last_updated": 1786437824.5799396,
      "provenance": "inferred",
      "confidence": 0.0,
      "current_game": null,
      "recent_run_context_facts": []
    },
    "social_state": {
      "recent_active_users": [],
      "recent_chat_count": 0,
      "last_raid": {},
      "last_cheer": {}
    },
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
    "database_watermarks": {
      "counts": {
        "chat_log": 0,
        "memory_facts": 0,
        "memory_chunks": 0,
        "stream_sessions": 0,
        "stream_chat_messages": 0,
        "stream_events": 0,
        "live_session_timeline": 0,
        "promotion_events": 0,
        "viewer_promotion_profiles": 0,
        "schema_migrations": 2,
        "conversations": 1,
        "open_threads": 1
      },
      "schema_migrations": [
        {
          "component": "cognitive_replay",
          "version": 1,
          "name": "replay_metadata",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "applied_at": "2026-08-11T08:43:44.574388+00:00"
        },
        {
          "component": "conversation_continuity",
          "version": 1,
          "name": "conversation_and_open_threads",
          "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
          "applied_at": "2026-08-11T08:43:44.641101+00:00"
        }
      ],
      "final_response_digest": "",
      "final_response_present": false
    }
  },
  "restart": {
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
      "enabled": false,
      "is_live": false,
      "live_status_known": false,
      "active_stream_session_id": null,
      "last_transition": null,
      "title": null,
      "game": null,
      "category": null
    },
    "current_scene": {},
    "pending": {
      "clarification": null,
      "conversation_turn": null
    },
    "conversation": {
      "active": {},
      "latest": {
        "id": "conv_d35db56c3bdf59faa1125861e60a4875",
        "context_kind": "owner_local",
        "context_id": "leo_local",
        "attention_state": "RELEASED",
        "turn_owner": "leo",
        "expected_reply_type": "yes_no",
        "topic": "replay_handoff",
        "origin_event_id": "open",
        "last_event_id": "open",
        "opened_at": 1000.0,
        "last_turn_at": 1000.0,
        "expires_at": 1060.0,
        "status": "INTERRUPTED",
        "closure_reason": "runtime_restart",
        "version": 2,
        "participants": [
          "leo",
          "hebe"
        ],
        "domain_payload": {},
        "consumed_event_ids": []
      },
      "all": [
        {
          "id": "conv_d35db56c3bdf59faa1125861e60a4875",
          "context_kind": "owner_local",
          "context_id": "leo_local",
          "attention_state": "RELEASED",
          "turn_owner": "leo",
          "expected_reply_type": "yes_no",
          "topic": "replay_handoff",
          "origin_event_id": "open",
          "last_event_id": "open",
          "opened_at": 1000.0,
          "last_turn_at": 1000.0,
          "expires_at": 1060.0,
          "status": "INTERRUPTED",
          "closure_reason": "runtime_restart",
          "version": 2,
          "participants": [
            "leo",
            "hebe"
          ],
          "domain_payload": {},
          "consumed_event_ids": []
        }
      ],
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
    "open_threads": [
      {
        "id": "thread_44cc61231e8b530093accff83a7368a5",
        "thread_type": "clarification",
        "scope_kind": "owner_local",
        "scope_id": "leo_local",
        "subject_ref": "conv_d35db56c3bdf59faa1125861e60a4875",
        "summary": "Unresolved replay_handoff clarification",
        "origin_event_id": "open",
        "latest_event_id": "runtime_restart",
        "status": "ARCHIVED",
        "priority": 50,
        "created_at": 1000.0,
        "relevance_until": 1060.0,
        "valid_until": 1060.0,
        "resolved_at": 1001.0,
        "resolution_event_id": "runtime_restart",
        "sensitivity": "normal",
        "version": 2,
        "participant_ids": [
          "leo",
          "hebe"
        ]
      }
    ],
    "memory": {
      "facts_count": 0,
      "chunks_count": 0
    },
    "beliefs": [],
    "game_state": {
      "game": "",
      "platform_version": "",
      "playthrough_type": "casual",
      "spoiler_policy": "spoiler_safe_hints",
      "current_location": null,
      "current_character": "",
      "party_members": [],
      "last_confirmed_progress": "",
      "current_objective": null,
      "challenge": "",
      "known_constraints": [],
      "last_updated": 1786437824.7248497,
      "provenance": "inferred",
      "confidence": 0.0,
      "current_game": null,
      "recent_run_context_facts": []
    },
    "social_state": {
      "recent_active_users": [],
      "recent_chat_count": 0,
      "last_raid": {},
      "last_cheer": {}
    },
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
    "database_watermarks": {
      "counts": {
        "chat_log": 0,
        "memory_facts": 0,
        "memory_chunks": 0,
        "stream_sessions": 0,
        "stream_chat_messages": 0,
        "stream_events": 0,
        "live_session_timeline": 0,
        "promotion_events": 0,
        "viewer_promotion_profiles": 0,
        "schema_migrations": 2,
        "conversations": 1,
        "open_threads": 1
      },
      "schema_migrations": [
        {
          "component": "cognitive_replay",
          "version": 1,
          "name": "replay_metadata",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "applied_at": "2026-08-11T08:43:44.574388+00:00"
        },
        {
          "component": "conversation_continuity",
          "version": 1,
          "name": "conversation_and_open_threads",
          "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
          "applied_at": "2026-08-11T08:43:44.641101+00:00"
        }
      ],
      "final_response_digest": "",
      "final_response_present": false
    }
  },
  "stale": {
    "runtime": {
      "mode": "active",
      "hebe_sleeping": false,
      "is_running": false,
      "last_input_source": null,
      "last_intent": null,
      "last_firewall": {
        "source": "ambient_stt",
        "authority": "ambient",
        "decision": "ignore",
        "reason": "offline_stream"
      },
      "last_policy": {
        "source": null,
        "authority": null,
        "decision": null,
        "reason": null
      }
    },
    "stream_session": {
      "enabled": false,
      "is_live": false,
      "live_status_known": false,
      "active_stream_session_id": null,
      "last_transition": null,
      "title": null,
      "game": null,
      "category": null
    },
    "current_scene": {},
    "pending": {
      "clarification": null,
      "conversation_turn": null
    },
    "conversation": {
      "active": {},
      "latest": {
        "id": "conv_d35db56c3bdf59faa1125861e60a4875",
        "context_kind": "owner_local",
        "context_id": "leo_local",
        "attention_state": "RELEASED",
        "turn_owner": "leo",
        "expected_reply_type": "yes_no",
        "topic": "replay_handoff",
        "origin_event_id": "open",
        "last_event_id": "open",
        "opened_at": 1000.0,
        "last_turn_at": 1000.0,
        "expires_at": 1060.0,
        "status": "INTERRUPTED",
        "closure_reason": "runtime_restart",
        "version": 2,
        "participants": [
          "leo",
          "hebe"
        ],
        "domain_payload": {},
        "consumed_event_ids": []
      },
      "all": [
        {
          "id": "conv_d35db56c3bdf59faa1125861e60a4875",
          "context_kind": "owner_local",
          "context_id": "leo_local",
          "attention_state": "RELEASED",
          "turn_owner": "leo",
          "expected_reply_type": "yes_no",
          "topic": "replay_handoff",
          "origin_event_id": "open",
          "last_event_id": "open",
          "opened_at": 1000.0,
          "last_turn_at": 1000.0,
          "expires_at": 1060.0,
          "status": "INTERRUPTED",
          "closure_reason": "runtime_restart",
          "version": 2,
          "participants": [
            "leo",
            "hebe"
          ],
          "domain_payload": {},
          "consumed_event_ids": []
        }
      ],
      "last_resolution": {
        "consumed": false,
        "decision": "no_conversation",
        "reason": "no_compatible_active_conversation",
        "conversation_id": "",
        "reply_act": "UNKNOWN",
        "payload": {},
        "conversation": null,
        "latency_ms": 1.6565999831072986
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
        "p50_ms": 1.6566,
        "p95_ms": 1.6566
      }
    },
    "open_threads": [
      {
        "id": "thread_44cc61231e8b530093accff83a7368a5",
        "thread_type": "clarification",
        "scope_kind": "owner_local",
        "scope_id": "leo_local",
        "subject_ref": "conv_d35db56c3bdf59faa1125861e60a4875",
        "summary": "Unresolved replay_handoff clarification",
        "origin_event_id": "open",
        "latest_event_id": "runtime_restart",
        "status": "ARCHIVED",
        "priority": 50,
        "created_at": 1000.0,
        "relevance_until": 1060.0,
        "valid_until": 1060.0,
        "resolved_at": 1001.0,
        "resolution_event_id": "runtime_restart",
        "sensitivity": "normal",
        "version": 2,
        "participant_ids": [
          "leo",
          "hebe"
        ]
      }
    ],
    "memory": {
      "facts_count": 0,
      "chunks_count": 0
    },
    "beliefs": [],
    "game_state": {
      "game": "",
      "platform_version": "",
      "playthrough_type": "casual",
      "spoiler_policy": "spoiler_safe_hints",
      "current_location": null,
      "current_character": "",
      "party_members": [],
      "last_confirmed_progress": "",
      "current_objective": null,
      "challenge": "",
      "known_constraints": [],
      "last_updated": 1786437824.7248497,
      "provenance": "inferred",
      "confidence": 0.0,
      "current_game": null,
      "recent_run_context_facts": []
    },
    "social_state": {
      "recent_active_users": [],
      "recent_chat_count": 0,
      "last_raid": {},
      "last_cheer": {}
    },
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
    "database_watermarks": {
      "counts": {
        "chat_log": 0,
        "memory_facts": 0,
        "memory_chunks": 0,
        "stream_sessions": 0,
        "stream_chat_messages": 0,
        "stream_events": 0,
        "live_session_timeline": 0,
        "promotion_events": 0,
        "viewer_promotion_profiles": 0,
        "schema_migrations": 2,
        "conversations": 1,
        "open_threads": 1
      },
      "schema_migrations": [
        {
          "component": "cognitive_replay",
          "version": 1,
          "name": "replay_metadata",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "applied_at": "2026-08-11T08:43:44.574388+00:00"
        },
        {
          "component": "conversation_continuity",
          "version": 1,
          "name": "conversation_and_open_threads",
          "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
          "applied_at": "2026-08-11T08:43:44.641101+00:00"
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
      "source": "ambient_stt",
      "authority": "ambient",
      "decision": "ignore",
      "reason": "offline_stream"
    },
    "last_policy": {
      "source": null,
      "authority": null,
      "decision": null,
      "reason": null
    }
  },
  "stream_session": {
    "enabled": false,
    "is_live": false,
    "live_status_known": false,
    "active_stream_session_id": null,
    "last_transition": null,
    "title": null,
    "game": null,
    "category": null
  },
  "current_scene": {},
  "pending": {
    "clarification": null,
    "conversation_turn": null
  },
  "conversation": {
    "active": {},
    "latest": {
      "id": "conv_d35db56c3bdf59faa1125861e60a4875",
      "context_kind": "owner_local",
      "context_id": "leo_local",
      "attention_state": "RELEASED",
      "turn_owner": "leo",
      "expected_reply_type": "yes_no",
      "topic": "replay_handoff",
      "origin_event_id": "open",
      "last_event_id": "open",
      "opened_at": 1000.0,
      "last_turn_at": 1000.0,
      "expires_at": 1060.0,
      "status": "INTERRUPTED",
      "closure_reason": "runtime_restart",
      "version": 2,
      "participants": [
        "leo",
        "hebe"
      ],
      "domain_payload": {},
      "consumed_event_ids": []
    },
    "all": [
      {
        "id": "conv_d35db56c3bdf59faa1125861e60a4875",
        "context_kind": "owner_local",
        "context_id": "leo_local",
        "attention_state": "RELEASED",
        "turn_owner": "leo",
        "expected_reply_type": "yes_no",
        "topic": "replay_handoff",
        "origin_event_id": "open",
        "last_event_id": "open",
        "opened_at": 1000.0,
        "last_turn_at": 1000.0,
        "expires_at": 1060.0,
        "status": "INTERRUPTED",
        "closure_reason": "runtime_restart",
        "version": 2,
        "participants": [
          "leo",
          "hebe"
        ],
        "domain_payload": {},
        "consumed_event_ids": []
      }
    ],
    "last_resolution": {
      "consumed": false,
      "decision": "no_conversation",
      "reason": "no_compatible_active_conversation",
      "conversation_id": "",
      "reply_act": "UNKNOWN",
      "payload": {},
      "conversation": null,
      "latency_ms": 1.6565999831072986
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
      "p50_ms": 1.6566,
      "p95_ms": 1.6566
    }
  },
  "open_threads": [
    {
      "id": "thread_44cc61231e8b530093accff83a7368a5",
      "thread_type": "clarification",
      "scope_kind": "owner_local",
      "scope_id": "leo_local",
      "subject_ref": "conv_d35db56c3bdf59faa1125861e60a4875",
      "summary": "Unresolved replay_handoff clarification",
      "origin_event_id": "open",
      "latest_event_id": "runtime_restart",
      "status": "ARCHIVED",
      "priority": 50,
      "created_at": 1000.0,
      "relevance_until": 1060.0,
      "valid_until": 1060.0,
      "resolved_at": 1001.0,
      "resolution_event_id": "runtime_restart",
      "sensitivity": "normal",
      "version": 2,
      "participant_ids": [
        "leo",
        "hebe"
      ]
    }
  ],
  "memory": {
    "facts_count": 0,
    "chunks_count": 0
  },
  "beliefs": [],
  "game_state": {
    "game": "",
    "platform_version": "",
    "playthrough_type": "casual",
    "spoiler_policy": "spoiler_safe_hints",
    "current_location": null,
    "current_character": "",
    "party_members": [],
    "last_confirmed_progress": "",
    "current_objective": null,
    "challenge": "",
    "known_constraints": [],
    "last_updated": 1786437824.7248497,
    "provenance": "inferred",
    "confidence": 0.0,
    "current_game": null,
    "recent_run_context_facts": []
  },
  "social_state": {
    "recent_active_users": [],
    "recent_chat_count": 0,
    "last_raid": {},
    "last_cheer": {}
  },
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
  "database_watermarks": {
    "counts": {
      "chat_log": 0,
      "memory_facts": 0,
      "memory_chunks": 0,
      "stream_sessions": 0,
      "stream_chat_messages": 0,
      "stream_events": 0,
      "live_session_timeline": 0,
      "promotion_events": 0,
      "viewer_promotion_profiles": 0,
      "schema_migrations": 2,
      "conversations": 1,
      "open_threads": 1
    },
    "schema_migrations": [
      {
        "component": "cognitive_replay",
        "version": 1,
        "name": "replay_metadata",
        "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
        "applied_at": "2026-08-11T08:43:44.574388+00:00"
      },
      {
        "component": "conversation_continuity",
        "version": 1,
        "name": "conversation_and_open_threads",
        "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
        "applied_at": "2026-08-11T08:43:44.641101+00:00"
      }
    ],
    "final_response_digest": "",
    "final_response_present": false
  }
}
```

#### Restart evidence

```json
[
  {
    "event_id": "restart",
    "old_engine_id": 2096481957072,
    "new_engine_id": 2096480554832,
    "old_engine_collected": true,
    "same_database": "C:\\Users\\Leo Nifelheim\\Documents\\Hebe\\hebe-ui\\artifacts\\cognitive-continuity-phase1\\final\\workspaces\\phase1_h_restart_stale\\hebe-replay.sqlite3",
    "before_persisted_counts": {
      "chat_log": 0,
      "memory_facts": 0,
      "memory_chunks": 0,
      "stream_sessions": 0,
      "stream_chat_messages": 0,
      "stream_events": 0,
      "live_session_timeline": 0,
      "promotion_events": 0,
      "viewer_promotion_profiles": 0,
      "schema_migrations": 2,
      "conversations": 1,
      "open_threads": 1
    },
    "after_persisted_counts": {
      "chat_log": 0,
      "memory_facts": 0,
      "memory_chunks": 0,
      "stream_sessions": 0,
      "stream_chat_messages": 0,
      "stream_events": 0,
      "live_session_timeline": 0,
      "promotion_events": 0,
      "viewer_promotion_profiles": 0,
      "schema_migrations": 2,
      "conversations": 1,
      "open_threads": 1
    },
    "volatile_state_recreated": true
  }
]
```

### phase1_i_free_response

- Status: **VERIFIED**
- Events: 2
- Restarts: 0
- Duration: 0.391102s
- Assertions passed/failed/skipped: 2/0/0


#### Checkpoint state

```json
{
  "open": {
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
      "enabled": false,
      "is_live": false,
      "live_status_known": false,
      "active_stream_session_id": null,
      "last_transition": null,
      "title": null,
      "game": null,
      "category": null
    },
    "current_scene": {},
    "pending": {
      "clarification": null,
      "conversation_turn": null
    },
    "conversation": {
      "active": {
        "id": "conv_63a29e42bb0e5f12af178df18d0f7a34",
        "context_kind": "owner_local",
        "context_id": "leo_local",
        "attention_state": "HANDED_OFF",
        "turn_owner": "leo",
        "expected_reply_type": "free_response",
        "topic": "replay_handoff",
        "origin_event_id": "open",
        "last_event_id": "open",
        "opened_at": 1000.0,
        "last_turn_at": 1000.0,
        "expires_at": 1060.0,
        "status": "WAITING_ON_LEO",
        "closure_reason": "",
        "version": 1,
        "participants": [
          "leo",
          "hebe"
        ],
        "domain_payload": {},
        "consumed_event_ids": []
      },
      "latest": {
        "id": "conv_63a29e42bb0e5f12af178df18d0f7a34",
        "context_kind": "owner_local",
        "context_id": "leo_local",
        "attention_state": "HANDED_OFF",
        "turn_owner": "leo",
        "expected_reply_type": "free_response",
        "topic": "replay_handoff",
        "origin_event_id": "open",
        "last_event_id": "open",
        "opened_at": 1000.0,
        "last_turn_at": 1000.0,
        "expires_at": 1060.0,
        "status": "WAITING_ON_LEO",
        "closure_reason": "",
        "version": 1,
        "participants": [
          "leo",
          "hebe"
        ],
        "domain_payload": {},
        "consumed_event_ids": []
      },
      "all": [
        {
          "id": "conv_63a29e42bb0e5f12af178df18d0f7a34",
          "context_kind": "owner_local",
          "context_id": "leo_local",
          "attention_state": "HANDED_OFF",
          "turn_owner": "leo",
          "expected_reply_type": "free_response",
          "topic": "replay_handoff",
          "origin_event_id": "open",
          "last_event_id": "open",
          "opened_at": 1000.0,
          "last_turn_at": 1000.0,
          "expires_at": 1060.0,
          "status": "WAITING_ON_LEO",
          "closure_reason": "",
          "version": 1,
          "participants": [
            "leo",
            "hebe"
          ],
          "domain_payload": {},
          "consumed_event_ids": []
        }
      ],
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
    "open_threads": [
      {
        "id": "thread_97be53bf333d5f55a61f656208cd14a7",
        "thread_type": "clarification",
        "scope_kind": "owner_local",
        "scope_id": "leo_local",
        "subject_ref": "conv_63a29e42bb0e5f12af178df18d0f7a34",
        "summary": "Unresolved replay_handoff clarification",
        "origin_event_id": "open",
        "latest_event_id": "open",
        "status": "WAITING_ON_LEO",
        "priority": 50,
        "created_at": 1000.0,
        "relevance_until": 1060.0,
        "valid_until": 1060.0,
        "resolved_at": 0.0,
        "resolution_event_id": "",
        "sensitivity": "normal",
        "version": 1,
        "participant_ids": [
          "leo",
          "hebe"
        ]
      }
    ],
    "memory": {
      "facts_count": 0,
      "chunks_count": 0
    },
    "beliefs": [],
    "game_state": {
      "game": "",
      "platform_version": "",
      "playthrough_type": "casual",
      "spoiler_policy": "spoiler_safe_hints",
      "current_location": null,
      "current_character": "",
      "party_members": [],
      "last_confirmed_progress": "",
      "current_objective": null,
      "challenge": "",
      "known_constraints": [],
      "last_updated": 1786437825.1420646,
      "provenance": "inferred",
      "confidence": 0.0,
      "current_game": null,
      "recent_run_context_facts": []
    },
    "social_state": {
      "recent_active_users": [],
      "recent_chat_count": 0,
      "last_raid": {},
      "last_cheer": {}
    },
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
    "database_watermarks": {
      "counts": {
        "chat_log": 0,
        "memory_facts": 0,
        "memory_chunks": 0,
        "stream_sessions": 0,
        "stream_chat_messages": 0,
        "stream_events": 0,
        "live_session_timeline": 0,
        "promotion_events": 0,
        "viewer_promotion_profiles": 0,
        "schema_migrations": 2,
        "conversations": 1,
        "open_threads": 1
      },
      "schema_migrations": [
        {
          "component": "cognitive_replay",
          "version": 1,
          "name": "replay_metadata",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "applied_at": "2026-08-11T08:43:45.135435+00:00"
        },
        {
          "component": "conversation_continuity",
          "version": 1,
          "name": "conversation_and_open_threads",
          "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
          "applied_at": "2026-08-11T08:43:45.211938+00:00"
        }
      ],
      "final_response_digest": "",
      "final_response_present": false
    }
  },
  "reply": {
    "runtime": {
      "mode": "active",
      "hebe_sleeping": false,
      "is_running": false,
      "last_input_source": null,
      "last_intent": null,
      "last_firewall": {
        "source": "owner_stt_followup",
        "authority": "owner",
        "decision": "allow",
        "reason": "owner_related_followup"
      },
      "last_policy": {
        "source": null,
        "authority": null,
        "decision": null,
        "reason": null
      }
    },
    "stream_session": {
      "enabled": false,
      "is_live": false,
      "live_status_known": false,
      "active_stream_session_id": null,
      "last_transition": null,
      "title": null,
      "game": null,
      "category": null
    },
    "current_scene": {},
    "pending": {
      "clarification": null,
      "conversation_turn": null
    },
    "conversation": {
      "active": {},
      "latest": {
        "id": "conv_63a29e42bb0e5f12af178df18d0f7a34",
        "context_kind": "owner_local",
        "context_id": "leo_local",
        "attention_state": "RELEASED",
        "turn_owner": "leo",
        "expected_reply_type": "free_response",
        "topic": "replay_handoff",
        "origin_event_id": "open",
        "last_event_id": "reply",
        "opened_at": 1000.0,
        "last_turn_at": 1001.0,
        "expires_at": 1060.0,
        "status": "CLOSED",
        "closure_reason": "reply_consumed",
        "version": 2,
        "participants": [
          "leo",
          "hebe"
        ],
        "domain_payload": {},
        "consumed_event_ids": [
          "reply"
        ]
      },
      "all": [
        {
          "id": "conv_63a29e42bb0e5f12af178df18d0f7a34",
          "context_kind": "owner_local",
          "context_id": "leo_local",
          "attention_state": "RELEASED",
          "turn_owner": "leo",
          "expected_reply_type": "free_response",
          "topic": "replay_handoff",
          "origin_event_id": "open",
          "last_event_id": "reply",
          "opened_at": 1000.0,
          "last_turn_at": 1001.0,
          "expires_at": 1060.0,
          "status": "CLOSED",
          "closure_reason": "reply_consumed",
          "version": 2,
          "participants": [
            "leo",
            "hebe"
          ],
          "domain_payload": {},
          "consumed_event_ids": [
            "reply"
          ]
        }
      ],
      "last_resolution": {
        "consumed": true,
        "decision": "compatible_reply",
        "reason": "bounded_free_response",
        "conversation_id": "conv_63a29e42bb0e5f12af178df18d0f7a34",
        "reply_act": "FREE_RESPONSE",
        "payload": {
          "response_text": "porque me apetece",
          "domain": {},
          "expected_reply_type": "free_response"
        },
        "conversation": {
          "id": "conv_63a29e42bb0e5f12af178df18d0f7a34",
          "context_kind": "owner_local",
          "context_id": "leo_local",
          "participants": [
            "leo",
            "hebe"
          ],
          "attention_state": "RELEASED",
          "turn_owner": "leo",
          "expected_reply": {
            "type": "free_response",
            "allowed_sources": [
              "owner_stt"
            ],
            "allowed_participant": "leo",
            "semantic_constraints": {
              "min_words": 2,
              "max_words": 20
            },
            "candidate_refs": [],
            "expires_at": 1060.0,
            "consume_policy": "once"
          },
          "topic": "replay_handoff",
          "origin_event_id": "open",
          "last_event_id": "reply",
          "opened_at": 1000.0,
          "last_turn_at": 1001.0,
          "expires_at": 1060.0,
          "status": "CLOSED",
          "closure_reason": "reply_consumed",
          "version": 2,
          "domain_payload": {},
          "consumed_event_ids": [
            "reply"
          ]
        },
        "latency_ms": 9.84380004229024
      },
      "legacy_pending_projection": {},
      "continuity_shadow_diff": {
        "legacy_result": false,
        "v2_result": true,
        "match": false,
        "difference_reason": "bounded_free_response"
      },
      "shadow_metrics": {
        "total": 1,
        "matches": 0,
        "differences": 1,
        "match_rate": 0.0,
        "difference_reasons": {
          "bounded_free_response": 1
        }
      },
      "performance": {
        "count": 1,
        "p50_ms": 9.8438,
        "p95_ms": 9.8438
      }
    },
    "open_threads": [
      {
        "id": "thread_97be53bf333d5f55a61f656208cd14a7",
        "thread_type": "clarification",
        "scope_kind": "owner_local",
        "scope_id": "leo_local",
        "subject_ref": "conv_63a29e42bb0e5f12af178df18d0f7a34",
        "summary": "Unresolved replay_handoff clarification",
        "origin_event_id": "open",
        "latest_event_id": "reply",
        "status": "RESOLVED",
        "priority": 50,
        "created_at": 1000.0,
        "relevance_until": 1060.0,
        "valid_until": 1060.0,
        "resolved_at": 1001.0,
        "resolution_event_id": "reply",
        "sensitivity": "normal",
        "version": 2,
        "participant_ids": [
          "leo",
          "hebe"
        ]
      }
    ],
    "memory": {
      "facts_count": 0,
      "chunks_count": 0
    },
    "beliefs": [],
    "game_state": {
      "game": "",
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
      "last_updated": 1786437825.1420646,
      "provenance": "current_live_session",
      "confidence": 0.0,
      "current_game": null,
      "recent_run_context_facts": []
    },
    "social_state": {
      "recent_active_users": [],
      "recent_chat_count": 0,
      "last_raid": {},
      "last_cheer": {}
    },
    "promotion_profiles": [],
    "actions": {
      "attempts": [],
      "speech_requests": [],
      "model_calls": [
        {
          "key": "stream_response:v1:none",
          "method": "chat"
        }
      ],
      "research_calls": []
    },
    "receipts": [],
    "emitted_outputs": [
      {
        "event_id": "1786437825.2302346",
        "emitted": true,
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "reason": "",
        "text_digest": "",
        "text_present": false
      }
    ],
    "final_emission_results": [
      {
        "event_id": "1786437825.2302346",
        "emitted": true,
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "reason": "",
        "text_digest": "",
        "text_present": false
      }
    ],
    "database_watermarks": {
      "counts": {
        "chat_log": 1,
        "memory_facts": 0,
        "memory_chunks": 0,
        "stream_sessions": 0,
        "stream_chat_messages": 0,
        "stream_events": 0,
        "live_session_timeline": 2,
        "promotion_events": 0,
        "viewer_promotion_profiles": 0,
        "schema_migrations": 2,
        "conversations": 1,
        "open_threads": 1
      },
      "schema_migrations": [
        {
          "component": "cognitive_replay",
          "version": 1,
          "name": "replay_metadata",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "applied_at": "2026-08-11T08:43:45.135435+00:00"
        },
        {
          "component": "conversation_continuity",
          "version": 1,
          "name": "conversation_and_open_threads",
          "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
          "applied_at": "2026-08-11T08:43:45.211938+00:00"
        }
      ],
      "final_response_digest": "7e07c6a5322789f6",
      "final_response_present": true
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
      "source": "owner_stt_followup",
      "authority": "owner",
      "decision": "allow",
      "reason": "owner_related_followup"
    },
    "last_policy": {
      "source": null,
      "authority": null,
      "decision": null,
      "reason": null
    }
  },
  "stream_session": {
    "enabled": false,
    "is_live": false,
    "live_status_known": false,
    "active_stream_session_id": null,
    "last_transition": null,
    "title": null,
    "game": null,
    "category": null
  },
  "current_scene": {},
  "pending": {
    "clarification": null,
    "conversation_turn": null
  },
  "conversation": {
    "active": {},
    "latest": {
      "id": "conv_63a29e42bb0e5f12af178df18d0f7a34",
      "context_kind": "owner_local",
      "context_id": "leo_local",
      "attention_state": "RELEASED",
      "turn_owner": "leo",
      "expected_reply_type": "free_response",
      "topic": "replay_handoff",
      "origin_event_id": "open",
      "last_event_id": "reply",
      "opened_at": 1000.0,
      "last_turn_at": 1001.0,
      "expires_at": 1060.0,
      "status": "CLOSED",
      "closure_reason": "reply_consumed",
      "version": 2,
      "participants": [
        "leo",
        "hebe"
      ],
      "domain_payload": {},
      "consumed_event_ids": [
        "reply"
      ]
    },
    "all": [
      {
        "id": "conv_63a29e42bb0e5f12af178df18d0f7a34",
        "context_kind": "owner_local",
        "context_id": "leo_local",
        "attention_state": "RELEASED",
        "turn_owner": "leo",
        "expected_reply_type": "free_response",
        "topic": "replay_handoff",
        "origin_event_id": "open",
        "last_event_id": "reply",
        "opened_at": 1000.0,
        "last_turn_at": 1001.0,
        "expires_at": 1060.0,
        "status": "CLOSED",
        "closure_reason": "reply_consumed",
        "version": 2,
        "participants": [
          "leo",
          "hebe"
        ],
        "domain_payload": {},
        "consumed_event_ids": [
          "reply"
        ]
      }
    ],
    "last_resolution": {
      "consumed": true,
      "decision": "compatible_reply",
      "reason": "bounded_free_response",
      "conversation_id": "conv_63a29e42bb0e5f12af178df18d0f7a34",
      "reply_act": "FREE_RESPONSE",
      "payload": {
        "response_text": "porque me apetece",
        "domain": {},
        "expected_reply_type": "free_response"
      },
      "conversation": {
        "id": "conv_63a29e42bb0e5f12af178df18d0f7a34",
        "context_kind": "owner_local",
        "context_id": "leo_local",
        "participants": [
          "leo",
          "hebe"
        ],
        "attention_state": "RELEASED",
        "turn_owner": "leo",
        "expected_reply": {
          "type": "free_response",
          "allowed_sources": [
            "owner_stt"
          ],
          "allowed_participant": "leo",
          "semantic_constraints": {
            "min_words": 2,
            "max_words": 20
          },
          "candidate_refs": [],
          "expires_at": 1060.0,
          "consume_policy": "once"
        },
        "topic": "replay_handoff",
        "origin_event_id": "open",
        "last_event_id": "reply",
        "opened_at": 1000.0,
        "last_turn_at": 1001.0,
        "expires_at": 1060.0,
        "status": "CLOSED",
        "closure_reason": "reply_consumed",
        "version": 2,
        "domain_payload": {},
        "consumed_event_ids": [
          "reply"
        ]
      },
      "latency_ms": 9.84380004229024
    },
    "legacy_pending_projection": {},
    "continuity_shadow_diff": {
      "legacy_result": false,
      "v2_result": true,
      "match": false,
      "difference_reason": "bounded_free_response"
    },
    "shadow_metrics": {
      "total": 1,
      "matches": 0,
      "differences": 1,
      "match_rate": 0.0,
      "difference_reasons": {
        "bounded_free_response": 1
      }
    },
    "performance": {
      "count": 1,
      "p50_ms": 9.8438,
      "p95_ms": 9.8438
    }
  },
  "open_threads": [
    {
      "id": "thread_97be53bf333d5f55a61f656208cd14a7",
      "thread_type": "clarification",
      "scope_kind": "owner_local",
      "scope_id": "leo_local",
      "subject_ref": "conv_63a29e42bb0e5f12af178df18d0f7a34",
      "summary": "Unresolved replay_handoff clarification",
      "origin_event_id": "open",
      "latest_event_id": "reply",
      "status": "RESOLVED",
      "priority": 50,
      "created_at": 1000.0,
      "relevance_until": 1060.0,
      "valid_until": 1060.0,
      "resolved_at": 1001.0,
      "resolution_event_id": "reply",
      "sensitivity": "normal",
      "version": 2,
      "participant_ids": [
        "leo",
        "hebe"
      ]
    }
  ],
  "memory": {
    "facts_count": 0,
    "chunks_count": 0
  },
  "beliefs": [],
  "game_state": {
    "game": "",
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
    "last_updated": 1786437825.1420646,
    "provenance": "current_live_session",
    "confidence": 0.0,
    "current_game": null,
    "recent_run_context_facts": []
  },
  "social_state": {
    "recent_active_users": [],
    "recent_chat_count": 0,
    "last_raid": {},
    "last_cheer": {}
  },
  "promotion_profiles": [],
  "actions": {
    "attempts": [],
    "speech_requests": [],
    "model_calls": [
      {
        "key": "stream_response:v1:none",
        "method": "chat"
      }
    ],
    "research_calls": []
  },
  "receipts": [],
  "emitted_outputs": [
    {
      "event_id": "1786437825.2302346",
      "emitted": true,
      "route": "local_owner_reply",
      "targets": [
        "local_ui"
      ],
      "reason": "",
      "text_digest": "",
      "text_present": false
    }
  ],
  "final_emission_results": [
    {
      "event_id": "1786437825.2302346",
      "emitted": true,
      "route": "local_owner_reply",
      "targets": [
        "local_ui"
      ],
      "reason": "",
      "text_digest": "",
      "text_present": false
    }
  ],
  "database_watermarks": {
    "counts": {
      "chat_log": 1,
      "memory_facts": 0,
      "memory_chunks": 0,
      "stream_sessions": 0,
      "stream_chat_messages": 0,
      "stream_events": 0,
      "live_session_timeline": 2,
      "promotion_events": 0,
      "viewer_promotion_profiles": 0,
      "schema_migrations": 2,
      "conversations": 1,
      "open_threads": 1
    },
    "schema_migrations": [
      {
        "component": "cognitive_replay",
        "version": 1,
        "name": "replay_metadata",
        "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
        "applied_at": "2026-08-11T08:43:45.135435+00:00"
      },
      {
        "component": "conversation_continuity",
        "version": 1,
        "name": "conversation_and_open_threads",
        "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
        "applied_at": "2026-08-11T08:43:45.211938+00:00"
      }
    ],
    "final_response_digest": "7e07c6a5322789f6",
    "final_response_present": true
  }
}
```

#### Restart evidence

```json
[]
```

### phase1_j_cancel

- Status: **VERIFIED**
- Events: 2
- Restarts: 0
- Duration: 0.315228s
- Assertions passed/failed/skipped: 3/0/0


#### Checkpoint state

```json
{
  "open": {
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
      "enabled": false,
      "is_live": false,
      "live_status_known": false,
      "active_stream_session_id": null,
      "last_transition": null,
      "title": null,
      "game": null,
      "category": null
    },
    "current_scene": {},
    "pending": {
      "clarification": null,
      "conversation_turn": null
    },
    "conversation": {
      "active": {
        "id": "conv_3a416da4996951c183aa7a53ce636885",
        "context_kind": "owner_local",
        "context_id": "leo_local",
        "attention_state": "HANDED_OFF",
        "turn_owner": "leo",
        "expected_reply_type": "yes_no",
        "topic": "replay_handoff",
        "origin_event_id": "open",
        "last_event_id": "open",
        "opened_at": 1000.0,
        "last_turn_at": 1000.0,
        "expires_at": 1060.0,
        "status": "WAITING_ON_LEO",
        "closure_reason": "",
        "version": 1,
        "participants": [
          "leo",
          "hebe"
        ],
        "domain_payload": {},
        "consumed_event_ids": []
      },
      "latest": {
        "id": "conv_3a416da4996951c183aa7a53ce636885",
        "context_kind": "owner_local",
        "context_id": "leo_local",
        "attention_state": "HANDED_OFF",
        "turn_owner": "leo",
        "expected_reply_type": "yes_no",
        "topic": "replay_handoff",
        "origin_event_id": "open",
        "last_event_id": "open",
        "opened_at": 1000.0,
        "last_turn_at": 1000.0,
        "expires_at": 1060.0,
        "status": "WAITING_ON_LEO",
        "closure_reason": "",
        "version": 1,
        "participants": [
          "leo",
          "hebe"
        ],
        "domain_payload": {},
        "consumed_event_ids": []
      },
      "all": [
        {
          "id": "conv_3a416da4996951c183aa7a53ce636885",
          "context_kind": "owner_local",
          "context_id": "leo_local",
          "attention_state": "HANDED_OFF",
          "turn_owner": "leo",
          "expected_reply_type": "yes_no",
          "topic": "replay_handoff",
          "origin_event_id": "open",
          "last_event_id": "open",
          "opened_at": 1000.0,
          "last_turn_at": 1000.0,
          "expires_at": 1060.0,
          "status": "WAITING_ON_LEO",
          "closure_reason": "",
          "version": 1,
          "participants": [
            "leo",
            "hebe"
          ],
          "domain_payload": {},
          "consumed_event_ids": []
        }
      ],
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
    "open_threads": [
      {
        "id": "thread_85517a7978f35c94b95f692db9bd066b",
        "thread_type": "clarification",
        "scope_kind": "owner_local",
        "scope_id": "leo_local",
        "subject_ref": "conv_3a416da4996951c183aa7a53ce636885",
        "summary": "Unresolved replay_handoff clarification",
        "origin_event_id": "open",
        "latest_event_id": "open",
        "status": "WAITING_ON_LEO",
        "priority": 50,
        "created_at": 1000.0,
        "relevance_until": 1060.0,
        "valid_until": 1060.0,
        "resolved_at": 0.0,
        "resolution_event_id": "",
        "sensitivity": "normal",
        "version": 1,
        "participant_ids": [
          "leo",
          "hebe"
        ]
      }
    ],
    "memory": {
      "facts_count": 0,
      "chunks_count": 0
    },
    "beliefs": [],
    "game_state": {
      "game": "",
      "platform_version": "",
      "playthrough_type": "casual",
      "spoiler_policy": "spoiler_safe_hints",
      "current_location": null,
      "current_character": "",
      "party_members": [],
      "last_confirmed_progress": "",
      "current_objective": null,
      "challenge": "",
      "known_constraints": [],
      "last_updated": 1786437825.5728567,
      "provenance": "inferred",
      "confidence": 0.0,
      "current_game": null,
      "recent_run_context_facts": []
    },
    "social_state": {
      "recent_active_users": [],
      "recent_chat_count": 0,
      "last_raid": {},
      "last_cheer": {}
    },
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
    "database_watermarks": {
      "counts": {
        "chat_log": 0,
        "memory_facts": 0,
        "memory_chunks": 0,
        "stream_sessions": 0,
        "stream_chat_messages": 0,
        "stream_events": 0,
        "live_session_timeline": 0,
        "promotion_events": 0,
        "viewer_promotion_profiles": 0,
        "schema_migrations": 2,
        "conversations": 1,
        "open_threads": 1
      },
      "schema_migrations": [
        {
          "component": "cognitive_replay",
          "version": 1,
          "name": "replay_metadata",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "applied_at": "2026-08-11T08:43:45.565504+00:00"
        },
        {
          "component": "conversation_continuity",
          "version": 1,
          "name": "conversation_and_open_threads",
          "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
          "applied_at": "2026-08-11T08:43:45.643400+00:00"
        }
      ],
      "final_response_digest": "",
      "final_response_present": false
    }
  },
  "cancel": {
    "runtime": {
      "mode": "active",
      "hebe_sleeping": false,
      "is_running": false,
      "last_input_source": null,
      "last_intent": null,
      "last_firewall": {
        "source": "ambient_stt",
        "authority": "ambient",
        "decision": "ignore",
        "reason": "offline_stream"
      },
      "last_policy": {
        "source": null,
        "authority": null,
        "decision": null,
        "reason": null
      }
    },
    "stream_session": {
      "enabled": false,
      "is_live": false,
      "live_status_known": false,
      "active_stream_session_id": null,
      "last_transition": null,
      "title": null,
      "game": null,
      "category": null
    },
    "current_scene": {},
    "pending": {
      "clarification": null,
      "conversation_turn": null
    },
    "conversation": {
      "active": {},
      "latest": {
        "id": "conv_3a416da4996951c183aa7a53ce636885",
        "context_kind": "owner_local",
        "context_id": "leo_local",
        "attention_state": "RELEASED",
        "turn_owner": "leo",
        "expected_reply_type": "yes_no",
        "topic": "replay_handoff",
        "origin_event_id": "open",
        "last_event_id": "cancel",
        "opened_at": 1000.0,
        "last_turn_at": 1001.0,
        "expires_at": 1060.0,
        "status": "CANCELLED",
        "closure_reason": "cancelled_by_owner",
        "version": 2,
        "participants": [
          "leo",
          "hebe"
        ],
        "domain_payload": {},
        "consumed_event_ids": [
          "cancel"
        ]
      },
      "all": [
        {
          "id": "conv_3a416da4996951c183aa7a53ce636885",
          "context_kind": "owner_local",
          "context_id": "leo_local",
          "attention_state": "RELEASED",
          "turn_owner": "leo",
          "expected_reply_type": "yes_no",
          "topic": "replay_handoff",
          "origin_event_id": "open",
          "last_event_id": "cancel",
          "opened_at": 1000.0,
          "last_turn_at": 1001.0,
          "expires_at": 1060.0,
          "status": "CANCELLED",
          "closure_reason": "cancelled_by_owner",
          "version": 2,
          "participants": [
            "leo",
            "hebe"
          ],
          "domain_payload": {},
          "consumed_event_ids": [
            "cancel"
          ]
        }
      ],
      "last_resolution": {
        "consumed": true,
        "decision": "compatible_reply",
        "reason": "deterministic_cancel",
        "conversation_id": "conv_3a416da4996951c183aa7a53ce636885",
        "reply_act": "CANCEL",
        "payload": {
          "domain": {},
          "expected_reply_type": "yes_no"
        },
        "conversation": {
          "id": "conv_3a416da4996951c183aa7a53ce636885",
          "context_kind": "owner_local",
          "context_id": "leo_local",
          "participants": [
            "leo",
            "hebe"
          ],
          "attention_state": "RELEASED",
          "turn_owner": "leo",
          "expected_reply": {
            "type": "yes_no",
            "allowed_sources": [
              "owner_stt"
            ],
            "allowed_participant": "leo",
            "semantic_constraints": {},
            "candidate_refs": [],
            "expires_at": 1060.0,
            "consume_policy": "once"
          },
          "topic": "replay_handoff",
          "origin_event_id": "open",
          "last_event_id": "cancel",
          "opened_at": 1000.0,
          "last_turn_at": 1001.0,
          "expires_at": 1060.0,
          "status": "CANCELLED",
          "closure_reason": "cancelled_by_owner",
          "version": 2,
          "domain_payload": {},
          "consumed_event_ids": [
            "cancel"
          ]
        },
        "latency_ms": 9.679699956905097
      },
      "legacy_pending_projection": {},
      "continuity_shadow_diff": {
        "legacy_result": false,
        "v2_result": true,
        "match": false,
        "difference_reason": "deterministic_cancel"
      },
      "shadow_metrics": {
        "total": 1,
        "matches": 0,
        "differences": 1,
        "match_rate": 0.0,
        "difference_reasons": {
          "deterministic_cancel": 1
        }
      },
      "performance": {
        "count": 1,
        "p50_ms": 9.6797,
        "p95_ms": 9.6797
      }
    },
    "open_threads": [
      {
        "id": "thread_85517a7978f35c94b95f692db9bd066b",
        "thread_type": "clarification",
        "scope_kind": "owner_local",
        "scope_id": "leo_local",
        "subject_ref": "conv_3a416da4996951c183aa7a53ce636885",
        "summary": "Unresolved replay_handoff clarification",
        "origin_event_id": "open",
        "latest_event_id": "cancel",
        "status": "ARCHIVED",
        "priority": 50,
        "created_at": 1000.0,
        "relevance_until": 1060.0,
        "valid_until": 1060.0,
        "resolved_at": 1001.0,
        "resolution_event_id": "cancel",
        "sensitivity": "normal",
        "version": 2,
        "participant_ids": [
          "leo",
          "hebe"
        ]
      }
    ],
    "memory": {
      "facts_count": 0,
      "chunks_count": 0
    },
    "beliefs": [],
    "game_state": {
      "game": "",
      "platform_version": "",
      "playthrough_type": "casual",
      "spoiler_policy": "spoiler_safe_hints",
      "current_location": null,
      "current_character": "",
      "party_members": [],
      "last_confirmed_progress": "",
      "current_objective": null,
      "challenge": "",
      "known_constraints": [],
      "last_updated": 1786437825.5728567,
      "provenance": "inferred",
      "confidence": 0.0,
      "current_game": null,
      "recent_run_context_facts": []
    },
    "social_state": {
      "recent_active_users": [],
      "recent_chat_count": 0,
      "last_raid": {},
      "last_cheer": {}
    },
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
    "database_watermarks": {
      "counts": {
        "chat_log": 0,
        "memory_facts": 0,
        "memory_chunks": 0,
        "stream_sessions": 0,
        "stream_chat_messages": 0,
        "stream_events": 0,
        "live_session_timeline": 0,
        "promotion_events": 0,
        "viewer_promotion_profiles": 0,
        "schema_migrations": 2,
        "conversations": 1,
        "open_threads": 1
      },
      "schema_migrations": [
        {
          "component": "cognitive_replay",
          "version": 1,
          "name": "replay_metadata",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "applied_at": "2026-08-11T08:43:45.565504+00:00"
        },
        {
          "component": "conversation_continuity",
          "version": 1,
          "name": "conversation_and_open_threads",
          "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
          "applied_at": "2026-08-11T08:43:45.643400+00:00"
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
      "source": "ambient_stt",
      "authority": "ambient",
      "decision": "ignore",
      "reason": "offline_stream"
    },
    "last_policy": {
      "source": null,
      "authority": null,
      "decision": null,
      "reason": null
    }
  },
  "stream_session": {
    "enabled": false,
    "is_live": false,
    "live_status_known": false,
    "active_stream_session_id": null,
    "last_transition": null,
    "title": null,
    "game": null,
    "category": null
  },
  "current_scene": {},
  "pending": {
    "clarification": null,
    "conversation_turn": null
  },
  "conversation": {
    "active": {},
    "latest": {
      "id": "conv_3a416da4996951c183aa7a53ce636885",
      "context_kind": "owner_local",
      "context_id": "leo_local",
      "attention_state": "RELEASED",
      "turn_owner": "leo",
      "expected_reply_type": "yes_no",
      "topic": "replay_handoff",
      "origin_event_id": "open",
      "last_event_id": "cancel",
      "opened_at": 1000.0,
      "last_turn_at": 1001.0,
      "expires_at": 1060.0,
      "status": "CANCELLED",
      "closure_reason": "cancelled_by_owner",
      "version": 2,
      "participants": [
        "leo",
        "hebe"
      ],
      "domain_payload": {},
      "consumed_event_ids": [
        "cancel"
      ]
    },
    "all": [
      {
        "id": "conv_3a416da4996951c183aa7a53ce636885",
        "context_kind": "owner_local",
        "context_id": "leo_local",
        "attention_state": "RELEASED",
        "turn_owner": "leo",
        "expected_reply_type": "yes_no",
        "topic": "replay_handoff",
        "origin_event_id": "open",
        "last_event_id": "cancel",
        "opened_at": 1000.0,
        "last_turn_at": 1001.0,
        "expires_at": 1060.0,
        "status": "CANCELLED",
        "closure_reason": "cancelled_by_owner",
        "version": 2,
        "participants": [
          "leo",
          "hebe"
        ],
        "domain_payload": {},
        "consumed_event_ids": [
          "cancel"
        ]
      }
    ],
    "last_resolution": {
      "consumed": true,
      "decision": "compatible_reply",
      "reason": "deterministic_cancel",
      "conversation_id": "conv_3a416da4996951c183aa7a53ce636885",
      "reply_act": "CANCEL",
      "payload": {
        "domain": {},
        "expected_reply_type": "yes_no"
      },
      "conversation": {
        "id": "conv_3a416da4996951c183aa7a53ce636885",
        "context_kind": "owner_local",
        "context_id": "leo_local",
        "participants": [
          "leo",
          "hebe"
        ],
        "attention_state": "RELEASED",
        "turn_owner": "leo",
        "expected_reply": {
          "type": "yes_no",
          "allowed_sources": [
            "owner_stt"
          ],
          "allowed_participant": "leo",
          "semantic_constraints": {},
          "candidate_refs": [],
          "expires_at": 1060.0,
          "consume_policy": "once"
        },
        "topic": "replay_handoff",
        "origin_event_id": "open",
        "last_event_id": "cancel",
        "opened_at": 1000.0,
        "last_turn_at": 1001.0,
        "expires_at": 1060.0,
        "status": "CANCELLED",
        "closure_reason": "cancelled_by_owner",
        "version": 2,
        "domain_payload": {},
        "consumed_event_ids": [
          "cancel"
        ]
      },
      "latency_ms": 9.679699956905097
    },
    "legacy_pending_projection": {},
    "continuity_shadow_diff": {
      "legacy_result": false,
      "v2_result": true,
      "match": false,
      "difference_reason": "deterministic_cancel"
    },
    "shadow_metrics": {
      "total": 1,
      "matches": 0,
      "differences": 1,
      "match_rate": 0.0,
      "difference_reasons": {
        "deterministic_cancel": 1
      }
    },
    "performance": {
      "count": 1,
      "p50_ms": 9.6797,
      "p95_ms": 9.6797
    }
  },
  "open_threads": [
    {
      "id": "thread_85517a7978f35c94b95f692db9bd066b",
      "thread_type": "clarification",
      "scope_kind": "owner_local",
      "scope_id": "leo_local",
      "subject_ref": "conv_3a416da4996951c183aa7a53ce636885",
      "summary": "Unresolved replay_handoff clarification",
      "origin_event_id": "open",
      "latest_event_id": "cancel",
      "status": "ARCHIVED",
      "priority": 50,
      "created_at": 1000.0,
      "relevance_until": 1060.0,
      "valid_until": 1060.0,
      "resolved_at": 1001.0,
      "resolution_event_id": "cancel",
      "sensitivity": "normal",
      "version": 2,
      "participant_ids": [
        "leo",
        "hebe"
      ]
    }
  ],
  "memory": {
    "facts_count": 0,
    "chunks_count": 0
  },
  "beliefs": [],
  "game_state": {
    "game": "",
    "platform_version": "",
    "playthrough_type": "casual",
    "spoiler_policy": "spoiler_safe_hints",
    "current_location": null,
    "current_character": "",
    "party_members": [],
    "last_confirmed_progress": "",
    "current_objective": null,
    "challenge": "",
    "known_constraints": [],
    "last_updated": 1786437825.5728567,
    "provenance": "inferred",
    "confidence": 0.0,
    "current_game": null,
    "recent_run_context_facts": []
  },
  "social_state": {
    "recent_active_users": [],
    "recent_chat_count": 0,
    "last_raid": {},
    "last_cheer": {}
  },
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
  "database_watermarks": {
    "counts": {
      "chat_log": 0,
      "memory_facts": 0,
      "memory_chunks": 0,
      "stream_sessions": 0,
      "stream_chat_messages": 0,
      "stream_events": 0,
      "live_session_timeline": 0,
      "promotion_events": 0,
      "viewer_promotion_profiles": 0,
      "schema_migrations": 2,
      "conversations": 1,
      "open_threads": 1
    },
    "schema_migrations": [
      {
        "component": "cognitive_replay",
        "version": 1,
        "name": "replay_metadata",
        "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
        "applied_at": "2026-08-11T08:43:45.565504+00:00"
      },
      {
        "component": "conversation_continuity",
        "version": 1,
        "name": "conversation_and_open_threads",
        "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
        "applied_at": "2026-08-11T08:43:45.643400+00:00"
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
    "C:\\Users\\Leo Nifelheim\\Documents\\Hebe\\hebe-ui\\artifacts\\cognitive-continuity-phase1\\final\\workspaces\\ambient_false_positive_foundation\\hebe-replay.sqlite3",
    "C:\\Users\\Leo Nifelheim\\Documents\\Hebe\\hebe-ui\\artifacts\\cognitive-continuity-phase1\\final\\workspaces\\consolidation_format\\hebe-replay.sqlite3",
    "C:\\Users\\Leo Nifelheim\\Documents\\Hebe\\hebe-ui\\artifacts\\cognitive-continuity-phase1\\final\\workspaces\\ffv_durable_run_format\\hebe-replay.sqlite3",
    "C:\\Users\\Leo Nifelheim\\Documents\\Hebe\\hebe-ui\\artifacts\\cognitive-continuity-phase1\\final\\workspaces\\ivanxi_resub_promo_restart\\hebe-replay.sqlite3",
    "C:\\Users\\Leo Nifelheim\\Documents\\Hebe\\hebe-ui\\artifacts\\cognitive-continuity-phase1\\final\\workspaces\\owner_correction_format\\hebe-replay.sqlite3",
    "C:\\Users\\Leo Nifelheim\\Documents\\Hebe\\hebe-ui\\artifacts\\cognitive-continuity-phase1\\final\\workspaces\\raid_transition_foundation\\hebe-replay.sqlite3",
    "C:\\Users\\Leo Nifelheim\\Documents\\Hebe\\hebe-ui\\artifacts\\cognitive-continuity-phase1\\final\\workspaces\\receipt_truth\\hebe-replay.sqlite3",
    "C:\\Users\\Leo Nifelheim\\Documents\\Hebe\\hebe-ui\\artifacts\\cognitive-continuity-phase1\\final\\workspaces\\research_fixture_foundation\\hebe-replay.sqlite3",
    "C:\\Users\\Leo Nifelheim\\Documents\\Hebe\\hebe-ui\\artifacts\\cognitive-continuity-phase1\\final\\workspaces\\temporal_social_thread_format\\hebe-replay.sqlite3",
    "C:\\Users\\Leo Nifelheim\\Documents\\Hebe\\hebe-ui\\artifacts\\cognitive-continuity-phase1\\final\\workspaces\\phase1_a_ivanxi_wake_free\\hebe-replay.sqlite3",
    "C:\\Users\\Leo Nifelheim\\Documents\\Hebe\\hebe-ui\\artifacts\\cognitive-continuity-phase1\\final\\workspaces\\phase1_b_ambient_owner\\hebe-replay.sqlite3",
    "C:\\Users\\Leo Nifelheim\\Documents\\Hebe\\hebe-ui\\artifacts\\cognitive-continuity-phase1\\final\\workspaces\\phase1_c_expired\\hebe-replay.sqlite3",
    "C:\\Users\\Leo Nifelheim\\Documents\\Hebe\\hebe-ui\\artifacts\\cognitive-continuity-phase1\\final\\workspaces\\phase1_d_interruption\\hebe-replay.sqlite3",
    "C:\\Users\\Leo Nifelheim\\Documents\\Hebe\\hebe-ui\\artifacts\\cognitive-continuity-phase1\\final\\workspaces\\phase1_e_entity_selection\\hebe-replay.sqlite3",
    "C:\\Users\\Leo Nifelheim\\Documents\\Hebe\\hebe-ui\\artifacts\\cognitive-continuity-phase1\\final\\workspaces\\phase1_f_wrong_context\\hebe-replay.sqlite3",
    "C:\\Users\\Leo Nifelheim\\Documents\\Hebe\\hebe-ui\\artifacts\\cognitive-continuity-phase1\\final\\workspaces\\phase1_g_duplicate\\hebe-replay.sqlite3",
    "C:\\Users\\Leo Nifelheim\\Documents\\Hebe\\hebe-ui\\artifacts\\cognitive-continuity-phase1\\final\\workspaces\\phase1_h_restart_stale\\hebe-replay.sqlite3",
    "C:\\Users\\Leo Nifelheim\\Documents\\Hebe\\hebe-ui\\artifacts\\cognitive-continuity-phase1\\final\\workspaces\\phase1_i_free_response\\hebe-replay.sqlite3",
    "C:\\Users\\Leo Nifelheim\\Documents\\Hebe\\hebe-ui\\artifacts\\cognitive-continuity-phase1\\final\\workspaces\\phase1_j_cancel\\hebe-replay.sqlite3"
  ],
  "restart_points": 7,
  "schema_migrations": [
    [
      {
        "component": "cognitive_replay",
        "version": 1,
        "name": "replay_metadata",
        "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
        "applied_at": "2026-08-11T08:43:13.502031+00:00"
      },
      {
        "component": "conversation_continuity",
        "version": 1,
        "name": "conversation_and_open_threads",
        "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
        "applied_at": "2026-08-11T08:43:13.615668+00:00"
      }
    ],
    [
      {
        "component": "cognitive_replay",
        "version": 1,
        "name": "replay_metadata",
        "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
        "applied_at": "2026-08-11T08:43:14.018660+00:00"
      },
      {
        "component": "conversation_continuity",
        "version": 1,
        "name": "conversation_and_open_threads",
        "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
        "applied_at": "2026-08-11T08:43:14.084294+00:00"
      }
    ],
    [
      {
        "component": "cognitive_replay",
        "version": 1,
        "name": "replay_metadata",
        "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
        "applied_at": "2026-08-11T08:43:20.140479+00:00"
      },
      {
        "component": "conversation_continuity",
        "version": 1,
        "name": "conversation_and_open_threads",
        "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
        "applied_at": "2026-08-11T08:43:20.205670+00:00"
      }
    ],
    [
      {
        "component": "cognitive_replay",
        "version": 1,
        "name": "replay_metadata",
        "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
        "applied_at": "2026-08-11T08:43:21.032363+00:00"
      },
      {
        "component": "conversation_continuity",
        "version": 1,
        "name": "conversation_and_open_threads",
        "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
        "applied_at": "2026-08-11T08:43:21.097850+00:00"
      }
    ],
    [
      {
        "component": "cognitive_replay",
        "version": 1,
        "name": "replay_metadata",
        "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
        "applied_at": "2026-08-11T08:43:21.843413+00:00"
      },
      {
        "component": "conversation_continuity",
        "version": 1,
        "name": "conversation_and_open_threads",
        "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
        "applied_at": "2026-08-11T08:43:21.928483+00:00"
      }
    ],
    [
      {
        "component": "cognitive_replay",
        "version": 1,
        "name": "replay_metadata",
        "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
        "applied_at": "2026-08-11T08:43:28.208504+00:00"
      },
      {
        "component": "conversation_continuity",
        "version": 1,
        "name": "conversation_and_open_threads",
        "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
        "applied_at": "2026-08-11T08:43:28.277977+00:00"
      }
    ],
    [
      {
        "component": "cognitive_replay",
        "version": 1,
        "name": "replay_metadata",
        "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
        "applied_at": "2026-08-11T08:43:34.167927+00:00"
      },
      {
        "component": "conversation_continuity",
        "version": 1,
        "name": "conversation_and_open_threads",
        "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
        "applied_at": "2026-08-11T08:43:34.237646+00:00"
      }
    ],
    [
      {
        "component": "cognitive_replay",
        "version": 1,
        "name": "replay_metadata",
        "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
        "applied_at": "2026-08-11T08:43:34.883619+00:00"
      },
      {
        "component": "conversation_continuity",
        "version": 1,
        "name": "conversation_and_open_threads",
        "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
        "applied_at": "2026-08-11T08:43:34.947337+00:00"
      }
    ],
    [
      {
        "component": "cognitive_replay",
        "version": 1,
        "name": "replay_metadata",
        "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
        "applied_at": "2026-08-11T08:43:35.231477+00:00"
      },
      {
        "component": "conversation_continuity",
        "version": 1,
        "name": "conversation_and_open_threads",
        "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
        "applied_at": "2026-08-11T08:43:35.330380+00:00"
      }
    ],
    [
      {
        "component": "cognitive_replay",
        "version": 1,
        "name": "replay_metadata",
        "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
        "applied_at": "2026-08-11T08:43:41.462632+00:00"
      },
      {
        "component": "conversation_continuity",
        "version": 1,
        "name": "conversation_and_open_threads",
        "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
        "applied_at": "2026-08-11T08:43:41.531488+00:00"
      }
    ],
    [
      {
        "component": "cognitive_replay",
        "version": 1,
        "name": "replay_metadata",
        "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
        "applied_at": "2026-08-11T08:43:42.350123+00:00"
      },
      {
        "component": "conversation_continuity",
        "version": 1,
        "name": "conversation_and_open_threads",
        "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
        "applied_at": "2026-08-11T08:43:42.415717+00:00"
      }
    ],
    [
      {
        "component": "cognitive_replay",
        "version": 1,
        "name": "replay_metadata",
        "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
        "applied_at": "2026-08-11T08:43:42.699121+00:00"
      },
      {
        "component": "conversation_continuity",
        "version": 1,
        "name": "conversation_and_open_threads",
        "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
        "applied_at": "2026-08-11T08:43:42.765257+00:00"
      }
    ],
    [
      {
        "component": "cognitive_replay",
        "version": 1,
        "name": "replay_metadata",
        "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
        "applied_at": "2026-08-11T08:43:43.051464+00:00"
      },
      {
        "component": "conversation_continuity",
        "version": 1,
        "name": "conversation_and_open_threads",
        "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
        "applied_at": "2026-08-11T08:43:43.120091+00:00"
      }
    ],
    [
      {
        "component": "cognitive_replay",
        "version": 1,
        "name": "replay_metadata",
        "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
        "applied_at": "2026-08-11T08:43:43.476168+00:00"
      },
      {
        "component": "conversation_continuity",
        "version": 1,
        "name": "conversation_and_open_threads",
        "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
        "applied_at": "2026-08-11T08:43:43.553023+00:00"
      }
    ],
    [
      {
        "component": "cognitive_replay",
        "version": 1,
        "name": "replay_metadata",
        "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
        "applied_at": "2026-08-11T08:43:43.844013+00:00"
      },
      {
        "component": "conversation_continuity",
        "version": 1,
        "name": "conversation_and_open_threads",
        "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
        "applied_at": "2026-08-11T08:43:43.922920+00:00"
      }
    ],
    [
      {
        "component": "cognitive_replay",
        "version": 1,
        "name": "replay_metadata",
        "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
        "applied_at": "2026-08-11T08:43:44.212972+00:00"
      },
      {
        "component": "conversation_continuity",
        "version": 1,
        "name": "conversation_and_open_threads",
        "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
        "applied_at": "2026-08-11T08:43:44.296748+00:00"
      }
    ],
    [
      {
        "component": "cognitive_replay",
        "version": 1,
        "name": "replay_metadata",
        "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
        "applied_at": "2026-08-11T08:43:44.574388+00:00"
      },
      {
        "component": "conversation_continuity",
        "version": 1,
        "name": "conversation_and_open_threads",
        "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
        "applied_at": "2026-08-11T08:43:44.641101+00:00"
      }
    ],
    [
      {
        "component": "cognitive_replay",
        "version": 1,
        "name": "replay_metadata",
        "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
        "applied_at": "2026-08-11T08:43:45.135435+00:00"
      },
      {
        "component": "conversation_continuity",
        "version": 1,
        "name": "conversation_and_open_threads",
        "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
        "applied_at": "2026-08-11T08:43:45.211938+00:00"
      }
    ],
    [
      {
        "component": "cognitive_replay",
        "version": 1,
        "name": "replay_metadata",
        "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
        "applied_at": "2026-08-11T08:43:45.565504+00:00"
      },
      {
        "component": "conversation_continuity",
        "version": 1,
        "name": "conversation_and_open_threads",
        "checksum": "687d2c78577ff15489d0c45cd188f8ad9acf98bf95f40916e53bc8af0fc01e41",
        "applied_at": "2026-08-11T08:43:45.643400+00:00"
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
  "baseline_commit": "a1e05d626c0ac2335f590b2179b6eaca6d5af4d1",
  "baseline_python": "3.11.0",
  "baseline_platform": "Windows-10-10.0.26200-SP0",
  "baseline_command": "",
  "baseline_command_duration_seconds": 0.0,
  "baseline_loader_errors": 1,
  "baseline_tests_passed": 0,
  "baseline_tests_failed": 13,
  "baseline_new_module_unavailable": "backend.tests.test_cognitive_replay",
  "phase_1_tests_passed": 0,
  "phase_1_tests_failed": 13,
  "new_regressions": 0,
  "pre_existing_failures": 13,
  "fixed_existing_failures": 0,
  "classification_counts": {
    "PASS_BOTH": 0,
    "PRE_EXISTING_FAILURE": 13,
    "NEW_PHASE_1_REGRESSION": 0,
    "FIXED_BY_PHASE_1": 0,
    "FAILURE_CHANGED": 0
  },
  "tests": [
    {
      "test": "backend.tests.test_game_knowledge.GameKnowledgeTests.test_response_synthesizer_handles_game_knowledge_command_result",
      "subsystem": "Game Knowledge response guard",
      "baseline_status": "FAIL",
      "phase_1_status": "FAIL",
      "baseline_exception_or_assertion": "AssertionError: 'Persona 5 Royal' not found in <captured value>",
      "phase_1_exception_or_assertion": "AssertionError: 'Persona 5 Royal' not found in <captured value>",
      "phase_1_code_on_failing_path": "NO: the Phase 1 production seam changes are not on this failing path",
      "classification": "PRE_EXISTING_FAILURE"
    },
    {
      "test": "backend.tests.test_voice_command_pipeline.VoiceCommandPipelineTests.test_high_value_game_tip_can_reply_without_hebe_mention",
      "subsystem": "Twitch no-mention/presence",
      "baseline_status": "FAIL",
      "phase_1_status": "FAIL",
      "baseline_exception_or_assertion": "AssertionError: 0 != 1",
      "phase_1_exception_or_assertion": "AssertionError: 0 != 1",
      "phase_1_code_on_failing_path": "YES: Phase 1 shares this path, but the baseline and current terminal assertion are identical",
      "classification": "PRE_EXISTING_FAILURE"
    },
    {
      "test": "backend.tests.test_voice_command_pipeline.VoiceCommandPipelineTests.test_no_wake_whitelisted_app_command_routes_while_stream_offline",
      "subsystem": "local application/capability resolution",
      "baseline_status": "ERROR",
      "phase_1_status": "ERROR",
      "baseline_exception_or_assertion": "IndexError: list index out of range",
      "phase_1_exception_or_assertion": "IndexError: list index out of range",
      "phase_1_code_on_failing_path": "ENTRY ONLY: the Phase 1 STT seam is on the path; the unchanged local capability resolver produces the identical failure",
      "classification": "PRE_EXISTING_FAILURE"
    },
    {
      "test": "backend.tests.test_voice_command_pipeline.VoiceCommandPipelineTests.test_obs_path_missing_returns_structured_action_result_not_generic_advice",
      "subsystem": "local application/capability resolution",
      "baseline_status": "FAIL",
      "phase_1_status": "FAIL",
      "baseline_exception_or_assertion": "AssertionError: 'HEBE_APP_OBS_PATH' not found in <captured value>",
      "phase_1_exception_or_assertion": "AssertionError: 'HEBE_APP_OBS_PATH' not found in <captured value>",
      "phase_1_code_on_failing_path": "NO: the Phase 1 production seam changes are not on this failing path",
      "classification": "PRE_EXISTING_FAILURE"
    },
    {
      "test": "backend.tests.test_voice_command_pipeline.VoiceCommandPipelineTests.test_stt_canonical_melonds_command_executes_once (transcript='Ebe, abre Melón DS')",
      "subsystem": "local application/capability resolution",
      "baseline_status": "FAIL",
      "phase_1_status": "FAIL",
      "baseline_exception_or_assertion": "AssertionError: 0 != 1",
      "phase_1_exception_or_assertion": "AssertionError: 0 != 1",
      "phase_1_code_on_failing_path": "ENTRY ONLY: the Phase 1 STT seam is on the path; the unchanged local capability resolver produces the identical failure",
      "classification": "PRE_EXISTING_FAILURE"
    },
    {
      "test": "backend.tests.test_voice_command_pipeline.VoiceCommandPipelineTests.test_stt_canonical_melonds_command_executes_once (transcript='Eve, abre Melón de Ese')",
      "subsystem": "local application/capability resolution",
      "baseline_status": "FAIL",
      "phase_1_status": "FAIL",
      "baseline_exception_or_assertion": "AssertionError: 0 != 1",
      "phase_1_exception_or_assertion": "AssertionError: 0 != 1",
      "phase_1_code_on_failing_path": "ENTRY ONLY: the Phase 1 STT seam is on the path; the unchanged local capability resolver produces the identical failure",
      "classification": "PRE_EXISTING_FAILURE"
    },
    {
      "test": "backend.tests.test_voice_command_pipeline.VoiceCommandPipelineTests.test_stt_canonical_melonds_command_executes_once (transcript='Hebe, abre melonDS')",
      "subsystem": "local application/capability resolution",
      "baseline_status": "FAIL",
      "phase_1_status": "FAIL",
      "baseline_exception_or_assertion": "AssertionError: 0 != 1",
      "phase_1_exception_or_assertion": "AssertionError: 0 != 1",
      "phase_1_code_on_failing_path": "ENTRY ONLY: the Phase 1 STT seam is on the path; the unchanged local capability resolver produces the identical failure",
      "classification": "PRE_EXISTING_FAILURE"
    },
    {
      "test": "backend.tests.test_voice_command_pipeline.VoiceCommandPipelineTests.test_stt_hebe_abre_obs_uses_same_open_application_pipeline",
      "subsystem": "local application/capability resolution",
      "baseline_status": "ERROR",
      "phase_1_status": "ERROR",
      "baseline_exception_or_assertion": "IndexError: list index out of range",
      "phase_1_exception_or_assertion": "IndexError: list index out of range",
      "phase_1_code_on_failing_path": "ENTRY ONLY: the Phase 1 STT seam is on the path; the unchanged local capability resolver produces the identical failure",
      "classification": "PRE_EXISTING_FAILURE"
    },
    {
      "test": "backend.tests.test_voice_command_pipeline.VoiceCommandPipelineTests.test_twitch_normal_no_mention_chat_reaches_presence_observe",
      "subsystem": "Twitch no-mention/presence",
      "baseline_status": "FAIL",
      "phase_1_status": "FAIL",
      "baseline_exception_or_assertion": "AssertionError: '[HEBE][TWITCH_PIPELINE_CLASSIFY] category=normal_no_mention_chat' not found in <captured value>",
      "phase_1_exception_or_assertion": "AssertionError: '[HEBE][TWITCH_PIPELINE_CLASSIFY] category=normal_no_mention_chat' not found in <captured value>",
      "phase_1_code_on_failing_path": "YES: Phase 1 shares this path, but the baseline and current terminal assertion are identical",
      "classification": "PRE_EXISTING_FAILURE"
    },
    {
      "test": "backend.tests.test_voice_command_pipeline.VoiceCommandPipelineTests.test_twitch_pipeline_health_counts_messages",
      "subsystem": "Twitch no-mention/presence",
      "baseline_status": "FAIL",
      "phase_1_status": "FAIL",
      "baseline_exception_or_assertion": "AssertionError: 0 != 1",
      "phase_1_exception_or_assertion": "AssertionError: 0 != 1",
      "phase_1_code_on_failing_path": "YES: Phase 1 shares this path, but the baseline and current terminal assertion are identical",
      "classification": "PRE_EXISTING_FAILURE"
    },
    {
      "test": "backend.tests.test_voice_command_pipeline.VoiceCommandPipelineTests.test_ui_abre_obs_creates_open_application_when_awake_and_whitelisted",
      "subsystem": "local application/capability resolution",
      "baseline_status": "ERROR",
      "phase_1_status": "ERROR",
      "baseline_exception_or_assertion": "IndexError: list index out of range",
      "phase_1_exception_or_assertion": "IndexError: list index out of range",
      "phase_1_code_on_failing_path": "NO: the Phase 1 production seam changes are not on this failing path",
      "classification": "PRE_EXISTING_FAILURE"
    },
    {
      "test": "backend.tests.test_voice_command_pipeline.VoiceCommandPipelineTests.test_ui_hebe_abre_obs_creates_open_application_action_plan",
      "subsystem": "local application/capability resolution",
      "baseline_status": "ERROR",
      "phase_1_status": "ERROR",
      "baseline_exception_or_assertion": "IndexError: list index out of range",
      "phase_1_exception_or_assertion": "IndexError: list index out of range",
      "phase_1_code_on_failing_path": "NO: the Phase 1 production seam changes are not on this failing path",
      "classification": "PRE_EXISTING_FAILURE"
    },
    {
      "test": "backend.tests.test_voice_command_pipeline.VoiceCommandPipelineTests.test_unrelated_action_during_pending_conversation_still_uses_action_flow",
      "subsystem": "local application/capability resolution",
      "baseline_status": "ERROR",
      "phase_1_status": "ERROR",
      "baseline_exception_or_assertion": "IndexError: list index out of range",
      "phase_1_exception_or_assertion": "IndexError: list index out of range",
      "phase_1_code_on_failing_path": "ENTRY ONLY: the Phase 1 STT seam is on the path; the unchanged local capability resolver produces the identical failure",
      "classification": "PRE_EXISTING_FAILURE"
    }
  ]
}
```

## Human evaluation boundary

This harness verifies cognitive/state prerequisites. Naturalness, personality, comedic timing, and social appropriateness still require human judgment.
