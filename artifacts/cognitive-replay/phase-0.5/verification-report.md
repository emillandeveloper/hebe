# Cognitive Replay Verification Report

Overall status: **VERIFIED**

## Repository and environment

- Commit: `9f91948003ec545fbf7ae045cf78b3affb3451d7`
- Working tree: `a3006bc4c10b`
- Platform: `Windows-10-10.0.26200-SP0`
- Python: `3.11.0`

## Commands

- `C:\Users\Leo Nifelheim\Documents\Hebe\hebe-ui\backend\.venv\Scripts\python.exe -m unittest backend.tests.test_cognitive_replay backend.tests.test_voice_command_pipeline backend.tests.test_cognitive_twitch backend.tests.test_stream_presence backend.tests.test_hebe_live_v1 backend.tests.test_hebe_live_v11 backend.tests.test_hebe_live_v12 backend.tests.test_hebe_live_v12_followup backend.tests.test_hebe_live_20260809_followup backend.tests.test_final_emission_gate backend.tests.test_cognitive_execution_guard backend.tests.test_game_knowledge backend.tests.test_stream_session_primer backend.tests.test_live_session_brain` → exit 1 (92.817306s)
- `python -m app.replay --suite cognitive-v2 --run-phase-tests --baseline-differential artifacts/cognitive-replay/phase-0.5/regression-differential.json --output artifacts/cognitive-replay/phase-0.5` → exit 0 (124.287116s)

## Tests

```json
{
  "failed": 13,
  "replay": {
    "duration_seconds": 30.711262,
    "expected_failures": 0,
    "expected_future_gaps": 4,
    "failed": 0,
    "passed": 5,
    "skipped": 4
  },
  "required_layer_missing": false,
  "unit_integration_regression": {
    "duration_seconds": 92.817306,
    "expected_failures": 0,
    "failed": 13,
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
    "output_digest": "77a152173895b8e0",
    "passed": 471,
    "required_layer_missing": false,
    "skipped": 0,
    "total": 484
  }
}
```

## Replay scenarios

### ambient_false_positive_foundation

- Status: **VERIFIED**
- Events: 4
- Restarts: 0
- Duration: 0.759448s
- Assertions passed/failed/skipped: 5/0/0


#### Checkpoint state

```json
{
  "ambient": {
    "actions": {
      "attempts": [],
      "model_calls": [
        {
          "key": "stream_response:v1:none",
          "method": "chat"
        }
      ],
      "research_calls": [],
      "speech_requests": []
    },
    "beliefs": [],
    "current_scene": {},
    "database_watermarks": {
      "counts": {
        "chat_log": 1,
        "live_session_timeline": 4,
        "memory_chunks": 0,
        "memory_facts": 0,
        "promotion_events": 0,
        "schema_migrations": 1,
        "stream_chat_messages": 0,
        "stream_events": 1,
        "stream_sessions": 1,
        "viewer_promotion_profiles": 0
      },
      "final_response_digest": "7e07c6a5322789f6",
      "final_response_present": true,
      "schema_migrations": [
        {
          "applied_at": "2026-08-11T07:28:25.873704+00:00",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "component": "cognitive_replay",
          "name": "replay_metadata",
          "version": 1
        }
      ]
    },
    "emitted_outputs": [
      {
        "emitted": true,
        "event_id": "1786433306.1621642",
        "reason": "",
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "text_digest": "",
        "text_present": false
      }
    ],
    "final_emission_results": [
      {
        "emitted": true,
        "event_id": "1786433306.1621642",
        "reason": "",
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "text_digest": "",
        "text_present": false
      }
    ],
    "game_state": {
      "challenge": "",
      "confidence": 0.0,
      "current_character": "",
      "current_game": null,
      "current_location": null,
      "current_objective": null,
      "game": "",
      "known_constraints": [],
      "last_confirmed_progress": "",
      "last_updated": 1786433305.8814347,
      "party_members": [],
      "platform_version": "",
      "playthrough_type": "casual",
      "provenance": "current_live_session",
      "recent_run_context_facts": [],
      "spoiler_policy": "no_spoilers"
    },
    "memory": {
      "chunks_count": 0,
      "facts_count": 0
    },
    "open_threads": [],
    "pending": {
      "clarification": null,
      "conversation_turn": null
    },
    "promotion_profiles": [],
    "receipts": [],
    "runtime": {
      "hebe_sleeping": false,
      "is_running": false,
      "last_firewall": {
        "authority": "ambient",
        "decision": "allow_context_only",
        "reason": "ambient_context_only",
        "source": "ambient_stt"
      },
      "last_input_source": null,
      "last_intent": null,
      "last_policy": {
        "authority": null,
        "decision": null,
        "reason": null,
        "source": null
      },
      "mode": "active"
    },
    "social_state": {
      "last_cheer": {},
      "last_raid": {},
      "recent_active_users": [],
      "recent_chat_count": 0
    },
    "stream_session": {
      "active_stream_session_id": 1,
      "category": null,
      "enabled": true,
      "game": null,
      "is_live": true,
      "last_transition": "online",
      "live_status_known": true,
      "title": null
    }
  },
  "owner": {
    "actions": {
      "attempts": [],
      "model_calls": [
        {
          "key": "stream_response:v1:none",
          "method": "chat"
        }
      ],
      "research_calls": [],
      "speech_requests": []
    },
    "beliefs": [],
    "current_scene": {},
    "database_watermarks": {
      "counts": {
        "chat_log": 1,
        "live_session_timeline": 4,
        "memory_chunks": 0,
        "memory_facts": 0,
        "promotion_events": 0,
        "schema_migrations": 1,
        "stream_chat_messages": 0,
        "stream_events": 1,
        "stream_sessions": 1,
        "viewer_promotion_profiles": 0
      },
      "final_response_digest": "7e07c6a5322789f6",
      "final_response_present": true,
      "schema_migrations": [
        {
          "applied_at": "2026-08-11T07:28:25.873704+00:00",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "component": "cognitive_replay",
          "name": "replay_metadata",
          "version": 1
        }
      ]
    },
    "emitted_outputs": [
      {
        "emitted": true,
        "event_id": "1786433306.1621642",
        "reason": "",
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "text_digest": "",
        "text_present": false
      }
    ],
    "final_emission_results": [
      {
        "emitted": true,
        "event_id": "1786433306.1621642",
        "reason": "",
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "text_digest": "",
        "text_present": false
      }
    ],
    "game_state": {
      "challenge": "",
      "confidence": 0.0,
      "current_character": "",
      "current_game": null,
      "current_location": null,
      "current_objective": null,
      "game": "",
      "known_constraints": [],
      "last_confirmed_progress": "",
      "last_updated": 1786433305.8814347,
      "party_members": [],
      "platform_version": "",
      "playthrough_type": "casual",
      "provenance": "current_live_session",
      "recent_run_context_facts": [],
      "spoiler_policy": "no_spoilers"
    },
    "memory": {
      "chunks_count": 0,
      "facts_count": 0
    },
    "open_threads": [],
    "pending": {
      "clarification": null,
      "conversation_turn": null
    },
    "promotion_profiles": [],
    "receipts": [],
    "runtime": {
      "hebe_sleeping": false,
      "is_running": false,
      "last_firewall": {
        "authority": "owner",
        "decision": "allow",
        "reason": "owner_direct",
        "source": "owner_stt_direct"
      },
      "last_input_source": null,
      "last_intent": null,
      "last_policy": {
        "authority": null,
        "decision": null,
        "reason": null,
        "source": null
      },
      "mode": "active"
    },
    "social_state": {
      "last_cheer": {},
      "last_raid": {},
      "recent_active_users": [],
      "recent_chat_count": 0
    },
    "stream_session": {
      "active_stream_session_id": 1,
      "category": null,
      "enabled": true,
      "game": null,
      "is_live": true,
      "last_transition": "online",
      "live_status_known": true,
      "title": null
    }
  },
  "start": {
    "actions": {
      "attempts": [],
      "model_calls": [],
      "research_calls": [],
      "speech_requests": []
    },
    "beliefs": [],
    "current_scene": {},
    "database_watermarks": {
      "counts": {
        "chat_log": 0,
        "live_session_timeline": 2,
        "memory_chunks": 0,
        "memory_facts": 0,
        "promotion_events": 0,
        "schema_migrations": 1,
        "stream_chat_messages": 0,
        "stream_events": 1,
        "stream_sessions": 1,
        "viewer_promotion_profiles": 0
      },
      "final_response_digest": "",
      "final_response_present": false,
      "schema_migrations": [
        {
          "applied_at": "2026-08-11T07:28:25.873704+00:00",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "component": "cognitive_replay",
          "name": "replay_metadata",
          "version": 1
        }
      ]
    },
    "emitted_outputs": [],
    "final_emission_results": [],
    "game_state": {
      "challenge": "",
      "confidence": 0.0,
      "current_character": "",
      "current_game": null,
      "current_location": null,
      "current_objective": null,
      "game": "",
      "known_constraints": [],
      "last_confirmed_progress": "",
      "last_updated": 1786433305.8814347,
      "party_members": [],
      "platform_version": "",
      "playthrough_type": "casual",
      "provenance": "inferred",
      "recent_run_context_facts": [],
      "spoiler_policy": "spoiler_safe_hints"
    },
    "memory": {
      "chunks_count": 0,
      "facts_count": 0
    },
    "open_threads": [],
    "pending": {
      "clarification": null,
      "conversation_turn": null
    },
    "promotion_profiles": [],
    "receipts": [],
    "runtime": {
      "hebe_sleeping": false,
      "is_running": false,
      "last_firewall": {
        "authority": null,
        "decision": null,
        "reason": null,
        "source": null
      },
      "last_input_source": null,
      "last_intent": null,
      "last_policy": {
        "authority": null,
        "decision": null,
        "reason": null,
        "source": null
      },
      "mode": "active"
    },
    "social_state": {
      "last_cheer": {},
      "last_raid": {},
      "recent_active_users": [],
      "recent_chat_count": 0
    },
    "stream_session": {
      "active_stream_session_id": 1,
      "category": null,
      "enabled": true,
      "game": null,
      "is_live": true,
      "last_transition": "online",
      "live_status_known": true,
      "title": null
    }
  },
  "time": {
    "actions": {
      "attempts": [],
      "model_calls": [
        {
          "key": "stream_response:v1:none",
          "method": "chat"
        }
      ],
      "research_calls": [],
      "speech_requests": []
    },
    "beliefs": [],
    "current_scene": {},
    "database_watermarks": {
      "counts": {
        "chat_log": 1,
        "live_session_timeline": 5,
        "memory_chunks": 0,
        "memory_facts": 0,
        "promotion_events": 0,
        "schema_migrations": 1,
        "stream_chat_messages": 0,
        "stream_events": 1,
        "stream_sessions": 1,
        "viewer_promotion_profiles": 0
      },
      "final_response_digest": "7e07c6a5322789f6",
      "final_response_present": true,
      "schema_migrations": [
        {
          "applied_at": "2026-08-11T07:28:25.873704+00:00",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "component": "cognitive_replay",
          "name": "replay_metadata",
          "version": 1
        }
      ]
    },
    "emitted_outputs": [
      {
        "emitted": true,
        "event_id": "1786433306.1621642",
        "reason": "",
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "text_digest": "",
        "text_present": false
      }
    ],
    "final_emission_results": [
      {
        "emitted": true,
        "event_id": "1786433306.1621642",
        "reason": "",
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "text_digest": "",
        "text_present": false
      }
    ],
    "game_state": {
      "challenge": "",
      "confidence": 0.0,
      "current_character": "",
      "current_game": null,
      "current_location": null,
      "current_objective": null,
      "game": "",
      "known_constraints": [],
      "last_confirmed_progress": "",
      "last_updated": 1786433305.8814347,
      "party_members": [],
      "platform_version": "",
      "playthrough_type": "casual",
      "provenance": "current_live_session",
      "recent_run_context_facts": [],
      "spoiler_policy": "no_spoilers"
    },
    "memory": {
      "chunks_count": 0,
      "facts_count": 0
    },
    "open_threads": [],
    "pending": {
      "clarification": null,
      "conversation_turn": null
    },
    "promotion_profiles": [],
    "receipts": [],
    "runtime": {
      "hebe_sleeping": false,
      "is_running": false,
      "last_firewall": {
        "authority": "ambient",
        "decision": "allow_context_only",
        "reason": "ambient_context_only",
        "source": "ambient_stt"
      },
      "last_input_source": null,
      "last_intent": null,
      "last_policy": {
        "authority": null,
        "decision": null,
        "reason": null,
        "source": null
      },
      "mode": "active"
    },
    "social_state": {
      "last_cheer": {},
      "last_raid": {},
      "recent_active_users": [],
      "recent_chat_count": 0
    },
    "stream_session": {
      "active_stream_session_id": 1,
      "category": null,
      "enabled": true,
      "game": null,
      "is_live": true,
      "last_transition": "online",
      "live_status_known": true,
      "title": null
    }
  }
}
```

#### Final state and side effects

```json
{
  "actions": {
    "attempts": [],
    "model_calls": [
      {
        "key": "stream_response:v1:none",
        "method": "chat"
      }
    ],
    "research_calls": [],
    "speech_requests": []
  },
  "beliefs": [],
  "current_scene": {},
  "database_watermarks": {
    "counts": {
      "chat_log": 1,
      "live_session_timeline": 5,
      "memory_chunks": 0,
      "memory_facts": 0,
      "promotion_events": 0,
      "schema_migrations": 1,
      "stream_chat_messages": 0,
      "stream_events": 1,
      "stream_sessions": 1,
      "viewer_promotion_profiles": 0
    },
    "final_response_digest": "7e07c6a5322789f6",
    "final_response_present": true,
    "schema_migrations": [
      {
        "applied_at": "2026-08-11T07:28:25.873704+00:00",
        "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
        "component": "cognitive_replay",
        "name": "replay_metadata",
        "version": 1
      }
    ]
  },
  "emitted_outputs": [
    {
      "emitted": true,
      "event_id": "1786433306.1621642",
      "reason": "",
      "route": "local_owner_reply",
      "targets": [
        "local_ui"
      ],
      "text_digest": "",
      "text_present": false
    }
  ],
  "final_emission_results": [
    {
      "emitted": true,
      "event_id": "1786433306.1621642",
      "reason": "",
      "route": "local_owner_reply",
      "targets": [
        "local_ui"
      ],
      "text_digest": "",
      "text_present": false
    }
  ],
  "game_state": {
    "challenge": "",
    "confidence": 0.0,
    "current_character": "",
    "current_game": null,
    "current_location": null,
    "current_objective": null,
    "game": "",
    "known_constraints": [],
    "last_confirmed_progress": "",
    "last_updated": 1786433305.8814347,
    "party_members": [],
    "platform_version": "",
    "playthrough_type": "casual",
    "provenance": "current_live_session",
    "recent_run_context_facts": [],
    "spoiler_policy": "no_spoilers"
  },
  "memory": {
    "chunks_count": 0,
    "facts_count": 0
  },
  "open_threads": [],
  "pending": {
    "clarification": null,
    "conversation_turn": null
  },
  "promotion_profiles": [],
  "receipts": [],
  "runtime": {
    "hebe_sleeping": false,
    "is_running": false,
    "last_firewall": {
      "authority": "ambient",
      "decision": "allow_context_only",
      "reason": "ambient_context_only",
      "source": "ambient_stt"
    },
    "last_input_source": null,
    "last_intent": null,
    "last_policy": {
      "authority": null,
      "decision": null,
      "reason": null,
      "source": null
    },
    "mode": "active"
  },
  "social_state": {
    "last_cheer": {},
    "last_raid": {},
    "recent_active_users": [],
    "recent_chat_count": 0
  },
  "stream_session": {
    "active_stream_session_id": 1,
    "category": null,
    "enabled": true,
    "game": null,
    "is_live": true,
    "last_transition": "online",
    "live_status_known": true,
    "title": null
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
- Duration: 6.327205s
- Assertions passed/failed/skipped: 0/0/1


#### Checkpoint state

```json
{
  "chat": {
    "actions": {
      "attempts": [],
      "model_calls": [],
      "research_calls": [],
      "speech_requests": []
    },
    "beliefs": [],
    "current_scene": {},
    "database_watermarks": {
      "counts": {
        "chat_log": 0,
        "live_session_timeline": 3,
        "memory_chunks": 0,
        "memory_facts": 0,
        "promotion_events": 0,
        "schema_migrations": 1,
        "stream_chat_messages": 1,
        "stream_events": 1,
        "stream_sessions": 1,
        "viewer_promotion_profiles": 0
      },
      "final_response_digest": "",
      "final_response_present": false,
      "schema_migrations": [
        {
          "applied_at": "2026-08-11T07:28:26.700985+00:00",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "component": "cognitive_replay",
          "name": "replay_metadata",
          "version": 1
        }
      ]
    },
    "emitted_outputs": [],
    "final_emission_results": [],
    "game_state": {
      "challenge": "",
      "confidence": 0.0,
      "current_character": "",
      "current_game": null,
      "current_location": null,
      "current_objective": null,
      "game": "",
      "known_constraints": [],
      "last_confirmed_progress": "",
      "last_updated": 1786433306.7116303,
      "party_members": [],
      "platform_version": "",
      "playthrough_type": "casual",
      "provenance": "inferred",
      "recent_run_context_facts": [],
      "spoiler_policy": "spoiler_safe_hints"
    },
    "memory": {
      "chunks_count": 0,
      "facts_count": 0
    },
    "open_threads": [],
    "pending": {
      "clarification": null,
      "conversation_turn": null
    },
    "promotion_profiles": [],
    "receipts": [],
    "runtime": {
      "hebe_sleeping": false,
      "is_running": false,
      "last_firewall": {
        "authority": "viewer",
        "decision": "allow",
        "reason": "live_viewer_message",
        "source": "twitch_viewer"
      },
      "last_input_source": null,
      "last_intent": null,
      "last_policy": {
        "authority": null,
        "decision": null,
        "reason": null,
        "source": null
      },
      "mode": "active"
    },
    "social_state": {
      "last_cheer": {},
      "last_raid": {},
      "recent_active_users": [
        "synthetic_chatter"
      ],
      "recent_chat_count": 1
    },
    "stream_session": {
      "active_stream_session_id": 1,
      "category": null,
      "enabled": true,
      "game": null,
      "is_live": true,
      "last_transition": "online",
      "live_status_known": true,
      "title": null
    }
  },
  "end": {
    "actions": {
      "attempts": [],
      "model_calls": [],
      "research_calls": [],
      "speech_requests": []
    },
    "beliefs": [],
    "current_scene": {},
    "database_watermarks": {
      "counts": {
        "chat_log": 0,
        "live_session_timeline": 4,
        "memory_chunks": 1,
        "memory_facts": 0,
        "promotion_events": 0,
        "schema_migrations": 1,
        "stream_chat_messages": 1,
        "stream_events": 2,
        "stream_sessions": 1,
        "viewer_promotion_profiles": 0
      },
      "final_response_digest": "",
      "final_response_present": false,
      "schema_migrations": [
        {
          "applied_at": "2026-08-11T07:28:26.700985+00:00",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "component": "cognitive_replay",
          "name": "replay_metadata",
          "version": 1
        }
      ]
    },
    "emitted_outputs": [],
    "final_emission_results": [],
    "game_state": {
      "challenge": "",
      "confidence": 0.0,
      "current_character": "",
      "current_game": null,
      "current_location": null,
      "current_objective": null,
      "game": "",
      "known_constraints": [],
      "last_confirmed_progress": "",
      "last_updated": 1786433306.7116303,
      "party_members": [],
      "platform_version": "",
      "playthrough_type": "casual",
      "provenance": "inferred",
      "recent_run_context_facts": [],
      "spoiler_policy": "spoiler_safe_hints"
    },
    "memory": {
      "chunks_count": 1,
      "facts_count": 0
    },
    "open_threads": [],
    "pending": {
      "clarification": null,
      "conversation_turn": null
    },
    "promotion_profiles": [],
    "receipts": [],
    "runtime": {
      "hebe_sleeping": false,
      "is_running": false,
      "last_firewall": {
        "authority": "viewer",
        "decision": "allow",
        "reason": "live_viewer_message",
        "source": "twitch_viewer"
      },
      "last_input_source": null,
      "last_intent": null,
      "last_policy": {
        "authority": null,
        "decision": null,
        "reason": null,
        "source": null
      },
      "mode": "active"
    },
    "social_state": {
      "last_cheer": {},
      "last_raid": {},
      "recent_active_users": [
        "synthetic_chatter"
      ],
      "recent_chat_count": 1
    },
    "stream_session": {
      "active_stream_session_id": null,
      "category": null,
      "enabled": true,
      "game": null,
      "is_live": false,
      "last_transition": "offline",
      "live_status_known": true,
      "title": null
    }
  },
  "maintenance": {
    "actions": {
      "attempts": [],
      "model_calls": [],
      "research_calls": [],
      "speech_requests": []
    },
    "beliefs": [],
    "current_scene": {},
    "database_watermarks": {
      "counts": {
        "chat_log": 0,
        "live_session_timeline": 5,
        "memory_chunks": 1,
        "memory_facts": 0,
        "promotion_events": 0,
        "schema_migrations": 1,
        "stream_chat_messages": 1,
        "stream_events": 2,
        "stream_sessions": 1,
        "viewer_promotion_profiles": 0
      },
      "final_response_digest": "",
      "final_response_present": false,
      "schema_migrations": [
        {
          "applied_at": "2026-08-11T07:28:26.700985+00:00",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "component": "cognitive_replay",
          "name": "replay_metadata",
          "version": 1
        }
      ]
    },
    "emitted_outputs": [],
    "final_emission_results": [],
    "game_state": {
      "challenge": "",
      "confidence": 0.0,
      "current_character": "",
      "current_game": null,
      "current_location": null,
      "current_objective": null,
      "game": "",
      "known_constraints": [],
      "last_confirmed_progress": "",
      "last_updated": 1786433312.648548,
      "party_members": [],
      "platform_version": "",
      "playthrough_type": "casual",
      "provenance": "inferred",
      "recent_run_context_facts": [],
      "spoiler_policy": "spoiler_safe_hints"
    },
    "memory": {
      "chunks_count": 1,
      "facts_count": 0
    },
    "open_threads": [],
    "pending": {
      "clarification": null,
      "conversation_turn": null
    },
    "promotion_profiles": [],
    "receipts": [],
    "runtime": {
      "hebe_sleeping": false,
      "is_running": false,
      "last_firewall": {
        "authority": null,
        "decision": null,
        "reason": null,
        "source": null
      },
      "last_input_source": null,
      "last_intent": null,
      "last_policy": {
        "authority": null,
        "decision": null,
        "reason": null,
        "source": null
      },
      "mode": "active"
    },
    "social_state": {
      "last_cheer": {},
      "last_raid": {},
      "recent_active_users": [],
      "recent_chat_count": 0
    },
    "stream_session": {
      "active_stream_session_id": null,
      "category": null,
      "enabled": false,
      "game": null,
      "is_live": false,
      "last_transition": null,
      "live_status_known": true,
      "title": null
    }
  },
  "restart": {
    "actions": {
      "attempts": [],
      "model_calls": [],
      "research_calls": [],
      "speech_requests": []
    },
    "beliefs": [],
    "current_scene": {},
    "database_watermarks": {
      "counts": {
        "chat_log": 0,
        "live_session_timeline": 4,
        "memory_chunks": 1,
        "memory_facts": 0,
        "promotion_events": 0,
        "schema_migrations": 1,
        "stream_chat_messages": 1,
        "stream_events": 2,
        "stream_sessions": 1,
        "viewer_promotion_profiles": 0
      },
      "final_response_digest": "",
      "final_response_present": false,
      "schema_migrations": [
        {
          "applied_at": "2026-08-11T07:28:26.700985+00:00",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "component": "cognitive_replay",
          "name": "replay_metadata",
          "version": 1
        }
      ]
    },
    "emitted_outputs": [],
    "final_emission_results": [],
    "game_state": {
      "challenge": "",
      "confidence": 0.0,
      "current_character": "",
      "current_game": null,
      "current_location": null,
      "current_objective": null,
      "game": "",
      "known_constraints": [],
      "last_confirmed_progress": "",
      "last_updated": 1786433312.648548,
      "party_members": [],
      "platform_version": "",
      "playthrough_type": "casual",
      "provenance": "inferred",
      "recent_run_context_facts": [],
      "spoiler_policy": "spoiler_safe_hints"
    },
    "memory": {
      "chunks_count": 1,
      "facts_count": 0
    },
    "open_threads": [],
    "pending": {
      "clarification": null,
      "conversation_turn": null
    },
    "promotion_profiles": [],
    "receipts": [],
    "runtime": {
      "hebe_sleeping": false,
      "is_running": false,
      "last_firewall": {
        "authority": null,
        "decision": null,
        "reason": null,
        "source": null
      },
      "last_input_source": null,
      "last_intent": null,
      "last_policy": {
        "authority": null,
        "decision": null,
        "reason": null,
        "source": null
      },
      "mode": "active"
    },
    "social_state": {
      "last_cheer": {},
      "last_raid": {},
      "recent_active_users": [],
      "recent_chat_count": 0
    },
    "stream_session": {
      "active_stream_session_id": null,
      "category": null,
      "enabled": false,
      "game": null,
      "is_live": false,
      "last_transition": null,
      "live_status_known": false,
      "title": null
    }
  },
  "start": {
    "actions": {
      "attempts": [],
      "model_calls": [],
      "research_calls": [],
      "speech_requests": []
    },
    "beliefs": [],
    "current_scene": {},
    "database_watermarks": {
      "counts": {
        "chat_log": 0,
        "live_session_timeline": 2,
        "memory_chunks": 0,
        "memory_facts": 0,
        "promotion_events": 0,
        "schema_migrations": 1,
        "stream_chat_messages": 0,
        "stream_events": 1,
        "stream_sessions": 1,
        "viewer_promotion_profiles": 0
      },
      "final_response_digest": "",
      "final_response_present": false,
      "schema_migrations": [
        {
          "applied_at": "2026-08-11T07:28:26.700985+00:00",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "component": "cognitive_replay",
          "name": "replay_metadata",
          "version": 1
        }
      ]
    },
    "emitted_outputs": [],
    "final_emission_results": [],
    "game_state": {
      "challenge": "",
      "confidence": 0.0,
      "current_character": "",
      "current_game": null,
      "current_location": null,
      "current_objective": null,
      "game": "",
      "known_constraints": [],
      "last_confirmed_progress": "",
      "last_updated": 1786433306.7116303,
      "party_members": [],
      "platform_version": "",
      "playthrough_type": "casual",
      "provenance": "inferred",
      "recent_run_context_facts": [],
      "spoiler_policy": "spoiler_safe_hints"
    },
    "memory": {
      "chunks_count": 0,
      "facts_count": 0
    },
    "open_threads": [],
    "pending": {
      "clarification": null,
      "conversation_turn": null
    },
    "promotion_profiles": [],
    "receipts": [],
    "runtime": {
      "hebe_sleeping": false,
      "is_running": false,
      "last_firewall": {
        "authority": null,
        "decision": null,
        "reason": null,
        "source": null
      },
      "last_input_source": null,
      "last_intent": null,
      "last_policy": {
        "authority": null,
        "decision": null,
        "reason": null,
        "source": null
      },
      "mode": "active"
    },
    "social_state": {
      "last_cheer": {},
      "last_raid": {},
      "recent_active_users": [],
      "recent_chat_count": 0
    },
    "stream_session": {
      "active_stream_session_id": 1,
      "category": null,
      "enabled": true,
      "game": null,
      "is_live": true,
      "last_transition": "online",
      "live_status_known": true,
      "title": null
    }
  }
}
```

#### Final state and side effects

```json
{
  "actions": {
    "attempts": [],
    "model_calls": [],
    "research_calls": [],
    "speech_requests": []
  },
  "beliefs": [],
  "current_scene": {},
  "database_watermarks": {
    "counts": {
      "chat_log": 0,
      "live_session_timeline": 5,
      "memory_chunks": 1,
      "memory_facts": 0,
      "promotion_events": 0,
      "schema_migrations": 1,
      "stream_chat_messages": 1,
      "stream_events": 2,
      "stream_sessions": 1,
      "viewer_promotion_profiles": 0
    },
    "final_response_digest": "",
    "final_response_present": false,
    "schema_migrations": [
      {
        "applied_at": "2026-08-11T07:28:26.700985+00:00",
        "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
        "component": "cognitive_replay",
        "name": "replay_metadata",
        "version": 1
      }
    ]
  },
  "emitted_outputs": [],
  "final_emission_results": [],
  "game_state": {
    "challenge": "",
    "confidence": 0.0,
    "current_character": "",
    "current_game": null,
    "current_location": null,
    "current_objective": null,
    "game": "",
    "known_constraints": [],
    "last_confirmed_progress": "",
    "last_updated": 1786433312.648548,
    "party_members": [],
    "platform_version": "",
    "playthrough_type": "casual",
    "provenance": "inferred",
    "recent_run_context_facts": [],
    "spoiler_policy": "spoiler_safe_hints"
  },
  "memory": {
    "chunks_count": 1,
    "facts_count": 0
  },
  "open_threads": [],
  "pending": {
    "clarification": null,
    "conversation_turn": null
  },
  "promotion_profiles": [],
  "receipts": [],
  "runtime": {
    "hebe_sleeping": false,
    "is_running": false,
    "last_firewall": {
      "authority": null,
      "decision": null,
      "reason": null,
      "source": null
    },
    "last_input_source": null,
    "last_intent": null,
    "last_policy": {
      "authority": null,
      "decision": null,
      "reason": null,
      "source": null
    },
    "mode": "active"
  },
  "social_state": {
    "last_cheer": {},
    "last_raid": {},
    "recent_active_users": [],
    "recent_chat_count": 0
  },
  "stream_session": {
    "active_stream_session_id": null,
    "category": null,
    "enabled": false,
    "game": null,
    "is_live": false,
    "last_transition": null,
    "live_status_known": true,
    "title": null
  }
}
```

#### Restart evidence

```json
[
  {
    "after_persisted_counts": {
      "chat_log": 0,
      "live_session_timeline": 4,
      "memory_chunks": 1,
      "memory_facts": 0,
      "promotion_events": 0,
      "schema_migrations": 1,
      "stream_chat_messages": 1,
      "stream_events": 2,
      "stream_sessions": 1,
      "viewer_promotion_profiles": 0
    },
    "before_persisted_counts": {
      "chat_log": 0,
      "live_session_timeline": 4,
      "memory_chunks": 1,
      "memory_facts": 0,
      "promotion_events": 0,
      "schema_migrations": 1,
      "stream_chat_messages": 1,
      "stream_events": 2,
      "stream_sessions": 1,
      "viewer_promotion_profiles": 0
    },
    "event_id": "restart",
    "new_engine_id": 2823652612496,
    "old_engine_collected": true,
    "old_engine_id": 2823652955152,
    "same_database": "C:\\Users\\Leo Nifelheim\\Documents\\Hebe\\hebe-ui\\artifacts\\cognitive-replay\\phase-0.5\\workspaces\\consolidation_format\\hebe-replay.sqlite3",
    "volatile_state_recreated": true
  }
]
```

### ffv_durable_run_format

- Status: **VERIFICATION_INCOMPLETE**
- Events: 7
- Restarts: 1
- Duration: 1.315635s
- Assertions passed/failed/skipped: 0/0/1


#### Checkpoint state

```json
{
  "chat": {
    "actions": {
      "attempts": [],
      "model_calls": [
        {
          "key": "stream_response:v1:none",
          "method": "chat"
        }
      ],
      "research_calls": [],
      "speech_requests": []
    },
    "beliefs": [],
    "current_scene": {},
    "database_watermarks": {
      "counts": {
        "chat_log": 1,
        "live_session_timeline": 5,
        "memory_chunks": 0,
        "memory_facts": 0,
        "promotion_events": 0,
        "schema_migrations": 1,
        "stream_chat_messages": 1,
        "stream_events": 1,
        "stream_sessions": 1,
        "viewer_promotion_profiles": 0
      },
      "final_response_digest": "",
      "final_response_present": false,
      "schema_migrations": [
        {
          "applied_at": "2026-08-11T07:28:33.114708+00:00",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "component": "cognitive_replay",
          "name": "replay_metadata",
          "version": 1
        }
      ]
    },
    "emitted_outputs": [
      {
        "emitted": true,
        "event_id": "1786433313.4060206",
        "reason": "",
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "text_digest": "",
        "text_present": false
      }
    ],
    "final_emission_results": [
      {
        "emitted": true,
        "event_id": "1786433313.4060206",
        "reason": "",
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "text_digest": "",
        "text_present": false
      }
    ],
    "game_state": {
      "challenge": "",
      "confidence": 0.75,
      "current_character": "",
      "current_game": "Final Fantasy V",
      "current_location": null,
      "current_objective": null,
      "game": "Final Fantasy V",
      "known_constraints": [],
      "last_confirmed_progress": "",
      "last_updated": 1786464000.0,
      "party_members": [],
      "platform_version": "",
      "playthrough_type": "casual",
      "provenance": "stream_context_sync",
      "recent_run_context_facts": [],
      "spoiler_policy": "no_spoilers"
    },
    "memory": {
      "chunks_count": 0,
      "facts_count": 0
    },
    "open_threads": [],
    "pending": {
      "clarification": null,
      "conversation_turn": null
    },
    "promotion_profiles": [],
    "receipts": [],
    "runtime": {
      "hebe_sleeping": false,
      "is_running": false,
      "last_firewall": {
        "authority": "viewer",
        "decision": "allow",
        "reason": "live_viewer_message",
        "source": "twitch_viewer"
      },
      "last_input_source": null,
      "last_intent": null,
      "last_policy": {
        "authority": null,
        "decision": null,
        "reason": null,
        "source": null
      },
      "mode": "active"
    },
    "social_state": {
      "last_cheer": {},
      "last_raid": {},
      "recent_active_users": [
        "synthetic_viewer"
      ],
      "recent_chat_count": 1
    },
    "stream_session": {
      "active_stream_session_id": 1,
      "category": "Final Fantasy V",
      "enabled": true,
      "game": "Final Fantasy V",
      "is_live": true,
      "last_transition": "online",
      "live_status_known": true,
      "title": "Crystal Roulette"
    }
  },
  "end-1": {
    "actions": {
      "attempts": [],
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
      "research_calls": [],
      "speech_requests": []
    },
    "beliefs": [],
    "current_scene": {},
    "database_watermarks": {
      "counts": {
        "chat_log": 2,
        "live_session_timeline": 8,
        "memory_chunks": 1,
        "memory_facts": 0,
        "promotion_events": 0,
        "schema_migrations": 1,
        "stream_chat_messages": 1,
        "stream_events": 2,
        "stream_sessions": 1,
        "viewer_promotion_profiles": 0
      },
      "final_response_digest": "7e07c6a5322789f6",
      "final_response_present": true,
      "schema_migrations": [
        {
          "applied_at": "2026-08-11T07:28:33.114708+00:00",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "component": "cognitive_replay",
          "name": "replay_metadata",
          "version": 1
        }
      ]
    },
    "emitted_outputs": [
      {
        "emitted": true,
        "event_id": "1786433313.4060206",
        "reason": "",
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "text_digest": "",
        "text_present": false
      },
      {
        "emitted": true,
        "event_id": "1786433313.6396518",
        "reason": "",
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "text_digest": "",
        "text_present": false
      }
    ],
    "final_emission_results": [
      {
        "emitted": true,
        "event_id": "1786433313.4060206",
        "reason": "",
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "text_digest": "",
        "text_present": false
      },
      {
        "emitted": true,
        "event_id": "1786433313.6396518",
        "reason": "",
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "text_digest": "",
        "text_present": false
      }
    ],
    "game_state": {
      "challenge": "",
      "confidence": 0.75,
      "current_character": "",
      "current_game": "Final Fantasy V",
      "current_location": null,
      "current_objective": null,
      "game": "Final Fantasy V",
      "known_constraints": [],
      "last_confirmed_progress": "",
      "last_updated": 1786464000.0,
      "party_members": [],
      "platform_version": "",
      "playthrough_type": "casual",
      "provenance": "stream_context_sync",
      "recent_run_context_facts": [],
      "spoiler_policy": "no_spoilers"
    },
    "memory": {
      "chunks_count": 1,
      "facts_count": 0
    },
    "open_threads": [],
    "pending": {
      "clarification": null,
      "conversation_turn": null
    },
    "promotion_profiles": [],
    "receipts": [],
    "runtime": {
      "hebe_sleeping": false,
      "is_running": false,
      "last_firewall": {
        "authority": "owner",
        "decision": "allow",
        "reason": "owner_direct",
        "source": "owner_stt_direct"
      },
      "last_input_source": null,
      "last_intent": null,
      "last_policy": {
        "authority": null,
        "decision": null,
        "reason": null,
        "source": null
      },
      "mode": "active"
    },
    "social_state": {
      "last_cheer": {},
      "last_raid": {},
      "recent_active_users": [
        "synthetic_viewer"
      ],
      "recent_chat_count": 1
    },
    "stream_session": {
      "active_stream_session_id": null,
      "category": "Final Fantasy V",
      "enabled": true,
      "game": "Final Fantasy V",
      "is_live": false,
      "last_transition": "offline",
      "live_status_known": true,
      "title": "Crystal Roulette"
    }
  },
  "reference": {
    "actions": {
      "attempts": [],
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
      "research_calls": [],
      "speech_requests": []
    },
    "beliefs": [],
    "current_scene": {},
    "database_watermarks": {
      "counts": {
        "chat_log": 2,
        "live_session_timeline": 7,
        "memory_chunks": 0,
        "memory_facts": 0,
        "promotion_events": 0,
        "schema_migrations": 1,
        "stream_chat_messages": 1,
        "stream_events": 1,
        "stream_sessions": 1,
        "viewer_promotion_profiles": 0
      },
      "final_response_digest": "7e07c6a5322789f6",
      "final_response_present": true,
      "schema_migrations": [
        {
          "applied_at": "2026-08-11T07:28:33.114708+00:00",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "component": "cognitive_replay",
          "name": "replay_metadata",
          "version": 1
        }
      ]
    },
    "emitted_outputs": [
      {
        "emitted": true,
        "event_id": "1786433313.4060206",
        "reason": "",
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "text_digest": "",
        "text_present": false
      },
      {
        "emitted": true,
        "event_id": "1786433313.6396518",
        "reason": "",
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "text_digest": "",
        "text_present": false
      }
    ],
    "final_emission_results": [
      {
        "emitted": true,
        "event_id": "1786433313.4060206",
        "reason": "",
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "text_digest": "",
        "text_present": false
      },
      {
        "emitted": true,
        "event_id": "1786433313.6396518",
        "reason": "",
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "text_digest": "",
        "text_present": false
      }
    ],
    "game_state": {
      "challenge": "",
      "confidence": 0.75,
      "current_character": "",
      "current_game": "Final Fantasy V",
      "current_location": null,
      "current_objective": null,
      "game": "Final Fantasy V",
      "known_constraints": [],
      "last_confirmed_progress": "",
      "last_updated": 1786464000.0,
      "party_members": [],
      "platform_version": "",
      "playthrough_type": "casual",
      "provenance": "stream_context_sync",
      "recent_run_context_facts": [],
      "spoiler_policy": "no_spoilers"
    },
    "memory": {
      "chunks_count": 0,
      "facts_count": 0
    },
    "open_threads": [],
    "pending": {
      "clarification": null,
      "conversation_turn": null
    },
    "promotion_profiles": [],
    "receipts": [],
    "runtime": {
      "hebe_sleeping": false,
      "is_running": false,
      "last_firewall": {
        "authority": "owner",
        "decision": "allow",
        "reason": "owner_direct",
        "source": "owner_stt_direct"
      },
      "last_input_source": null,
      "last_intent": null,
      "last_policy": {
        "authority": null,
        "decision": null,
        "reason": null,
        "source": null
      },
      "mode": "active"
    },
    "social_state": {
      "last_cheer": {},
      "last_raid": {},
      "recent_active_users": [
        "synthetic_viewer"
      ],
      "recent_chat_count": 1
    },
    "stream_session": {
      "active_stream_session_id": 1,
      "category": "Final Fantasy V",
      "enabled": true,
      "game": "Final Fantasy V",
      "is_live": true,
      "last_transition": "online",
      "live_status_known": true,
      "title": "Crystal Roulette"
    }
  },
  "restart": {
    "actions": {
      "attempts": [],
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
      "research_calls": [],
      "speech_requests": []
    },
    "beliefs": [],
    "current_scene": {},
    "database_watermarks": {
      "counts": {
        "chat_log": 2,
        "live_session_timeline": 8,
        "memory_chunks": 1,
        "memory_facts": 0,
        "promotion_events": 0,
        "schema_migrations": 1,
        "stream_chat_messages": 1,
        "stream_events": 2,
        "stream_sessions": 1,
        "viewer_promotion_profiles": 0
      },
      "final_response_digest": "",
      "final_response_present": false,
      "schema_migrations": [
        {
          "applied_at": "2026-08-11T07:28:33.114708+00:00",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "component": "cognitive_replay",
          "name": "replay_metadata",
          "version": 1
        }
      ]
    },
    "emitted_outputs": [
      {
        "emitted": true,
        "event_id": "1786433313.4060206",
        "reason": "",
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "text_digest": "",
        "text_present": false
      },
      {
        "emitted": true,
        "event_id": "1786433313.6396518",
        "reason": "",
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "text_digest": "",
        "text_present": false
      }
    ],
    "final_emission_results": [
      {
        "emitted": true,
        "event_id": "1786433313.4060206",
        "reason": "",
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "text_digest": "",
        "text_present": false
      },
      {
        "emitted": true,
        "event_id": "1786433313.6396518",
        "reason": "",
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "text_digest": "",
        "text_present": false
      }
    ],
    "game_state": {
      "challenge": "",
      "confidence": 0.0,
      "current_character": "",
      "current_game": null,
      "current_location": null,
      "current_objective": null,
      "game": "",
      "known_constraints": [],
      "last_confirmed_progress": "",
      "last_updated": 1786433313.961805,
      "party_members": [],
      "platform_version": "",
      "playthrough_type": "casual",
      "provenance": "inferred",
      "recent_run_context_facts": [],
      "spoiler_policy": "spoiler_safe_hints"
    },
    "memory": {
      "chunks_count": 1,
      "facts_count": 0
    },
    "open_threads": [],
    "pending": {
      "clarification": null,
      "conversation_turn": null
    },
    "promotion_profiles": [],
    "receipts": [],
    "runtime": {
      "hebe_sleeping": false,
      "is_running": false,
      "last_firewall": {
        "authority": null,
        "decision": null,
        "reason": null,
        "source": null
      },
      "last_input_source": null,
      "last_intent": null,
      "last_policy": {
        "authority": null,
        "decision": null,
        "reason": null,
        "source": null
      },
      "mode": "active"
    },
    "social_state": {
      "last_cheer": {},
      "last_raid": {},
      "recent_active_users": [],
      "recent_chat_count": 0
    },
    "stream_session": {
      "active_stream_session_id": null,
      "category": null,
      "enabled": false,
      "game": null,
      "is_live": false,
      "last_transition": null,
      "live_status_known": false,
      "title": null
    }
  },
  "roll": {
    "actions": {
      "attempts": [],
      "model_calls": [
        {
          "key": "stream_response:v1:none",
          "method": "chat"
        }
      ],
      "research_calls": [],
      "speech_requests": []
    },
    "beliefs": [],
    "current_scene": {},
    "database_watermarks": {
      "counts": {
        "chat_log": 1,
        "live_session_timeline": 4,
        "memory_chunks": 0,
        "memory_facts": 0,
        "promotion_events": 0,
        "schema_migrations": 1,
        "stream_chat_messages": 0,
        "stream_events": 1,
        "stream_sessions": 1,
        "viewer_promotion_profiles": 0
      },
      "final_response_digest": "7e07c6a5322789f6",
      "final_response_present": true,
      "schema_migrations": [
        {
          "applied_at": "2026-08-11T07:28:33.114708+00:00",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "component": "cognitive_replay",
          "name": "replay_metadata",
          "version": 1
        }
      ]
    },
    "emitted_outputs": [
      {
        "emitted": true,
        "event_id": "1786433313.4060206",
        "reason": "",
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "text_digest": "",
        "text_present": false
      }
    ],
    "final_emission_results": [
      {
        "emitted": true,
        "event_id": "1786433313.4060206",
        "reason": "",
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "text_digest": "",
        "text_present": false
      }
    ],
    "game_state": {
      "challenge": "",
      "confidence": 0.75,
      "current_character": "",
      "current_game": "Final Fantasy V",
      "current_location": null,
      "current_objective": null,
      "game": "Final Fantasy V",
      "known_constraints": [],
      "last_confirmed_progress": "",
      "last_updated": 1786464000.0,
      "party_members": [],
      "platform_version": "",
      "playthrough_type": "casual",
      "provenance": "stream_context_sync",
      "recent_run_context_facts": [],
      "spoiler_policy": "no_spoilers"
    },
    "memory": {
      "chunks_count": 0,
      "facts_count": 0
    },
    "open_threads": [],
    "pending": {
      "clarification": null,
      "conversation_turn": null
    },
    "promotion_profiles": [],
    "receipts": [],
    "runtime": {
      "hebe_sleeping": false,
      "is_running": false,
      "last_firewall": {
        "authority": "owner",
        "decision": "allow",
        "reason": "owner_direct",
        "source": "owner_stt_direct"
      },
      "last_input_source": null,
      "last_intent": null,
      "last_policy": {
        "authority": null,
        "decision": null,
        "reason": null,
        "source": null
      },
      "mode": "active"
    },
    "social_state": {
      "last_cheer": {},
      "last_raid": {},
      "recent_active_users": [],
      "recent_chat_count": 0
    },
    "stream_session": {
      "active_stream_session_id": 1,
      "category": "Final Fantasy V",
      "enabled": true,
      "game": "Final Fantasy V",
      "is_live": true,
      "last_transition": "online",
      "live_status_known": true,
      "title": "Crystal Roulette"
    }
  },
  "start-1": {
    "actions": {
      "attempts": [],
      "model_calls": [],
      "research_calls": [],
      "speech_requests": []
    },
    "beliefs": [],
    "current_scene": {},
    "database_watermarks": {
      "counts": {
        "chat_log": 0,
        "live_session_timeline": 2,
        "memory_chunks": 0,
        "memory_facts": 0,
        "promotion_events": 0,
        "schema_migrations": 1,
        "stream_chat_messages": 0,
        "stream_events": 1,
        "stream_sessions": 1,
        "viewer_promotion_profiles": 0
      },
      "final_response_digest": "",
      "final_response_present": false,
      "schema_migrations": [
        {
          "applied_at": "2026-08-11T07:28:33.114708+00:00",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "component": "cognitive_replay",
          "name": "replay_metadata",
          "version": 1
        }
      ]
    },
    "emitted_outputs": [],
    "final_emission_results": [],
    "game_state": {
      "challenge": "",
      "confidence": 0.75,
      "current_character": "",
      "current_game": "Final Fantasy V",
      "current_location": null,
      "current_objective": null,
      "game": "Final Fantasy V",
      "known_constraints": [],
      "last_confirmed_progress": "",
      "last_updated": 1786464000.0,
      "party_members": [],
      "platform_version": "",
      "playthrough_type": "casual",
      "provenance": "stream_context_sync",
      "recent_run_context_facts": [],
      "spoiler_policy": "no_spoilers"
    },
    "memory": {
      "chunks_count": 0,
      "facts_count": 0
    },
    "open_threads": [],
    "pending": {
      "clarification": null,
      "conversation_turn": null
    },
    "promotion_profiles": [],
    "receipts": [],
    "runtime": {
      "hebe_sleeping": false,
      "is_running": false,
      "last_firewall": {
        "authority": null,
        "decision": null,
        "reason": null,
        "source": null
      },
      "last_input_source": null,
      "last_intent": null,
      "last_policy": {
        "authority": null,
        "decision": null,
        "reason": null,
        "source": null
      },
      "mode": "active"
    },
    "social_state": {
      "last_cheer": {},
      "last_raid": {},
      "recent_active_users": [],
      "recent_chat_count": 0
    },
    "stream_session": {
      "active_stream_session_id": 1,
      "category": "Final Fantasy V",
      "enabled": true,
      "game": "Final Fantasy V",
      "is_live": true,
      "last_transition": "online",
      "live_status_known": true,
      "title": "Crystal Roulette"
    }
  },
  "start-2": {
    "actions": {
      "attempts": [],
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
      "research_calls": [],
      "speech_requests": []
    },
    "beliefs": [],
    "current_scene": {},
    "database_watermarks": {
      "counts": {
        "chat_log": 2,
        "live_session_timeline": 10,
        "memory_chunks": 1,
        "memory_facts": 0,
        "promotion_events": 0,
        "schema_migrations": 1,
        "stream_chat_messages": 1,
        "stream_events": 3,
        "stream_sessions": 2,
        "viewer_promotion_profiles": 0
      },
      "final_response_digest": "",
      "final_response_present": false,
      "schema_migrations": [
        {
          "applied_at": "2026-08-11T07:28:33.114708+00:00",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "component": "cognitive_replay",
          "name": "replay_metadata",
          "version": 1
        }
      ]
    },
    "emitted_outputs": [
      {
        "emitted": true,
        "event_id": "1786433313.4060206",
        "reason": "",
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "text_digest": "",
        "text_present": false
      },
      {
        "emitted": true,
        "event_id": "1786433313.6396518",
        "reason": "",
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "text_digest": "",
        "text_present": false
      }
    ],
    "final_emission_results": [
      {
        "emitted": true,
        "event_id": "1786433313.4060206",
        "reason": "",
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "text_digest": "",
        "text_present": false
      },
      {
        "emitted": true,
        "event_id": "1786433313.6396518",
        "reason": "",
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "text_digest": "",
        "text_present": false
      }
    ],
    "game_state": {
      "challenge": "",
      "confidence": 0.75,
      "current_character": "",
      "current_game": "Final Fantasy V",
      "current_location": null,
      "current_objective": null,
      "game": "Final Fantasy V",
      "known_constraints": [],
      "last_confirmed_progress": "",
      "last_updated": 1786550400.0,
      "party_members": [],
      "platform_version": "",
      "playthrough_type": "casual",
      "provenance": "stream_context_sync",
      "recent_run_context_facts": [],
      "spoiler_policy": "no_spoilers"
    },
    "memory": {
      "chunks_count": 1,
      "facts_count": 0
    },
    "open_threads": [],
    "pending": {
      "clarification": null,
      "conversation_turn": null
    },
    "promotion_profiles": [],
    "receipts": [],
    "runtime": {
      "hebe_sleeping": false,
      "is_running": false,
      "last_firewall": {
        "authority": null,
        "decision": null,
        "reason": null,
        "source": null
      },
      "last_input_source": null,
      "last_intent": null,
      "last_policy": {
        "authority": null,
        "decision": null,
        "reason": null,
        "source": null
      },
      "mode": "active"
    },
    "social_state": {
      "last_cheer": {},
      "last_raid": {},
      "recent_active_users": [],
      "recent_chat_count": 0
    },
    "stream_session": {
      "active_stream_session_id": 2,
      "category": "Final Fantasy V",
      "enabled": true,
      "game": "Final Fantasy V",
      "is_live": true,
      "last_transition": "online",
      "live_status_known": true,
      "title": "Crystal Roulette continues"
    }
  }
}
```

#### Final state and side effects

```json
{
  "actions": {
    "attempts": [],
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
    "research_calls": [],
    "speech_requests": []
  },
  "beliefs": [],
  "current_scene": {},
  "database_watermarks": {
    "counts": {
      "chat_log": 2,
      "live_session_timeline": 10,
      "memory_chunks": 1,
      "memory_facts": 0,
      "promotion_events": 0,
      "schema_migrations": 1,
      "stream_chat_messages": 1,
      "stream_events": 3,
      "stream_sessions": 2,
      "viewer_promotion_profiles": 0
    },
    "final_response_digest": "",
    "final_response_present": false,
    "schema_migrations": [
      {
        "applied_at": "2026-08-11T07:28:33.114708+00:00",
        "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
        "component": "cognitive_replay",
        "name": "replay_metadata",
        "version": 1
      }
    ]
  },
  "emitted_outputs": [
    {
      "emitted": true,
      "event_id": "1786433313.4060206",
      "reason": "",
      "route": "local_owner_reply",
      "targets": [
        "local_ui"
      ],
      "text_digest": "",
      "text_present": false
    },
    {
      "emitted": true,
      "event_id": "1786433313.6396518",
      "reason": "",
      "route": "local_owner_reply",
      "targets": [
        "local_ui"
      ],
      "text_digest": "",
      "text_present": false
    }
  ],
  "final_emission_results": [
    {
      "emitted": true,
      "event_id": "1786433313.4060206",
      "reason": "",
      "route": "local_owner_reply",
      "targets": [
        "local_ui"
      ],
      "text_digest": "",
      "text_present": false
    },
    {
      "emitted": true,
      "event_id": "1786433313.6396518",
      "reason": "",
      "route": "local_owner_reply",
      "targets": [
        "local_ui"
      ],
      "text_digest": "",
      "text_present": false
    }
  ],
  "game_state": {
    "challenge": "",
    "confidence": 0.75,
    "current_character": "",
    "current_game": "Final Fantasy V",
    "current_location": null,
    "current_objective": null,
    "game": "Final Fantasy V",
    "known_constraints": [],
    "last_confirmed_progress": "",
    "last_updated": 1786550400.0,
    "party_members": [],
    "platform_version": "",
    "playthrough_type": "casual",
    "provenance": "stream_context_sync",
    "recent_run_context_facts": [],
    "spoiler_policy": "no_spoilers"
  },
  "memory": {
    "chunks_count": 1,
    "facts_count": 0
  },
  "open_threads": [],
  "pending": {
    "clarification": null,
    "conversation_turn": null
  },
  "promotion_profiles": [],
  "receipts": [],
  "runtime": {
    "hebe_sleeping": false,
    "is_running": false,
    "last_firewall": {
      "authority": null,
      "decision": null,
      "reason": null,
      "source": null
    },
    "last_input_source": null,
    "last_intent": null,
    "last_policy": {
      "authority": null,
      "decision": null,
      "reason": null,
      "source": null
    },
    "mode": "active"
  },
  "social_state": {
    "last_cheer": {},
    "last_raid": {},
    "recent_active_users": [],
    "recent_chat_count": 0
  },
  "stream_session": {
    "active_stream_session_id": 2,
    "category": "Final Fantasy V",
    "enabled": true,
    "game": "Final Fantasy V",
    "is_live": true,
    "last_transition": "online",
    "live_status_known": true,
    "title": "Crystal Roulette continues"
  }
}
```

#### Restart evidence

```json
[
  {
    "after_persisted_counts": {
      "chat_log": 2,
      "live_session_timeline": 8,
      "memory_chunks": 1,
      "memory_facts": 0,
      "promotion_events": 0,
      "schema_migrations": 1,
      "stream_chat_messages": 1,
      "stream_events": 2,
      "stream_sessions": 1,
      "viewer_promotion_profiles": 0
    },
    "before_persisted_counts": {
      "chat_log": 2,
      "live_session_timeline": 8,
      "memory_chunks": 1,
      "memory_facts": 0,
      "promotion_events": 0,
      "schema_migrations": 1,
      "stream_chat_messages": 1,
      "stream_events": 2,
      "stream_sessions": 1,
      "viewer_promotion_profiles": 0
    },
    "event_id": "restart",
    "new_engine_id": 2823644916176,
    "old_engine_collected": true,
    "old_engine_id": 2823652763216,
    "same_database": "C:\\Users\\Leo Nifelheim\\Documents\\Hebe\\hebe-ui\\artifacts\\cognitive-replay\\phase-0.5\\workspaces\\ffv_durable_run_format\\hebe-replay.sqlite3",
    "volatile_state_recreated": true
  }
]
```

### ivanxi_resub_promo_restart

- Status: **VERIFIED**
- Events: 7
- Restarts: 1
- Duration: 1.215351s
- Assertions passed/failed/skipped: 5/0/0


#### Checkpoint state

```json
{
  "ivanxi-chat": {
    "actions": {
      "attempts": [],
      "model_calls": [],
      "research_calls": [],
      "speech_requests": []
    },
    "beliefs": [],
    "current_scene": {},
    "database_watermarks": {
      "counts": {
        "chat_log": 0,
        "live_session_timeline": 3,
        "memory_chunks": 0,
        "memory_facts": 0,
        "promotion_events": 0,
        "schema_migrations": 1,
        "stream_chat_messages": 1,
        "stream_events": 1,
        "stream_sessions": 1,
        "viewer_promotion_profiles": 0
      },
      "final_response_digest": "",
      "final_response_present": false,
      "schema_migrations": [
        {
          "applied_at": "2026-08-11T07:28:34.501984+00:00",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "component": "cognitive_replay",
          "name": "replay_metadata",
          "version": 1
        }
      ]
    },
    "emitted_outputs": [],
    "final_emission_results": [],
    "game_state": {
      "challenge": "",
      "confidence": 0.75,
      "current_character": "",
      "current_game": "Final Fantasy V",
      "current_location": null,
      "current_objective": null,
      "game": "Final Fantasy V",
      "known_constraints": [],
      "last_confirmed_progress": "",
      "last_updated": 1786464000.0,
      "party_members": [],
      "platform_version": "",
      "playthrough_type": "casual",
      "provenance": "stream_context_sync",
      "recent_run_context_facts": [],
      "spoiler_policy": "no_spoilers"
    },
    "memory": {
      "chunks_count": 0,
      "facts_count": 0
    },
    "open_threads": [],
    "pending": {
      "clarification": null,
      "conversation_turn": null
    },
    "promotion_profiles": [],
    "receipts": [],
    "runtime": {
      "hebe_sleeping": false,
      "is_running": false,
      "last_firewall": {
        "authority": "viewer",
        "decision": "allow",
        "reason": "live_viewer_message",
        "source": "twitch_viewer"
      },
      "last_input_source": null,
      "last_intent": null,
      "last_policy": {
        "authority": null,
        "decision": null,
        "reason": null,
        "source": null
      },
      "mode": "active"
    },
    "social_state": {
      "last_cheer": {},
      "last_raid": {},
      "recent_active_users": [
        "ivanxi_kun"
      ],
      "recent_chat_count": 1
    },
    "stream_session": {
      "active_stream_session_id": 1,
      "category": "Final Fantasy V",
      "enabled": true,
      "game": "Final Fantasy V",
      "is_live": true,
      "last_transition": "online",
      "live_status_known": true,
      "title": "[ENG/ESP] Crystal Roulette — Final Fantasy V"
    }
  },
  "ivanxi-resub": {
    "actions": {
      "attempts": [],
      "model_calls": [],
      "research_calls": [],
      "speech_requests": []
    },
    "beliefs": [],
    "current_scene": {},
    "database_watermarks": {
      "counts": {
        "chat_log": 0,
        "live_session_timeline": 2,
        "memory_chunks": 0,
        "memory_facts": 0,
        "promotion_events": 0,
        "schema_migrations": 1,
        "stream_chat_messages": 0,
        "stream_events": 1,
        "stream_sessions": 1,
        "viewer_promotion_profiles": 0
      },
      "final_response_digest": "",
      "final_response_present": false,
      "schema_migrations": [
        {
          "applied_at": "2026-08-11T07:28:34.501984+00:00",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "component": "cognitive_replay",
          "name": "replay_metadata",
          "version": 1
        }
      ]
    },
    "emitted_outputs": [],
    "final_emission_results": [],
    "game_state": {
      "challenge": "",
      "confidence": 0.75,
      "current_character": "",
      "current_game": "Final Fantasy V",
      "current_location": null,
      "current_objective": null,
      "game": "Final Fantasy V",
      "known_constraints": [],
      "last_confirmed_progress": "",
      "last_updated": 1786464000.0,
      "party_members": [],
      "platform_version": "",
      "playthrough_type": "casual",
      "provenance": "stream_context_sync",
      "recent_run_context_facts": [],
      "spoiler_policy": "no_spoilers"
    },
    "memory": {
      "chunks_count": 0,
      "facts_count": 0
    },
    "open_threads": [],
    "pending": {
      "clarification": null,
      "conversation_turn": null
    },
    "promotion_profiles": [],
    "receipts": [],
    "runtime": {
      "hebe_sleeping": false,
      "is_running": false,
      "last_firewall": {
        "authority": "system",
        "decision": "allow",
        "reason": "system_event",
        "source": "twitch_system"
      },
      "last_input_source": null,
      "last_intent": null,
      "last_policy": {
        "authority": null,
        "decision": null,
        "reason": null,
        "source": null
      },
      "mode": "active"
    },
    "social_state": {
      "last_cheer": {},
      "last_raid": {},
      "recent_active_users": [],
      "recent_chat_count": 0
    },
    "stream_session": {
      "active_stream_session_id": 1,
      "category": "Final Fantasy V",
      "enabled": true,
      "game": "Final Fantasy V",
      "is_live": true,
      "last_transition": "online",
      "live_status_known": true,
      "title": "[ENG/ESP] Crystal Roulette — Final Fantasy V"
    }
  },
  "owner-promo": {
    "actions": {
      "attempts": [
        {
          "operation": "twitch.shoutout",
          "outcome": {
            "message_id": "replay-so-1",
            "status": "sent",
            "success": true
          },
          "payload": {
            "command": "!so ivanxi_kun",
            "target": "ivanxi_kun"
          }
        }
      ],
      "model_calls": [],
      "research_calls": [],
      "speech_requests": []
    },
    "beliefs": [],
    "current_scene": {},
    "database_watermarks": {
      "counts": {
        "chat_log": 1,
        "live_session_timeline": 5,
        "memory_chunks": 0,
        "memory_facts": 0,
        "promotion_events": 1,
        "schema_migrations": 1,
        "stream_chat_messages": 1,
        "stream_events": 1,
        "stream_sessions": 1,
        "viewer_promotion_profiles": 1
      },
      "final_response_digest": "",
      "final_response_present": false,
      "schema_migrations": [
        {
          "applied_at": "2026-08-11T07:28:34.501984+00:00",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "component": "cognitive_replay",
          "name": "replay_metadata",
          "version": 1
        }
      ]
    },
    "emitted_outputs": [
      {
        "emitted": true,
        "event_id": "1786433314.861827",
        "reason": "",
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "text_digest": "",
        "text_present": false
      }
    ],
    "final_emission_results": [
      {
        "emitted": true,
        "event_id": "1786433314.861827",
        "reason": "",
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "text_digest": "",
        "text_present": false
      }
    ],
    "game_state": {
      "challenge": "",
      "confidence": 0.75,
      "current_character": "",
      "current_game": "Final Fantasy V",
      "current_location": null,
      "current_objective": null,
      "game": "Final Fantasy V",
      "known_constraints": [],
      "last_confirmed_progress": "",
      "last_updated": 1786464000.0,
      "party_members": [],
      "platform_version": "",
      "playthrough_type": "casual",
      "provenance": "stream_context_sync",
      "recent_run_context_facts": [],
      "spoiler_policy": "no_spoilers"
    },
    "memory": {
      "chunks_count": 0,
      "facts_count": 0
    },
    "open_threads": [],
    "pending": {
      "clarification": null,
      "conversation_turn": null
    },
    "promotion_profiles": [
      {
        "active": 1,
        "auto_promo_mode": "first_message_each_stream",
        "created_by": "owner_command",
        "current_login": "ivanxi_kun",
        "display_name": "ivanxi_kun",
        "last_promoted_stream_id": "1",
        "owner_locked": 1,
        "twitch_user_id": "42"
      }
    ],
    "receipts": [
      {
        "execution_status": "sent",
        "failure_reason": "",
        "id": "promo_4706d2db511c500f94a3499dce27d6a1",
        "requested_by": "leo",
        "resolved_login": "ivanxi_kun",
        "resolved_twitch_user_id": "42",
        "source_event_id": "1786433314.861827",
        "stream_session_id": "1",
        "trigger_type": "owner_learn_and_execute",
        "twitch_message_id": "replay-so-1"
      }
    ],
    "runtime": {
      "hebe_sleeping": false,
      "is_running": false,
      "last_firewall": {
        "authority": "owner",
        "decision": "allow",
        "reason": "owner_direct",
        "source": "owner_stt_direct"
      },
      "last_input_source": null,
      "last_intent": null,
      "last_policy": {
        "authority": null,
        "decision": null,
        "reason": null,
        "source": null
      },
      "mode": "active"
    },
    "social_state": {
      "last_cheer": {},
      "last_raid": {},
      "recent_active_users": [
        "ivanxi_kun"
      ],
      "recent_chat_count": 1
    },
    "stream_session": {
      "active_stream_session_id": 1,
      "category": "Final Fantasy V",
      "enabled": true,
      "game": "Final Fantasy V",
      "is_live": true,
      "last_transition": "online",
      "live_status_known": true,
      "title": "[ENG/ESP] Crystal Roulette — Final Fantasy V"
    }
  },
  "restart-1": {
    "actions": {
      "attempts": [
        {
          "operation": "twitch.shoutout",
          "outcome": {
            "message_id": "replay-so-1",
            "status": "sent",
            "success": true
          },
          "payload": {
            "command": "!so ivanxi_kun",
            "target": "ivanxi_kun"
          }
        }
      ],
      "model_calls": [],
      "research_calls": [],
      "speech_requests": []
    },
    "beliefs": [],
    "current_scene": {},
    "database_watermarks": {
      "counts": {
        "chat_log": 1,
        "live_session_timeline": 6,
        "memory_chunks": 1,
        "memory_facts": 0,
        "promotion_events": 1,
        "schema_migrations": 1,
        "stream_chat_messages": 1,
        "stream_events": 2,
        "stream_sessions": 1,
        "viewer_promotion_profiles": 1
      },
      "final_response_digest": "",
      "final_response_present": false,
      "schema_migrations": [
        {
          "applied_at": "2026-08-11T07:28:34.501984+00:00",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "component": "cognitive_replay",
          "name": "replay_metadata",
          "version": 1
        }
      ]
    },
    "emitted_outputs": [
      {
        "emitted": true,
        "event_id": "1786433314.861827",
        "reason": "",
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "text_digest": "",
        "text_present": false
      }
    ],
    "final_emission_results": [
      {
        "emitted": true,
        "event_id": "1786433314.861827",
        "reason": "",
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "text_digest": "",
        "text_present": false
      }
    ],
    "game_state": {
      "challenge": "",
      "confidence": 0.0,
      "current_character": "",
      "current_game": null,
      "current_location": null,
      "current_objective": null,
      "game": "",
      "known_constraints": [],
      "last_confirmed_progress": "",
      "last_updated": 1786433315.243552,
      "party_members": [],
      "platform_version": "",
      "playthrough_type": "casual",
      "provenance": "inferred",
      "recent_run_context_facts": [],
      "spoiler_policy": "spoiler_safe_hints"
    },
    "memory": {
      "chunks_count": 1,
      "facts_count": 0
    },
    "open_threads": [],
    "pending": {
      "clarification": null,
      "conversation_turn": null
    },
    "promotion_profiles": [
      {
        "active": 1,
        "auto_promo_mode": "first_message_each_stream",
        "created_by": "owner_command",
        "current_login": "ivanxi_kun",
        "display_name": "ivanxi_kun",
        "last_promoted_stream_id": "1",
        "owner_locked": 1,
        "twitch_user_id": "42"
      }
    ],
    "receipts": [
      {
        "execution_status": "sent",
        "failure_reason": "",
        "id": "promo_4706d2db511c500f94a3499dce27d6a1",
        "requested_by": "leo",
        "resolved_login": "ivanxi_kun",
        "resolved_twitch_user_id": "42",
        "source_event_id": "1786433314.861827",
        "stream_session_id": "1",
        "trigger_type": "owner_learn_and_execute",
        "twitch_message_id": "replay-so-1"
      }
    ],
    "runtime": {
      "hebe_sleeping": false,
      "is_running": false,
      "last_firewall": {
        "authority": null,
        "decision": null,
        "reason": null,
        "source": null
      },
      "last_input_source": null,
      "last_intent": null,
      "last_policy": {
        "authority": null,
        "decision": null,
        "reason": null,
        "source": null
      },
      "mode": "active"
    },
    "social_state": {
      "last_cheer": {},
      "last_raid": {},
      "recent_active_users": [],
      "recent_chat_count": 0
    },
    "stream_session": {
      "active_stream_session_id": null,
      "category": null,
      "enabled": false,
      "game": null,
      "is_live": false,
      "last_transition": null,
      "live_status_known": false,
      "title": null
    }
  },
  "stream-1-end": {
    "actions": {
      "attempts": [
        {
          "operation": "twitch.shoutout",
          "outcome": {
            "message_id": "replay-so-1",
            "status": "sent",
            "success": true
          },
          "payload": {
            "command": "!so ivanxi_kun",
            "target": "ivanxi_kun"
          }
        }
      ],
      "model_calls": [],
      "research_calls": [],
      "speech_requests": []
    },
    "beliefs": [],
    "current_scene": {},
    "database_watermarks": {
      "counts": {
        "chat_log": 1,
        "live_session_timeline": 6,
        "memory_chunks": 1,
        "memory_facts": 0,
        "promotion_events": 1,
        "schema_migrations": 1,
        "stream_chat_messages": 1,
        "stream_events": 2,
        "stream_sessions": 1,
        "viewer_promotion_profiles": 1
      },
      "final_response_digest": "",
      "final_response_present": false,
      "schema_migrations": [
        {
          "applied_at": "2026-08-11T07:28:34.501984+00:00",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "component": "cognitive_replay",
          "name": "replay_metadata",
          "version": 1
        }
      ]
    },
    "emitted_outputs": [
      {
        "emitted": true,
        "event_id": "1786433314.861827",
        "reason": "",
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "text_digest": "",
        "text_present": false
      }
    ],
    "final_emission_results": [
      {
        "emitted": true,
        "event_id": "1786433314.861827",
        "reason": "",
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "text_digest": "",
        "text_present": false
      }
    ],
    "game_state": {
      "challenge": "",
      "confidence": 0.75,
      "current_character": "",
      "current_game": "Final Fantasy V",
      "current_location": null,
      "current_objective": null,
      "game": "Final Fantasy V",
      "known_constraints": [],
      "last_confirmed_progress": "",
      "last_updated": 1786464000.0,
      "party_members": [],
      "platform_version": "",
      "playthrough_type": "casual",
      "provenance": "stream_context_sync",
      "recent_run_context_facts": [],
      "spoiler_policy": "no_spoilers"
    },
    "memory": {
      "chunks_count": 1,
      "facts_count": 0
    },
    "open_threads": [],
    "pending": {
      "clarification": null,
      "conversation_turn": null
    },
    "promotion_profiles": [
      {
        "active": 1,
        "auto_promo_mode": "first_message_each_stream",
        "created_by": "owner_command",
        "current_login": "ivanxi_kun",
        "display_name": "ivanxi_kun",
        "last_promoted_stream_id": "1",
        "owner_locked": 1,
        "twitch_user_id": "42"
      }
    ],
    "receipts": [
      {
        "execution_status": "sent",
        "failure_reason": "",
        "id": "promo_4706d2db511c500f94a3499dce27d6a1",
        "requested_by": "leo",
        "resolved_login": "ivanxi_kun",
        "resolved_twitch_user_id": "42",
        "source_event_id": "1786433314.861827",
        "stream_session_id": "1",
        "trigger_type": "owner_learn_and_execute",
        "twitch_message_id": "replay-so-1"
      }
    ],
    "runtime": {
      "hebe_sleeping": false,
      "is_running": false,
      "last_firewall": {
        "authority": "owner",
        "decision": "allow",
        "reason": "owner_direct",
        "source": "owner_stt_direct"
      },
      "last_input_source": null,
      "last_intent": null,
      "last_policy": {
        "authority": null,
        "decision": null,
        "reason": null,
        "source": null
      },
      "mode": "active"
    },
    "social_state": {
      "last_cheer": {},
      "last_raid": {},
      "recent_active_users": [
        "ivanxi_kun"
      ],
      "recent_chat_count": 1
    },
    "stream_session": {
      "active_stream_session_id": null,
      "category": "Final Fantasy V",
      "enabled": true,
      "game": "Final Fantasy V",
      "is_live": false,
      "last_transition": "offline",
      "live_status_known": true,
      "title": "[ENG/ESP] Crystal Roulette — Final Fantasy V"
    }
  },
  "stream-1-start": {
    "actions": {
      "attempts": [],
      "model_calls": [],
      "research_calls": [],
      "speech_requests": []
    },
    "beliefs": [],
    "current_scene": {},
    "database_watermarks": {
      "counts": {
        "chat_log": 0,
        "live_session_timeline": 2,
        "memory_chunks": 0,
        "memory_facts": 0,
        "promotion_events": 0,
        "schema_migrations": 1,
        "stream_chat_messages": 0,
        "stream_events": 1,
        "stream_sessions": 1,
        "viewer_promotion_profiles": 0
      },
      "final_response_digest": "",
      "final_response_present": false,
      "schema_migrations": [
        {
          "applied_at": "2026-08-11T07:28:34.501984+00:00",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "component": "cognitive_replay",
          "name": "replay_metadata",
          "version": 1
        }
      ]
    },
    "emitted_outputs": [],
    "final_emission_results": [],
    "game_state": {
      "challenge": "",
      "confidence": 0.75,
      "current_character": "",
      "current_game": "Final Fantasy V",
      "current_location": null,
      "current_objective": null,
      "game": "Final Fantasy V",
      "known_constraints": [],
      "last_confirmed_progress": "",
      "last_updated": 1786464000.0,
      "party_members": [],
      "platform_version": "",
      "playthrough_type": "casual",
      "provenance": "stream_context_sync",
      "recent_run_context_facts": [],
      "spoiler_policy": "no_spoilers"
    },
    "memory": {
      "chunks_count": 0,
      "facts_count": 0
    },
    "open_threads": [],
    "pending": {
      "clarification": null,
      "conversation_turn": null
    },
    "promotion_profiles": [],
    "receipts": [],
    "runtime": {
      "hebe_sleeping": false,
      "is_running": false,
      "last_firewall": {
        "authority": null,
        "decision": null,
        "reason": null,
        "source": null
      },
      "last_input_source": null,
      "last_intent": null,
      "last_policy": {
        "authority": null,
        "decision": null,
        "reason": null,
        "source": null
      },
      "mode": "active"
    },
    "social_state": {
      "last_cheer": {},
      "last_raid": {},
      "recent_active_users": [],
      "recent_chat_count": 0
    },
    "stream_session": {
      "active_stream_session_id": 1,
      "category": "Final Fantasy V",
      "enabled": true,
      "game": "Final Fantasy V",
      "is_live": true,
      "last_transition": "online",
      "live_status_known": true,
      "title": "[ENG/ESP] Crystal Roulette — Final Fantasy V"
    }
  },
  "stream-2-start": {
    "actions": {
      "attempts": [
        {
          "operation": "twitch.shoutout",
          "outcome": {
            "message_id": "replay-so-1",
            "status": "sent",
            "success": true
          },
          "payload": {
            "command": "!so ivanxi_kun",
            "target": "ivanxi_kun"
          }
        }
      ],
      "model_calls": [],
      "research_calls": [],
      "speech_requests": []
    },
    "beliefs": [],
    "current_scene": {},
    "database_watermarks": {
      "counts": {
        "chat_log": 1,
        "live_session_timeline": 8,
        "memory_chunks": 1,
        "memory_facts": 0,
        "promotion_events": 1,
        "schema_migrations": 1,
        "stream_chat_messages": 1,
        "stream_events": 3,
        "stream_sessions": 2,
        "viewer_promotion_profiles": 1
      },
      "final_response_digest": "",
      "final_response_present": false,
      "schema_migrations": [
        {
          "applied_at": "2026-08-11T07:28:34.501984+00:00",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "component": "cognitive_replay",
          "name": "replay_metadata",
          "version": 1
        }
      ]
    },
    "emitted_outputs": [
      {
        "emitted": true,
        "event_id": "1786433314.861827",
        "reason": "",
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "text_digest": "",
        "text_present": false
      }
    ],
    "final_emission_results": [
      {
        "emitted": true,
        "event_id": "1786433314.861827",
        "reason": "",
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "text_digest": "",
        "text_present": false
      }
    ],
    "game_state": {
      "challenge": "",
      "confidence": 0.75,
      "current_character": "",
      "current_game": "Final Fantasy V",
      "current_location": null,
      "current_objective": null,
      "game": "Final Fantasy V",
      "known_constraints": [],
      "last_confirmed_progress": "",
      "last_updated": 1786550400.0,
      "party_members": [],
      "platform_version": "",
      "playthrough_type": "casual",
      "provenance": "stream_context_sync",
      "recent_run_context_facts": [],
      "spoiler_policy": "no_spoilers"
    },
    "memory": {
      "chunks_count": 1,
      "facts_count": 0
    },
    "open_threads": [],
    "pending": {
      "clarification": null,
      "conversation_turn": null
    },
    "promotion_profiles": [
      {
        "active": 1,
        "auto_promo_mode": "first_message_each_stream",
        "created_by": "owner_command",
        "current_login": "ivanxi_kun",
        "display_name": "ivanxi_kun",
        "last_promoted_stream_id": "1",
        "owner_locked": 1,
        "twitch_user_id": "42"
      }
    ],
    "receipts": [
      {
        "execution_status": "sent",
        "failure_reason": "",
        "id": "promo_4706d2db511c500f94a3499dce27d6a1",
        "requested_by": "leo",
        "resolved_login": "ivanxi_kun",
        "resolved_twitch_user_id": "42",
        "source_event_id": "1786433314.861827",
        "stream_session_id": "1",
        "trigger_type": "owner_learn_and_execute",
        "twitch_message_id": "replay-so-1"
      }
    ],
    "runtime": {
      "hebe_sleeping": false,
      "is_running": false,
      "last_firewall": {
        "authority": null,
        "decision": null,
        "reason": null,
        "source": null
      },
      "last_input_source": null,
      "last_intent": null,
      "last_policy": {
        "authority": null,
        "decision": null,
        "reason": null,
        "source": null
      },
      "mode": "active"
    },
    "social_state": {
      "last_cheer": {},
      "last_raid": {},
      "recent_active_users": [],
      "recent_chat_count": 0
    },
    "stream_session": {
      "active_stream_session_id": 2,
      "category": "Final Fantasy V",
      "enabled": true,
      "game": "Final Fantasy V",
      "is_live": true,
      "last_transition": "online",
      "live_status_known": true,
      "title": "[ENG/ESP] Crystal Roulette continues — Final Fantasy V"
    }
  }
}
```

#### Final state and side effects

```json
{
  "actions": {
    "attempts": [
      {
        "operation": "twitch.shoutout",
        "outcome": {
          "message_id": "replay-so-1",
          "status": "sent",
          "success": true
        },
        "payload": {
          "command": "!so ivanxi_kun",
          "target": "ivanxi_kun"
        }
      }
    ],
    "model_calls": [],
    "research_calls": [],
    "speech_requests": []
  },
  "beliefs": [],
  "current_scene": {},
  "database_watermarks": {
    "counts": {
      "chat_log": 1,
      "live_session_timeline": 8,
      "memory_chunks": 1,
      "memory_facts": 0,
      "promotion_events": 1,
      "schema_migrations": 1,
      "stream_chat_messages": 1,
      "stream_events": 3,
      "stream_sessions": 2,
      "viewer_promotion_profiles": 1
    },
    "final_response_digest": "",
    "final_response_present": false,
    "schema_migrations": [
      {
        "applied_at": "2026-08-11T07:28:34.501984+00:00",
        "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
        "component": "cognitive_replay",
        "name": "replay_metadata",
        "version": 1
      }
    ]
  },
  "emitted_outputs": [
    {
      "emitted": true,
      "event_id": "1786433314.861827",
      "reason": "",
      "route": "local_owner_reply",
      "targets": [
        "local_ui"
      ],
      "text_digest": "",
      "text_present": false
    }
  ],
  "final_emission_results": [
    {
      "emitted": true,
      "event_id": "1786433314.861827",
      "reason": "",
      "route": "local_owner_reply",
      "targets": [
        "local_ui"
      ],
      "text_digest": "",
      "text_present": false
    }
  ],
  "game_state": {
    "challenge": "",
    "confidence": 0.75,
    "current_character": "",
    "current_game": "Final Fantasy V",
    "current_location": null,
    "current_objective": null,
    "game": "Final Fantasy V",
    "known_constraints": [],
    "last_confirmed_progress": "",
    "last_updated": 1786550400.0,
    "party_members": [],
    "platform_version": "",
    "playthrough_type": "casual",
    "provenance": "stream_context_sync",
    "recent_run_context_facts": [],
    "spoiler_policy": "no_spoilers"
  },
  "memory": {
    "chunks_count": 1,
    "facts_count": 0
  },
  "open_threads": [],
  "pending": {
    "clarification": null,
    "conversation_turn": null
  },
  "promotion_profiles": [
    {
      "active": 1,
      "auto_promo_mode": "first_message_each_stream",
      "created_by": "owner_command",
      "current_login": "ivanxi_kun",
      "display_name": "ivanxi_kun",
      "last_promoted_stream_id": "1",
      "owner_locked": 1,
      "twitch_user_id": "42"
    }
  ],
  "receipts": [
    {
      "execution_status": "sent",
      "failure_reason": "",
      "id": "promo_4706d2db511c500f94a3499dce27d6a1",
      "requested_by": "leo",
      "resolved_login": "ivanxi_kun",
      "resolved_twitch_user_id": "42",
      "source_event_id": "1786433314.861827",
      "stream_session_id": "1",
      "trigger_type": "owner_learn_and_execute",
      "twitch_message_id": "replay-so-1"
    }
  ],
  "runtime": {
    "hebe_sleeping": false,
    "is_running": false,
    "last_firewall": {
      "authority": null,
      "decision": null,
      "reason": null,
      "source": null
    },
    "last_input_source": null,
    "last_intent": null,
    "last_policy": {
      "authority": null,
      "decision": null,
      "reason": null,
      "source": null
    },
    "mode": "active"
  },
  "social_state": {
    "last_cheer": {},
    "last_raid": {},
    "recent_active_users": [],
    "recent_chat_count": 0
  },
  "stream_session": {
    "active_stream_session_id": 2,
    "category": "Final Fantasy V",
    "enabled": true,
    "game": "Final Fantasy V",
    "is_live": true,
    "last_transition": "online",
    "live_status_known": true,
    "title": "[ENG/ESP] Crystal Roulette continues — Final Fantasy V"
  }
}
```

#### Restart evidence

```json
[
  {
    "after_persisted_counts": {
      "chat_log": 1,
      "live_session_timeline": 6,
      "memory_chunks": 1,
      "memory_facts": 0,
      "promotion_events": 1,
      "schema_migrations": 1,
      "stream_chat_messages": 1,
      "stream_events": 2,
      "stream_sessions": 1,
      "viewer_promotion_profiles": 1
    },
    "before_persisted_counts": {
      "chat_log": 1,
      "live_session_timeline": 6,
      "memory_chunks": 1,
      "memory_facts": 0,
      "promotion_events": 1,
      "schema_migrations": 1,
      "stream_chat_messages": 1,
      "stream_events": 2,
      "stream_sessions": 1,
      "viewer_promotion_profiles": 1
    },
    "event_id": "restart-1",
    "new_engine_id": 2823652618128,
    "old_engine_collected": true,
    "old_engine_id": 2823652893520,
    "same_database": "C:\\Users\\Leo Nifelheim\\Documents\\Hebe\\hebe-ui\\artifacts\\cognitive-replay\\phase-0.5\\workspaces\\ivanxi_resub_promo_restart\\hebe-replay.sqlite3",
    "volatile_state_recreated": true
  }
]
```

### owner_correction_format

- Status: **VERIFICATION_INCOMPLETE**
- Events: 6
- Restarts: 1
- Duration: 6.728503s
- Assertions passed/failed/skipped: 0/0/1


#### Checkpoint state

```json
{
  "correction": {
    "actions": {
      "attempts": [],
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
      "research_calls": [],
      "speech_requests": []
    },
    "beliefs": [],
    "current_scene": {},
    "database_watermarks": {
      "counts": {
        "chat_log": 2,
        "live_session_timeline": 5,
        "memory_chunks": 0,
        "memory_facts": 0,
        "promotion_events": 0,
        "schema_migrations": 1,
        "stream_chat_messages": 0,
        "stream_events": 1,
        "stream_sessions": 1,
        "viewer_promotion_profiles": 0
      },
      "final_response_digest": "7e07c6a5322789f6",
      "final_response_present": true,
      "schema_migrations": [
        {
          "applied_at": "2026-08-11T07:28:35.821984+00:00",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "component": "cognitive_replay",
          "name": "replay_metadata",
          "version": 1
        }
      ]
    },
    "emitted_outputs": [
      {
        "emitted": true,
        "event_id": "1786433316.1203387",
        "reason": "",
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "text_digest": "",
        "text_present": false
      }
    ],
    "final_emission_results": [
      {
        "emitted": true,
        "event_id": "1786433316.1203387",
        "reason": "",
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "text_digest": "",
        "text_present": false
      }
    ],
    "game_state": {
      "challenge": "",
      "confidence": 0.75,
      "current_character": "",
      "current_game": "Final Fantasy V",
      "current_location": null,
      "current_objective": null,
      "game": "Final Fantasy V",
      "known_constraints": [],
      "last_confirmed_progress": "",
      "last_updated": 1786464000.0,
      "party_members": [],
      "platform_version": "",
      "playthrough_type": "casual",
      "provenance": "stream_context_sync",
      "recent_run_context_facts": [],
      "spoiler_policy": "no_spoilers"
    },
    "memory": {
      "chunks_count": 0,
      "facts_count": 0
    },
    "open_threads": [],
    "pending": {
      "clarification": null,
      "conversation_turn": null
    },
    "promotion_profiles": [],
    "receipts": [],
    "runtime": {
      "hebe_sleeping": false,
      "is_running": false,
      "last_firewall": {
        "authority": "owner",
        "decision": "allow",
        "reason": "owner_direct",
        "source": "owner_stt_direct"
      },
      "last_input_source": null,
      "last_intent": null,
      "last_policy": {
        "authority": null,
        "decision": null,
        "reason": null,
        "source": null
      },
      "mode": "active"
    },
    "social_state": {
      "last_cheer": {},
      "last_raid": {},
      "recent_active_users": [],
      "recent_chat_count": 0
    },
    "stream_session": {
      "active_stream_session_id": 1,
      "category": "Final Fantasy V",
      "enabled": true,
      "game": "Final Fantasy V",
      "is_live": true,
      "last_transition": "online",
      "live_status_known": true,
      "title": null
    }
  },
  "end": {
    "actions": {
      "attempts": [],
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
      "research_calls": [],
      "speech_requests": []
    },
    "beliefs": [],
    "current_scene": {},
    "database_watermarks": {
      "counts": {
        "chat_log": 2,
        "live_session_timeline": 6,
        "memory_chunks": 1,
        "memory_facts": 0,
        "promotion_events": 0,
        "schema_migrations": 1,
        "stream_chat_messages": 0,
        "stream_events": 2,
        "stream_sessions": 1,
        "viewer_promotion_profiles": 0
      },
      "final_response_digest": "7e07c6a5322789f6",
      "final_response_present": true,
      "schema_migrations": [
        {
          "applied_at": "2026-08-11T07:28:35.821984+00:00",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "component": "cognitive_replay",
          "name": "replay_metadata",
          "version": 1
        }
      ]
    },
    "emitted_outputs": [
      {
        "emitted": true,
        "event_id": "1786433316.1203387",
        "reason": "",
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "text_digest": "",
        "text_present": false
      }
    ],
    "final_emission_results": [
      {
        "emitted": true,
        "event_id": "1786433316.1203387",
        "reason": "",
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "text_digest": "",
        "text_present": false
      }
    ],
    "game_state": {
      "challenge": "",
      "confidence": 0.75,
      "current_character": "",
      "current_game": "Final Fantasy V",
      "current_location": null,
      "current_objective": null,
      "game": "Final Fantasy V",
      "known_constraints": [],
      "last_confirmed_progress": "",
      "last_updated": 1786464000.0,
      "party_members": [],
      "platform_version": "",
      "playthrough_type": "casual",
      "provenance": "stream_context_sync",
      "recent_run_context_facts": [],
      "spoiler_policy": "no_spoilers"
    },
    "memory": {
      "chunks_count": 1,
      "facts_count": 0
    },
    "open_threads": [],
    "pending": {
      "clarification": null,
      "conversation_turn": null
    },
    "promotion_profiles": [],
    "receipts": [],
    "runtime": {
      "hebe_sleeping": false,
      "is_running": false,
      "last_firewall": {
        "authority": "owner",
        "decision": "allow",
        "reason": "owner_direct",
        "source": "owner_stt_direct"
      },
      "last_input_source": null,
      "last_intent": null,
      "last_policy": {
        "authority": null,
        "decision": null,
        "reason": null,
        "source": null
      },
      "mode": "active"
    },
    "social_state": {
      "last_cheer": {},
      "last_raid": {},
      "recent_active_users": [],
      "recent_chat_count": 0
    },
    "stream_session": {
      "active_stream_session_id": null,
      "category": "Final Fantasy V",
      "enabled": true,
      "game": "Final Fantasy V",
      "is_live": false,
      "last_transition": "offline",
      "live_status_known": true,
      "title": null
    }
  },
  "later": {
    "actions": {
      "attempts": [],
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
      "research_calls": [],
      "speech_requests": []
    },
    "beliefs": [],
    "current_scene": {},
    "database_watermarks": {
      "counts": {
        "chat_log": 2,
        "live_session_timeline": 8,
        "memory_chunks": 1,
        "memory_facts": 0,
        "promotion_events": 0,
        "schema_migrations": 1,
        "stream_chat_messages": 0,
        "stream_events": 3,
        "stream_sessions": 2,
        "viewer_promotion_profiles": 0
      },
      "final_response_digest": "",
      "final_response_present": false,
      "schema_migrations": [
        {
          "applied_at": "2026-08-11T07:28:35.821984+00:00",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "component": "cognitive_replay",
          "name": "replay_metadata",
          "version": 1
        }
      ]
    },
    "emitted_outputs": [
      {
        "emitted": true,
        "event_id": "1786433316.1203387",
        "reason": "",
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "text_digest": "",
        "text_present": false
      }
    ],
    "final_emission_results": [
      {
        "emitted": true,
        "event_id": "1786433316.1203387",
        "reason": "",
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "text_digest": "",
        "text_present": false
      }
    ],
    "game_state": {
      "challenge": "",
      "confidence": 0.75,
      "current_character": "",
      "current_game": "Final Fantasy V",
      "current_location": null,
      "current_objective": null,
      "game": "Final Fantasy V",
      "known_constraints": [],
      "last_confirmed_progress": "",
      "last_updated": 1786550400.0,
      "party_members": [],
      "platform_version": "",
      "playthrough_type": "casual",
      "provenance": "stream_context_sync",
      "recent_run_context_facts": [],
      "spoiler_policy": "no_spoilers"
    },
    "memory": {
      "chunks_count": 1,
      "facts_count": 0
    },
    "open_threads": [],
    "pending": {
      "clarification": null,
      "conversation_turn": null
    },
    "promotion_profiles": [],
    "receipts": [],
    "runtime": {
      "hebe_sleeping": false,
      "is_running": false,
      "last_firewall": {
        "authority": null,
        "decision": null,
        "reason": null,
        "source": null
      },
      "last_input_source": null,
      "last_intent": null,
      "last_policy": {
        "authority": null,
        "decision": null,
        "reason": null,
        "source": null
      },
      "mode": "active"
    },
    "social_state": {
      "last_cheer": {},
      "last_raid": {},
      "recent_active_users": [],
      "recent_chat_count": 0
    },
    "stream_session": {
      "active_stream_session_id": 2,
      "category": "Final Fantasy V",
      "enabled": true,
      "game": "Final Fantasy V",
      "is_live": true,
      "last_transition": "online",
      "live_status_known": true,
      "title": null
    }
  },
  "restart": {
    "actions": {
      "attempts": [],
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
      "research_calls": [],
      "speech_requests": []
    },
    "beliefs": [],
    "current_scene": {},
    "database_watermarks": {
      "counts": {
        "chat_log": 2,
        "live_session_timeline": 6,
        "memory_chunks": 1,
        "memory_facts": 0,
        "promotion_events": 0,
        "schema_migrations": 1,
        "stream_chat_messages": 0,
        "stream_events": 2,
        "stream_sessions": 1,
        "viewer_promotion_profiles": 0
      },
      "final_response_digest": "",
      "final_response_present": false,
      "schema_migrations": [
        {
          "applied_at": "2026-08-11T07:28:35.821984+00:00",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "component": "cognitive_replay",
          "name": "replay_metadata",
          "version": 1
        }
      ]
    },
    "emitted_outputs": [
      {
        "emitted": true,
        "event_id": "1786433316.1203387",
        "reason": "",
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "text_digest": "",
        "text_present": false
      }
    ],
    "final_emission_results": [
      {
        "emitted": true,
        "event_id": "1786433316.1203387",
        "reason": "",
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "text_digest": "",
        "text_present": false
      }
    ],
    "game_state": {
      "challenge": "",
      "confidence": 0.0,
      "current_character": "",
      "current_game": null,
      "current_location": null,
      "current_objective": null,
      "game": "",
      "known_constraints": [],
      "last_confirmed_progress": "",
      "last_updated": 1786433322.0587761,
      "party_members": [],
      "platform_version": "",
      "playthrough_type": "casual",
      "provenance": "inferred",
      "recent_run_context_facts": [],
      "spoiler_policy": "spoiler_safe_hints"
    },
    "memory": {
      "chunks_count": 1,
      "facts_count": 0
    },
    "open_threads": [],
    "pending": {
      "clarification": null,
      "conversation_turn": null
    },
    "promotion_profiles": [],
    "receipts": [],
    "runtime": {
      "hebe_sleeping": false,
      "is_running": false,
      "last_firewall": {
        "authority": null,
        "decision": null,
        "reason": null,
        "source": null
      },
      "last_input_source": null,
      "last_intent": null,
      "last_policy": {
        "authority": null,
        "decision": null,
        "reason": null,
        "source": null
      },
      "mode": "active"
    },
    "social_state": {
      "last_cheer": {},
      "last_raid": {},
      "recent_active_users": [],
      "recent_chat_count": 0
    },
    "stream_session": {
      "active_stream_session_id": null,
      "category": null,
      "enabled": false,
      "game": null,
      "is_live": false,
      "last_transition": null,
      "live_status_known": false,
      "title": null
    }
  },
  "start": {
    "actions": {
      "attempts": [],
      "model_calls": [],
      "research_calls": [],
      "speech_requests": []
    },
    "beliefs": [],
    "current_scene": {},
    "database_watermarks": {
      "counts": {
        "chat_log": 0,
        "live_session_timeline": 2,
        "memory_chunks": 0,
        "memory_facts": 0,
        "promotion_events": 0,
        "schema_migrations": 1,
        "stream_chat_messages": 0,
        "stream_events": 1,
        "stream_sessions": 1,
        "viewer_promotion_profiles": 0
      },
      "final_response_digest": "",
      "final_response_present": false,
      "schema_migrations": [
        {
          "applied_at": "2026-08-11T07:28:35.821984+00:00",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "component": "cognitive_replay",
          "name": "replay_metadata",
          "version": 1
        }
      ]
    },
    "emitted_outputs": [],
    "final_emission_results": [],
    "game_state": {
      "challenge": "",
      "confidence": 0.75,
      "current_character": "",
      "current_game": "Final Fantasy V",
      "current_location": null,
      "current_objective": null,
      "game": "Final Fantasy V",
      "known_constraints": [],
      "last_confirmed_progress": "",
      "last_updated": 1786464000.0,
      "party_members": [],
      "platform_version": "",
      "playthrough_type": "casual",
      "provenance": "stream_context_sync",
      "recent_run_context_facts": [],
      "spoiler_policy": "no_spoilers"
    },
    "memory": {
      "chunks_count": 0,
      "facts_count": 0
    },
    "open_threads": [],
    "pending": {
      "clarification": null,
      "conversation_turn": null
    },
    "promotion_profiles": [],
    "receipts": [],
    "runtime": {
      "hebe_sleeping": false,
      "is_running": false,
      "last_firewall": {
        "authority": null,
        "decision": null,
        "reason": null,
        "source": null
      },
      "last_input_source": null,
      "last_intent": null,
      "last_policy": {
        "authority": null,
        "decision": null,
        "reason": null,
        "source": null
      },
      "mode": "active"
    },
    "social_state": {
      "last_cheer": {},
      "last_raid": {},
      "recent_active_users": [],
      "recent_chat_count": 0
    },
    "stream_session": {
      "active_stream_session_id": 1,
      "category": "Final Fantasy V",
      "enabled": true,
      "game": "Final Fantasy V",
      "is_live": true,
      "last_transition": "online",
      "live_status_known": true,
      "title": null
    }
  },
  "third": {
    "actions": {
      "attempts": [],
      "model_calls": [
        {
          "key": "stream_response:v1:none",
          "method": "chat"
        }
      ],
      "research_calls": [],
      "speech_requests": []
    },
    "beliefs": [],
    "current_scene": {},
    "database_watermarks": {
      "counts": {
        "chat_log": 1,
        "live_session_timeline": 4,
        "memory_chunks": 0,
        "memory_facts": 0,
        "promotion_events": 0,
        "schema_migrations": 1,
        "stream_chat_messages": 0,
        "stream_events": 1,
        "stream_sessions": 1,
        "viewer_promotion_profiles": 0
      },
      "final_response_digest": "7e07c6a5322789f6",
      "final_response_present": true,
      "schema_migrations": [
        {
          "applied_at": "2026-08-11T07:28:35.821984+00:00",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "component": "cognitive_replay",
          "name": "replay_metadata",
          "version": 1
        }
      ]
    },
    "emitted_outputs": [
      {
        "emitted": true,
        "event_id": "1786433316.1203387",
        "reason": "",
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "text_digest": "",
        "text_present": false
      }
    ],
    "final_emission_results": [
      {
        "emitted": true,
        "event_id": "1786433316.1203387",
        "reason": "",
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "text_digest": "",
        "text_present": false
      }
    ],
    "game_state": {
      "challenge": "",
      "confidence": 0.75,
      "current_character": "",
      "current_game": "Final Fantasy V",
      "current_location": null,
      "current_objective": null,
      "game": "Final Fantasy V",
      "known_constraints": [],
      "last_confirmed_progress": "",
      "last_updated": 1786464000.0,
      "party_members": [],
      "platform_version": "",
      "playthrough_type": "casual",
      "provenance": "stream_context_sync",
      "recent_run_context_facts": [],
      "spoiler_policy": "no_spoilers"
    },
    "memory": {
      "chunks_count": 0,
      "facts_count": 0
    },
    "open_threads": [],
    "pending": {
      "clarification": null,
      "conversation_turn": null
    },
    "promotion_profiles": [],
    "receipts": [],
    "runtime": {
      "hebe_sleeping": false,
      "is_running": false,
      "last_firewall": {
        "authority": "owner",
        "decision": "allow",
        "reason": "owner_direct",
        "source": "owner_stt_direct"
      },
      "last_input_source": null,
      "last_intent": null,
      "last_policy": {
        "authority": null,
        "decision": null,
        "reason": null,
        "source": null
      },
      "mode": "active"
    },
    "social_state": {
      "last_cheer": {},
      "last_raid": {},
      "recent_active_users": [],
      "recent_chat_count": 0
    },
    "stream_session": {
      "active_stream_session_id": 1,
      "category": "Final Fantasy V",
      "enabled": true,
      "game": "Final Fantasy V",
      "is_live": true,
      "last_transition": "online",
      "live_status_known": true,
      "title": null
    }
  }
}
```

#### Final state and side effects

```json
{
  "actions": {
    "attempts": [],
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
    "research_calls": [],
    "speech_requests": []
  },
  "beliefs": [],
  "current_scene": {},
  "database_watermarks": {
    "counts": {
      "chat_log": 2,
      "live_session_timeline": 8,
      "memory_chunks": 1,
      "memory_facts": 0,
      "promotion_events": 0,
      "schema_migrations": 1,
      "stream_chat_messages": 0,
      "stream_events": 3,
      "stream_sessions": 2,
      "viewer_promotion_profiles": 0
    },
    "final_response_digest": "",
    "final_response_present": false,
    "schema_migrations": [
      {
        "applied_at": "2026-08-11T07:28:35.821984+00:00",
        "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
        "component": "cognitive_replay",
        "name": "replay_metadata",
        "version": 1
      }
    ]
  },
  "emitted_outputs": [
    {
      "emitted": true,
      "event_id": "1786433316.1203387",
      "reason": "",
      "route": "local_owner_reply",
      "targets": [
        "local_ui"
      ],
      "text_digest": "",
      "text_present": false
    }
  ],
  "final_emission_results": [
    {
      "emitted": true,
      "event_id": "1786433316.1203387",
      "reason": "",
      "route": "local_owner_reply",
      "targets": [
        "local_ui"
      ],
      "text_digest": "",
      "text_present": false
    }
  ],
  "game_state": {
    "challenge": "",
    "confidence": 0.75,
    "current_character": "",
    "current_game": "Final Fantasy V",
    "current_location": null,
    "current_objective": null,
    "game": "Final Fantasy V",
    "known_constraints": [],
    "last_confirmed_progress": "",
    "last_updated": 1786550400.0,
    "party_members": [],
    "platform_version": "",
    "playthrough_type": "casual",
    "provenance": "stream_context_sync",
    "recent_run_context_facts": [],
    "spoiler_policy": "no_spoilers"
  },
  "memory": {
    "chunks_count": 1,
    "facts_count": 0
  },
  "open_threads": [],
  "pending": {
    "clarification": null,
    "conversation_turn": null
  },
  "promotion_profiles": [],
  "receipts": [],
  "runtime": {
    "hebe_sleeping": false,
    "is_running": false,
    "last_firewall": {
      "authority": null,
      "decision": null,
      "reason": null,
      "source": null
    },
    "last_input_source": null,
    "last_intent": null,
    "last_policy": {
      "authority": null,
      "decision": null,
      "reason": null,
      "source": null
    },
    "mode": "active"
  },
  "social_state": {
    "last_cheer": {},
    "last_raid": {},
    "recent_active_users": [],
    "recent_chat_count": 0
  },
  "stream_session": {
    "active_stream_session_id": 2,
    "category": "Final Fantasy V",
    "enabled": true,
    "game": "Final Fantasy V",
    "is_live": true,
    "last_transition": "online",
    "live_status_known": true,
    "title": null
  }
}
```

#### Restart evidence

```json
[
  {
    "after_persisted_counts": {
      "chat_log": 2,
      "live_session_timeline": 6,
      "memory_chunks": 1,
      "memory_facts": 0,
      "promotion_events": 0,
      "schema_migrations": 1,
      "stream_chat_messages": 0,
      "stream_events": 2,
      "stream_sessions": 1,
      "viewer_promotion_profiles": 0
    },
    "before_persisted_counts": {
      "chat_log": 2,
      "live_session_timeline": 6,
      "memory_chunks": 1,
      "memory_facts": 0,
      "promotion_events": 0,
      "schema_migrations": 1,
      "stream_chat_messages": 0,
      "stream_events": 2,
      "stream_sessions": 1,
      "viewer_promotion_profiles": 0
    },
    "event_id": "restart",
    "new_engine_id": 2823652886224,
    "old_engine_collected": true,
    "old_engine_id": 2823652961808,
    "same_database": "C:\\Users\\Leo Nifelheim\\Documents\\Hebe\\hebe-ui\\artifacts\\cognitive-replay\\phase-0.5\\workspaces\\owner_correction_format\\hebe-replay.sqlite3",
    "volatile_state_recreated": true
  }
]
```

### raid_transition_foundation

- Status: **VERIFIED**
- Events: 3
- Restarts: 0
- Duration: 6.047282s
- Assertions passed/failed/skipped: 2/0/0


#### Checkpoint state

```json
{
  "end": {
    "actions": {
      "attempts": [
        {
          "operation": "twitch.send_message",
          "outcome": {
            "message_id": "raid-thanks-1",
            "status": "sent",
            "success": true
          },
          "payload": {
            "text": "Gracias por la raid, SyntheticRaider."
          }
        }
      ],
      "model_calls": [
        {
          "key": "stream_response:v1:none",
          "method": "chat"
        }
      ],
      "research_calls": [],
      "speech_requests": []
    },
    "beliefs": [],
    "current_scene": {},
    "database_watermarks": {
      "counts": {
        "chat_log": 0,
        "live_session_timeline": 4,
        "memory_chunks": 0,
        "memory_facts": 0,
        "promotion_events": 0,
        "schema_migrations": 1,
        "stream_chat_messages": 0,
        "stream_events": 3,
        "stream_sessions": 1,
        "viewer_promotion_profiles": 0
      },
      "final_response_digest": "",
      "final_response_present": false,
      "schema_migrations": [
        {
          "applied_at": "2026-08-11T07:28:42.618630+00:00",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "component": "cognitive_replay",
          "name": "replay_metadata",
          "version": 1
        }
      ]
    },
    "emitted_outputs": [
      {
        "emitted": true,
        "event_id": "raid",
        "reason": "",
        "route": "twitch_text_reply",
        "targets": [
          "twitch_chat"
        ],
        "text_digest": "",
        "text_present": false
      }
    ],
    "final_emission_results": [
      {
        "emitted": true,
        "event_id": "raid",
        "reason": "",
        "route": "twitch_text_reply",
        "targets": [
          "twitch_chat"
        ],
        "text_digest": "",
        "text_present": false
      }
    ],
    "game_state": {
      "challenge": "",
      "confidence": 0.0,
      "current_character": "",
      "current_game": null,
      "current_location": null,
      "current_objective": null,
      "game": "",
      "known_constraints": [],
      "last_confirmed_progress": "",
      "last_updated": 1786433322.6289573,
      "party_members": [],
      "platform_version": "",
      "playthrough_type": "casual",
      "provenance": "current_live_session",
      "recent_run_context_facts": [],
      "spoiler_policy": "no_spoilers"
    },
    "memory": {
      "chunks_count": 0,
      "facts_count": 0
    },
    "open_threads": [],
    "pending": {
      "clarification": null,
      "conversation_turn": null
    },
    "promotion_profiles": [],
    "receipts": [],
    "runtime": {
      "hebe_sleeping": false,
      "is_running": false,
      "last_firewall": {
        "authority": "system",
        "decision": "allow",
        "reason": "system_event",
        "source": "twitch_system"
      },
      "last_input_source": null,
      "last_intent": null,
      "last_policy": {
        "authority": null,
        "decision": null,
        "reason": null,
        "source": null
      },
      "mode": "active"
    },
    "social_state": {
      "last_cheer": {},
      "last_raid": {
        "display_name": "SyntheticRaider",
        "source": "eventsub",
        "ts": 1786464005.0,
        "user_login": "synthetic_raider",
        "viewer_count": 12
      },
      "recent_active_users": [],
      "recent_chat_count": 0
    },
    "stream_session": {
      "active_stream_session_id": null,
      "category": null,
      "enabled": true,
      "game": null,
      "is_live": false,
      "last_transition": "offline",
      "live_status_known": true,
      "title": null
    }
  },
  "raid": {
    "actions": {
      "attempts": [
        {
          "operation": "twitch.send_message",
          "outcome": {
            "message_id": "raid-thanks-1",
            "status": "sent",
            "success": true
          },
          "payload": {
            "text": "Gracias por la raid, SyntheticRaider."
          }
        }
      ],
      "model_calls": [
        {
          "key": "stream_response:v1:none",
          "method": "chat"
        }
      ],
      "research_calls": [],
      "speech_requests": []
    },
    "beliefs": [],
    "current_scene": {},
    "database_watermarks": {
      "counts": {
        "chat_log": 0,
        "live_session_timeline": 3,
        "memory_chunks": 0,
        "memory_facts": 0,
        "promotion_events": 0,
        "schema_migrations": 1,
        "stream_chat_messages": 0,
        "stream_events": 2,
        "stream_sessions": 1,
        "viewer_promotion_profiles": 0
      },
      "final_response_digest": "",
      "final_response_present": false,
      "schema_migrations": [
        {
          "applied_at": "2026-08-11T07:28:42.618630+00:00",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "component": "cognitive_replay",
          "name": "replay_metadata",
          "version": 1
        }
      ]
    },
    "emitted_outputs": [
      {
        "emitted": true,
        "event_id": "raid",
        "reason": "",
        "route": "twitch_text_reply",
        "targets": [
          "twitch_chat"
        ],
        "text_digest": "",
        "text_present": false
      }
    ],
    "final_emission_results": [
      {
        "emitted": true,
        "event_id": "raid",
        "reason": "",
        "route": "twitch_text_reply",
        "targets": [
          "twitch_chat"
        ],
        "text_digest": "",
        "text_present": false
      }
    ],
    "game_state": {
      "challenge": "",
      "confidence": 0.0,
      "current_character": "",
      "current_game": null,
      "current_location": null,
      "current_objective": null,
      "game": "",
      "known_constraints": [],
      "last_confirmed_progress": "",
      "last_updated": 1786433322.6289573,
      "party_members": [],
      "platform_version": "",
      "playthrough_type": "casual",
      "provenance": "current_live_session",
      "recent_run_context_facts": [],
      "spoiler_policy": "no_spoilers"
    },
    "memory": {
      "chunks_count": 0,
      "facts_count": 0
    },
    "open_threads": [],
    "pending": {
      "clarification": null,
      "conversation_turn": null
    },
    "promotion_profiles": [],
    "receipts": [],
    "runtime": {
      "hebe_sleeping": false,
      "is_running": false,
      "last_firewall": {
        "authority": "system",
        "decision": "allow",
        "reason": "system_event",
        "source": "twitch_system"
      },
      "last_input_source": null,
      "last_intent": null,
      "last_policy": {
        "authority": null,
        "decision": null,
        "reason": null,
        "source": null
      },
      "mode": "active"
    },
    "social_state": {
      "last_cheer": {},
      "last_raid": {
        "display_name": "SyntheticRaider",
        "source": "eventsub",
        "ts": 1786464005.0,
        "user_login": "synthetic_raider",
        "viewer_count": 12
      },
      "recent_active_users": [],
      "recent_chat_count": 0
    },
    "stream_session": {
      "active_stream_session_id": 1,
      "category": null,
      "enabled": true,
      "game": null,
      "is_live": true,
      "last_transition": "online",
      "live_status_known": true,
      "title": null
    }
  },
  "start": {
    "actions": {
      "attempts": [],
      "model_calls": [],
      "research_calls": [],
      "speech_requests": []
    },
    "beliefs": [],
    "current_scene": {},
    "database_watermarks": {
      "counts": {
        "chat_log": 0,
        "live_session_timeline": 2,
        "memory_chunks": 0,
        "memory_facts": 0,
        "promotion_events": 0,
        "schema_migrations": 1,
        "stream_chat_messages": 0,
        "stream_events": 1,
        "stream_sessions": 1,
        "viewer_promotion_profiles": 0
      },
      "final_response_digest": "",
      "final_response_present": false,
      "schema_migrations": [
        {
          "applied_at": "2026-08-11T07:28:42.618630+00:00",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "component": "cognitive_replay",
          "name": "replay_metadata",
          "version": 1
        }
      ]
    },
    "emitted_outputs": [],
    "final_emission_results": [],
    "game_state": {
      "challenge": "",
      "confidence": 0.0,
      "current_character": "",
      "current_game": null,
      "current_location": null,
      "current_objective": null,
      "game": "",
      "known_constraints": [],
      "last_confirmed_progress": "",
      "last_updated": 1786433322.6289573,
      "party_members": [],
      "platform_version": "",
      "playthrough_type": "casual",
      "provenance": "inferred",
      "recent_run_context_facts": [],
      "spoiler_policy": "spoiler_safe_hints"
    },
    "memory": {
      "chunks_count": 0,
      "facts_count": 0
    },
    "open_threads": [],
    "pending": {
      "clarification": null,
      "conversation_turn": null
    },
    "promotion_profiles": [],
    "receipts": [],
    "runtime": {
      "hebe_sleeping": false,
      "is_running": false,
      "last_firewall": {
        "authority": null,
        "decision": null,
        "reason": null,
        "source": null
      },
      "last_input_source": null,
      "last_intent": null,
      "last_policy": {
        "authority": null,
        "decision": null,
        "reason": null,
        "source": null
      },
      "mode": "active"
    },
    "social_state": {
      "last_cheer": {},
      "last_raid": {},
      "recent_active_users": [],
      "recent_chat_count": 0
    },
    "stream_session": {
      "active_stream_session_id": 1,
      "category": null,
      "enabled": true,
      "game": null,
      "is_live": true,
      "last_transition": "online",
      "live_status_known": true,
      "title": null
    }
  }
}
```

#### Final state and side effects

```json
{
  "actions": {
    "attempts": [
      {
        "operation": "twitch.send_message",
        "outcome": {
          "message_id": "raid-thanks-1",
          "status": "sent",
          "success": true
        },
        "payload": {
          "text": "Gracias por la raid, SyntheticRaider."
        }
      }
    ],
    "model_calls": [
      {
        "key": "stream_response:v1:none",
        "method": "chat"
      }
    ],
    "research_calls": [],
    "speech_requests": []
  },
  "beliefs": [],
  "current_scene": {},
  "database_watermarks": {
    "counts": {
      "chat_log": 0,
      "live_session_timeline": 4,
      "memory_chunks": 0,
      "memory_facts": 0,
      "promotion_events": 0,
      "schema_migrations": 1,
      "stream_chat_messages": 0,
      "stream_events": 3,
      "stream_sessions": 1,
      "viewer_promotion_profiles": 0
    },
    "final_response_digest": "",
    "final_response_present": false,
    "schema_migrations": [
      {
        "applied_at": "2026-08-11T07:28:42.618630+00:00",
        "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
        "component": "cognitive_replay",
        "name": "replay_metadata",
        "version": 1
      }
    ]
  },
  "emitted_outputs": [
    {
      "emitted": true,
      "event_id": "raid",
      "reason": "",
      "route": "twitch_text_reply",
      "targets": [
        "twitch_chat"
      ],
      "text_digest": "",
      "text_present": false
    }
  ],
  "final_emission_results": [
    {
      "emitted": true,
      "event_id": "raid",
      "reason": "",
      "route": "twitch_text_reply",
      "targets": [
        "twitch_chat"
      ],
      "text_digest": "",
      "text_present": false
    }
  ],
  "game_state": {
    "challenge": "",
    "confidence": 0.0,
    "current_character": "",
    "current_game": null,
    "current_location": null,
    "current_objective": null,
    "game": "",
    "known_constraints": [],
    "last_confirmed_progress": "",
    "last_updated": 1786433322.6289573,
    "party_members": [],
    "platform_version": "",
    "playthrough_type": "casual",
    "provenance": "current_live_session",
    "recent_run_context_facts": [],
    "spoiler_policy": "no_spoilers"
  },
  "memory": {
    "chunks_count": 0,
    "facts_count": 0
  },
  "open_threads": [],
  "pending": {
    "clarification": null,
    "conversation_turn": null
  },
  "promotion_profiles": [],
  "receipts": [],
  "runtime": {
    "hebe_sleeping": false,
    "is_running": false,
    "last_firewall": {
      "authority": "system",
      "decision": "allow",
      "reason": "system_event",
      "source": "twitch_system"
    },
    "last_input_source": null,
    "last_intent": null,
    "last_policy": {
      "authority": null,
      "decision": null,
      "reason": null,
      "source": null
    },
    "mode": "active"
  },
  "social_state": {
    "last_cheer": {},
    "last_raid": {
      "display_name": "SyntheticRaider",
      "source": "eventsub",
      "ts": 1786464005.0,
      "user_login": "synthetic_raider",
      "viewer_count": 12
    },
    "recent_active_users": [],
    "recent_chat_count": 0
  },
  "stream_session": {
    "active_stream_session_id": null,
    "category": null,
    "enabled": true,
    "game": null,
    "is_live": false,
    "last_transition": "offline",
    "live_status_known": true,
    "title": null
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
- Duration: 1.278153s
- Assertions passed/failed/skipped: 4/0/0


#### Checkpoint state

```json
{
  "alice-chat": {
    "actions": {
      "attempts": [],
      "model_calls": [],
      "research_calls": [],
      "speech_requests": []
    },
    "beliefs": [],
    "current_scene": {},
    "database_watermarks": {
      "counts": {
        "chat_log": 0,
        "live_session_timeline": 3,
        "memory_chunks": 0,
        "memory_facts": 0,
        "promotion_events": 0,
        "schema_migrations": 1,
        "stream_chat_messages": 1,
        "stream_events": 1,
        "stream_sessions": 1,
        "viewer_promotion_profiles": 0
      },
      "final_response_digest": "",
      "final_response_present": false,
      "schema_migrations": [
        {
          "applied_at": "2026-08-11T07:28:48.816593+00:00",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "component": "cognitive_replay",
          "name": "replay_metadata",
          "version": 1
        }
      ]
    },
    "emitted_outputs": [],
    "final_emission_results": [],
    "game_state": {
      "challenge": "",
      "confidence": 0.0,
      "current_character": "",
      "current_game": null,
      "current_location": null,
      "current_objective": null,
      "game": "",
      "known_constraints": [],
      "last_confirmed_progress": "",
      "last_updated": 1786433328.8273897,
      "party_members": [],
      "platform_version": "",
      "playthrough_type": "casual",
      "provenance": "inferred",
      "recent_run_context_facts": [],
      "spoiler_policy": "spoiler_safe_hints"
    },
    "memory": {
      "chunks_count": 0,
      "facts_count": 0
    },
    "open_threads": [],
    "pending": {
      "clarification": null,
      "conversation_turn": null
    },
    "promotion_profiles": [],
    "receipts": [],
    "runtime": {
      "hebe_sleeping": false,
      "is_running": false,
      "last_firewall": {
        "authority": "viewer",
        "decision": "allow",
        "reason": "live_viewer_message",
        "source": "twitch_viewer"
      },
      "last_input_source": null,
      "last_intent": null,
      "last_policy": {
        "authority": null,
        "decision": null,
        "reason": null,
        "source": null
      },
      "mode": "active"
    },
    "social_state": {
      "last_cheer": {},
      "last_raid": {},
      "recent_active_users": [
        "alice_test"
      ],
      "recent_chat_count": 1
    },
    "stream_session": {
      "active_stream_session_id": 1,
      "category": null,
      "enabled": true,
      "game": null,
      "is_live": true,
      "last_transition": "online",
      "live_status_known": true,
      "title": null
    }
  },
  "bob-chat": {
    "actions": {
      "attempts": [],
      "model_calls": [],
      "research_calls": [],
      "speech_requests": []
    },
    "beliefs": [],
    "current_scene": {},
    "database_watermarks": {
      "counts": {
        "chat_log": 0,
        "live_session_timeline": 4,
        "memory_chunks": 0,
        "memory_facts": 0,
        "promotion_events": 0,
        "schema_migrations": 1,
        "stream_chat_messages": 2,
        "stream_events": 1,
        "stream_sessions": 1,
        "viewer_promotion_profiles": 0
      },
      "final_response_digest": "",
      "final_response_present": false,
      "schema_migrations": [
        {
          "applied_at": "2026-08-11T07:28:48.816593+00:00",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "component": "cognitive_replay",
          "name": "replay_metadata",
          "version": 1
        }
      ]
    },
    "emitted_outputs": [],
    "final_emission_results": [],
    "game_state": {
      "challenge": "",
      "confidence": 0.0,
      "current_character": "",
      "current_game": null,
      "current_location": null,
      "current_objective": null,
      "game": "",
      "known_constraints": [],
      "last_confirmed_progress": "",
      "last_updated": 1786433328.8273897,
      "party_members": [],
      "platform_version": "",
      "playthrough_type": "casual",
      "provenance": "inferred",
      "recent_run_context_facts": [],
      "spoiler_policy": "spoiler_safe_hints"
    },
    "memory": {
      "chunks_count": 0,
      "facts_count": 0
    },
    "open_threads": [],
    "pending": {
      "clarification": null,
      "conversation_turn": null
    },
    "promotion_profiles": [],
    "receipts": [],
    "runtime": {
      "hebe_sleeping": false,
      "is_running": false,
      "last_firewall": {
        "authority": "viewer",
        "decision": "allow",
        "reason": "live_viewer_message",
        "source": "twitch_viewer"
      },
      "last_input_source": null,
      "last_intent": null,
      "last_policy": {
        "authority": null,
        "decision": null,
        "reason": null,
        "source": null
      },
      "mode": "active"
    },
    "social_state": {
      "last_cheer": {},
      "last_raid": {},
      "recent_active_users": [
        "alice_test",
        "bob_test"
      ],
      "recent_chat_count": 2
    },
    "stream_session": {
      "active_stream_session_id": 1,
      "category": null,
      "enabled": true,
      "game": null,
      "is_live": true,
      "last_transition": "online",
      "live_status_known": true,
      "title": null
    }
  },
  "carol-chat": {
    "actions": {
      "attempts": [],
      "model_calls": [],
      "research_calls": [],
      "speech_requests": []
    },
    "beliefs": [],
    "current_scene": {},
    "database_watermarks": {
      "counts": {
        "chat_log": 0,
        "live_session_timeline": 5,
        "memory_chunks": 0,
        "memory_facts": 0,
        "promotion_events": 0,
        "schema_migrations": 1,
        "stream_chat_messages": 3,
        "stream_events": 1,
        "stream_sessions": 1,
        "viewer_promotion_profiles": 0
      },
      "final_response_digest": "",
      "final_response_present": false,
      "schema_migrations": [
        {
          "applied_at": "2026-08-11T07:28:48.816593+00:00",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "component": "cognitive_replay",
          "name": "replay_metadata",
          "version": 1
        }
      ]
    },
    "emitted_outputs": [],
    "final_emission_results": [],
    "game_state": {
      "challenge": "",
      "confidence": 0.0,
      "current_character": "",
      "current_game": null,
      "current_location": null,
      "current_objective": null,
      "game": "",
      "known_constraints": [],
      "last_confirmed_progress": "",
      "last_updated": 1786433328.8273897,
      "party_members": [],
      "platform_version": "",
      "playthrough_type": "casual",
      "provenance": "inferred",
      "recent_run_context_facts": [],
      "spoiler_policy": "spoiler_safe_hints"
    },
    "memory": {
      "chunks_count": 0,
      "facts_count": 0
    },
    "open_threads": [],
    "pending": {
      "clarification": null,
      "conversation_turn": null
    },
    "promotion_profiles": [],
    "receipts": [],
    "runtime": {
      "hebe_sleeping": false,
      "is_running": false,
      "last_firewall": {
        "authority": "viewer",
        "decision": "allow",
        "reason": "live_viewer_message",
        "source": "twitch_viewer"
      },
      "last_input_source": null,
      "last_intent": null,
      "last_policy": {
        "authority": null,
        "decision": null,
        "reason": null,
        "source": null
      },
      "mode": "active"
    },
    "social_state": {
      "last_cheer": {},
      "last_raid": {},
      "recent_active_users": [
        "alice_test",
        "bob_test",
        "carol_test"
      ],
      "recent_chat_count": 3
    },
    "stream_session": {
      "active_stream_session_id": 1,
      "category": null,
      "enabled": true,
      "game": null,
      "is_live": true,
      "last_transition": "online",
      "live_status_known": true,
      "title": null
    }
  },
  "failure": {
    "actions": {
      "attempts": [
        {
          "operation": "twitch.shoutout",
          "outcome": {
            "message_id": "receipt-success",
            "status": "sent",
            "success": true
          },
          "payload": {
            "command": "!so alice_test",
            "target": "alice_test"
          }
        },
        {
          "operation": "twitch.shoutout",
          "outcome": {
            "reason": "synthetic_failure",
            "status": "failed",
            "success": false
          },
          "payload": {
            "command": "!so bob_test",
            "target": "bob_test"
          }
        }
      ],
      "model_calls": [],
      "research_calls": [],
      "speech_requests": []
    },
    "beliefs": [],
    "current_scene": {},
    "database_watermarks": {
      "counts": {
        "chat_log": 2,
        "live_session_timeline": 9,
        "memory_chunks": 0,
        "memory_facts": 0,
        "promotion_events": 2,
        "schema_migrations": 1,
        "stream_chat_messages": 3,
        "stream_events": 1,
        "stream_sessions": 1,
        "viewer_promotion_profiles": 1
      },
      "final_response_digest": "",
      "final_response_present": false,
      "schema_migrations": [
        {
          "applied_at": "2026-08-11T07:28:48.816593+00:00",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "component": "cognitive_replay",
          "name": "replay_metadata",
          "version": 1
        }
      ]
    },
    "emitted_outputs": [
      {
        "emitted": true,
        "event_id": "1786433329.253763",
        "reason": "",
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "text_digest": "",
        "text_present": false
      },
      {
        "emitted": true,
        "event_id": "1786433329.4493551",
        "reason": "",
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "text_digest": "",
        "text_present": false
      }
    ],
    "final_emission_results": [
      {
        "emitted": true,
        "event_id": "1786433329.253763",
        "reason": "",
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "text_digest": "",
        "text_present": false
      },
      {
        "emitted": true,
        "event_id": "1786433329.4493551",
        "reason": "",
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "text_digest": "",
        "text_present": false
      }
    ],
    "game_state": {
      "challenge": "",
      "confidence": 0.0,
      "current_character": "",
      "current_game": null,
      "current_location": null,
      "current_objective": null,
      "game": "",
      "known_constraints": [],
      "last_confirmed_progress": "",
      "last_updated": 1786433328.8273897,
      "party_members": [],
      "platform_version": "",
      "playthrough_type": "casual",
      "provenance": "current_live_session",
      "recent_run_context_facts": [],
      "spoiler_policy": "no_spoilers"
    },
    "memory": {
      "chunks_count": 0,
      "facts_count": 0
    },
    "open_threads": [],
    "pending": {
      "clarification": null,
      "conversation_turn": null
    },
    "promotion_profiles": [
      {
        "active": 1,
        "auto_promo_mode": "first_message_each_stream",
        "created_by": "owner_command",
        "current_login": "alice_test",
        "display_name": "alice_test",
        "last_promoted_stream_id": "1",
        "owner_locked": 1,
        "twitch_user_id": "501"
      }
    ],
    "receipts": [
      {
        "execution_status": "sent",
        "failure_reason": "",
        "id": "promo_9957df32356355528825eedeb574e7bb",
        "requested_by": "leo",
        "resolved_login": "alice_test",
        "resolved_twitch_user_id": "501",
        "source_event_id": "1786433329.253763",
        "stream_session_id": "1",
        "trigger_type": "owner_learn_and_execute",
        "twitch_message_id": "receipt-success"
      },
      {
        "execution_status": "failed",
        "failure_reason": "send_failed: RuntimeError: Twitch shoutout command returned false",
        "id": "promo_50152700fa2556f585f71dd7baab9482",
        "requested_by": "leo",
        "resolved_login": "bob_test",
        "resolved_twitch_user_id": "502",
        "source_event_id": "1786433329.4493551",
        "stream_session_id": "1",
        "trigger_type": "owner_learn_and_execute",
        "twitch_message_id": ""
      }
    ],
    "runtime": {
      "hebe_sleeping": false,
      "is_running": false,
      "last_firewall": {
        "authority": "owner",
        "decision": "allow",
        "reason": "owner_direct",
        "source": "owner_stt_direct"
      },
      "last_input_source": null,
      "last_intent": null,
      "last_policy": {
        "authority": null,
        "decision": null,
        "reason": null,
        "source": null
      },
      "mode": "active"
    },
    "social_state": {
      "last_cheer": {},
      "last_raid": {},
      "recent_active_users": [
        "alice_test",
        "bob_test",
        "carol_test"
      ],
      "recent_chat_count": 3
    },
    "stream_session": {
      "active_stream_session_id": 1,
      "category": null,
      "enabled": true,
      "game": null,
      "is_live": true,
      "last_transition": "online",
      "live_status_known": true,
      "title": null
    }
  },
  "start": {
    "actions": {
      "attempts": [],
      "model_calls": [],
      "research_calls": [],
      "speech_requests": []
    },
    "beliefs": [],
    "current_scene": {},
    "database_watermarks": {
      "counts": {
        "chat_log": 0,
        "live_session_timeline": 2,
        "memory_chunks": 0,
        "memory_facts": 0,
        "promotion_events": 0,
        "schema_migrations": 1,
        "stream_chat_messages": 0,
        "stream_events": 1,
        "stream_sessions": 1,
        "viewer_promotion_profiles": 0
      },
      "final_response_digest": "",
      "final_response_present": false,
      "schema_migrations": [
        {
          "applied_at": "2026-08-11T07:28:48.816593+00:00",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "component": "cognitive_replay",
          "name": "replay_metadata",
          "version": 1
        }
      ]
    },
    "emitted_outputs": [],
    "final_emission_results": [],
    "game_state": {
      "challenge": "",
      "confidence": 0.0,
      "current_character": "",
      "current_game": null,
      "current_location": null,
      "current_objective": null,
      "game": "",
      "known_constraints": [],
      "last_confirmed_progress": "",
      "last_updated": 1786433328.8273897,
      "party_members": [],
      "platform_version": "",
      "playthrough_type": "casual",
      "provenance": "inferred",
      "recent_run_context_facts": [],
      "spoiler_policy": "spoiler_safe_hints"
    },
    "memory": {
      "chunks_count": 0,
      "facts_count": 0
    },
    "open_threads": [],
    "pending": {
      "clarification": null,
      "conversation_turn": null
    },
    "promotion_profiles": [],
    "receipts": [],
    "runtime": {
      "hebe_sleeping": false,
      "is_running": false,
      "last_firewall": {
        "authority": null,
        "decision": null,
        "reason": null,
        "source": null
      },
      "last_input_source": null,
      "last_intent": null,
      "last_policy": {
        "authority": null,
        "decision": null,
        "reason": null,
        "source": null
      },
      "mode": "active"
    },
    "social_state": {
      "last_cheer": {},
      "last_raid": {},
      "recent_active_users": [],
      "recent_chat_count": 0
    },
    "stream_session": {
      "active_stream_session_id": 1,
      "category": null,
      "enabled": true,
      "game": null,
      "is_live": true,
      "last_transition": "online",
      "live_status_known": true,
      "title": null
    }
  },
  "success": {
    "actions": {
      "attempts": [
        {
          "operation": "twitch.shoutout",
          "outcome": {
            "message_id": "receipt-success",
            "status": "sent",
            "success": true
          },
          "payload": {
            "command": "!so alice_test",
            "target": "alice_test"
          }
        }
      ],
      "model_calls": [],
      "research_calls": [],
      "speech_requests": []
    },
    "beliefs": [],
    "current_scene": {},
    "database_watermarks": {
      "counts": {
        "chat_log": 1,
        "live_session_timeline": 7,
        "memory_chunks": 0,
        "memory_facts": 0,
        "promotion_events": 1,
        "schema_migrations": 1,
        "stream_chat_messages": 3,
        "stream_events": 1,
        "stream_sessions": 1,
        "viewer_promotion_profiles": 1
      },
      "final_response_digest": "",
      "final_response_present": false,
      "schema_migrations": [
        {
          "applied_at": "2026-08-11T07:28:48.816593+00:00",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "component": "cognitive_replay",
          "name": "replay_metadata",
          "version": 1
        }
      ]
    },
    "emitted_outputs": [
      {
        "emitted": true,
        "event_id": "1786433329.253763",
        "reason": "",
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "text_digest": "",
        "text_present": false
      }
    ],
    "final_emission_results": [
      {
        "emitted": true,
        "event_id": "1786433329.253763",
        "reason": "",
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "text_digest": "",
        "text_present": false
      }
    ],
    "game_state": {
      "challenge": "",
      "confidence": 0.0,
      "current_character": "",
      "current_game": null,
      "current_location": null,
      "current_objective": null,
      "game": "",
      "known_constraints": [],
      "last_confirmed_progress": "",
      "last_updated": 1786433328.8273897,
      "party_members": [],
      "platform_version": "",
      "playthrough_type": "casual",
      "provenance": "current_live_session",
      "recent_run_context_facts": [],
      "spoiler_policy": "no_spoilers"
    },
    "memory": {
      "chunks_count": 0,
      "facts_count": 0
    },
    "open_threads": [],
    "pending": {
      "clarification": null,
      "conversation_turn": null
    },
    "promotion_profiles": [
      {
        "active": 1,
        "auto_promo_mode": "first_message_each_stream",
        "created_by": "owner_command",
        "current_login": "alice_test",
        "display_name": "alice_test",
        "last_promoted_stream_id": "1",
        "owner_locked": 1,
        "twitch_user_id": "501"
      }
    ],
    "receipts": [
      {
        "execution_status": "sent",
        "failure_reason": "",
        "id": "promo_9957df32356355528825eedeb574e7bb",
        "requested_by": "leo",
        "resolved_login": "alice_test",
        "resolved_twitch_user_id": "501",
        "source_event_id": "1786433329.253763",
        "stream_session_id": "1",
        "trigger_type": "owner_learn_and_execute",
        "twitch_message_id": "receipt-success"
      }
    ],
    "runtime": {
      "hebe_sleeping": false,
      "is_running": false,
      "last_firewall": {
        "authority": "owner",
        "decision": "allow",
        "reason": "owner_direct",
        "source": "owner_stt_direct"
      },
      "last_input_source": null,
      "last_intent": null,
      "last_policy": {
        "authority": null,
        "decision": null,
        "reason": null,
        "source": null
      },
      "mode": "active"
    },
    "social_state": {
      "last_cheer": {},
      "last_raid": {},
      "recent_active_users": [
        "alice_test",
        "bob_test",
        "carol_test"
      ],
      "recent_chat_count": 3
    },
    "stream_session": {
      "active_stream_session_id": 1,
      "category": null,
      "enabled": true,
      "game": null,
      "is_live": true,
      "last_transition": "online",
      "live_status_known": true,
      "title": null
    }
  },
  "timeout": {
    "actions": {
      "attempts": [
        {
          "operation": "twitch.shoutout",
          "outcome": {
            "message_id": "receipt-success",
            "status": "sent",
            "success": true
          },
          "payload": {
            "command": "!so alice_test",
            "target": "alice_test"
          }
        },
        {
          "operation": "twitch.shoutout",
          "outcome": {
            "reason": "synthetic_failure",
            "status": "failed",
            "success": false
          },
          "payload": {
            "command": "!so bob_test",
            "target": "bob_test"
          }
        },
        {
          "operation": "twitch.shoutout",
          "outcome": {
            "reason": "synthetic_timeout",
            "status": "timeout",
            "success": false
          },
          "payload": {
            "command": "!so carol_test",
            "target": "carol_test"
          }
        }
      ],
      "model_calls": [],
      "research_calls": [],
      "speech_requests": []
    },
    "beliefs": [],
    "current_scene": {},
    "database_watermarks": {
      "counts": {
        "chat_log": 3,
        "live_session_timeline": 11,
        "memory_chunks": 0,
        "memory_facts": 0,
        "promotion_events": 3,
        "schema_migrations": 1,
        "stream_chat_messages": 3,
        "stream_events": 1,
        "stream_sessions": 1,
        "viewer_promotion_profiles": 1
      },
      "final_response_digest": "",
      "final_response_present": false,
      "schema_migrations": [
        {
          "applied_at": "2026-08-11T07:28:48.816593+00:00",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "component": "cognitive_replay",
          "name": "replay_metadata",
          "version": 1
        }
      ]
    },
    "emitted_outputs": [
      {
        "emitted": true,
        "event_id": "1786433329.253763",
        "reason": "",
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "text_digest": "",
        "text_present": false
      },
      {
        "emitted": true,
        "event_id": "1786433329.4493551",
        "reason": "",
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "text_digest": "",
        "text_present": false
      },
      {
        "emitted": true,
        "event_id": "1786433329.6374128",
        "reason": "",
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "text_digest": "",
        "text_present": false
      }
    ],
    "final_emission_results": [
      {
        "emitted": true,
        "event_id": "1786433329.253763",
        "reason": "",
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "text_digest": "",
        "text_present": false
      },
      {
        "emitted": true,
        "event_id": "1786433329.4493551",
        "reason": "",
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "text_digest": "",
        "text_present": false
      },
      {
        "emitted": true,
        "event_id": "1786433329.6374128",
        "reason": "",
        "route": "local_owner_reply",
        "targets": [
          "local_ui"
        ],
        "text_digest": "",
        "text_present": false
      }
    ],
    "game_state": {
      "challenge": "",
      "confidence": 0.0,
      "current_character": "",
      "current_game": null,
      "current_location": null,
      "current_objective": null,
      "game": "",
      "known_constraints": [],
      "last_confirmed_progress": "",
      "last_updated": 1786433328.8273897,
      "party_members": [],
      "platform_version": "",
      "playthrough_type": "casual",
      "provenance": "current_live_session",
      "recent_run_context_facts": [],
      "spoiler_policy": "no_spoilers"
    },
    "memory": {
      "chunks_count": 0,
      "facts_count": 0
    },
    "open_threads": [],
    "pending": {
      "clarification": null,
      "conversation_turn": null
    },
    "promotion_profiles": [
      {
        "active": 1,
        "auto_promo_mode": "first_message_each_stream",
        "created_by": "owner_command",
        "current_login": "alice_test",
        "display_name": "alice_test",
        "last_promoted_stream_id": "1",
        "owner_locked": 1,
        "twitch_user_id": "501"
      }
    ],
    "receipts": [
      {
        "execution_status": "sent",
        "failure_reason": "",
        "id": "promo_9957df32356355528825eedeb574e7bb",
        "requested_by": "leo",
        "resolved_login": "alice_test",
        "resolved_twitch_user_id": "501",
        "source_event_id": "1786433329.253763",
        "stream_session_id": "1",
        "trigger_type": "owner_learn_and_execute",
        "twitch_message_id": "receipt-success"
      },
      {
        "execution_status": "failed",
        "failure_reason": "send_failed: RuntimeError: Twitch shoutout command returned false",
        "id": "promo_50152700fa2556f585f71dd7baab9482",
        "requested_by": "leo",
        "resolved_login": "bob_test",
        "resolved_twitch_user_id": "502",
        "source_event_id": "1786433329.4493551",
        "stream_session_id": "1",
        "trigger_type": "owner_learn_and_execute",
        "twitch_message_id": ""
      },
      {
        "execution_status": "failed",
        "failure_reason": "send_failed: TimeoutError: synthetic_timeout",
        "id": "promo_91c44fbd1d025a0aafdcce244be1fd04",
        "requested_by": "leo",
        "resolved_login": "carol_test",
        "resolved_twitch_user_id": "503",
        "source_event_id": "1786433329.6374128",
        "stream_session_id": "1",
        "trigger_type": "owner_learn_and_execute",
        "twitch_message_id": ""
      }
    ],
    "runtime": {
      "hebe_sleeping": false,
      "is_running": false,
      "last_firewall": {
        "authority": "owner",
        "decision": "allow",
        "reason": "owner_direct",
        "source": "owner_stt_direct"
      },
      "last_input_source": null,
      "last_intent": null,
      "last_policy": {
        "authority": null,
        "decision": null,
        "reason": null,
        "source": null
      },
      "mode": "active"
    },
    "social_state": {
      "last_cheer": {},
      "last_raid": {},
      "recent_active_users": [
        "alice_test",
        "bob_test",
        "carol_test"
      ],
      "recent_chat_count": 3
    },
    "stream_session": {
      "active_stream_session_id": 1,
      "category": null,
      "enabled": true,
      "game": null,
      "is_live": true,
      "last_transition": "online",
      "live_status_known": true,
      "title": null
    }
  }
}
```

#### Final state and side effects

```json
{
  "actions": {
    "attempts": [
      {
        "operation": "twitch.shoutout",
        "outcome": {
          "message_id": "receipt-success",
          "status": "sent",
          "success": true
        },
        "payload": {
          "command": "!so alice_test",
          "target": "alice_test"
        }
      },
      {
        "operation": "twitch.shoutout",
        "outcome": {
          "reason": "synthetic_failure",
          "status": "failed",
          "success": false
        },
        "payload": {
          "command": "!so bob_test",
          "target": "bob_test"
        }
      },
      {
        "operation": "twitch.shoutout",
        "outcome": {
          "reason": "synthetic_timeout",
          "status": "timeout",
          "success": false
        },
        "payload": {
          "command": "!so carol_test",
          "target": "carol_test"
        }
      }
    ],
    "model_calls": [],
    "research_calls": [],
    "speech_requests": []
  },
  "beliefs": [],
  "current_scene": {},
  "database_watermarks": {
    "counts": {
      "chat_log": 3,
      "live_session_timeline": 11,
      "memory_chunks": 0,
      "memory_facts": 0,
      "promotion_events": 3,
      "schema_migrations": 1,
      "stream_chat_messages": 3,
      "stream_events": 1,
      "stream_sessions": 1,
      "viewer_promotion_profiles": 1
    },
    "final_response_digest": "",
    "final_response_present": false,
    "schema_migrations": [
      {
        "applied_at": "2026-08-11T07:28:48.816593+00:00",
        "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
        "component": "cognitive_replay",
        "name": "replay_metadata",
        "version": 1
      }
    ]
  },
  "emitted_outputs": [
    {
      "emitted": true,
      "event_id": "1786433329.253763",
      "reason": "",
      "route": "local_owner_reply",
      "targets": [
        "local_ui"
      ],
      "text_digest": "",
      "text_present": false
    },
    {
      "emitted": true,
      "event_id": "1786433329.4493551",
      "reason": "",
      "route": "local_owner_reply",
      "targets": [
        "local_ui"
      ],
      "text_digest": "",
      "text_present": false
    },
    {
      "emitted": true,
      "event_id": "1786433329.6374128",
      "reason": "",
      "route": "local_owner_reply",
      "targets": [
        "local_ui"
      ],
      "text_digest": "",
      "text_present": false
    }
  ],
  "final_emission_results": [
    {
      "emitted": true,
      "event_id": "1786433329.253763",
      "reason": "",
      "route": "local_owner_reply",
      "targets": [
        "local_ui"
      ],
      "text_digest": "",
      "text_present": false
    },
    {
      "emitted": true,
      "event_id": "1786433329.4493551",
      "reason": "",
      "route": "local_owner_reply",
      "targets": [
        "local_ui"
      ],
      "text_digest": "",
      "text_present": false
    },
    {
      "emitted": true,
      "event_id": "1786433329.6374128",
      "reason": "",
      "route": "local_owner_reply",
      "targets": [
        "local_ui"
      ],
      "text_digest": "",
      "text_present": false
    }
  ],
  "game_state": {
    "challenge": "",
    "confidence": 0.0,
    "current_character": "",
    "current_game": null,
    "current_location": null,
    "current_objective": null,
    "game": "",
    "known_constraints": [],
    "last_confirmed_progress": "",
    "last_updated": 1786433328.8273897,
    "party_members": [],
    "platform_version": "",
    "playthrough_type": "casual",
    "provenance": "current_live_session",
    "recent_run_context_facts": [],
    "spoiler_policy": "no_spoilers"
  },
  "memory": {
    "chunks_count": 0,
    "facts_count": 0
  },
  "open_threads": [],
  "pending": {
    "clarification": null,
    "conversation_turn": null
  },
  "promotion_profiles": [
    {
      "active": 1,
      "auto_promo_mode": "first_message_each_stream",
      "created_by": "owner_command",
      "current_login": "alice_test",
      "display_name": "alice_test",
      "last_promoted_stream_id": "1",
      "owner_locked": 1,
      "twitch_user_id": "501"
    }
  ],
  "receipts": [
    {
      "execution_status": "sent",
      "failure_reason": "",
      "id": "promo_9957df32356355528825eedeb574e7bb",
      "requested_by": "leo",
      "resolved_login": "alice_test",
      "resolved_twitch_user_id": "501",
      "source_event_id": "1786433329.253763",
      "stream_session_id": "1",
      "trigger_type": "owner_learn_and_execute",
      "twitch_message_id": "receipt-success"
    },
    {
      "execution_status": "failed",
      "failure_reason": "send_failed: RuntimeError: Twitch shoutout command returned false",
      "id": "promo_50152700fa2556f585f71dd7baab9482",
      "requested_by": "leo",
      "resolved_login": "bob_test",
      "resolved_twitch_user_id": "502",
      "source_event_id": "1786433329.4493551",
      "stream_session_id": "1",
      "trigger_type": "owner_learn_and_execute",
      "twitch_message_id": ""
    },
    {
      "execution_status": "failed",
      "failure_reason": "send_failed: TimeoutError: synthetic_timeout",
      "id": "promo_91c44fbd1d025a0aafdcce244be1fd04",
      "requested_by": "leo",
      "resolved_login": "carol_test",
      "resolved_twitch_user_id": "503",
      "source_event_id": "1786433329.6374128",
      "stream_session_id": "1",
      "trigger_type": "owner_learn_and_execute",
      "twitch_message_id": ""
    }
  ],
  "runtime": {
    "hebe_sleeping": false,
    "is_running": false,
    "last_firewall": {
      "authority": "owner",
      "decision": "allow",
      "reason": "owner_direct",
      "source": "owner_stt_direct"
    },
    "last_input_source": null,
    "last_intent": null,
    "last_policy": {
      "authority": null,
      "decision": null,
      "reason": null,
      "source": null
    },
    "mode": "active"
  },
  "social_state": {
    "last_cheer": {},
    "last_raid": {},
    "recent_active_users": [
      "alice_test",
      "bob_test",
      "carol_test"
    ],
    "recent_chat_count": 3
  },
  "stream_session": {
    "active_stream_session_id": 1,
    "category": null,
    "enabled": true,
    "game": null,
    "is_live": true,
    "last_transition": "online",
    "live_status_known": true,
    "title": null
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
- Duration: 0.42873s
- Assertions passed/failed/skipped: 2/0/0


#### Checkpoint state

```json
{
  "research": {
    "actions": {
      "attempts": [],
      "model_calls": [],
      "research_calls": [
        {
          "constraints": "{\"entity\": \"core systems and spoiler-free premise\", \"expected_fact_type\": \"general_mechanics\", \"spoiler_limit\": \"strict\", \"strict_first_playthrough\": true}",
          "key": "Final Fantasy V core systems and spoiler-free premise spoiler-safe no future story information",
          "query": "Final Fantasy V core systems and spoiler-free premise spoiler-safe no future story information"
        }
      ],
      "speech_requests": []
    },
    "beliefs": [],
    "current_scene": {},
    "database_watermarks": {
      "counts": {
        "chat_log": 0,
        "live_session_timeline": 0,
        "memory_chunks": 0,
        "memory_facts": 0,
        "promotion_events": 0,
        "schema_migrations": 1,
        "stream_chat_messages": 0,
        "stream_events": 0,
        "stream_sessions": 0,
        "viewer_promotion_profiles": 0
      },
      "final_response_digest": "",
      "final_response_present": false,
      "schema_migrations": [
        {
          "applied_at": "2026-08-11T07:28:50.201630+00:00",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "component": "cognitive_replay",
          "name": "replay_metadata",
          "version": 1
        }
      ]
    },
    "emitted_outputs": [],
    "final_emission_results": [],
    "game_state": {
      "challenge": "",
      "confidence": 0.0,
      "current_character": "",
      "current_game": null,
      "current_location": null,
      "current_objective": null,
      "game": "",
      "known_constraints": [],
      "last_confirmed_progress": "",
      "last_updated": 1786433330.2143204,
      "party_members": [],
      "platform_version": "",
      "playthrough_type": "casual",
      "provenance": "inferred",
      "recent_run_context_facts": [],
      "spoiler_policy": "spoiler_safe_hints"
    },
    "memory": {
      "chunks_count": 0,
      "facts_count": 0
    },
    "open_threads": [],
    "pending": {
      "clarification": null,
      "conversation_turn": null
    },
    "promotion_profiles": [],
    "receipts": [],
    "runtime": {
      "hebe_sleeping": false,
      "is_running": false,
      "last_firewall": {
        "authority": null,
        "decision": null,
        "reason": null,
        "source": null
      },
      "last_input_source": null,
      "last_intent": null,
      "last_policy": {
        "authority": null,
        "decision": null,
        "reason": null,
        "source": null
      },
      "mode": "active"
    },
    "social_state": {
      "last_cheer": {},
      "last_raid": {},
      "recent_active_users": [],
      "recent_chat_count": 0
    },
    "stream_session": {
      "active_stream_session_id": null,
      "category": null,
      "enabled": false,
      "game": null,
      "is_live": false,
      "last_transition": null,
      "live_status_known": false,
      "title": null
    }
  }
}
```

#### Final state and side effects

```json
{
  "actions": {
    "attempts": [],
    "model_calls": [],
    "research_calls": [
      {
        "constraints": "{\"entity\": \"core systems and spoiler-free premise\", \"expected_fact_type\": \"general_mechanics\", \"spoiler_limit\": \"strict\", \"strict_first_playthrough\": true}",
        "key": "Final Fantasy V core systems and spoiler-free premise spoiler-safe no future story information",
        "query": "Final Fantasy V core systems and spoiler-free premise spoiler-safe no future story information"
      }
    ],
    "speech_requests": []
  },
  "beliefs": [],
  "current_scene": {},
  "database_watermarks": {
    "counts": {
      "chat_log": 0,
      "live_session_timeline": 0,
      "memory_chunks": 0,
      "memory_facts": 0,
      "promotion_events": 0,
      "schema_migrations": 1,
      "stream_chat_messages": 0,
      "stream_events": 0,
      "stream_sessions": 0,
      "viewer_promotion_profiles": 0
    },
    "final_response_digest": "",
    "final_response_present": false,
    "schema_migrations": [
      {
        "applied_at": "2026-08-11T07:28:50.201630+00:00",
        "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
        "component": "cognitive_replay",
        "name": "replay_metadata",
        "version": 1
      }
    ]
  },
  "emitted_outputs": [],
  "final_emission_results": [],
  "game_state": {
    "challenge": "",
    "confidence": 0.0,
    "current_character": "",
    "current_game": null,
    "current_location": null,
    "current_objective": null,
    "game": "",
    "known_constraints": [],
    "last_confirmed_progress": "",
    "last_updated": 1786433330.2143204,
    "party_members": [],
    "platform_version": "",
    "playthrough_type": "casual",
    "provenance": "inferred",
    "recent_run_context_facts": [],
    "spoiler_policy": "spoiler_safe_hints"
  },
  "memory": {
    "chunks_count": 0,
    "facts_count": 0
  },
  "open_threads": [],
  "pending": {
    "clarification": null,
    "conversation_turn": null
  },
  "promotion_profiles": [],
  "receipts": [],
  "runtime": {
    "hebe_sleeping": false,
    "is_running": false,
    "last_firewall": {
      "authority": null,
      "decision": null,
      "reason": null,
      "source": null
    },
    "last_input_source": null,
    "last_intent": null,
    "last_policy": {
      "authority": null,
      "decision": null,
      "reason": null,
      "source": null
    },
    "mode": "active"
  },
  "social_state": {
    "last_cheer": {},
    "last_raid": {},
    "recent_active_users": [],
    "recent_chat_count": 0
  },
  "stream_session": {
    "active_stream_session_id": null,
    "category": null,
    "enabled": false,
    "game": null,
    "is_live": false,
    "last_transition": null,
    "live_status_known": false,
    "title": null
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
- Duration: 6.610955s
- Assertions passed/failed/skipped: 0/0/1


#### Checkpoint state

```json
{
  "end-1": {
    "actions": {
      "attempts": [],
      "model_calls": [],
      "research_calls": [],
      "speech_requests": []
    },
    "beliefs": [],
    "current_scene": {},
    "database_watermarks": {
      "counts": {
        "chat_log": 0,
        "live_session_timeline": 4,
        "memory_chunks": 1,
        "memory_facts": 0,
        "promotion_events": 0,
        "schema_migrations": 1,
        "stream_chat_messages": 1,
        "stream_events": 2,
        "stream_sessions": 1,
        "viewer_promotion_profiles": 0
      },
      "final_response_digest": "",
      "final_response_present": false,
      "schema_migrations": [
        {
          "applied_at": "2026-08-11T07:28:50.638959+00:00",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "component": "cognitive_replay",
          "name": "replay_metadata",
          "version": 1
        }
      ]
    },
    "emitted_outputs": [],
    "final_emission_results": [],
    "game_state": {
      "challenge": "",
      "confidence": 0.0,
      "current_character": "",
      "current_game": null,
      "current_location": null,
      "current_objective": null,
      "game": "",
      "known_constraints": [],
      "last_confirmed_progress": "",
      "last_updated": 1786433330.6558828,
      "party_members": [],
      "platform_version": "",
      "playthrough_type": "casual",
      "provenance": "inferred",
      "recent_run_context_facts": [],
      "spoiler_policy": "spoiler_safe_hints"
    },
    "memory": {
      "chunks_count": 1,
      "facts_count": 0
    },
    "open_threads": [],
    "pending": {
      "clarification": null,
      "conversation_turn": null
    },
    "promotion_profiles": [],
    "receipts": [],
    "runtime": {
      "hebe_sleeping": false,
      "is_running": false,
      "last_firewall": {
        "authority": "viewer",
        "decision": "allow",
        "reason": "live_viewer_message",
        "source": "twitch_viewer"
      },
      "last_input_source": null,
      "last_intent": null,
      "last_policy": {
        "authority": null,
        "decision": null,
        "reason": null,
        "source": null
      },
      "mode": "active"
    },
    "social_state": {
      "last_cheer": {},
      "last_raid": {},
      "recent_active_users": [
        "synthetic_student"
      ],
      "recent_chat_count": 1
    },
    "stream_session": {
      "active_stream_session_id": null,
      "category": null,
      "enabled": true,
      "game": null,
      "is_live": false,
      "last_transition": "offline",
      "live_status_known": true,
      "title": null
    }
  },
  "expire": {
    "actions": {
      "attempts": [],
      "model_calls": [],
      "research_calls": [],
      "speech_requests": []
    },
    "beliefs": [],
    "current_scene": {},
    "database_watermarks": {
      "counts": {
        "chat_log": 0,
        "live_session_timeline": 8,
        "memory_chunks": 2,
        "memory_facts": 0,
        "promotion_events": 0,
        "schema_migrations": 1,
        "stream_chat_messages": 2,
        "stream_events": 3,
        "stream_sessions": 2,
        "viewer_promotion_profiles": 0
      },
      "final_response_digest": "",
      "final_response_present": false,
      "schema_migrations": [
        {
          "applied_at": "2026-08-11T07:28:50.638959+00:00",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "component": "cognitive_replay",
          "name": "replay_metadata",
          "version": 1
        }
      ]
    },
    "emitted_outputs": [],
    "final_emission_results": [],
    "game_state": {
      "challenge": "",
      "confidence": 0.0,
      "current_character": "",
      "current_game": null,
      "current_location": null,
      "current_objective": null,
      "game": "",
      "known_constraints": [],
      "last_confirmed_progress": "",
      "last_updated": 1786433336.666283,
      "party_members": [],
      "platform_version": "",
      "playthrough_type": "casual",
      "provenance": "inferred",
      "recent_run_context_facts": [],
      "spoiler_policy": "spoiler_safe_hints"
    },
    "memory": {
      "chunks_count": 2,
      "facts_count": 0
    },
    "open_threads": [],
    "pending": {
      "clarification": null,
      "conversation_turn": null
    },
    "promotion_profiles": [],
    "receipts": [],
    "runtime": {
      "hebe_sleeping": false,
      "is_running": false,
      "last_firewall": {
        "authority": "viewer",
        "decision": "allow",
        "reason": "live_viewer_message",
        "source": "twitch_viewer"
      },
      "last_input_source": null,
      "last_intent": null,
      "last_policy": {
        "authority": null,
        "decision": null,
        "reason": null,
        "source": null
      },
      "mode": "active"
    },
    "social_state": {
      "last_cheer": {},
      "last_raid": {},
      "recent_active_users": [
        "synthetic_student"
      ],
      "recent_chat_count": 1
    },
    "stream_session": {
      "active_stream_session_id": 2,
      "category": null,
      "enabled": true,
      "game": null,
      "is_live": true,
      "last_transition": "online",
      "live_status_known": true,
      "title": null
    }
  },
  "ill": {
    "actions": {
      "attempts": [],
      "model_calls": [],
      "research_calls": [],
      "speech_requests": []
    },
    "beliefs": [],
    "current_scene": {},
    "database_watermarks": {
      "counts": {
        "chat_log": 0,
        "live_session_timeline": 3,
        "memory_chunks": 0,
        "memory_facts": 0,
        "promotion_events": 0,
        "schema_migrations": 1,
        "stream_chat_messages": 1,
        "stream_events": 1,
        "stream_sessions": 1,
        "viewer_promotion_profiles": 0
      },
      "final_response_digest": "",
      "final_response_present": false,
      "schema_migrations": [
        {
          "applied_at": "2026-08-11T07:28:50.638959+00:00",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "component": "cognitive_replay",
          "name": "replay_metadata",
          "version": 1
        }
      ]
    },
    "emitted_outputs": [],
    "final_emission_results": [],
    "game_state": {
      "challenge": "",
      "confidence": 0.0,
      "current_character": "",
      "current_game": null,
      "current_location": null,
      "current_objective": null,
      "game": "",
      "known_constraints": [],
      "last_confirmed_progress": "",
      "last_updated": 1786433330.6558828,
      "party_members": [],
      "platform_version": "",
      "playthrough_type": "casual",
      "provenance": "inferred",
      "recent_run_context_facts": [],
      "spoiler_policy": "spoiler_safe_hints"
    },
    "memory": {
      "chunks_count": 0,
      "facts_count": 0
    },
    "open_threads": [],
    "pending": {
      "clarification": null,
      "conversation_turn": null
    },
    "promotion_profiles": [],
    "receipts": [],
    "runtime": {
      "hebe_sleeping": false,
      "is_running": false,
      "last_firewall": {
        "authority": "viewer",
        "decision": "allow",
        "reason": "live_viewer_message",
        "source": "twitch_viewer"
      },
      "last_input_source": null,
      "last_intent": null,
      "last_policy": {
        "authority": null,
        "decision": null,
        "reason": null,
        "source": null
      },
      "mode": "active"
    },
    "social_state": {
      "last_cheer": {},
      "last_raid": {},
      "recent_active_users": [
        "synthetic_student"
      ],
      "recent_chat_count": 1
    },
    "stream_session": {
      "active_stream_session_id": 1,
      "category": null,
      "enabled": true,
      "game": null,
      "is_live": true,
      "last_transition": "online",
      "live_status_known": true,
      "title": null
    }
  },
  "restart": {
    "actions": {
      "attempts": [],
      "model_calls": [],
      "research_calls": [],
      "speech_requests": []
    },
    "beliefs": [],
    "current_scene": {},
    "database_watermarks": {
      "counts": {
        "chat_log": 0,
        "live_session_timeline": 4,
        "memory_chunks": 1,
        "memory_facts": 0,
        "promotion_events": 0,
        "schema_migrations": 1,
        "stream_chat_messages": 1,
        "stream_events": 2,
        "stream_sessions": 1,
        "viewer_promotion_profiles": 0
      },
      "final_response_digest": "",
      "final_response_present": false,
      "schema_migrations": [
        {
          "applied_at": "2026-08-11T07:28:50.638959+00:00",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "component": "cognitive_replay",
          "name": "replay_metadata",
          "version": 1
        }
      ]
    },
    "emitted_outputs": [],
    "final_emission_results": [],
    "game_state": {
      "challenge": "",
      "confidence": 0.0,
      "current_character": "",
      "current_game": null,
      "current_location": null,
      "current_objective": null,
      "game": "",
      "known_constraints": [],
      "last_confirmed_progress": "",
      "last_updated": 1786433336.666283,
      "party_members": [],
      "platform_version": "",
      "playthrough_type": "casual",
      "provenance": "inferred",
      "recent_run_context_facts": [],
      "spoiler_policy": "spoiler_safe_hints"
    },
    "memory": {
      "chunks_count": 1,
      "facts_count": 0
    },
    "open_threads": [],
    "pending": {
      "clarification": null,
      "conversation_turn": null
    },
    "promotion_profiles": [],
    "receipts": [],
    "runtime": {
      "hebe_sleeping": false,
      "is_running": false,
      "last_firewall": {
        "authority": null,
        "decision": null,
        "reason": null,
        "source": null
      },
      "last_input_source": null,
      "last_intent": null,
      "last_policy": {
        "authority": null,
        "decision": null,
        "reason": null,
        "source": null
      },
      "mode": "active"
    },
    "social_state": {
      "last_cheer": {},
      "last_raid": {},
      "recent_active_users": [],
      "recent_chat_count": 0
    },
    "stream_session": {
      "active_stream_session_id": null,
      "category": null,
      "enabled": false,
      "game": null,
      "is_live": false,
      "last_transition": null,
      "live_status_known": false,
      "title": null
    }
  },
  "return": {
    "actions": {
      "attempts": [],
      "model_calls": [],
      "research_calls": [],
      "speech_requests": []
    },
    "beliefs": [],
    "current_scene": {},
    "database_watermarks": {
      "counts": {
        "chat_log": 0,
        "live_session_timeline": 7,
        "memory_chunks": 1,
        "memory_facts": 0,
        "promotion_events": 0,
        "schema_migrations": 1,
        "stream_chat_messages": 2,
        "stream_events": 3,
        "stream_sessions": 2,
        "viewer_promotion_profiles": 0
      },
      "final_response_digest": "",
      "final_response_present": false,
      "schema_migrations": [
        {
          "applied_at": "2026-08-11T07:28:50.638959+00:00",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "component": "cognitive_replay",
          "name": "replay_metadata",
          "version": 1
        }
      ]
    },
    "emitted_outputs": [],
    "final_emission_results": [],
    "game_state": {
      "challenge": "",
      "confidence": 0.0,
      "current_character": "",
      "current_game": null,
      "current_location": null,
      "current_objective": null,
      "game": "",
      "known_constraints": [],
      "last_confirmed_progress": "",
      "last_updated": 1786433336.666283,
      "party_members": [],
      "platform_version": "",
      "playthrough_type": "casual",
      "provenance": "inferred",
      "recent_run_context_facts": [],
      "spoiler_policy": "spoiler_safe_hints"
    },
    "memory": {
      "chunks_count": 1,
      "facts_count": 0
    },
    "open_threads": [],
    "pending": {
      "clarification": null,
      "conversation_turn": null
    },
    "promotion_profiles": [],
    "receipts": [],
    "runtime": {
      "hebe_sleeping": false,
      "is_running": false,
      "last_firewall": {
        "authority": "viewer",
        "decision": "allow",
        "reason": "live_viewer_message",
        "source": "twitch_viewer"
      },
      "last_input_source": null,
      "last_intent": null,
      "last_policy": {
        "authority": null,
        "decision": null,
        "reason": null,
        "source": null
      },
      "mode": "active"
    },
    "social_state": {
      "last_cheer": {},
      "last_raid": {},
      "recent_active_users": [
        "synthetic_student"
      ],
      "recent_chat_count": 1
    },
    "stream_session": {
      "active_stream_session_id": 2,
      "category": null,
      "enabled": true,
      "game": null,
      "is_live": true,
      "last_transition": "online",
      "live_status_known": true,
      "title": null
    }
  },
  "start-1": {
    "actions": {
      "attempts": [],
      "model_calls": [],
      "research_calls": [],
      "speech_requests": []
    },
    "beliefs": [],
    "current_scene": {},
    "database_watermarks": {
      "counts": {
        "chat_log": 0,
        "live_session_timeline": 2,
        "memory_chunks": 0,
        "memory_facts": 0,
        "promotion_events": 0,
        "schema_migrations": 1,
        "stream_chat_messages": 0,
        "stream_events": 1,
        "stream_sessions": 1,
        "viewer_promotion_profiles": 0
      },
      "final_response_digest": "",
      "final_response_present": false,
      "schema_migrations": [
        {
          "applied_at": "2026-08-11T07:28:50.638959+00:00",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "component": "cognitive_replay",
          "name": "replay_metadata",
          "version": 1
        }
      ]
    },
    "emitted_outputs": [],
    "final_emission_results": [],
    "game_state": {
      "challenge": "",
      "confidence": 0.0,
      "current_character": "",
      "current_game": null,
      "current_location": null,
      "current_objective": null,
      "game": "",
      "known_constraints": [],
      "last_confirmed_progress": "",
      "last_updated": 1786433330.6558828,
      "party_members": [],
      "platform_version": "",
      "playthrough_type": "casual",
      "provenance": "inferred",
      "recent_run_context_facts": [],
      "spoiler_policy": "spoiler_safe_hints"
    },
    "memory": {
      "chunks_count": 0,
      "facts_count": 0
    },
    "open_threads": [],
    "pending": {
      "clarification": null,
      "conversation_turn": null
    },
    "promotion_profiles": [],
    "receipts": [],
    "runtime": {
      "hebe_sleeping": false,
      "is_running": false,
      "last_firewall": {
        "authority": null,
        "decision": null,
        "reason": null,
        "source": null
      },
      "last_input_source": null,
      "last_intent": null,
      "last_policy": {
        "authority": null,
        "decision": null,
        "reason": null,
        "source": null
      },
      "mode": "active"
    },
    "social_state": {
      "last_cheer": {},
      "last_raid": {},
      "recent_active_users": [],
      "recent_chat_count": 0
    },
    "stream_session": {
      "active_stream_session_id": 1,
      "category": null,
      "enabled": true,
      "game": null,
      "is_live": true,
      "last_transition": "online",
      "live_status_known": true,
      "title": null
    }
  },
  "start-2": {
    "actions": {
      "attempts": [],
      "model_calls": [],
      "research_calls": [],
      "speech_requests": []
    },
    "beliefs": [],
    "current_scene": {},
    "database_watermarks": {
      "counts": {
        "chat_log": 0,
        "live_session_timeline": 6,
        "memory_chunks": 1,
        "memory_facts": 0,
        "promotion_events": 0,
        "schema_migrations": 1,
        "stream_chat_messages": 1,
        "stream_events": 3,
        "stream_sessions": 2,
        "viewer_promotion_profiles": 0
      },
      "final_response_digest": "",
      "final_response_present": false,
      "schema_migrations": [
        {
          "applied_at": "2026-08-11T07:28:50.638959+00:00",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "component": "cognitive_replay",
          "name": "replay_metadata",
          "version": 1
        }
      ]
    },
    "emitted_outputs": [],
    "final_emission_results": [],
    "game_state": {
      "challenge": "",
      "confidence": 0.0,
      "current_character": "",
      "current_game": null,
      "current_location": null,
      "current_objective": null,
      "game": "",
      "known_constraints": [],
      "last_confirmed_progress": "",
      "last_updated": 1786433336.666283,
      "party_members": [],
      "platform_version": "",
      "playthrough_type": "casual",
      "provenance": "inferred",
      "recent_run_context_facts": [],
      "spoiler_policy": "spoiler_safe_hints"
    },
    "memory": {
      "chunks_count": 1,
      "facts_count": 0
    },
    "open_threads": [],
    "pending": {
      "clarification": null,
      "conversation_turn": null
    },
    "promotion_profiles": [],
    "receipts": [],
    "runtime": {
      "hebe_sleeping": false,
      "is_running": false,
      "last_firewall": {
        "authority": null,
        "decision": null,
        "reason": null,
        "source": null
      },
      "last_input_source": null,
      "last_intent": null,
      "last_policy": {
        "authority": null,
        "decision": null,
        "reason": null,
        "source": null
      },
      "mode": "active"
    },
    "social_state": {
      "last_cheer": {},
      "last_raid": {},
      "recent_active_users": [],
      "recent_chat_count": 0
    },
    "stream_session": {
      "active_stream_session_id": 2,
      "category": null,
      "enabled": true,
      "game": null,
      "is_live": true,
      "last_transition": "online",
      "live_status_known": true,
      "title": null
    }
  }
}
```

#### Final state and side effects

```json
{
  "actions": {
    "attempts": [],
    "model_calls": [],
    "research_calls": [],
    "speech_requests": []
  },
  "beliefs": [],
  "current_scene": {},
  "database_watermarks": {
    "counts": {
      "chat_log": 0,
      "live_session_timeline": 8,
      "memory_chunks": 2,
      "memory_facts": 0,
      "promotion_events": 0,
      "schema_migrations": 1,
      "stream_chat_messages": 2,
      "stream_events": 3,
      "stream_sessions": 2,
      "viewer_promotion_profiles": 0
    },
    "final_response_digest": "",
    "final_response_present": false,
    "schema_migrations": [
      {
        "applied_at": "2026-08-11T07:28:50.638959+00:00",
        "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
        "component": "cognitive_replay",
        "name": "replay_metadata",
        "version": 1
      }
    ]
  },
  "emitted_outputs": [],
  "final_emission_results": [],
  "game_state": {
    "challenge": "",
    "confidence": 0.0,
    "current_character": "",
    "current_game": null,
    "current_location": null,
    "current_objective": null,
    "game": "",
    "known_constraints": [],
    "last_confirmed_progress": "",
    "last_updated": 1786433336.666283,
    "party_members": [],
    "platform_version": "",
    "playthrough_type": "casual",
    "provenance": "inferred",
    "recent_run_context_facts": [],
    "spoiler_policy": "spoiler_safe_hints"
  },
  "memory": {
    "chunks_count": 2,
    "facts_count": 0
  },
  "open_threads": [],
  "pending": {
    "clarification": null,
    "conversation_turn": null
  },
  "promotion_profiles": [],
  "receipts": [],
  "runtime": {
    "hebe_sleeping": false,
    "is_running": false,
    "last_firewall": {
      "authority": "viewer",
      "decision": "allow",
      "reason": "live_viewer_message",
      "source": "twitch_viewer"
    },
    "last_input_source": null,
    "last_intent": null,
    "last_policy": {
      "authority": null,
      "decision": null,
      "reason": null,
      "source": null
    },
    "mode": "active"
  },
  "social_state": {
    "last_cheer": {},
    "last_raid": {},
    "recent_active_users": [
      "synthetic_student"
    ],
    "recent_chat_count": 1
  },
  "stream_session": {
    "active_stream_session_id": 2,
    "category": null,
    "enabled": true,
    "game": null,
    "is_live": true,
    "last_transition": "online",
    "live_status_known": true,
    "title": null
  }
}
```

#### Restart evidence

```json
[
  {
    "after_persisted_counts": {
      "chat_log": 0,
      "live_session_timeline": 4,
      "memory_chunks": 1,
      "memory_facts": 0,
      "promotion_events": 0,
      "schema_migrations": 1,
      "stream_chat_messages": 1,
      "stream_events": 2,
      "stream_sessions": 1,
      "viewer_promotion_profiles": 0
    },
    "before_persisted_counts": {
      "chat_log": 0,
      "live_session_timeline": 4,
      "memory_chunks": 1,
      "memory_facts": 0,
      "promotion_events": 0,
      "schema_migrations": 1,
      "stream_chat_messages": 1,
      "stream_events": 2,
      "stream_sessions": 1,
      "viewer_promotion_profiles": 0
    },
    "event_id": "restart",
    "new_engine_id": 2823653062544,
    "old_engine_collected": true,
    "old_engine_id": 2823653360656,
    "same_database": "C:\\Users\\Leo Nifelheim\\Documents\\Hebe\\hebe-ui\\artifacts\\cognitive-replay\\phase-0.5\\workspaces\\temporal_social_thread_format\\hebe-replay.sqlite3",
    "volatile_state_recreated": true
  }
]
```

## External boundaries

- desktop: `fake`
- game_research_web: `fixture`
- llm_model: `fixture`
- network: `blocked_by_design`
- tts_audio: `fake`
- twitch: `fake`

## Persistence

```json
{
  "database_paths": [
    "C:\\Users\\Leo Nifelheim\\Documents\\Hebe\\hebe-ui\\artifacts\\cognitive-replay\\phase-0.5\\workspaces\\ambient_false_positive_foundation\\hebe-replay.sqlite3",
    "C:\\Users\\Leo Nifelheim\\Documents\\Hebe\\hebe-ui\\artifacts\\cognitive-replay\\phase-0.5\\workspaces\\consolidation_format\\hebe-replay.sqlite3",
    "C:\\Users\\Leo Nifelheim\\Documents\\Hebe\\hebe-ui\\artifacts\\cognitive-replay\\phase-0.5\\workspaces\\ffv_durable_run_format\\hebe-replay.sqlite3",
    "C:\\Users\\Leo Nifelheim\\Documents\\Hebe\\hebe-ui\\artifacts\\cognitive-replay\\phase-0.5\\workspaces\\ivanxi_resub_promo_restart\\hebe-replay.sqlite3",
    "C:\\Users\\Leo Nifelheim\\Documents\\Hebe\\hebe-ui\\artifacts\\cognitive-replay\\phase-0.5\\workspaces\\owner_correction_format\\hebe-replay.sqlite3",
    "C:\\Users\\Leo Nifelheim\\Documents\\Hebe\\hebe-ui\\artifacts\\cognitive-replay\\phase-0.5\\workspaces\\raid_transition_foundation\\hebe-replay.sqlite3",
    "C:\\Users\\Leo Nifelheim\\Documents\\Hebe\\hebe-ui\\artifacts\\cognitive-replay\\phase-0.5\\workspaces\\receipt_truth\\hebe-replay.sqlite3",
    "C:\\Users\\Leo Nifelheim\\Documents\\Hebe\\hebe-ui\\artifacts\\cognitive-replay\\phase-0.5\\workspaces\\research_fixture_foundation\\hebe-replay.sqlite3",
    "C:\\Users\\Leo Nifelheim\\Documents\\Hebe\\hebe-ui\\artifacts\\cognitive-replay\\phase-0.5\\workspaces\\temporal_social_thread_format\\hebe-replay.sqlite3"
  ],
  "database_type": "isolated_sqlite",
  "restart_points": 5,
  "schema_migrations": [
    [
      {
        "already_applied": false,
        "applied_at": "2026-08-11T07:28:25.873704+00:00",
        "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
        "component": "cognitive_replay",
        "name": "replay_metadata",
        "version": 1
      }
    ],
    [
      {
        "already_applied": false,
        "applied_at": "2026-08-11T07:28:26.700985+00:00",
        "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
        "component": "cognitive_replay",
        "name": "replay_metadata",
        "version": 1
      }
    ],
    [
      {
        "already_applied": false,
        "applied_at": "2026-08-11T07:28:33.114708+00:00",
        "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
        "component": "cognitive_replay",
        "name": "replay_metadata",
        "version": 1
      }
    ],
    [
      {
        "already_applied": false,
        "applied_at": "2026-08-11T07:28:34.501984+00:00",
        "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
        "component": "cognitive_replay",
        "name": "replay_metadata",
        "version": 1
      }
    ],
    [
      {
        "already_applied": false,
        "applied_at": "2026-08-11T07:28:35.821984+00:00",
        "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
        "component": "cognitive_replay",
        "name": "replay_metadata",
        "version": 1
      }
    ],
    [
      {
        "already_applied": false,
        "applied_at": "2026-08-11T07:28:42.618630+00:00",
        "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
        "component": "cognitive_replay",
        "name": "replay_metadata",
        "version": 1
      }
    ],
    [
      {
        "already_applied": false,
        "applied_at": "2026-08-11T07:28:48.816593+00:00",
        "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
        "component": "cognitive_replay",
        "name": "replay_metadata",
        "version": 1
      }
    ],
    [
      {
        "already_applied": false,
        "applied_at": "2026-08-11T07:28:50.201630+00:00",
        "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
        "component": "cognitive_replay",
        "name": "replay_metadata",
        "version": 1
      }
    ],
    [
      {
        "already_applied": false,
        "applied_at": "2026-08-11T07:28:50.638959+00:00",
        "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
        "component": "cognitive_replay",
        "name": "replay_metadata",
        "version": 1
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
  "baseline_commit": "9f91948003ec545fbf7ae045cf78b3affb3451d7",
  "baseline_python": "3.11.0",
  "baseline_platform": "Windows-10-10.0.26200-SP0",
  "baseline_command": "C:\\Users\\Leo Nifelheim\\Documents\\Hebe\\hebe-ui\\backend\\.venv\\Scripts\\python.exe -m unittest backend.tests.test_cognitive_replay backend.tests.test_voice_command_pipeline backend.tests.test_cognitive_twitch backend.tests.test_stream_presence backend.tests.test_hebe_live_v1 backend.tests.test_hebe_live_v11 backend.tests.test_hebe_live_v12 backend.tests.test_hebe_live_v12_followup backend.tests.test_hebe_live_20260809_followup backend.tests.test_final_emission_gate backend.tests.test_cognitive_execution_guard backend.tests.test_game_knowledge backend.tests.test_stream_session_primer backend.tests.test_live_session_brain",
  "baseline_command_duration_seconds": 104.272,
  "baseline_loader_errors": 1,
  "baseline_tests_passed": 451,
  "baseline_tests_failed": 13,
  "baseline_new_module_unavailable": "backend.tests.test_cognitive_replay",
  "phase_0_5_tests_passed": 471,
  "phase_0_5_tests_failed": 13,
  "new_regressions": 0,
  "pre_existing_failures": 13,
  "fixed_existing_failures": 0,
  "classification_counts": {
    "PASS_BOTH": 0,
    "PRE_EXISTING_FAILURE": 13,
    "NEW_PHASE_0_5_REGRESSION": 0,
    "FIXED_BY_PHASE_0_5": 0,
    "FAILURE_CHANGED": 0
  },
  "tests": [
    {
      "test": "backend.tests.test_game_knowledge.GameKnowledgeTests.test_response_synthesizer_handles_game_knowledge_command_result",
      "subsystem": "Game Knowledge response guard",
      "baseline_status": "FAIL",
      "phase_0_5_status": "FAIL",
      "baseline_exception_or_assertion": "AssertionError: 'Persona 5 Royal' not found in <captured value>",
      "phase_0_5_exception_or_assertion": "AssertionError: 'Persona 5 Royal' not found in <captured value>",
      "phase_0_5_code_on_failing_path": "NO: the Phase 0.5 production seam changes are not on this failing path",
      "classification": "PRE_EXISTING_FAILURE"
    },
    {
      "test": "backend.tests.test_voice_command_pipeline.VoiceCommandPipelineTests.test_high_value_game_tip_can_reply_without_hebe_mention",
      "subsystem": "Twitch no-mention/presence",
      "baseline_status": "FAIL",
      "phase_0_5_status": "FAIL",
      "baseline_exception_or_assertion": "AssertionError: 0 != 1",
      "phase_0_5_exception_or_assertion": "AssertionError: 0 != 1",
      "phase_0_5_code_on_failing_path": "YES: handle_twitch_chat_event was refactored, but the baseline and current terminal assertion are identical",
      "classification": "PRE_EXISTING_FAILURE"
    },
    {
      "test": "backend.tests.test_voice_command_pipeline.VoiceCommandPipelineTests.test_no_wake_whitelisted_app_command_routes_while_stream_offline",
      "subsystem": "local application/capability resolution",
      "baseline_status": "ERROR",
      "phase_0_5_status": "ERROR",
      "baseline_exception_or_assertion": "IndexError: list index out of range",
      "phase_0_5_exception_or_assertion": "IndexError: list index out of range",
      "phase_0_5_code_on_failing_path": "ENTRY ONLY: the shared STT ingress wrapper is on the path; the unchanged local capability resolver produces the identical failure",
      "classification": "PRE_EXISTING_FAILURE"
    },
    {
      "test": "backend.tests.test_voice_command_pipeline.VoiceCommandPipelineTests.test_obs_path_missing_returns_structured_action_result_not_generic_advice",
      "subsystem": "local application/capability resolution",
      "baseline_status": "FAIL",
      "phase_0_5_status": "FAIL",
      "baseline_exception_or_assertion": "AssertionError: 'HEBE_APP_OBS_PATH' not found in <captured value>",
      "phase_0_5_exception_or_assertion": "AssertionError: 'HEBE_APP_OBS_PATH' not found in <captured value>",
      "phase_0_5_code_on_failing_path": "NO: the Phase 0.5 production seam changes are not on this failing path",
      "classification": "PRE_EXISTING_FAILURE"
    },
    {
      "test": "backend.tests.test_voice_command_pipeline.VoiceCommandPipelineTests.test_stt_canonical_melonds_command_executes_once (transcript='Ebe, abre Melón DS')",
      "subsystem": "local application/capability resolution",
      "baseline_status": "FAIL",
      "phase_0_5_status": "FAIL",
      "baseline_exception_or_assertion": "AssertionError: 0 != 1",
      "phase_0_5_exception_or_assertion": "AssertionError: 0 != 1",
      "phase_0_5_code_on_failing_path": "ENTRY ONLY: the shared STT ingress wrapper is on the path; the unchanged local capability resolver produces the identical failure",
      "classification": "PRE_EXISTING_FAILURE"
    },
    {
      "test": "backend.tests.test_voice_command_pipeline.VoiceCommandPipelineTests.test_stt_canonical_melonds_command_executes_once (transcript='Eve, abre Melón de Ese')",
      "subsystem": "local application/capability resolution",
      "baseline_status": "FAIL",
      "phase_0_5_status": "FAIL",
      "baseline_exception_or_assertion": "AssertionError: 0 != 1",
      "phase_0_5_exception_or_assertion": "AssertionError: 0 != 1",
      "phase_0_5_code_on_failing_path": "ENTRY ONLY: the shared STT ingress wrapper is on the path; the unchanged local capability resolver produces the identical failure",
      "classification": "PRE_EXISTING_FAILURE"
    },
    {
      "test": "backend.tests.test_voice_command_pipeline.VoiceCommandPipelineTests.test_stt_canonical_melonds_command_executes_once (transcript='Hebe, abre melonDS')",
      "subsystem": "local application/capability resolution",
      "baseline_status": "FAIL",
      "phase_0_5_status": "FAIL",
      "baseline_exception_or_assertion": "AssertionError: 0 != 1",
      "phase_0_5_exception_or_assertion": "AssertionError: 0 != 1",
      "phase_0_5_code_on_failing_path": "ENTRY ONLY: the shared STT ingress wrapper is on the path; the unchanged local capability resolver produces the identical failure",
      "classification": "PRE_EXISTING_FAILURE"
    },
    {
      "test": "backend.tests.test_voice_command_pipeline.VoiceCommandPipelineTests.test_stt_hebe_abre_obs_uses_same_open_application_pipeline",
      "subsystem": "local application/capability resolution",
      "baseline_status": "ERROR",
      "phase_0_5_status": "ERROR",
      "baseline_exception_or_assertion": "IndexError: list index out of range",
      "phase_0_5_exception_or_assertion": "IndexError: list index out of range",
      "phase_0_5_code_on_failing_path": "ENTRY ONLY: the shared STT ingress wrapper is on the path; the unchanged local capability resolver produces the identical failure",
      "classification": "PRE_EXISTING_FAILURE"
    },
    {
      "test": "backend.tests.test_voice_command_pipeline.VoiceCommandPipelineTests.test_twitch_normal_no_mention_chat_reaches_presence_observe",
      "subsystem": "Twitch no-mention/presence",
      "baseline_status": "FAIL",
      "phase_0_5_status": "FAIL",
      "baseline_exception_or_assertion": "AssertionError: '[HEBE][TWITCH_PIPELINE_CLASSIFY] category=normal_no_mention_chat' not found in <captured value>",
      "phase_0_5_exception_or_assertion": "AssertionError: '[HEBE][TWITCH_PIPELINE_CLASSIFY] category=normal_no_mention_chat' not found in <captured value>",
      "phase_0_5_code_on_failing_path": "YES: handle_twitch_chat_event was refactored, but the baseline and current terminal assertion are identical",
      "classification": "PRE_EXISTING_FAILURE"
    },
    {
      "test": "backend.tests.test_voice_command_pipeline.VoiceCommandPipelineTests.test_twitch_pipeline_health_counts_messages",
      "subsystem": "Twitch no-mention/presence",
      "baseline_status": "FAIL",
      "phase_0_5_status": "FAIL",
      "baseline_exception_or_assertion": "AssertionError: 0 != 1",
      "phase_0_5_exception_or_assertion": "AssertionError: 0 != 1",
      "phase_0_5_code_on_failing_path": "YES: handle_twitch_chat_event was refactored, but the baseline and current terminal assertion are identical",
      "classification": "PRE_EXISTING_FAILURE"
    },
    {
      "test": "backend.tests.test_voice_command_pipeline.VoiceCommandPipelineTests.test_ui_abre_obs_creates_open_application_when_awake_and_whitelisted",
      "subsystem": "local application/capability resolution",
      "baseline_status": "ERROR",
      "phase_0_5_status": "ERROR",
      "baseline_exception_or_assertion": "IndexError: list index out of range",
      "phase_0_5_exception_or_assertion": "IndexError: list index out of range",
      "phase_0_5_code_on_failing_path": "NO: the Phase 0.5 production seam changes are not on this failing path",
      "classification": "PRE_EXISTING_FAILURE"
    },
    {
      "test": "backend.tests.test_voice_command_pipeline.VoiceCommandPipelineTests.test_ui_hebe_abre_obs_creates_open_application_action_plan",
      "subsystem": "local application/capability resolution",
      "baseline_status": "ERROR",
      "phase_0_5_status": "ERROR",
      "baseline_exception_or_assertion": "IndexError: list index out of range",
      "phase_0_5_exception_or_assertion": "IndexError: list index out of range",
      "phase_0_5_code_on_failing_path": "NO: the Phase 0.5 production seam changes are not on this failing path",
      "classification": "PRE_EXISTING_FAILURE"
    },
    {
      "test": "backend.tests.test_voice_command_pipeline.VoiceCommandPipelineTests.test_unrelated_action_during_pending_conversation_still_uses_action_flow",
      "subsystem": "local application/capability resolution",
      "baseline_status": "ERROR",
      "phase_0_5_status": "ERROR",
      "baseline_exception_or_assertion": "IndexError: list index out of range",
      "phase_0_5_exception_or_assertion": "IndexError: list index out of range",
      "phase_0_5_code_on_failing_path": "ENTRY ONLY: the shared STT ingress wrapper is on the path; the unchanged local capability resolver produces the identical failure",
      "classification": "PRE_EXISTING_FAILURE"
    }
  ]
}
```

## Human evaluation boundary

This harness verifies cognitive/state prerequisites. Naturalness, personality, comedic timing, and social appropriateness still require human judgment.
