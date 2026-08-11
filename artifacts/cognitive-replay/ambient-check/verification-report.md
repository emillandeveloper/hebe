# Cognitive Replay Verification Report

Overall status: **VERIFICATION_INCOMPLETE**

## Repository and environment

- Commit: `9f91948003ec545fbf7ae045cf78b3affb3451d7`
- Working tree: `a3006bc4c10b`
- Platform: `Windows-10-10.0.26200-SP0`
- Python: `3.11.0`

## Commands

- `python -m app.replay --scenario ambient_false_positive_foundation --output artifacts/cognitive-replay/ambient-check` → exit 2 (0.913806s)

## Tests

```json
{
  "unit_integration_regression": {
    "passed": 0,
    "failed": 0,
    "skipped": 0,
    "total": 0,
    "required_layer_missing": true
  },
  "replay": {
    "passed": 1,
    "failed": 0,
    "skipped": 0,
    "expected_future_gaps": 0,
    "expected_failures": 0,
    "duration_seconds": 0.821061
  },
  "failed": 0,
  "required_layer_missing": true
}
```

## Replay scenarios

### ambient_false_positive_foundation

- Status: **VERIFIED**
- Events: 4
- Restarts: 0
- Duration: 0.821061s
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
      "last_updated": 1786433183.9470952,
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
        "schema_migrations": 1
      },
      "schema_migrations": [
        {
          "component": "cognitive_replay",
          "version": 1,
          "name": "replay_metadata",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "applied_at": "2026-08-11T07:26:23.937967+00:00"
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
      "last_updated": 1786433183.9470952,
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
        "event_id": "1786433184.2178054",
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
        "event_id": "1786433184.2178054",
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
        "schema_migrations": 1
      },
      "schema_migrations": [
        {
          "component": "cognitive_replay",
          "version": 1,
          "name": "replay_metadata",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "applied_at": "2026-08-11T07:26:23.937967+00:00"
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
      "last_updated": 1786433183.9470952,
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
        "event_id": "1786433184.2178054",
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
        "event_id": "1786433184.2178054",
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
        "schema_migrations": 1
      },
      "schema_migrations": [
        {
          "component": "cognitive_replay",
          "version": 1,
          "name": "replay_metadata",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "applied_at": "2026-08-11T07:26:23.937967+00:00"
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
      "last_updated": 1786433183.9470952,
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
        "event_id": "1786433184.2178054",
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
        "event_id": "1786433184.2178054",
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
        "schema_migrations": 1
      },
      "schema_migrations": [
        {
          "component": "cognitive_replay",
          "version": 1,
          "name": "replay_metadata",
          "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
          "applied_at": "2026-08-11T07:26:23.937967+00:00"
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
    "last_updated": 1786433183.9470952,
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
      "event_id": "1786433184.2178054",
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
      "event_id": "1786433184.2178054",
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
      "schema_migrations": 1
    },
    "schema_migrations": [
      {
        "component": "cognitive_replay",
        "version": 1,
        "name": "replay_metadata",
        "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
        "applied_at": "2026-08-11T07:26:23.937967+00:00"
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
    "C:\\Users\\Leo Nifelheim\\Documents\\Hebe\\hebe-ui\\artifacts\\cognitive-replay\\ambient-check\\workspaces\\ambient_false_positive_foundation\\hebe-replay.sqlite3"
  ],
  "restart_points": 0,
  "schema_migrations": [
    [
      {
        "component": "cognitive_replay",
        "version": 1,
        "name": "replay_metadata",
        "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
        "applied_at": "2026-08-11T07:26:23.937967+00:00",
        "already_applied": false
      }
    ]
  ]
}
```

## Limitations

- datetime.now() reads in legacy persistence remain wall-clock based; behavioral TTL/cooldown time.time() reads are controlled during replay dispatch
- faster-whisper audio decoding is outside the cognitive replay boundary and requires its separate integration suite

## Baseline differential

- No baseline differential was attached.

## Human evaluation boundary

This harness verifies cognitive/state prerequisites. Naturalness, personality, comedic timing, and social appropriateness still require human judgment.
