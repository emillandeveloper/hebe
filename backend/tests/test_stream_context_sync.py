import unittest

from app.integrations.twitch.helix_client import TwitchHelixClient
from app.integrations.twitch.event_adapter import TwitchEventAdapter
from app.stream.context_sync import StreamContextSyncService
from app.stream.state import StreamSessionState
from app.stream.title_parser import parse_stream_title


class FakeResponse:
    def __init__(self, payload, ok=True, status_code=200, text="OK"):
        self._payload = payload
        self.ok = ok
        self.status_code = status_code
        self.text = text

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, headers=None, params=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "params": params, "timeout": timeout})
        return self.responses.pop(0)


class FakePostSession:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return self.response


class FakeTwitchApi:
    def __init__(self, *, stream=None, channel=None, error=None):
        self.stream = stream
        self.channel = channel
        self.error = error

    def get_channel_info(self):
        if self.error:
            raise RuntimeError(self.error)
        return self.channel

    def get_stream(self):
        if self.error:
            raise RuntimeError(self.error)
        return self.stream


class StreamContextSyncTests(unittest.TestCase):
    def test_helix_get_streams_live_response_sets_live_context(self):
        session = FakeSession(
            [
                FakeResponse(
                    {
                        "data": [
                            {
                                "title": "First Playthrough [ENG/ESP]",
                                "game_name": "Final Fantasy X",
                                "tags": ["JRPG", "NoSpoilers"],
                                "started_at": "2026-05-31T18:00:00Z",
                            }
                        ]
                    }
                )
            ]
        )
        client = TwitchHelixClient(
            client_id="client",
            oauth_token="token",
            broadcaster_id="123",
            session=session,
        )

        stream = client.get_stream()

        self.assertEqual(stream["game_name"], "Final Fantasy X")
        self.assertEqual(session.calls[0]["params"], {"user_id": "123"})

    def test_helix_token_strips_oauth_prefix(self):
        session = FakeSession([FakeResponse({"data": []})])
        client = TwitchHelixClient(
            client_id="client",
            oauth_token="oauth:abc123",
            broadcaster_id="123",
            session=session,
        )

        client.get_stream()

        self.assertEqual(session.calls[0]["headers"]["Authorization"], "Bearer abc123")

    def test_context_sync_live_response_sets_state(self):
        now = 1_000_000.0
        state = StreamSessionState(enabled=True)
        api = FakeTwitchApi(
            channel={"title": "Old title", "game_name": "Old Game", "tags": []},
            stream={
                "title": "First Playthrough [ENG/ESP]",
                "game_name": "Final Fantasy X",
                "tags": ["JRPG", "NoSpoilers"],
                "started_at": "2026-05-31T18:00:00Z",
            },
        )

        ok = StreamContextSyncService(twitch_api=api, now_fn=lambda: now).sync(state)

        self.assertTrue(ok)
        self.assertTrue(state.is_live)
        self.assertTrue(state.live_status_known)
        self.assertEqual(state.current_stream_title, "First Playthrough [ENG/ESP]")
        self.assertEqual(state.current_category, "Final Fantasy X")
        self.assertEqual(state.current_game, "Final Fantasy X")
        self.assertEqual(state.current_tags, ["JRPG", "NoSpoilers"])
        self.assertEqual(state.stream_started_at, "2026-05-31T18:00:00Z")
        self.assertEqual(state.current_playthrough_type, "first_playthrough")
        self.assertEqual(state.language_mode, "ENG/ESP")
        self.assertTrue(state.bilingual_mode)
        self.assertGreaterEqual(state.stream_spontaneity_grace_until_ts, now + 4 * 60)

    def test_get_streams_empty_response_sets_offline(self):
        now = 1_000_000.0
        state = StreamSessionState(enabled=True)
        api = FakeTwitchApi(channel={"title": "Offline title", "game_name": "Retro"}, stream=None)

        ok = StreamContextSyncService(twitch_api=api, now_fn=lambda: now).sync(state)

        self.assertTrue(ok)
        self.assertFalse(state.is_live)
        self.assertTrue(state.live_status_known)
        self.assertEqual(state.current_stream_title, "Offline title")
        self.assertEqual(state.current_category, "Retro")

    def test_channel_info_updates_offline_title_category_safely(self):
        now = 1_000_000.0
        state = StreamSessionState(enabled=True)
        api = FakeTwitchApi(
            channel={"title": "Challenge Playthrough Level 1", "game_name": "JRPG", "tags": ["ESP"]},
            stream=None,
        )

        StreamContextSyncService(twitch_api=api, now_fn=lambda: now).sync(state)

        self.assertEqual(state.current_stream_title, "Challenge Playthrough Level 1")
        self.assertEqual(state.current_category, "JRPG")
        self.assertEqual(state.current_playthrough_type, "challenge")
        self.assertEqual(state.current_challenge, "level_1")

    def test_twitch_api_failure_does_not_crash_and_marks_stale_unknown(self):
        now = 1_000_000.0
        state = StreamSessionState(enabled=True)
        state.is_live = True
        state.live_status_known = True
        state.stream_context_updated_ts = now - 10 * 60
        api = FakeTwitchApi(error="boom")

        ok = StreamContextSyncService(twitch_api=api, now_fn=lambda: now).sync(state)

        self.assertFalse(ok)
        self.assertFalse(state.is_live)
        self.assertFalse(state.live_status_known)
        self.assertIn("boom", state.last_stream_context_error)

    def test_helix_401_response_stores_readable_error(self):
        session = FakeSession(
            [
                FakeResponse(
                    {"error": "Unauthorized", "message": "Invalid OAuth token"},
                    ok=False,
                    status_code=401,
                    text='{"error":"Unauthorized"}',
                )
            ]
        )
        client = TwitchHelixClient(
            client_id="client",
            oauth_token="token",
            broadcaster_id="123",
            session=session,
        )
        state = StreamSessionState(enabled=True)

        ok = StreamContextSyncService(twitch_api=client, now_fn=lambda: 1_000_000.0).sync(state)

        self.assertFalse(ok)
        self.assertIn("Helix get_streams failed: 401 Unauthorized", state.last_stream_context_error)
        self.assertIn("Invalid OAuth token", state.last_stream_context_error)

    def test_missing_config_stores_readable_error(self):
        client = TwitchHelixClient(
            client_id="",
            oauth_token="token",
            broadcaster_id="123",
            session=FakeSession([]),
        )
        state = StreamSessionState(enabled=True)

        ok = StreamContextSyncService(twitch_api=client, now_fn=lambda: 1_000_000.0).sync(state)

        self.assertFalse(ok)
        self.assertIn("Missing Twitch config: TWITCH_CLIENT_ID", state.last_stream_context_error)


class StreamTitleParserTests(unittest.TestCase):
    def test_first_playthrough_sets_no_spoilers(self):
        parsed = parse_stream_title("Final Fantasy X - First Playthrough [ENG/ESP]")

        self.assertEqual(parsed.playthrough_type, "first_playthrough")
        self.assertEqual(parsed.spoiler_policy, "no_spoilers")
        self.assertTrue(parsed.bilingual_mode)
        self.assertEqual(parsed.language_mode, "ENG/ESP")

    def test_retro_weekend_first_playthrough_title_detects_context(self):
        parsed = parse_stream_title(
            "[ENG/ESP] Retro Weekend: Food for Leveling! That's Zwei — Zwei: The Arges Adventure | First Playthrough"
        )

        self.assertTrue(parsed.bilingual_mode)
        self.assertEqual(parsed.language_mode, "ENG/ESP")
        self.assertEqual(parsed.stream_slot, "retro_weekend")
        self.assertEqual(parsed.playthrough_type, "first_playthrough")
        self.assertEqual(parsed.spoiler_policy, "no_spoilers")

    def test_retro_weeeekend_typo_detects_retro_weekend(self):
        parsed = parse_stream_title(
            "[ENG/ESP] Retro Weeeekend: Food for Leveling! That's Zwei | First Playthrough"
        )

        self.assertEqual(parsed.stream_slot, "retro_weekend")

    def test_chat_playthrough_detected(self):
        parsed = parse_stream_title("Chat Playthrough")

        self.assertEqual(parsed.playthrough_type, "chat_playthrough")

    def test_challenge_playthrough_detected(self):
        parsed = parse_stream_title("Challenge Playthrough")

        self.assertEqual(parsed.playthrough_type, "challenge")

    def test_level_1_detected(self):
        parsed = parse_stream_title("Challenge Playthrough Level 1")

        self.assertIn("level_1", parsed.challenges)

    def test_level_1_challenge_playthrough_detects_challenge_type_and_challenge(self):
        parsed = parse_stream_title("Level 1 Challenge Playthrough")

        self.assertEqual(parsed.playthrough_type, "challenge")
        self.assertIn("level_1", parsed.challenges)

    def test_no_sphere_grid_detected(self):
        parsed = parse_stream_title("No Sphere Grid")

        self.assertIn("no_sphere_grid", parsed.challenges)

    def test_no_shops_detected(self):
        parsed = parse_stream_title("No Shops")

        self.assertIn("no_shops", parsed.challenges)

    def test_challenge_monday_detected(self):
        parsed = parse_stream_title("Challenge Monday: No Shops")

        self.assertEqual(parsed.stream_slot, "challenge_monday")
        self.assertIn("no_shops", parsed.challenges)


class EventSubStreamContextSeparationTests(unittest.TestCase):
    def test_eventsub_chat_message_403_is_optional_and_does_not_prevent_context_sync(self):
        adapter = TwitchEventAdapter(
            client_id="client",
            user_oauth_token="token",
            broadcaster_user_id="123",
            bot_user_id="456",
            twitch_service=object(),
            session=FakePostSession(
                FakeResponse(
                    {"error": "Forbidden", "message": "subscription missing proper authorization"},
                    ok=False,
                    status_code=403,
                    text='{"message":"subscription missing proper authorization"}',
                )
            ),
            subscribe_chat_messages=True,
        )
        adapter._session_id = "session"

        ok = adapter._create_subscription(
            sub_type="channel.chat.message",
            version="1",
            condition={"broadcaster_user_id": "123", "user_id": "456"},
        )

        self.assertFalse(ok)

        state = StreamSessionState(enabled=True)
        api = FakeTwitchApi(
            stream={"title": "Live", "game_name": "JRPG", "tags": [], "started_at": "2026-05-31T18:00:00Z"},
            channel={"title": "Live", "game_name": "JRPG", "tags": []},
        )
        self.assertTrue(StreamContextSyncService(twitch_api=api, now_fn=lambda: 1_000_000.0).sync(state))
        self.assertTrue(state.is_live)


if __name__ == "__main__":
    unittest.main()
