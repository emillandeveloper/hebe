export type EventType =
  | "stt.partial"
  | "stt.final"
  | "llm.partial"
  | "llm.final"
  | "tts.start"
  | "tts.end"
  | "avatar.state"
  | "status"
  | "error"
  | "chat.user"
  | "chat.assistant"
  // Futuro
  | "log"
  | (string & {});

export type HebeEvent = {
  type: EventType;
  data: any;
  ts: number;
};

export type ClientMsg =
  | { type: "client.message"; data: { text: string } }
  | { type: "client.command"; data: { name: string; payload?: Record<string, any> } };
