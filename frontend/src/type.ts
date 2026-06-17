export type ServerEvent = {
  type:
    | "stt.partial" | "stt.final"
    | "llm.partial" | "llm.final"
    | "tts.start" | "tts.end"
    | "avatar.state"
    | "status"
    | "error"
    | "chat_message"
    | "chat.user" | "chat.assistant";
  data: any;
  ts: number;
  event_id?: string;
};

export type ClientMsg =
  | { type: "client.message"; data: { text: string } }
  | { type: "client.command"; data: { name: string; payload?: any } };
