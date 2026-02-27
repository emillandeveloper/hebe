export type ServerEvent = {
  type:
    | "stt.partial" | "stt.final"
    | "llm.partial" | "llm.final"
    | "tts.start" | "tts.end"
    | "avatar.state"
    | "status"
    | "error"
    | "chat.user" | "chat.assistant";
  data: any;
  ts: number;
};

export type ClientMsg =
  | { type: "client.message"; data: { text: string } }
  | { type: "client.command"; data: { name: string; payload?: any } };
