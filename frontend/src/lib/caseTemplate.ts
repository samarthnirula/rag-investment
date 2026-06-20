export interface CaseTab {
  id: "chat" | "timeline" | "overview";
  label: string;
  icon: string;
}

export const CASE_TAB_TEMPLATE: CaseTab[] = [
  { id: "chat",     label: "Chat",     icon: "💬" },
  { id: "timeline", label: "Timeline", icon: "📅" },
  { id: "overview", label: "Overview", icon: "📋" },
];
