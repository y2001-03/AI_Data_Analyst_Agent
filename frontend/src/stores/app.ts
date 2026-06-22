import { defineStore } from "pinia";

interface AppState {
  activeSessionId: string;
}

export const useAppStore = defineStore("app", {
  state: (): AppState => ({
    activeSessionId: "demo-session"
  })
});

