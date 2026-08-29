"use client";

import { useState } from "react";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import {
  Activity,
  BookOpen,
  Brain,
  PanelRightOpen,
  UploadCloud,
  Workflow,
} from "lucide-react";
import { AgentStatusPanel } from "./AgentStatusPanel";
import { RAGSourceViewer } from "./RAGSourceViewer";
import { MemoryInspector } from "./MemoryInspector";
import { SupervisorPanel } from "./SupervisorPanel";
import { DocumentUploader } from "./DocumentUploader";

type NexusTab = "status" | "sources" | "memory" | "supervisor" | "documents";

const TABS: { id: NexusTab; label: string; icon: React.ReactNode }[] = [
  { id: "status", label: "Status", icon: <Activity className="size-4" /> },
  { id: "sources", label: "Sources", icon: <BookOpen className="size-4" /> },
  { id: "memory", label: "Memory", icon: <Brain className="size-4" /> },
  { id: "supervisor", label: "Agents", icon: <Workflow className="size-4" /> },
  {
    id: "documents",
    label: "Documents",
    icon: <UploadCloud className="size-4" />,
  },
];

/**
 * The single entry point for every NEXUS-specific extension on top of the
 * vendored agent-chat-ui (D3/D10). Kept as a slide-over `Sheet` rather than a
 * permanent side-by-side column so it never has to fight `Thread`'s own
 * `w-full h-screen` layout — this is additive to the upstream component tree,
 * not a rework of it.
 */
export function NexusSidebar() {
  const [tab, setTab] = useState<NexusTab>("status");

  return (
    <Sheet>
      <SheetTrigger asChild>
        <Button
          variant="outline"
          size="icon"
          className="fixed right-4 top-4 z-30"
          aria-label="Open NEXUS panel"
        >
          <PanelRightOpen className="size-4" />
        </Button>
      </SheetTrigger>
      <SheetContent
        side="right"
        className="w-full sm:max-w-md flex flex-col p-0"
      >
        <SheetHeader className="border-b">
          <SheetTitle>NEXUS</SheetTitle>
        </SheetHeader>

        <div className="flex border-b overflow-x-auto shrink-0">
          {TABS.map((t) => (
            <button
              key={t.id}
              type="button"
              onClick={() => setTab(t.id)}
              className={cn(
                "flex items-center gap-1.5 px-3 py-2 text-sm whitespace-nowrap border-b-2 -mb-px",
                tab === t.id
                  ? "border-primary text-foreground font-medium"
                  : "border-transparent text-muted-foreground hover:text-foreground",
              )}
            >
              {t.icon}
              {t.label}
            </button>
          ))}
        </div>

        <div className="flex-1 overflow-y-auto">
          {tab === "status" && <AgentStatusPanel />}
          {tab === "sources" && <RAGSourceViewer />}
          {tab === "memory" && <MemoryInspector />}
          {tab === "supervisor" && <SupervisorPanel />}
          {tab === "documents" && <DocumentUploader />}
        </div>
      </SheetContent>
    </Sheet>
  );
}
