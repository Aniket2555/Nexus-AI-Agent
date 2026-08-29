"use client";

import { useState } from "react";
import { useStreamContext } from "@/providers/Stream";
import type { MemoryFact } from "@/lib/nexus-types";
import { ChevronDown, ChevronRight, Brain, Network } from "lucide-react";
import { cn } from "@/lib/utils";

function FactRow({ fact }: { fact: MemoryFact }) {
  return (
    <div className="rounded-md border bg-background p-2 text-sm">
      <p>{fact.content}</p>
      <div className="mt-1 flex items-center justify-between text-xs text-muted-foreground">
        <span>{String(fact.metadata?.source ?? "unknown source")}</span>
        <span>score {fact.score.toFixed(2)}</span>
      </div>
    </div>
  );
}

function Section({
  title,
  icon,
  facts,
}: {
  title: string;
  icon: React.ReactNode;
  facts: MemoryFact[];
}) {
  const [open, setOpen] = useState(true);
  return (
    <div className="flex flex-col gap-2">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="flex items-center gap-2 text-sm font-medium"
      >
        {open ? (
          <ChevronDown className="size-4" />
        ) : (
          <ChevronRight className="size-4" />
        )}
        {icon}
        {title}
        <span className="text-muted-foreground font-normal">
          ({facts.length})
        </span>
      </button>
      {open && (
        <div
          className={cn(
            "flex flex-col gap-2 pl-6",
            facts.length === 0 && "hidden",
          )}
        >
          {facts.map((fact) => (
            <FactRow key={fact.id} fact={fact} />
          ))}
        </div>
      )}
    </div>
  );
}

/**
 * Renders `rag_chat` graph's `recalled_memory` verbatim — MemoryManager.recall()'s
 * output exposed separately from the retrieved_docs it also gets merged into
 * (see rag_chat.py's memory_node docstring), so this never needs a backend
 * change to reflect what memory actually contributed to an answer.
 */
export function MemoryInspector() {
  const stream = useStreamContext();
  const memory = stream.values.recalled_memory;

  if (
    !memory ||
    (memory.long_term_facts.length === 0 && memory.graph_context.length === 0)
  ) {
    return (
      <div className="p-4 text-sm text-muted-foreground">
        No memory was recalled for the current turn.
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-4 p-4">
      <Section
        title="Long-term facts"
        icon={<Brain className="size-4 text-muted-foreground" />}
        facts={memory.long_term_facts}
      />
      <Section
        title="Knowledge graph"
        icon={<Network className="size-4 text-muted-foreground" />}
        facts={memory.graph_context}
      />
    </div>
  );
}
