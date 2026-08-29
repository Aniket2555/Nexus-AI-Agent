"use client";

import { useStreamContext } from "@/providers/Stream";
import type { Citation } from "@/lib/nexus-types";
import { FileText } from "lucide-react";

function CitationCard({ citation }: { citation: Citation }) {
  return (
    <div className="flex flex-col gap-1 rounded-md border bg-background p-3 text-sm">
      <div className="flex items-center gap-2 font-medium">
        <FileText className="size-4 text-muted-foreground" />
        <span className="truncate">{citation.source}</span>
        {citation.page !== null && (
          <span className="text-muted-foreground font-normal">
            · Page {citation.page}
          </span>
        )}
      </div>
      <div className="text-xs text-muted-foreground truncate">
        {citation.chunk_id}
      </div>
      <div className="flex gap-3 text-xs text-muted-foreground">
        {citation.retrieval_score !== null && (
          <span>retrieval: {citation.retrieval_score.toFixed(4)}</span>
        )}
        {citation.rerank_score !== null && (
          <span>rerank: {citation.rerank_score.toFixed(4)}</span>
        )}
      </div>
    </div>
  );
}

/**
 * Renders `rag_chat` graph's `citations` — the same `Citation` objects
 * `citation_builder.build_citations()` produces server-side, keyed on
 * `chunk_id` (never a store's native id, so this survives dense/sparse
 * fusion — see hybrid.py). Empty when the last answer had no matching
 * content ("compress_node"'s no-hits path).
 */
export function RAGSourceViewer() {
  const stream = useStreamContext();
  const citations = stream.values.citations ?? [];

  if (citations.length === 0) {
    return (
      <div className="p-4 text-sm text-muted-foreground">
        No sources for the current answer yet.
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-2 p-4">
      {citations.map((citation, i) => (
        <CitationCard key={`${citation.chunk_id}-${i}`} citation={citation} />
      ))}
    </div>
  );
}
