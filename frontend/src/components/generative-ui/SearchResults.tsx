import type { SpecialistResult } from "@/lib/nexus-types";
import { Search, FileText } from "lucide-react";

/**
 * Renders a Research specialist's `SpecialistResult` (backend/app/graphs/
 * specialists/research.py) — its synthesized `summary` plus the citations it
 * grounded that summary in, built from the same hybrid-retrieval + rerank
 * pipeline `rag_chat.py` uses, just invoked as one dispatched sub-task rather
 * than the top-level chat turn.
 */
export function SearchResults({ result }: { result: SpecialistResult }) {
  return (
    <div className="flex flex-col gap-2 rounded-lg border p-3">
      <div className="flex items-center gap-2 text-sm font-medium">
        <Search className="size-4 text-muted-foreground" />
        Research: task #{result.task_id}
      </div>

      <p className="text-sm">{result.summary}</p>

      {result.citations.length > 0 && (
        <div className="flex flex-col gap-1 border-t pt-2">
          {result.citations.map((citation, i) => (
            <div
              key={`${citation.chunk_id}-${i}`}
              className="flex items-center gap-2 text-xs text-muted-foreground"
            >
              <FileText className="size-3.5 shrink-0" />
              <span className="truncate">{citation.source}</span>
              {citation.page !== null && <span>· p.{citation.page}</span>}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
