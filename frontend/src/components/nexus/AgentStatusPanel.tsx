"use client";

import { useStreamContext } from "@/providers/Stream";
import { Loader2, CheckCircle2, PauseCircle } from "lucide-react";

function StatRow({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between text-sm">
      <span className="text-muted-foreground">{label}</span>
      <span className="font-medium">{value}</span>
    </div>
  );
}

/**
 * A live, graph-agnostic status readout — both `rag_chat` (§3.2's reflection
 * loop) and `nexus_supervisor` (§6.2's escalation loop) reuse the same
 * `should_retry()` machinery and therefore the same `iteration_count`/
 * `quality_score` fields, so this panel renders identically for either graph
 * rather than needing a switch on which one is active.
 */
export function AgentStatusPanel() {
  const stream = useStreamContext();
  const { values, isLoading, interrupt } = stream;

  const iterationCount = values.iteration_count;
  const qualityScore = values.quality_score;
  const feedback = values.reflection_feedback || values.review_feedback;

  return (
    <div className="flex flex-col gap-3 p-4">
      <div className="flex items-center gap-2 text-sm font-medium">
        {interrupt ? (
          <>
            <PauseCircle className="size-4 text-amber-600" />
            Waiting on human approval
          </>
        ) : isLoading ? (
          <>
            <Loader2 className="size-4 animate-spin text-blue-600" />
            Running
          </>
        ) : (
          <>
            <CheckCircle2 className="size-4 text-green-600" />
            Idle
          </>
        )}
      </div>

      {iterationCount !== undefined && (
        <StatRow label="Reflection iterations" value={iterationCount} />
      )}
      {qualityScore !== undefined && (
        <StatRow label="Quality score" value={qualityScore.toFixed(2)} />
      )}

      {feedback && (
        <div className="rounded-md border bg-muted/50 p-2 text-xs text-muted-foreground">
          {feedback}
        </div>
      )}

      {!interrupt && !isLoading && iterationCount === undefined && (
        <p className="text-sm text-muted-foreground">
          Send a message to see live agent status here.
        </p>
      )}
    </div>
  );
}
