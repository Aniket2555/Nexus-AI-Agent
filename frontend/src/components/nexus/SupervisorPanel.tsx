"use client";

import { useState } from "react";
import { useStreamContext } from "@/providers/Stream";
import type {
  CodeSpecialistDetails,
  DataSpecialistDetails,
  SpecialistResult,
  SubTask,
} from "@/lib/nexus-types";
import {
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  CircleDashed,
  XCircle,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { CodeBlock } from "@/components/generative-ui/CodeBlock";
import { DataChart } from "@/components/generative-ui/DataChart";
import { SearchResults } from "@/components/generative-ui/SearchResults";

function ResultDetail({ result }: { result: SpecialistResult }) {
  if (result.specialist === "code") {
    return (
      <CodeBlock details={result.details as unknown as CodeSpecialistDetails} />
    );
  }
  if (result.specialist === "data") {
    return (
      <DataChart details={result.details as unknown as DataSpecialistDetails} />
    );
  }
  if (result.specialist === "research") {
    return <SearchResults result={result} />;
  }
  return <p className="text-sm text-muted-foreground">{result.summary}</p>;
}

const SPECIALIST_COLOR: Record<string, string> = {
  research: "bg-blue-100 text-blue-800 dark:bg-blue-950 dark:text-blue-300",
  code: "bg-purple-100 text-purple-800 dark:bg-purple-950 dark:text-purple-300",
  data: "bg-amber-100 text-amber-800 dark:bg-amber-950 dark:text-amber-300",
  browser: "bg-teal-100 text-teal-800 dark:bg-teal-950 dark:text-teal-300",
  vision: "bg-pink-100 text-pink-800 dark:bg-pink-950 dark:text-pink-300",
  report: "bg-slate-100 text-slate-800 dark:bg-slate-800 dark:text-slate-300",
};

function TaskRow({
  task,
  result,
}: {
  task: SubTask;
  result: SpecialistResult | undefined;
}) {
  const status = !result ? "pending" : result.success ? "done" : "failed";
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="flex flex-col gap-1 rounded-md border bg-background p-3 text-sm">
      <div className="flex items-center gap-2">
        {status === "pending" && (
          <CircleDashed className="size-4 text-muted-foreground" />
        )}
        {status === "done" && (
          <CheckCircle2 className="size-4 text-green-600" />
        )}
        {status === "failed" && <XCircle className="size-4 text-red-600" />}

        <span
          className={cn(
            "rounded px-1.5 py-0.5 text-xs font-medium",
            SPECIALIST_COLOR[task.specialist] ??
              "bg-muted text-muted-foreground",
          )}
        >
          {task.specialist}
        </span>
        <span className="text-muted-foreground text-xs">#{task.id}</span>

        {result && (
          <button
            type="button"
            onClick={() => setExpanded((e) => !e)}
            className="ml-auto text-muted-foreground"
            aria-label={expanded ? "Collapse" : "Expand"}
          >
            {expanded ? (
              <ChevronDown className="size-4" />
            ) : (
              <ChevronRight className="size-4" />
            )}
          </button>
        )}
      </div>

      <p>{task.description}</p>

      {task.depends_on.length > 0 && (
        <p className="text-xs text-muted-foreground">
          depends on: {task.depends_on.join(", ")}
        </p>
      )}

      {result && !expanded && (
        <p className="text-xs text-muted-foreground border-t pt-1 mt-1">
          {result.summary}
        </p>
      )}

      {result && expanded && (
        <div className="border-t pt-2 mt-1">
          <ResultDetail result={result} />
        </div>
      )}
    </div>
  );
}

/**
 * Renders the `nexus_supervisor` graph's `sub_tasks` (the planner's DAG) next
 * to each task's `agent_results` entry once dispatch_node has produced one —
 * agent-to-agent handoffs are expressed via `depends_on`, not a direct call
 * (§6.1/§6.2), so this reads as a checklist rather than a call trace.
 */
export function SupervisorPanel() {
  const stream = useStreamContext();
  const subTasks = stream.values.sub_tasks ?? [];
  const results = stream.values.agent_results ?? [];

  if (subTasks.length === 0) {
    return (
      <div className="p-4 text-sm text-muted-foreground">
        No plan has been generated yet.
      </div>
    );
  }

  const resultByTaskId = new Map(results.map((r) => [r.task_id, r]));

  return (
    <div className="flex flex-col gap-3 p-4">
      {stream.values.current_plan && (
        <p className="text-sm text-muted-foreground italic">
          {stream.values.current_plan}
        </p>
      )}
      {subTasks.map((task) => (
        <TaskRow
          key={task.id}
          task={task}
          result={resultByTaskId.get(task.id)}
        />
      ))}
      {stream.values.review_passed === false &&
        stream.values.review_feedback && (
          <div className="rounded-md border border-amber-300 bg-amber-50 p-3 text-sm text-amber-900 dark:border-amber-800 dark:bg-amber-950 dark:text-amber-200">
            Review failed: {stream.values.review_feedback}
          </div>
        )}
    </div>
  );
}
