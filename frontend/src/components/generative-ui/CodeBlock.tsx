import { SyntaxHighlighter } from "@/components/thread/syntax-highlighter";
import type { CodeSpecialistDetails } from "@/lib/nexus-types";
import { CheckCircle2, XCircle, Clock } from "lucide-react";

/**
 * Renders a Code specialist's `SpecialistResult.details` (backend/app/graphs/
 * specialists/code.py) — the generated code, its `ast.parse()` syntax-validity
 * verdict, and (Phase 7 sandbox) execution stdout/stderr if it actually ran.
 * Non-Python code is only syntax-checked, never executed — `note` carries
 * that distinction verbatim from the backend rather than being re-derived here.
 */
export function CodeBlock({ details }: { details: CodeSpecialistDetails }) {
  return (
    <div className="flex flex-col gap-2 rounded-lg border overflow-hidden">
      <div className="flex items-center justify-between bg-muted px-3 py-1.5 text-xs">
        <span className="font-mono">{details.language || "text"}</span>
        <div className="flex items-center gap-1">
          {details.syntax_valid ? (
            <CheckCircle2 className="size-3.5 text-green-600" />
          ) : (
            <XCircle className="size-3.5 text-red-600" />
          )}
          <span className="text-muted-foreground">
            {details.syntax_valid ? "syntax valid" : "syntax error"}
          </span>
        </div>
      </div>

      <SyntaxHighlighter language={details.language || "python"}>
        {details.code}
      </SyntaxHighlighter>

      {details.syntax_error && (
        <p className="px-3 pb-2 text-xs text-red-600 font-mono">
          {details.syntax_error}
        </p>
      )}

      {details.executed && (
        <div className="flex flex-col gap-1 border-t bg-muted/30 px-3 py-2 text-xs font-mono">
          <div className="flex items-center gap-1 text-muted-foreground">
            {details.execution_timed_out ? (
              <Clock className="size-3.5" />
            ) : details.execution_exit_code === 0 ? (
              <CheckCircle2 className="size-3.5 text-green-600" />
            ) : (
              <XCircle className="size-3.5 text-red-600" />
            )}
            exit code: {details.execution_exit_code ?? "—"}
            {details.execution_timed_out && " (timed out)"}
            {details.execution_truncated && " (output truncated)"}
          </div>
          {details.execution_stdout && (
            <pre className="whitespace-pre-wrap">
              {details.execution_stdout}
            </pre>
          )}
          {details.execution_stderr && (
            <pre className="whitespace-pre-wrap text-red-600">
              {details.execution_stderr}
            </pre>
          )}
        </div>
      )}

      <p className="px-3 pb-2 text-xs text-muted-foreground">{details.note}</p>
    </div>
  );
}
