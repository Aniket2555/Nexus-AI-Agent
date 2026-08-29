"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { DataSpecialistDetails } from "@/lib/nexus-types";

function pickAxes(rows: Record<string, unknown>[]) {
  if (rows.length === 0) return null;
  const keys = Object.keys(rows[0]);
  const numericKey = keys.find((k) => typeof rows[0][k] === "number");
  const categoryKey = keys.find((k) => k !== numericKey);
  if (!numericKey || !categoryKey) return null;
  return { categoryKey, numericKey };
}

/**
 * Renders a Data specialist's `SpecialistResult.details` (backend/app/graphs/
 * specialists/data.py) — the generated read-only SQL, and up to 200 result
 * rows (the same cap the backend applies before ever sending them here).
 * Charts when a numeric column is present; otherwise falls back to a table,
 * since not every query result is chartable (e.g. a single scalar count).
 */
export function DataChart({ details }: { details: DataSpecialistDetails }) {
  if (details.error) {
    return (
      <div className="rounded-md border border-red-300 bg-red-50 p-3 text-sm text-red-800 dark:border-red-800 dark:bg-red-950 dark:text-red-200">
        {details.error}
      </div>
    );
  }

  const axes = pickAxes(details.rows);

  return (
    <div className="flex flex-col gap-2">
      <p className="text-xs font-mono text-muted-foreground break-all">
        {details.sql_query}
      </p>

      {axes ? (
        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={details.rows}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey={axes.categoryKey} />
              <YAxis />
              <Tooltip />
              <Bar dataKey={axes.numericKey} fill="#2563eb" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      ) : (
        <div className="overflow-x-auto rounded-md border">
          <table className="w-full text-sm">
            <thead className="bg-muted">
              <tr>
                {details.rows[0] &&
                  Object.keys(details.rows[0]).map((col) => (
                    <th key={col} className="px-2 py-1 text-left font-medium">
                      {col}
                    </th>
                  ))}
              </tr>
            </thead>
            <tbody>
              {details.rows.map((row, i) => (
                <tr key={i} className="border-t">
                  {Object.values(row).map((value, j) => (
                    <td key={j} className="px-2 py-1">
                      {String(value)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <p className="text-xs text-muted-foreground">
        {details.row_count} row(s)
      </p>
    </div>
  );
}
