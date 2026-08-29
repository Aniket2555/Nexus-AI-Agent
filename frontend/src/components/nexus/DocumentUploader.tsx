"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Loader2,
  UploadCloud,
  CheckCircle2,
  XCircle,
  FileText,
} from "lucide-react";
import { cn } from "@/lib/utils";
import type {
  DocumentListResponse,
  DocumentRecord,
  DocumentStatus,
  UploadJobResponse,
} from "@/lib/nexus-types";

const REST_API_URL =
  process.env.NEXT_PUBLIC_NEXUS_API_URL || "http://localhost:8000";
const API_KEY = process.env.NEXT_PUBLIC_NEXUS_API_KEY;

const POLL_INTERVAL_MS = 1500;

function authHeaders(): HeadersInit {
  return API_KEY ? { "X-API-Key": API_KEY } : {};
}

function StatusIcon({ status }: { status: DocumentStatus }) {
  if (status === "succeeded")
    return <CheckCircle2 className="size-4 text-green-600" />;
  if (status === "failed") return <XCircle className="size-4 text-red-600" />;
  return <Loader2 className="size-4 animate-spin text-blue-600" />;
}

/**
 * Ingests a document through `POST /api/v1/documents/upload` (§1.6/§2.7's
 * ingestion pipeline — extract pages -> chunk -> dense + sparse upsert ->
 * Postgres registry record) and polls `GET /api/v1/documents/jobs/{job_id}`
 * until the background task finishes, then refreshes the registry list from
 * `GET /api/v1/documents` — the same list any tenant-scoped caller sees,
 * independent of this component's own in-memory job tracking.
 */
export function DocumentUploader() {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [uploading, setUploading] = useState(false);
  const [jobStatus, setJobStatus] = useState<UploadJobResponse | null>(null);
  const [documents, setDocuments] = useState<DocumentRecord[]>([]);
  const [error, setError] = useState<string | null>(null);

  const refreshDocuments = useCallback(async () => {
    try {
      const res = await fetch(`${REST_API_URL}/api/v1/documents`, {
        headers: authHeaders(),
      });
      if (!res.ok) return;
      const data: DocumentListResponse = await res.json();
      setDocuments(data.documents);
    } catch {
      // Best-effort refresh; upload flow surfaces its own errors separately.
    }
  }, []);

  useEffect(() => {
    refreshDocuments();
  }, [refreshDocuments]);

  useEffect(() => {
    if (
      !jobStatus ||
      jobStatus.status === "succeeded" ||
      jobStatus.status === "failed"
    ) {
      return;
    }

    const interval = setInterval(async () => {
      try {
        const res = await fetch(
          `${REST_API_URL}/api/v1/documents/jobs/${jobStatus.job_id}`,
          { headers: authHeaders() },
        );
        if (!res.ok) return;
        const data = await res.json();
        setJobStatus(data);
        if (data.status === "succeeded" || data.status === "failed") {
          refreshDocuments();
        }
      } catch {
        // Keep polling; a transient network error shouldn't kill the loop.
      }
    }, POLL_INTERVAL_MS);

    return () => clearInterval(interval);
  }, [jobStatus, refreshDocuments]);

  const handleUpload = async () => {
    const file = fileInputRef.current?.files?.[0];
    if (!file) return;

    setUploading(true);
    setError(null);
    setJobStatus(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const res = await fetch(`${REST_API_URL}/api/v1/documents/upload`, {
        method: "POST",
        headers: authHeaders(),
        body: formData,
      });

      if (!res.ok) {
        const detail = await res.text();
        throw new Error(detail || `Upload failed with status ${res.status}`);
      }

      const data: UploadJobResponse = await res.json();
      setJobStatus(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Upload failed.");
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  return (
    <div className="flex flex-col gap-4 p-4">
      <div className="flex flex-col gap-2">
        <Label htmlFor="nexus-doc-upload">Upload a document</Label>
        <Input
          id="nexus-doc-upload"
          type="file"
          ref={fileInputRef}
          accept=".pdf,.txt,.md,.docx,.html"
          disabled={uploading}
        />
        <Button
          onClick={handleUpload}
          disabled={uploading}
          className="self-start"
        >
          {uploading ? (
            <Loader2 className="size-4 animate-spin" />
          ) : (
            <UploadCloud className="size-4" />
          )}
          Ingest
        </Button>
      </div>

      {error && (
        <p className="text-sm text-red-600" role="alert">
          {error}
        </p>
      )}

      {jobStatus && (
        <div className="flex items-center gap-2 rounded-md border bg-background p-3 text-sm">
          <StatusIcon status={jobStatus.status} />
          <span className="truncate">{jobStatus.filename}</span>
          <span className="text-muted-foreground">— {jobStatus.status}</span>
        </div>
      )}

      <div className="flex flex-col gap-2">
        <p className="text-sm font-medium">Ingested documents</p>
        {documents.length === 0 ? (
          <p className="text-sm text-muted-foreground">No documents yet.</p>
        ) : (
          documents.map((doc) => (
            <div
              key={doc.doc_id}
              className={cn(
                "flex items-center justify-between rounded-md border bg-background p-2 text-sm",
              )}
            >
              <div className="flex items-center gap-2 truncate">
                <FileText className="size-4 text-muted-foreground shrink-0" />
                <span className="truncate">{doc.filename}</span>
              </div>
              <div className="flex items-center gap-2 text-xs text-muted-foreground shrink-0">
                <StatusIcon status={doc.status} />
                {doc.chunk_count} chunks
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
