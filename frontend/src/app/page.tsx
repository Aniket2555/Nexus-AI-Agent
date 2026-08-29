"use client";

import { Thread } from "@/components/thread";
import { StreamProvider } from "@/providers/Stream";
import { ThreadProvider } from "@/providers/Thread";
import { Toaster } from "@/components/ui/sonner";
import { NexusSidebar } from "@/components/nexus/NexusSidebar";
import React from "react";

export default function DemoPage(): React.ReactNode {
  return (
    <React.Suspense fallback={<div>Loading (layout)...</div>}>
      <Toaster />
      <ThreadProvider>
        <StreamProvider>
          <Thread />
          <NexusSidebar />
        </StreamProvider>
      </ThreadProvider>
    </React.Suspense>
  );
}
