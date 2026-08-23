'use client';

import { useMemo } from 'react';
import { Activity } from 'lucide-react';
import { useTextStream } from '@livekit/components-react';

interface MetricPayload {
  kind: 'stt' | 'tts';
  latency_ms: number;
  updated_at: number;
}

function parseLatestByKind(textStreams: { text: string }[]): {
  stt: MetricPayload | null;
  tts: MetricPayload | null;
} {
  let stt: MetricPayload | null = null;
  let tts: MetricPayload | null = null;
  for (const s of textStreams) {
    try {
      const payload = JSON.parse(s.text) as MetricPayload;
      if (payload.kind === 'stt') stt = payload;
      else if (payload.kind === 'tts') tts = payload;
    } catch {
      // Malformed payload — skip.
    }
  }
  return { stt, tts };
}

/**
 * Live STT/TTS pipeline latency, bottom-left — pulled from the agent's
 * `metrics_collected` events (see backend/src/agent.py) over a text-stream
 * data channel. STT = time from end of speech to transcript; TTS = time to
 * first audio byte. Not shown until at least one measurement has arrived.
 */
export function MetricsPanel() {
  const { textStreams } = useTextStream('healthmitra-metrics');
  const { stt, tts } = useMemo(() => parseLatestByKind(textStreams), [textStreams]);

  if (!stt && !tts) return null;

  return (
    <div className="border-border bg-card/90 text-muted-foreground fixed bottom-5 left-6 z-40 flex items-center gap-3 rounded-full border px-3 py-1.5 font-mono text-[11px] backdrop-blur-md">
      <Activity className="h-3 w-3 shrink-0" />
      <span>
        STT{' '}
        <span className="text-foreground font-semibold">
          {stt ? `${Math.round(stt.latency_ms)}ms` : '—'}
        </span>
      </span>
      <span className="border-border h-3 border-l" />
      <span>
        TTS{' '}
        <span className="text-foreground font-semibold">
          {tts ? `${Math.round(tts.latency_ms)}ms` : '—'}
        </span>
      </span>
    </div>
  );
}
