'use client';

import { useEffect, useMemo, useRef } from 'react';
import { AlertTriangle, Clock, Globe, HeartPulse, MapPin, Stethoscope } from 'lucide-react';
import { motion } from 'motion/react';
import { useTextStream } from '@livekit/components-react';
import { cn } from '@/lib/shadcn/utils';

// ─── Payload shapes — must match backend/src/health_tools.py + agent.py ──────

type TriageLevel = 'emergency' | 'phc' | 'home_care' | 'unclear';

interface TriagePayload {
  level: TriageLevel;
  matched_keyword: string | null;
  // Already in whichever language (English or Hindi) the call is
  // currently in — chosen server-side, see backend/src/health_tools.py.
  advice: string;
  language: 'en' | 'hi';
  source: string;
  ruleset_version: string;
}

interface Facility {
  name: string;
  type: string;
  area: string;
}

interface FacilityPayload {
  status: 'ok' | 'no_district' | 'not_found';
  district: string;
  facilities: Facility[];
  data_source: 'live:openstreetmap-nominatim' | 'local-fallback' | null;
  fetched_at: string | null;
}

type Card =
  | { id: string; timestamp: number; kind: 'triage'; data: TriagePayload }
  | { id: string; timestamp: number; kind: 'facility'; data: FacilityPayload };

function parseStreams<T>(
  textStreams: { text: string; streamInfo: { id: string; timestamp: number } }[]
): { id: string; timestamp: number; data: T }[] {
  const out: { id: string; timestamp: number; data: T }[] = [];
  for (const s of textStreams) {
    try {
      out.push({
        id: s.streamInfo.id,
        timestamp: s.streamInfo.timestamp,
        data: JSON.parse(s.text) as T,
      });
    } catch {
      // Malformed payload — skip rather than crash the panel.
    }
  }
  return out;
}

// ─── Triage card ───────────────────────────────────────────────────────────

const TRIAGE_CONFIG: Record<
  TriageLevel,
  { label: string; className: string; icon: React.ReactNode }
> = {
  emergency: {
    label: 'Emergency — call 108',
    className: 'border-destructive/40 bg-destructive/10 text-destructive',
    icon: <AlertTriangle className="h-4 w-4" />,
  },
  phc: {
    label: 'Visit a PHC today',
    className: 'border-amber-500/40 bg-amber-500/10 text-amber-700 dark:text-amber-400',
    icon: <Stethoscope className="h-4 w-4" />,
  },
  home_care: {
    label: 'Home care',
    className: 'border-emerald-500/40 bg-emerald-500/10 text-emerald-700 dark:text-emerald-400',
    icon: <HeartPulse className="h-4 w-4" />,
  },
  unclear: {
    label: 'Needs more detail',
    className: 'border-border bg-muted text-muted-foreground',
    icon: <HeartPulse className="h-4 w-4" />,
  },
};

function TriageCard({ data }: { data: TriagePayload }) {
  const cfg = TRIAGE_CONFIG[data.level];
  return (
    <div className="border-border bg-card h-full w-full rounded-lg border p-4 text-left">
      <div
        className={cn(
          'inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-semibold',
          cfg.className
        )}
      >
        {cfg.icon}
        {cfg.label}
      </div>
      <p className="text-foreground mt-2 text-sm leading-relaxed">{data.advice}</p>
      <p className="text-muted-foreground mt-2 text-[11px]">
        Screening aid only, not a diagnosis · local rule set {data.ruleset_version}
        {data.matched_keyword ? ` · matched "${data.matched_keyword}"` : ''}
      </p>
    </div>
  );
}

// ─── Facility card ─────────────────────────────────────────────────────────

function FacilityCard({ data }: { data: FacilityPayload }) {
  if (data.status === 'not_found') {
    return (
      <div className="border-border bg-card h-full w-full rounded-lg border p-4 text-left">
        <div className="text-foreground inline-flex items-center gap-1.5 text-sm font-semibold">
          <MapPin className="h-4 w-4" />
          No facilities found for {data.district || 'this area'}
        </div>
        <p className="text-muted-foreground mt-2 text-sm leading-relaxed">
          Try the 104 health helpline or a local ASHA worker for directions.
        </p>
      </div>
    );
  }

  const isLive = data.data_source === 'live:openstreetmap-nominatim';

  return (
    <div className="border-border bg-card h-full w-full rounded-lg border p-4 text-left">
      <div className="flex items-center justify-between gap-2">
        <div className="text-foreground inline-flex items-center gap-1.5 text-sm font-semibold">
          <MapPin className="h-4 w-4" />
          Near {data.district}
        </div>
        <div
          className={cn(
            'inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[10px] font-medium',
            isLive
              ? 'border-emerald-500/40 bg-emerald-500/10 text-emerald-700 dark:text-emerald-400'
              : 'border-border bg-muted text-muted-foreground'
          )}
        >
          <Globe className="h-3 w-3" />
          {isLive ? 'Live · OpenStreetMap' : 'Local reference list'}
        </div>
      </div>

      <ul className="mt-3 space-y-2">
        {data.facilities.map((f) => (
          <li key={f.name} className="text-sm">
            <span className="text-foreground font-medium">{f.name}</span>
            <span className="text-muted-foreground"> — {f.type}</span>
            <div className="text-muted-foreground text-xs">{f.area}</div>
          </li>
        ))}
      </ul>

      {data.fetched_at && (
        <p className="text-muted-foreground mt-3 inline-flex items-center gap-1 text-[11px]">
          <Clock className="h-3 w-3" />
          As of {data.fetched_at}
        </p>
      )}
    </div>
  );
}

// ─── Panel ─────────────────────────────────────────────────────────────────

/**
 * Shows every triage/facility result the agent's tools have fetched this
 * call, live, as a horizontal rail of cards — addresses and facility names
 * are hard to hold in your head from audio alone. Each new result is
 * appended as its own card on the right; the rail auto-scrolls to it, which
 * visually slides the earlier cards left to make room, rather than
 * replacing them or growing the page taller.
 */
export function HealthDataPanel() {
  const { textStreams: triageStreams } = useTextStream('healthmitra-triage');
  const { textStreams: facilityStreams } = useTextStream('healthmitra-facility');

  const cards = useMemo<Card[]>(() => {
    const triageCards = parseStreams<TriagePayload>(triageStreams).map((c) => ({
      ...c,
      kind: 'triage' as const,
    }));
    const facilityCards = parseStreams<FacilityPayload>(facilityStreams)
      .filter((c) => c.data.status !== 'no_district') // nothing to show yet — not a card
      .map((c) => ({ ...c, kind: 'facility' as const }));
    return [...triageCards, ...facilityCards].sort((a, b) => a.timestamp - b.timestamp);
  }, [triageStreams, facilityStreams]);

  const railRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = railRef.current;
    if (!el) return;
    el.scrollTo({ left: el.scrollWidth, behavior: 'smooth' });
  }, [cards.length]);

  if (cards.length === 0) return null;

  return (
    <div
      ref={railRef}
      className="flex w-full snap-x gap-3 overflow-x-auto scroll-smooth pb-2 [scrollbar-width:thin]"
    >
      {cards.map((card) => (
        <motion.div
          key={card.id}
          layout
          initial={{ opacity: 0, x: 32 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.35, ease: 'easeOut' }}
          className="w-[300px] shrink-0 snap-start"
        >
          {card.kind === 'triage' ? (
            <TriageCard data={card.data} />
          ) : (
            <FacilityCard data={card.data} />
          )}
        </motion.div>
      ))}
    </div>
  );
}
