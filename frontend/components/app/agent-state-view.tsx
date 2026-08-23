'use client';

import React from 'react';
import { Loader2, Mic, MicOff, PhoneOff } from 'lucide-react';
import { AnimatePresence, motion } from 'motion/react';
import {
  type ReceivedMessage,
  useLocalParticipant,
  useVoiceAssistant,
} from '@livekit/components-react';
import { AgentAudioVisualizerBar } from '@/components/agents-ui/agent-audio-visualizer-bar';
import { AgentChatTranscript } from '@/components/agents-ui/agent-chat-transcript';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/shadcn/utils';

// ─── Types ───────────────────────────────────────────────────────────────────

export type AgentSdkState =
  | 'disconnected'
  | 'connecting'
  | 'pre-connect-buffering'
  | 'initializing'
  | 'idle'
  | 'listening'
  | 'thinking'
  | 'speaking'
  | 'failed';

export type UiPhase = 'ready' | 'connecting' | 'listening' | 'speaking' | 'ended' | 'blocked';

export function deriveUiPhase(
  isConnected: boolean,
  agentState: AgentSdkState | undefined,
  showEnded: boolean
): UiPhase {
  if (!isConnected && showEnded) return 'ended';
  if (!isConnected) return 'ready';
  if (
    !agentState ||
    agentState === 'connecting' ||
    agentState === 'pre-connect-buffering' ||
    agentState === 'initializing' ||
    agentState === 'idle'
  )
    return 'connecting';
  if (agentState === 'speaking') return 'speaking';
  return 'listening'; // listening | thinking
}

// ─── Per-phase config ────────────────────────────────────────────────────────
// One semantic color per phase: neutral for idle states, primary (brand
// accent) while the agent talks, emerald while it listens, red for errors.

interface PhaseConfig {
  label: string;
  headline: string;
  sublabel: string;
  orb: string; // solid background for the orb
  dot: string; // solid background for the status dot
  pulsing: boolean;
}

const PHASE_CONFIG: Record<UiPhase, PhaseConfig> = {
  ready: {
    label: 'Ready',
    headline: "Hi, I'm HealthMitra",
    sublabel:
      'Your AI health companion — symptom guidance, PHC referrals, and scheme info. Tap below to begin; speak in Hindi or English.',
    orb: 'bg-primary text-primary-foreground',
    dot: 'bg-primary',
    pulsing: false,
  },
  connecting: {
    label: 'Connecting',
    headline: 'Connecting…',
    sublabel: 'Please wait while HealthMitra joins the call.',
    orb: 'bg-primary text-primary-foreground',
    dot: 'bg-primary',
    pulsing: false,
  },
  listening: {
    label: 'Listening',
    headline: 'Listening',
    sublabel: 'Go ahead — HealthMitra is listening to you.',
    orb: 'bg-emerald-600 text-white',
    dot: 'bg-emerald-600',
    pulsing: true,
  },
  speaking: {
    label: 'Speaking',
    headline: 'Speaking',
    sublabel: 'HealthMitra is replying to you.',
    orb: 'bg-primary text-primary-foreground',
    dot: 'bg-primary',
    pulsing: true,
  },
  ended: {
    label: 'Call ended',
    headline: 'Call ended',
    sublabel: "The conversation has ended. Start a new one whenever you're ready.",
    orb: 'bg-muted-foreground text-background',
    dot: 'bg-muted-foreground',
    pulsing: false,
  },
  blocked: {
    label: 'Microphone blocked',
    headline: 'Microphone access blocked',
    sublabel:
      "Your browser blocked microphone access, so HealthMitra can't hear you. Voice chat needs the microphone to work.",
    orb: 'bg-destructive text-white',
    dot: 'bg-destructive',
    pulsing: false,
  },
};

// ─── Orb icon per phase ───────────────────────────────────────────────────────
// Listening/speaking swap in a real, audio-reactive bar visualizer — driven
// by the agent's live audio track — so it's obvious at a glance who's
// talking, not just from the text label.

function OrbIcon({ phase }: { phase: UiPhase }) {
  if (phase === 'connecting') return <Loader2 className="h-8 w-8 animate-spin" />;
  if (phase === 'ended') return <PhoneOff className="h-8 w-8" />;
  if (phase === 'blocked') return <MicOff className="h-8 w-8" />;
  return <Mic className="h-8 w-8" />;
}

function Orb({ phase }: { phase: UiPhase }) {
  const { audioTrack } = useVoiceAssistant();
  const cfg = PHASE_CONFIG[phase];

  return (
    <div
      className={cn(
        'flex h-24 w-24 items-center justify-center rounded-full transition-colors duration-300',
        cfg.orb
      )}
    >
      {phase === 'listening' || phase === 'speaking' ? (
        <AgentAudioVisualizerBar
          state={phase === 'speaking' ? 'speaking' : 'listening'}
          audioTrack={audioTrack}
          size="icon"
          barCount={5}
          className="h-8"
        />
      ) : (
        <OrbIcon phase={phase} />
      )}
    </div>
  );
}

// ─── Mic mute toggle (only shown when connected) ──────────────────────────────

function MicToggle() {
  const { localParticipant, isMicrophoneEnabled } = useLocalParticipant();

  const toggle = () => {
    localParticipant.setMicrophoneEnabled(!isMicrophoneEnabled);
  };

  return (
    <button
      onClick={toggle}
      aria-label={isMicrophoneEnabled ? 'Mute microphone' : 'Unmute microphone'}
      className={cn(
        'flex h-10 w-10 items-center justify-center rounded-full border transition-colors',
        isMicrophoneEnabled
          ? 'border-border bg-secondary text-secondary-foreground hover:bg-accent'
          : 'border-destructive/40 bg-destructive/10 text-destructive hover:bg-destructive/20'
      )}
    >
      {isMicrophoneEnabled ? <Mic className="h-4 w-4" /> : <MicOff className="h-4 w-4" />}
    </button>
  );
}

// ─── Main component ───────────────────────────────────────────────────────────

export interface AgentStateViewProps {
  phase: UiPhase;
  isConnected: boolean;
  onStart: () => void;
  onEnd: () => void;
  startButtonText?: string;
  /** Live transcript of both sides of the call, shown once there's something to show. */
  messages?: ReceivedMessage[];
  agentState?: AgentSdkState;
}

export function AgentStateView({
  phase,
  isConnected,
  onStart,
  onEnd,
  startButtonText = 'Start talking',
  messages = [],
  agentState,
}: AgentStateViewProps) {
  const cfg = PHASE_CONFIG[phase];

  return (
    <div className="relative flex min-h-svh flex-col items-center px-4 pt-28 pb-16">
      {/*
        `my-auto` (not `justify-center` on the parent) is what keeps this
        safe as content grows: it centers the hero block only while there's
        spare room, and — because auto-margins can't go negative — it can
        never push the top of the headline up past the container's own
        padding-top and under the fixed header. `justify-center` alone lets
        overflowing content spill equally above and below, which is what
        was clipping text behind the header once the data/transcript panels
        made the page taller than the viewport.
      */}
      <section className="relative z-10 my-auto flex w-full max-w-md flex-col items-center gap-8 text-center">
        {/* ── Status badge ── */}
        <div className="border-border bg-secondary text-secondary-foreground inline-flex items-center gap-2 rounded-full border px-3 py-1 text-xs font-medium">
          <span className="relative flex h-2 w-2 shrink-0">
            {cfg.pulsing && (
              <span
                className={cn(
                  'absolute inline-flex h-full w-full animate-ping rounded-full opacity-60',
                  cfg.dot
                )}
              />
            )}
            <span className={cn('relative inline-flex h-2 w-2 rounded-full', cfg.dot)} />
          </span>
          {cfg.label}
        </div>

        {/* ── Orb — audio-reactive bars while listening/speaking ── */}
        <Orb phase={phase} />

        {/* ── Headline + sublabel ── */}
        <div className="flex flex-col gap-2">
          <h1 className="text-foreground text-2xl font-semibold tracking-tight sm:text-3xl">
            {cfg.headline}
          </h1>
          <p className="text-muted-foreground mx-auto max-w-xs text-sm leading-relaxed">
            {cfg.sublabel}
          </p>
        </div>

        {/* ── Blocked → how to fix instructions ── */}
        {phase === 'blocked' && (
          <div className="border-destructive/30 bg-destructive/5 w-full rounded-lg border p-4 text-left">
            <p className="text-foreground text-xs font-semibold">To enable your microphone:</p>
            <ol className="text-muted-foreground mt-2 list-inside list-decimal space-y-1 text-xs leading-relaxed">
              <li>Click the lock/site-info icon next to the address bar</li>
              <li>Find &quot;Microphone&quot; and set it to &quot;Allow&quot;</li>
              <li>Reload this page and press start again</li>
            </ol>
          </div>
        )}

        {/* ── Action area ── */}
        <AnimatePresence mode="wait">
          {/* Ready / Ended / Blocked → single Start button */}
          {(phase === 'ready' || phase === 'ended' || phase === 'blocked') && (
            <motion.div
              key="start-btn"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.2 }}
            >
              <Button
                size="lg"
                onClick={onStart}
                className="h-12 min-w-56 rounded-full px-8 text-base font-semibold"
              >
                <Mic className="mr-2 h-4 w-4" />
                {phase === 'blocked'
                  ? 'Try again'
                  : phase === 'ended'
                    ? 'Start again'
                    : startButtonText}
              </Button>
            </motion.div>
          )}

          {/* Connecting → hint text, no action */}
          {phase === 'connecting' && (
            <motion.p
              key="connecting-hint"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.2 }}
              className="text-muted-foreground text-xs"
            >
              This usually takes just a moment…
            </motion.p>
          )}

          {/* Listening / Speaking → mic toggle + end call */}
          {(phase === 'listening' || phase === 'speaking') && isConnected && (
            <motion.div
              key="session-controls"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.2 }}
              className="flex items-center gap-3"
            >
              <MicToggle />
              <button
                onClick={onEnd}
                aria-label="End call"
                className="border-destructive/40 bg-destructive/10 text-destructive hover:bg-destructive/20 flex h-10 items-center gap-2 rounded-full border px-4 text-sm font-medium transition-colors"
              >
                <PhoneOff className="h-4 w-4" />
                End call
              </button>
            </motion.div>
          )}
        </AnimatePresence>
      </section>

      {/* ── Chat transcript — what's been spoken so far ── */}
      {messages.length > 0 && (
        <div className="border-border bg-card relative z-10 mt-6 flex h-72 w-full max-w-md flex-col overflow-hidden rounded-lg border">
          <AgentChatTranscript agentState={agentState} messages={messages} className="flex-1" />
        </div>
      )}

      {/* ── Footer — a standing medical disclaimer, always visible ── */}
      <div className="relative z-10 mt-10 w-full max-w-sm px-4 text-center">
        <p className="text-muted-foreground text-xs leading-relaxed">
          This is an AI health assistant, not a substitute for a doctor. In an emergency, dial{' '}
          <span className="text-foreground font-semibold">108</span> immediately.
        </p>
      </div>
    </div>
  );
}
