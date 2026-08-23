'use client';

import { useEffect, useRef, useState } from 'react';
import { MediaDeviceFailure, RoomEvent } from 'livekit-client';
import { useAgent, useSessionContext, useSessionMessages } from '@livekit/components-react';
import type { AppConfig } from '@/app-config';
import { AgentStateView, deriveUiPhase } from '@/components/app/agent-state-view';

interface ViewControllerProps {
  appConfig: AppConfig;
}

export function ViewController({ appConfig }: ViewControllerProps) {
  const session = useSessionContext();
  const { isConnected, start, end, room } = session;
  const { state: agentState } = useAgent();
  const { messages } = useSessionMessages(session);

  // Track whether we were ever connected so we can show the "ended" state
  const wasConnected = useRef(false);
  const [showEnded, setShowEnded] = useState(false);
  const [micBlocked, setMicBlocked] = useState(false);

  useEffect(() => {
    if (isConnected) {
      wasConnected.current = true;
      setShowEnded(false);
    } else if (wasConnected.current) {
      setShowEnded(true);
    }
  }, [isConnected]);

  // Detect when the browser blocks microphone access so we can explain why
  // the call can't start instead of failing silently.
  useEffect(() => {
    const onMediaDevicesError = (error: Error, kind?: MediaDeviceKind) => {
      if (kind && kind !== 'audioinput') return;
      if (MediaDeviceFailure.getFailure(error) !== MediaDeviceFailure.PermissionDenied) return;

      setMicBlocked(true);
      end();
    };

    room.on(RoomEvent.MediaDevicesError, onMediaDevicesError);
    return () => {
      room.off(RoomEvent.MediaDevicesError, onMediaDevicesError);
    };
  }, [room, end]);

  const phase = micBlocked
    ? 'blocked'
    : deriveUiPhase(isConnected, agentState as Parameters<typeof deriveUiPhase>[1], showEnded);

  // Temporary console-only diagnostic for the transcript-not-appearing
  // report — not a UI element, just visible in DevTools. Safe to remove
  // once that's confirmed fixed.
  useEffect(() => {
    console.log('[HealthMitra debug] messages:', messages.length, 'phase:', phase);
  }, [messages, phase]);

  const handleStart = () => {
    setShowEnded(false);
    setMicBlocked(false);
    wasConnected.current = false;
    start();
  };

  return (
    <AgentStateView
      phase={phase}
      isConnected={isConnected}
      onStart={handleStart}
      onEnd={end}
      startButtonText={appConfig.startButtonText}
      messages={messages}
      agentState={agentState}
    />
  );
}
