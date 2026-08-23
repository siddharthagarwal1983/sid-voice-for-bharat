import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  eslint: {
    // These warnings come from upstream LiveKit/AI UI components, not our code.
    ignoreDuringBuilds: true,
  },
  // Hides the "N" dev-tools badge Next.js overlays bottom-left in dev mode.
  devIndicators: false,
};

export default nextConfig;
