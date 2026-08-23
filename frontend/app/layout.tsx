import type { Metadata } from 'next';
import { Public_Sans } from 'next/font/google';
import localFont from 'next/font/local';
import { headers } from 'next/headers';
import Link from 'next/link';
import { ThemeProvider } from '@/components/app/theme-provider';
import { ThemeToggle } from '@/components/app/theme-toggle';
import { cn } from '@/lib/shadcn/utils';
import { getAppConfig, getStyles } from '@/lib/utils';
import '@/styles/globals.css';

const publicSans = Public_Sans({
  variable: '--font-public-sans',
  subsets: ['latin'],
});

const commitMono = localFont({
  display: 'swap',
  variable: '--font-commit-mono',
  src: [
    {
      path: '../fonts/CommitMono-400-Regular.otf',
      weight: '400',
      style: 'normal',
    },
    {
      path: '../fonts/CommitMono-700-Regular.otf',
      weight: '700',
      style: 'normal',
    },
    {
      path: '../fonts/CommitMono-400-Italic.otf',
      weight: '400',
      style: 'italic',
    },
    {
      path: '../fonts/CommitMono-700-Italic.otf',
      weight: '700',
      style: 'italic',
    },
  ],
});

interface RootLayoutProps {
  children: React.ReactNode;
}

// Only metadataBase — title/description are set dynamically per-request in
// the JSX below (they depend on appConfig, which is resolved from request
// headers for sandbox/multi-tenant support). metadataBase itself is static,
// and is needed so opengraph-image resolves to an absolute URL in
// production instead of defaulting to localhost.
export const metadata: Metadata = {
  metadataBase: new URL(
    process.env.VERCEL_URL ? `https://${process.env.VERCEL_URL}` : 'http://localhost:3000'
  ),
};

export default async function RootLayout({ children }: RootLayoutProps) {
  const hdrs = await headers();
  const appConfig = await getAppConfig(hdrs);
  const styles = getStyles(appConfig);
  const { pageTitle, pageDescription, companyName, logo, logoDark } = appConfig;

  return (
    <html
      lang="en"
      suppressHydrationWarning
      className={cn(
        publicSans.variable,
        commitMono.variable,
        'scroll-smooth font-sans antialiased'
      )}
    >
      <head>
        {styles && <style>{styles}</style>}
        <title>{pageTitle}</title>
        <meta name="description" content={pageDescription} />
      </head>
      <body className="overflow-x-hidden">
        <ThemeProvider
          attribute="class"
          defaultTheme="system"
          enableSystem
          disableTransitionOnChange
        >
          <header className="border-border/50 bg-background/80 fixed top-0 left-0 z-50 flex w-full flex-row items-center justify-between border-b p-6 backdrop-blur-md">
            <Link href="/" className="flex items-center gap-2.5">
              {/* Full-color illustrated mark — same file in light/dark, see app-config.ts */}
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={logo ?? logoDark} alt={`${companyName} Logo`} className="h-10 w-auto" />
              {/* Bilingual wordmark — Roman + Devanagari, both part of the logo lockup */}
              <span className="text-foreground flex items-baseline gap-1.5 text-sm leading-none font-semibold">
                {companyName}
                <span className="text-muted-foreground text-xs font-normal">हेल्थमित्र</span>
              </span>
            </Link>

            <span className="text-muted-foreground font-mono text-xs font-medium tracking-wide">
              Voice by{' '}
              <a
                target="_blank"
                rel="noopener noreferrer"
                href="https://murf.ai"
                className="text-foreground underline underline-offset-4"
              >
                Murf Falcon
              </a>
            </span>
          </header>

          {children}
          <div className="fixed right-6 bottom-5 z-50">
            <ThemeToggle />
          </div>
        </ThemeProvider>
      </body>
    </html>
  );
}
