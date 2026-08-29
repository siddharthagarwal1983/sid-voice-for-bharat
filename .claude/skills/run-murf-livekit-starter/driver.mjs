// Drives the HealthMitra frontend (http://localhost:3000) in headless
// Chromium: loads the landing page, starts a call, waits for the agent
// to join and greet, screenshots, then ends the call cleanly.
//
// Usage:
//   node driver.mjs                 # full call smoke test
//   node driver.mjs --landing-only  # just load the page and screenshot
//
// Requires: `npm install` in this directory first (installs playwright +
// downloads its bundled Chromium), and the frontend dev server already
// running on :3000 (see SKILL.md "Run (agent path)").
//
// Screenshots land next to this script, in ./screenshots/.

import { chromium } from "playwright";
import { mkdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const HERE = dirname(fileURLToPath(import.meta.url));
const SHOTS = join(HERE, "screenshots");
mkdirSync(SHOTS, { recursive: true });

const landingOnly = process.argv.includes("--landing-only");

function shot(page, name) {
  return page.screenshot({ path: join(SHOTS, name), fullPage: true });
}

(async () => {
  const browser = await chromium.launch({
    args: [
      "--no-sandbox",
      "--use-fake-ui-for-media-stream", // auto-grant mic permission
      "--use-fake-device-for-media-stream", // synthetic mic input, no real audio needed
    ],
  });
  const page = await browser.newPage();
  const consoleErrors = [];
  page.on("console", (msg) => msg.type() === "error" && consoleErrors.push(msg.text()));
  page.on("pageerror", (err) => consoleErrors.push(String(err)));

  await page.goto("http://localhost:3000", { waitUntil: "networkidle", timeout: 30000 });
  await page.waitForSelector('button:has-text("Start talking")', { timeout: 15000 });
  await shot(page, "1-landing.png");
  console.log("[driver] landing page loaded");

  if (!landingOnly) {
    await page.locator('button:has-text("Start talking")').click();
    // "Connecting" is often too transient to catch - don't fail if it's
    // already past that state by the time we look.
    await page
      .waitForSelector("text=Connecting", { timeout: 3000 })
      .catch(() => {});
    await shot(page, "2-connecting.png");

    // The agent joins the LiveKit room and speaks a greeting; "Listening"
    // appears once its opening turn finishes and the mic is live.
    await page.waitForSelector("text=Listening", { timeout: 20000 });
    await page.waitForTimeout(1000); // let the greeting bubble render
    await shot(page, "3-connected.png");

    const greeting = await page.locator("main").innerText();
    console.log("[driver] agent joined and greeted. Visible text snippet:");
    console.log(greeting.slice(0, 300).replace(/\s+/g, " "));

    await page.locator('button:has-text("End call")').click();
    await page.waitForTimeout(1000);
    await shot(page, "4-ended.png");
    console.log("[driver] call ended cleanly");
  }

  if (consoleErrors.length) {
    console.log("[driver] CONSOLE ERRORS:", JSON.stringify(consoleErrors, null, 2));
  } else {
    console.log("[driver] no browser console errors");
  }

  await browser.close();
  process.exit(consoleErrors.length ? 1 : 0);
})().catch((err) => {
  console.error("[driver] FAILED:", err);
  process.exit(1);
});
