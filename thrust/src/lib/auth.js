/**
 * The site's login: one password (env SITE_PASSWORD) and a session cookie signed
 * with HMAC-SHA256 under env AUTH_SECRET. Web Crypto only, so the same code runs in
 * the edge middleware and in API routes. Rotating AUTH_SECRET logs everyone out.
 */

export const COOKIE = "bh_session";
export const SESSION_DAYS = 30;

const enc = new TextEncoder();

function hex(buf) {
  return Array.from(new Uint8Array(buf), (b) => b.toString(16).padStart(2, "0")).join("");
}

function sameString(a, b) {
  if (typeof a !== "string" || typeof b !== "string" || a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i++) diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return diff === 0;
}

async function hmac(secret, message) {
  const key = await crypto.subtle.importKey("raw", enc.encode(secret), { name: "HMAC", hash: "SHA-256" }, false, ["sign"]);
  return hex(await crypto.subtle.sign("HMAC", key, enc.encode(message)));
}

/** `<expiry>.<signature>`; the expiry is signed, so it cannot be extended. */
export async function makeSession(secret, days = SESSION_DAYS) {
  const exp = Math.floor(Date.now() / 1000) + days * 86400;
  return `${exp}.${await hmac(secret, `session:${exp}`)}`;
}

export async function checkSession(secret, value) {
  if (!secret || !value) return false;
  const [exp, sig] = value.split(".");
  if (!/^\d+$/.test(exp || "") || Number(exp) < Date.now() / 1000) return false;
  return sameString(sig, await hmac(secret, `session:${exp}`));
}

/** Compare a typed password with the configured one without leaking where they differ. */
export async function passwordMatches(given, expected) {
  if (typeof given !== "string" || !expected) return false;
  const digest = async (s) => hex(await crypto.subtle.digest("SHA-256", enc.encode(s)));
  return sameString(await digest(given), await digest(expected));
}

/** Only same-site paths may be returned to after login. */
export function safeNext(next) {
  return typeof next === "string" && next.startsWith("/") && !next.startsWith("//") && !next.startsWith("/\\") ? next : "/";
}
