import { COOKIE, SESSION_DAYS, makeSession, passwordMatches } from "@/lib/auth";

export default async function handler(req, res) {
  if (req.method !== "POST") {
    res.setHeader("Allow", "POST");
    return res.status(405).json({ error: "POST only" });
  }
  const secret = process.env.AUTH_SECRET;
  const expected = process.env.SITE_PASSWORD;
  if (!secret || !expected) return res.status(503).json({ error: "login is not configured on this deployment" });

  const password = req.body && typeof req.body === "object" ? req.body.password : undefined;
  if (!(await passwordMatches(password, expected))) {
    // A pause per wrong guess makes guessing slow.
    await new Promise((r) => setTimeout(r, 900));
    return res.status(401).json({ error: "wrong password" });
  }
  const secure = process.env.NODE_ENV === "production" ? "; Secure" : "";
  res.setHeader(
    "Set-Cookie",
    `${COOKIE}=${await makeSession(secret)}; Path=/; HttpOnly; SameSite=Lax; Max-Age=${SESSION_DAYS * 86400}${secure}`
  );
  res.setHeader("Cache-Control", "no-store");
  return res.status(200).json({ ok: true });
}
