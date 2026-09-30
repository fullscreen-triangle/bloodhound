import { COOKIE } from "@/lib/auth";

export default function handler(req, res) {
  const secure = process.env.NODE_ENV === "production" ? "; Secure" : "";
  res.setHeader("Set-Cookie", `${COOKIE}=; Path=/; HttpOnly; SameSite=Lax; Max-Age=0${secure}`);
  res.setHeader("Cache-Control", "no-store");
  res.redirect(303, "/login");
}
