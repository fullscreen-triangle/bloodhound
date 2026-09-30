import { NextResponse } from "next/server";
import { COOKIE, checkSession } from "@/lib/auth";

/**
 * Nothing on the site is public: every page, API route, data file and page bundle
 * needs a signed session cookie. Only what the login page itself needs is open —
 * the login page and its API, the framework's shared code, CSS and fonts.
 *
 * Without AUTH_SECRET and SITE_PASSWORD the site is closed in production and open
 * in local development.
 */

const OPEN = [
  /^\/login$/,
  /^\/api\/auth\/(login|logout)$/,
  /^\/favicon\.ico$/,
  /^\/_next\/static\/(css|media)\//,
  // shared framework and library chunks; each page's own bundle stays closed
  /^\/_next\/static\/chunks\/(?!pages\/)/,
  /^\/_next\/static\/chunks\/pages\/(_app|_error|login)-[^/]+\.js$/,
  /^\/_next\/static\/[^/]+\/_(buildManifest|ssgManifest)\.js$/,
  /^\/_next\/webpack-hmr/,
];

export async function middleware(req) {
  const { pathname, search } = req.nextUrl;
  const secret = process.env.AUTH_SECRET;
  const password = process.env.SITE_PASSWORD;

  if (!secret || !password) {
    if (process.env.NODE_ENV === "development") return NextResponse.next();
    return new NextResponse("This site is locked: AUTH_SECRET and SITE_PASSWORD are not configured.", { status: 503 });
  }
  if (OPEN.some((re) => re.test(pathname))) return NextResponse.next();
  if (await checkSession(secret, req.cookies.get(COOKIE)?.value)) return NextResponse.next();

  if (pathname.startsWith("/api/") || pathname.startsWith("/_next/")) {
    return new NextResponse(JSON.stringify({ error: "login required" }), {
      status: 401,
      headers: { "content-type": "application/json" },
    });
  }
  const url = req.nextUrl.clone();
  url.pathname = "/login";
  url.search = `?next=${encodeURIComponent(pathname + search)}`;
  return NextResponse.redirect(url);
}

export const config = { matcher: "/:path*" };
