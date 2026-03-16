// Redirect to phase-space (about page replaced by pillar pages)
import { useEffect } from "react";
import { useRouter } from "next/router";

export default function About() {
  const router = useRouter();
  useEffect(() => { router.replace("/phase-space"); }, [router]);
  return null;
}
