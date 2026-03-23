import { useEffect } from "react";
import { useRouter } from "next/router";

export default function Articles() {
  const router = useRouter();
  useEffect(() => { router.replace("/use-cases"); }, [router]);
  return null;
}
