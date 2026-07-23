import Footer from "@/components/Footer";
import Navbar from "@/components/Navbar";
import "@/styles/globals.css";
import { AnimatePresence } from "framer-motion";
import { Montserrat } from "next/font/google";
import Head from "next/head";
import { useRouter } from "next/router";
import { useEffect } from "react";

const montserrat = Montserrat({ subsets: ["latin"], variable: "--font-mont" });

// Pages that manage their own chrome (no global Navbar/Footer)
const BLANK_SCREEN_ROUTES = ["/", "/cytochrome"];

export default function App({ Component, pageProps }) {
  const router = useRouter();
  const isBlankScreen = BLANK_SCREEN_ROUTES.includes(router.pathname);

  // Force dark mode always
  useEffect(() => {
    document.documentElement.classList.add("dark");
  }, []);

  return (
    <>
      <Head>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <main
        className={`${montserrat.variable} font-mont bg-dark w-full min-h-screen h-full`}
      >
        {!isBlankScreen && <Navbar />}
        <AnimatePresence initial={false} mode="wait">
          <Component key={router.asPath} {...pageProps} />
        </AnimatePresence>
        {!isBlankScreen && <Footer />}
      </main>
    </>
  );
}
