import Link from "next/link";
import React from "react";
import Layout from "./Layout";

const Footer = () => {
  return (
    <footer className="w-full border-t border-solid border-primary/10 text-muted text-sm">
      <Layout className="!py-8 flex items-center justify-between lg:flex-col lg:py-6 lg:gap-4">
        <span>{new Date().getFullYear()} Bloodhound Framework. All rights reserved.</span>

        <nav className="flex items-center gap-6">
          <Link href="/phase-space" className="hover:text-primary transition-colors">
            Phase Space
          </Link>
          <Link href="/compilation" className="hover:text-primary transition-colors">
            Compilation
          </Link>
          <Link href="/federated" className="hover:text-primary transition-colors">
            Federated
          </Link>
          <Link href="/collaborate" className="hover:text-primary transition-colors">
            Collaborate
          </Link>
        </nav>

        <Link
          href="mailto:contact@bloodhound.dev"
          className="text-primary hover:text-primary/80 transition-colors"
        >
          contact@bloodhound.dev
        </Link>
      </Layout>
    </footer>
  );
};

export default Footer;
