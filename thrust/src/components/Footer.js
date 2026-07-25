import Link from "next/link";
import React from "react";
import Layout from "./Layout";

const Footer = () => {
  return (
    <footer className="w-full border-t border-solid border-primary/10 text-muted text-sm">
      <Layout className="!py-8">
        <div className="flex items-start justify-between lg:flex-col lg:gap-8">
          <div>
            <div className="font-bold text-light mb-3">Bloodhound Framework</div>
            <div className="text-xs text-muted max-w-xs leading-relaxed">
              A distributed virtual machine for automated deep research through trajectory completion in bounded phase space.
            </div>
          </div>

          <div className="flex gap-12 md:gap-8 sm:flex-col sm:gap-6">
            <div>
              <div className="text-xs font-mono text-muted uppercase tracking-widest mb-3">Framework</div>
              <nav className="flex flex-col gap-2">
                <Link href="/about" className="hover:text-primary transition-colors">About</Link>
                <Link href="/architecture" className="hover:text-primary transition-colors">Architecture</Link>
                <Link href="/use-cases" className="hover:text-primary transition-colors">Use Cases</Link>
                <Link href="/docs" className="hover:text-primary transition-colors">Docs</Link>
              </nav>
            </div>
            <div>
              <div className="text-xs font-mono text-muted uppercase tracking-widest mb-3">Pillars</div>
              <nav className="flex flex-col gap-2">
                <Link href="/phase-space" className="hover:text-primary transition-colors">Phase Space</Link>
                <Link href="/compilation" className="hover:text-primary transition-colors">Compilation</Link>
                <Link href="/federated" className="hover:text-primary transition-colors">Federated</Link>
                <Link href="/pipeline" className="hover:text-primary transition-colors">Pipeline</Link>
                <Link href="/validation" className="hover:text-primary transition-colors">Validation</Link>
              </nav>
            </div>
            <div>
              <div className="text-xs font-mono text-muted uppercase tracking-widest mb-3">Tools</div>
              <nav className="flex flex-col gap-2">
                <Link href="/repo-lens" className="text-primary hover:text-primary/80 transition-colors">Repo Lens</Link>
                <Link href="/cytochrome" className="text-primary hover:text-primary/80 transition-colors">CYP450 Engine</Link>
              </nav>
            </div>
            <div>
              <div className="text-xs font-mono text-muted uppercase tracking-widest mb-3">Connect</div>
              <nav className="flex flex-col gap-2">
                <Link href="/collaborate" className="hover:text-primary transition-colors">Collaborate</Link>
                <Link href="/roadmap" className="hover:text-primary transition-colors">Roadmap</Link>
                <Link href="mailto:contact@bloodhound.dev" className="text-primary hover:text-primary/80 transition-colors">
                  contact@bloodhound.dev
                </Link>
              </nav>
            </div>
          </div>
        </div>

        <div className="mt-8 pt-6 border-t border-primary/5 text-xs text-muted text-center">
          {new Date().getFullYear()} Bloodhound Framework. MIT License. All rights reserved.
        </div>
      </Layout>
    </footer>
  );
};

export default Footer;
