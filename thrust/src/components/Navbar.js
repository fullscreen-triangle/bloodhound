import Link from "next/link";
import React, { useState, useRef, useEffect } from "react";
import Logo from "./Logo";
import { useRouter } from "next/router";
import { motion, AnimatePresence } from "framer-motion";

const NavLink = ({ href, title, className = "" }) => {
  const router = useRouter();
  const isActive = router.asPath === href;

  return (
    <Link
      href={href}
      className={`${className} relative text-sm font-medium tracking-wide uppercase
        ${isActive ? "text-primary" : "text-muted hover:text-light"}
        transition-colors duration-300`}
    >
      {title}
      {isActive && (
        <motion.span
          layoutId="nav-indicator"
          className="absolute -bottom-1 left-0 w-full h-[2px] bg-primary"
        />
      )}
    </Link>
  );
};

const DropdownMenu = ({ label, items, isFrameworkActive }) => {
  const [open, setOpen] = useState(false);
  const ref = useRef(null);

  useEffect(() => {
    const handleClick = (e) => {
      if (ref.current && !ref.current.contains(e.target)) setOpen(false);
    };
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, []);

  return (
    <div className="relative" ref={ref}>
      <button
        className={`text-sm font-medium tracking-wide uppercase transition-colors duration-300
          ${isFrameworkActive ? "text-primary" : "text-muted hover:text-light"}`}
        onClick={() => setOpen(!open)}
      >
        {label}
        <span className="ml-1 text-xs">&#9662;</span>
      </button>
      <AnimatePresence>
        {open && (
          <motion.div
            className="absolute top-full left-0 mt-3 w-48 bg-darkSecondary/95 backdrop-blur-xl
              border border-primary/10 rounded-xl py-2 z-50 shadow-glow-lg"
            initial={{ opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.15 }}
          >
            {items.map((item) => (
              <Link
                key={item.href}
                href={item.href}
                className="block px-4 py-2.5 text-sm text-muted hover:text-light hover:bg-primary/5 transition-colors"
                onClick={() => setOpen(false)}
              >
                {item.title}
              </Link>
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

const MobileNavLink = ({ href, title, toggle }) => {
  const router = useRouter();
  const isActive = router.asPath === href;

  const handleClick = () => {
    toggle();
    router.push(href);
  };

  return (
    <button
      className={`text-lg font-medium tracking-wide
        ${isActive ? "text-primary" : "text-light/70 hover:text-light"}
        transition-colors duration-300`}
      onClick={handleClick}
    >
      {title}
    </button>
  );
};

const MobileSection = ({ title, items, toggle }) => (
  <div className="text-center">
    <div className="text-xs font-mono text-muted uppercase tracking-widest mb-2">{title}</div>
    <div className="flex flex-col gap-2">
      {items.map((item) => (
        <MobileNavLink key={item.href} toggle={toggle} href={item.href} title={item.title} />
      ))}
    </div>
  </div>
);

const frameworkItems = [
  { href: "/phase-space", title: "Phase Space" },
  { href: "/compilation", title: "Compilation" },
  { href: "/federated", title: "Federated" },
  { href: "/pipeline", title: "Pipeline" },
  { href: "/validation", title: "Validation" },
];

const frameworkPaths = frameworkItems.map((i) => i.href);

const Navbar = () => {
  const [isOpen, setIsOpen] = useState(false);
  const toggle = () => setIsOpen(!isOpen);
  const router = useRouter();
  const isFrameworkActive = frameworkPaths.includes(router.asPath);

  return (
    <header
      className="w-full flex items-center justify-between px-32 py-6 font-medium z-50
      lg:px-16 md:px-12 sm:px-8 relative bg-dark/80 backdrop-blur-md border-b border-primary/5"
    >
      <button
        type="button"
        className="flex-col items-center justify-center hidden lg:flex z-50"
        aria-controls="mobile-menu"
        aria-expanded={isOpen}
        onClick={toggle}
      >
        <span className="sr-only">Open main menu</span>
        <span className={`bg-light block h-0.5 w-6 rounded-sm transition-all duration-300 ease-out ${isOpen ? "rotate-45 translate-y-1" : "-translate-y-0.5"}`} />
        <span className={`bg-light block h-0.5 w-6 rounded-sm transition-all duration-300 ease-out ${isOpen ? "opacity-0" : "opacity-100"} my-0.5`} />
        <span className={`bg-light block h-0.5 w-6 rounded-sm transition-all duration-300 ease-out ${isOpen ? "-rotate-45 -translate-y-1" : "translate-y-0.5"}`} />
      </button>

      <div className="w-full flex justify-between items-center lg:hidden">
        <nav className="flex items-center gap-8">
          <NavLink href="/" title="Home" />
          <NavLink href="/about" title="About" />
          <NavLink href="/architecture" title="Architecture" />
          <DropdownMenu
            label="Pillars"
            isFrameworkActive={isFrameworkActive}
            items={frameworkItems}
          />
          <NavLink href="/use-cases" title="Use Cases" />
          <NavLink href="/docs" title="Docs" />
          <NavLink href="/roadmap" title="Roadmap" />
          <NavLink href="/collaborate" title="Collaborate" />
          <NavLink href="/repo-lens" title="Repo Lens" className="text-primary" />
          <NavLink href="/cytochrome" title="Demo" className="text-primary" />
        </nav>
      </div>

      <AnimatePresence>
        {isOpen && (
          <motion.div
            className="min-w-[70vw] sm:min-w-[90vw] flex flex-col items-center gap-6
              fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2
              py-12 bg-darkSecondary/95 rounded-2xl z-40 backdrop-blur-xl
              border border-primary/10"
            initial={{ scale: 0, x: "-50%", y: "-50%", opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0, opacity: 0 }}
          >
            <MobileNavLink toggle={toggle} href="/" title="Home" />
            <MobileNavLink toggle={toggle} href="/about" title="About" />
            <MobileNavLink toggle={toggle} href="/architecture" title="Architecture" />

            <MobileSection title="Framework Pillars" toggle={toggle} items={frameworkItems} />

            <MobileNavLink toggle={toggle} href="/use-cases" title="Use Cases" />
            <MobileNavLink toggle={toggle} href="/docs" title="Docs" />
            <MobileNavLink toggle={toggle} href="/roadmap" title="Roadmap" />
            <MobileNavLink toggle={toggle} href="/collaborate" title="Collaborate" />
            <MobileNavLink toggle={toggle} href="/repo-lens" title="Repo Lens" />
            <MobileNavLink toggle={toggle} href="/cytochrome" title="Demo" />
          </motion.div>
        )}
      </AnimatePresence>

      <div className="absolute left-[50%] top-2 translate-x-[-50%]">
        <Logo />
      </div>
    </header>
  );
};

export default Navbar;
