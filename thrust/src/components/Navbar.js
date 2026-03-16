import Link from "next/link";
import React, { useState } from "react";
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

const Navbar = () => {
  const [isOpen, setIsOpen] = useState(false);
  const toggle = () => setIsOpen(!isOpen);

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
          <NavLink href="/phase-space" title="Phase Space" />
          <NavLink href="/compilation" title="Compilation" />
          <NavLink href="/federated" title="Federated" />
          <NavLink href="/pipeline" title="Pipeline" />
          <NavLink href="/validation" title="Validation" />
          <NavLink href="/collaborate" title="Collaborate" />
        </nav>
      </div>

      <AnimatePresence>
        {isOpen && (
          <motion.div
            className="min-w-[70vw] sm:min-w-[90vw] flex flex-col items-center gap-6
              fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2
              py-16 bg-darkSecondary/95 rounded-2xl z-40 backdrop-blur-xl
              border border-primary/10"
            initial={{ scale: 0, x: "-50%", y: "-50%", opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0, opacity: 0 }}
          >
            <MobileNavLink toggle={toggle} href="/" title="Home" />
            <MobileNavLink toggle={toggle} href="/phase-space" title="Phase Space" />
            <MobileNavLink toggle={toggle} href="/compilation" title="Compilation" />
            <MobileNavLink toggle={toggle} href="/federated" title="Federated" />
            <MobileNavLink toggle={toggle} href="/pipeline" title="Pipeline" />
            <MobileNavLink toggle={toggle} href="/validation" title="Validation" />
            <MobileNavLink toggle={toggle} href="/collaborate" title="Collaborate" />
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
