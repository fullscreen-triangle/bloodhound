/** @type {import('tailwindcss').Config} */
const { fontFamily } = require("tailwindcss/defaultTheme");

module.exports = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx}",
    "./pages/**/*.{js,ts,jsx,tsx}",
    "./components/**/*.{js,ts,jsx,tsx}",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: "class",
  theme: {
    extend: {
      fontFamily: {
        mont: ["var(--font-mont)", ...fontFamily.sans],
      },
      colors: {
        dark: "#0a0a0f",
        darkSecondary: "#12121a",
        darkTertiary: "#1a1a28",
        light: "#f0f0f5",
        primary: "#2A9D8F",
        primaryDark: "#2A9D8F",
        accent: "#F4A261",
        danger: "#E63946",
        muted: "#8888aa",
        surface: "#16161f",
      },
      animation: {
        "spin-slow": "spin 8s linear infinite",
        "pulse-slow": "pulse 4s cubic-bezier(0.4, 0, 0.6, 1) infinite",
        "float": "float 6s ease-in-out infinite",
        "glow": "glow 2s ease-in-out infinite alternate",
      },
      keyframes: {
        float: {
          "0%, 100%": { transform: "translateY(0px)" },
          "50%": { transform: "translateY(-20px)" },
        },
        glow: {
          "0%": { boxShadow: "0 0 20px rgba(42, 157, 143, 0.2)" },
          "100%": { boxShadow: "0 0 40px rgba(42, 157, 143, 0.4)" },
        },
      },
      backgroundImage: {
        "grid-pattern":
          "linear-gradient(rgba(42,157,143,0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(42,157,143,0.03) 1px, transparent 1px)",
        "radial-dark":
          "radial-gradient(ellipse at center, rgba(42,157,143,0.08) 0%, transparent 70%)",
      },
      backgroundSize: {
        grid: "60px 60px",
      },
      boxShadow: {
        glow: "0 0 30px rgba(42, 157, 143, 0.15)",
        "glow-lg": "0 0 60px rgba(42, 157, 143, 0.2)",
      },
    },
    screens: {
      "2xl": { max: "1535px" },
      xl: { max: "1279px" },
      lg: { max: "1023px" },
      md: { max: "767px" },
      sm: { max: "639px" },
      xs: { max: "479px" },
    },
  },
  plugins: [
    function ({ addVariant }) {
      addVariant("child", "& > *");
      addVariant("child-hover", "& > *:hover");
    },
  ],
};
