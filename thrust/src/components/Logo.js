import { motion } from "framer-motion";
import Link from "next/link";
import Image from "next/image";

let MotionLink = motion(Link);

const Logo = () => {
  return (
    <div className="flex flex-col items-center justify-center mt-2">
      <MotionLink
        href="/"
        className="flex items-center justify-center w-12 h-12"
        whileHover={{ scale: 1.1 }}
        whileTap={{ scale: 0.95 }}
      >
        <Image
          src="/logo.png"
          alt="Bloodhound"
          width={48}
          height={48}
          className="rounded-full"
        />
      </MotionLink>
    </div>
  );
};

export default Logo;
