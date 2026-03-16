import React, { Suspense, useRef, useEffect, useState, useMemo } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { useGLTF, OrbitControls, Environment, ContactShadows } from "@react-three/drei";
import * as THREE from "three";

function Sailfish() {
  const ref = useRef();
  const { scene } = useGLTF("/sailfish.glb");

  // Clone the scene so we don't mutate the cached original
  const clonedScene = useMemo(() => {
    const clone = scene.clone(true);

    // Copy materials properly (clone doesn't deep-clone materials)
    clone.traverse((child) => {
      if (child.isMesh) {
        child.material = child.material.clone();
      }
    });

    // Compute bounding box to auto-fit
    const box = new THREE.Box3().setFromObject(clone);
    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());

    // Center the clone
    clone.position.sub(center);

    // Scale to fit nicely in view
    const maxDim = Math.max(size.x, size.y, size.z);
    if (maxDim > 0) {
      const scale = 3 / maxDim;
      clone.scale.setScalar(scale);
    }

    return clone;
  }, [scene]);

  useFrame((state) => {
    if (ref.current) {
      // Gentle swimming animation
      ref.current.rotation.y = Math.sin(state.clock.elapsedTime * 0.3) * 0.2 + Math.PI * 0.1;
      ref.current.position.y = Math.sin(state.clock.elapsedTime * 0.5) * 0.08;
      ref.current.position.x = Math.sin(state.clock.elapsedTime * 0.2) * 0.05;
    }
  });

  return (
    <group ref={ref}>
      <primitive object={clonedScene} />
    </group>
  );
}

function LoadingFallback() {
  const ref = useRef();
  useFrame((state) => {
    if (ref.current) {
      ref.current.rotation.x = state.clock.elapsedTime * 0.5;
      ref.current.rotation.y = state.clock.elapsedTime * 0.3;
    }
  });

  return (
    <mesh ref={ref}>
      <icosahedronGeometry args={[0.8, 1]} />
      <meshStandardMaterial color="#2A9D8F" wireframe transparent opacity={0.6} />
    </mesh>
  );
}

export default function SailfishModel({ className = "" }) {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) {
    return (
      <div className={`w-full h-full flex items-center justify-center ${className}`}>
        <div className="w-16 h-16 border-2 border-primary/30 border-t-primary rounded-full animate-spin" />
      </div>
    );
  }

  return (
    <div className={`w-full h-full ${className}`}>
      <Canvas
        camera={{ position: [0, 0.5, 4], fov: 45 }}
        gl={{ antialias: true, alpha: true }}
        style={{ background: "transparent" }}
        dpr={[1, 2]}
      >
        <color attach="background" args={["#0a0a0f"]} />

        <ambientLight intensity={0.5} />
        <directionalLight position={[5, 5, 5]} intensity={1.2} color="#ffffff" />
        <pointLight position={[-3, 2, -3]} intensity={0.8} color="#2A9D8F" />
        <pointLight position={[3, -1, 3]} intensity={0.4} color="#F4A261" />
        <spotLight position={[0, 5, 0]} intensity={0.5} angle={0.5} penumbra={1} color="#2A9D8F" />

        <Suspense fallback={<LoadingFallback />}>
          <Sailfish />
          <ContactShadows
            position={[0, -1.8, 0]}
            opacity={0.3}
            scale={8}
            blur={2.5}
            color="#2A9D8F"
          />
          <Environment preset="night" />
        </Suspense>

        <OrbitControls
          enableZoom={false}
          enablePan={false}
          autoRotate
          autoRotateSpeed={0.8}
          minPolarAngle={Math.PI / 3}
          maxPolarAngle={Math.PI / 1.8}
        />

        {/* Subtle fog for depth */}
        <fog attach="fog" args={["#0a0a0f", 6, 15]} />
      </Canvas>
    </div>
  );
}
