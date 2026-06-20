"use client";

import { useRef } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { Float, MeshDistortMaterial } from "@react-three/drei";
import * as THREE from "three";

function Scales({ mouse }: { mouse: React.RefObject<{ x: number; y: number } | null> }) {
  const groupRef = useRef<THREE.Group>(null);

  useFrame((_state, delta) => {
    if (!groupRef.current) return;
    groupRef.current.rotation.y += delta * 0.3;
    if (mouse.current) {
      groupRef.current.rotation.x = THREE.MathUtils.lerp(
        groupRef.current.rotation.x,
        mouse.current.y * 0.15,
        0.05
      );
      groupRef.current.rotation.z = THREE.MathUtils.lerp(
        groupRef.current.rotation.z,
        mouse.current.x * 0.1,
        0.05
      );
    }
  });

  const goldColor = "#C9A84C";

  return (
    <Float speed={2} rotationIntensity={0.3} floatIntensity={0.5}>
      <group ref={groupRef}>
        {/* Central pillar */}
        <mesh position={[0, 0, 0]}>
          <cylinderGeometry args={[0.05, 0.05, 2.5, 16]} />
          <meshStandardMaterial color={goldColor} metalness={0.8} roughness={0.2} />
        </mesh>

        {/* Crossbar */}
        <mesh position={[0, 1.2, 0]}>
          <boxGeometry args={[2.4, 0.06, 0.06]} />
          <meshStandardMaterial color={goldColor} metalness={0.8} roughness={0.2} />
        </mesh>

        {/* Left pan */}
        <mesh position={[-1.1, 0.5, 0]} rotation={[Math.PI / 2, 0, 0]}>
          <cylinderGeometry args={[0.4, 0.4, 0.04, 24]} />
          <MeshDistortMaterial color={goldColor} metalness={0.7} roughness={0.3} distort={0.05} speed={2} />
        </mesh>
        {/* Left chains */}
        {[-0.25, 0, 0.25].map((offset, i) => (
          <mesh key={`lc-${i}`} position={[-1.1 + offset * 0.8, 0.85, 0]}>
            <cylinderGeometry args={[0.012, 0.012, 0.7, 6]} />
            <meshStandardMaterial color={goldColor} metalness={0.9} roughness={0.1} />
          </mesh>
        ))}

        {/* Right pan */}
        <mesh position={[1.1, 0.7, 0]} rotation={[Math.PI / 2, 0, 0]}>
          <cylinderGeometry args={[0.4, 0.4, 0.04, 24]} />
          <MeshDistortMaterial color={goldColor} metalness={0.7} roughness={0.3} distort={0.05} speed={2} />
        </mesh>
        {/* Right chains */}
        {[-0.25, 0, 0.25].map((offset, i) => (
          <mesh key={`rc-${i}`} position={[1.1 + offset * 0.8, 0.95, 0]}>
            <cylinderGeometry args={[0.012, 0.012, 0.5, 6]} />
            <meshStandardMaterial color={goldColor} metalness={0.9} roughness={0.1} />
          </mesh>
        ))}

        {/* Base */}
        <mesh position={[0, -1.25, 0]}>
          <cylinderGeometry args={[0.6, 0.7, 0.15, 24]} />
          <meshStandardMaterial color={goldColor} metalness={0.8} roughness={0.2} />
        </mesh>
      </group>
    </Float>
  );
}

function Scene() {
  const mouse = useRef({ x: 0, y: 0 });
  const { viewport } = useThree();

  return (
    <group
      onPointerMove={(e) => {
        mouse.current = {
          x: (e.point.x / viewport.width) * 2,
          y: (e.point.y / viewport.height) * 2,
        };
      }}
    >
      <ambientLight intensity={0.4} />
      <directionalLight position={[5, 5, 5]} intensity={1} />
      <directionalLight position={[-3, 3, -3]} intensity={0.5} color="#C9A84C" />
      <pointLight position={[0, 3, 0]} intensity={0.6} color="#C9A84C" />
      <Scales mouse={mouse} />
    </group>
  );
}

export function ScalesOfJustice() {
  return (
    <div className="w-full h-full">
      <Canvas
        camera={{ position: [0, 0.5, 5], fov: 35 }}
        style={{ background: "transparent" }}
        gl={{ alpha: true, antialias: true }}
      >
        <Scene />
      </Canvas>
    </div>
  );
}
