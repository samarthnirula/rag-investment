import { initializeApp, getApps, FirebaseApp } from "firebase/app";
import {
  getAuth,
  signInWithEmailAndPassword,
  createUserWithEmailAndPassword,
  GoogleAuthProvider,
  signInWithPopup,
  signOut as firebaseSignOut,
  onAuthStateChanged,
  browserLocalPersistence,
  setPersistence,
} from "firebase/auth";
import type { User, UserCredential } from "firebase/auth";

const firebaseConfig = {
  apiKey:            process.env.NEXT_PUBLIC_FIREBASE_API_KEY,
  authDomain:        process.env.NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN,
  projectId:         process.env.NEXT_PUBLIC_FIREBASE_PROJECT_ID,
  storageBucket:     process.env.NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET,
  messagingSenderId: process.env.NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID,
  appId:             process.env.NEXT_PUBLIC_FIREBASE_APP_ID,
};

const hasConfig = !!process.env.NEXT_PUBLIC_FIREBASE_API_KEY;

let app: FirebaseApp;
if (hasConfig) {
  app = getApps().length ? getApps()[0] : initializeApp(firebaseConfig);
}

export const auth = hasConfig && getApps().length ? getAuth(getApps()[0]) : null;
export const googleProvider = hasConfig ? new GoogleAuthProvider() : null;

// Explicitly persist auth state in localStorage so users stay logged in across
// page refreshes and browser restarts. This call is idempotent and safe to
// repeat on hot-reload.
if (auth) {
  setPersistence(auth, browserLocalPersistence).catch(() => {});
}

export async function signInWithGoogle(): Promise<UserCredential> {
  if (!auth || !googleProvider) throw new Error("Firebase not configured");
  // signInWithPopup keeps the user on the current page (no full-page redirect).
  // The caller is responsible for navigating after the returned promise resolves.
  return signInWithPopup(auth, googleProvider);
}

export async function signOut(): Promise<void> {
  if (!auth) return;
  await firebaseSignOut(auth);
}

export { onAuthStateChanged, signInWithEmailAndPassword, createUserWithEmailAndPassword, User };
