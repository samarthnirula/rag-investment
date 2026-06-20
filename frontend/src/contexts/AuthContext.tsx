"use client";

import {
  createContext,
  useContext,
  useEffect,
  useState,
  ReactNode,
} from "react";
import type { User } from "firebase/auth";
import { onAuthStateChanged, auth } from "@/lib/firebase";
import { registerUser, getSubscriptionStatus, type SubscriptionStatus } from "@/lib/api";

interface AuthContextType {
  user: User | null;
  idToken: string | null;
  plan: string;
  loading: boolean;
  subscriptionStatus: SubscriptionStatus | null;
  isTrialExpired: boolean;
  daysRemaining: number;
  hoursRemaining: number;
  refreshSubscription: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType>({
  user: null,
  idToken: null,
  plan: "trial",
  loading: true,
  subscriptionStatus: null,
  isTrialExpired: false,
  daysRemaining: 4,
  hoursRemaining: 0,
  refreshSubscription: async () => {},
});

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser]       = useState<User | null>(null);
  const [idToken, setIdToken] = useState<string | null>(null);
  const [plan, setPlan]       = useState<string>("trial");
  const [loading, setLoading] = useState(true);
  const [subscriptionStatus, setSubscriptionStatus] = useState<SubscriptionStatus | null>(null);

  const isTrialExpired = subscriptionStatus?.is_trial_expired ?? false;
  const daysRemaining  = subscriptionStatus?.days_remaining  ?? 4;
  const hoursRemaining = subscriptionStatus?.hours_remaining ?? 0;

  async function refreshSubscription() {
    try {
      const status = await getSubscriptionStatus();
      setSubscriptionStatus(status);
    } catch {
      // Silently ignore — subscription status is non-critical
    }
  }

  useEffect(() => {
    if (!auth) {
      setLoading(false);
      return;
    }

    const unsubscribe = onAuthStateChanged(auth, async (u) => {
      setUser(u);

      if (u) {
        const token = await u.getIdToken(false);
        setIdToken(token);

        try {
          const result = await registerUser();
          setPlan(result.plan);

          if (result.is_new) {
            await u.getIdToken(true);
          }
        } catch {
          // Backend not yet ready or network error — silently ignore.
        }

        // Fetch subscription status after registration is confirmed
        try {
          const status = await getSubscriptionStatus();
          setSubscriptionStatus(status);
        } catch {
          // Non-critical — leave as null
        }
      } else {
        setIdToken(null);
        setPlan("trial");
        setSubscriptionStatus(null);
      }

      setLoading(false);
    });

    return unsubscribe;
  }, []);

  return (
    <AuthContext.Provider
      value={{
        user,
        idToken,
        plan,
        loading,
        subscriptionStatus,
        isTrialExpired,
        daysRemaining,
        hoursRemaining,
        refreshSubscription,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  return useContext(AuthContext);
}
