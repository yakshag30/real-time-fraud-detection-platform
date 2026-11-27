const sameOrigin =
  typeof window !== "undefined" && window.location.port === "8000";

export const API_BASE: string =
  (import.meta.env.VITE_API_URL as string | undefined) ??
  (sameOrigin ? "" : "http://127.0.0.1:8000");

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail ?? body);
    } catch {
      /* non-JSON error body */
    }
    throw new Error(detail);
  }
  return (await response.json()) as T;
}

export const api = {
  get: <T>(path: string) => request<T>(path),
  post: <T>(path: string, body: unknown) =>
    request<T>(path, { method: "POST", body: JSON.stringify(body) }),
  patch: <T>(path: string, body: unknown) =>
    request<T>(path, { method: "PATCH", body: JSON.stringify(body) }),
};

export function inr(value: number | null | undefined): string {
  const amount = Number(value ?? 0);
  const sign = amount < 0 ? "-" : "";
  const [whole, decimals] = Math.abs(amount).toFixed(2).split(".");
  let grouped = whole;
  if (whole.length > 3) {
    const last3 = whole.slice(-3);
    let rest = whole.slice(0, -3);
    const parts: string[] = [];
    while (rest.length > 2) {
      parts.unshift(rest.slice(-2));
      rest = rest.slice(0, -2);
    }
    if (rest) parts.unshift(rest);
    grouped = `${parts.join(",")},${last3}`;
  }
  return `${sign}₹${grouped}${decimals === "00" ? "" : `.${decimals}`}`;
}

export function compactInr(value: number | null | undefined): string {
  const amount = Number(value ?? 0);
  if (Math.abs(amount) >= 1e7) return `₹${(amount / 1e7).toFixed(2)} Cr`;
  if (Math.abs(amount) >= 1e5) return `₹${(amount / 1e5).toFixed(2)} L`;
  if (Math.abs(amount) >= 1e3) return `₹${(amount / 1e3).toFixed(1)}K`;
  return inr(amount);
}

export const pct = (value: number | null | undefined, digits = 0): string =>
  `${(Number(value ?? 0) * 100).toFixed(digits)}%`;

export const shortTime = (value?: string | null): string =>
  value ? String(value).replace("T", " ").slice(0, 16) : "—";
