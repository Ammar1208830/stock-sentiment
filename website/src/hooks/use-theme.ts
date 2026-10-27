import { useEffect, useState } from "react";

type Theme = "light" | "dark";
const key = "sentiment-lab-theme";

function storedTheme(): Theme | null {
  try {
    const value = localStorage.getItem(key);
    return value === "light" || value === "dark" ? value : null;
  } catch {
    return null;
  }
}

export function useTheme() {
  const [theme, setTheme] = useState<Theme>(() =>
    document.documentElement.classList.contains("dark") ? "dark" : "light",
  );
  useEffect(() => {
    document.documentElement.classList.toggle("dark", theme === "dark");
    document.documentElement.style.colorScheme = theme;
    document
      .querySelector('meta[name="theme-color"]')
      ?.setAttribute("content", theme === "dark" ? "#141b18" : "#f6f8f4");
  }, [theme]);
  useEffect(() => {
    const media = matchMedia("(prefers-color-scheme: dark)");
    const systemChanged = () => {
      if (!storedTheme()) setTheme(media.matches ? "dark" : "light");
    };
    const storageChanged = (event: StorageEvent) => {
      if (event.key === key || event.key === null)
        setTheme(storedTheme() ?? (media.matches ? "dark" : "light"));
    };
    media.addEventListener("change", systemChanged);
    addEventListener("storage", storageChanged);
    return () => {
      media.removeEventListener("change", systemChanged);
      removeEventListener("storage", storageChanged);
    };
  }, []);
  function toggleTheme() {
    const next = theme === "dark" ? "light" : "dark";
    try {
      localStorage.setItem(key, next);
    } catch {
      /* Theme still works if browser storage is unavailable. */
    }
    setTheme(next);
  }
  return { theme, toggleTheme };
}
