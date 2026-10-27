import { useEffect, useLayoutEffect, useRef, useState } from "react";

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
  const transitionTimer = useRef<ReturnType<typeof setTimeout> | undefined>(
    undefined,
  );
  useLayoutEffect(() => {
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
      clearTimeout(transitionTimer.current);
      document.documentElement.removeAttribute("data-theme-transition");
    };
  }, []);
  function toggleTheme() {
    const next = theme === "dark" ? "light" : "dark";
    const root = document.documentElement;
    clearTimeout(transitionTimer.current);
    if (!matchMedia("(prefers-reduced-motion: reduce)").matches) {
      root.dataset.themeTransition = "colors";
      transitionTimer.current = setTimeout(() => {
        root.removeAttribute("data-theme-transition");
      }, 550);
    } else {
      root.removeAttribute("data-theme-transition");
    }
    try {
      localStorage.setItem(key, next);
    } catch {
      /* Theme still works if browser storage is unavailable. */
    }
    setTheme(next);
  }
  return { theme, toggleTheme };
}
