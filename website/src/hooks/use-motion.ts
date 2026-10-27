import { useEffect, type RefObject } from "react";
import gsap from "gsap";
import { useGSAP } from "@gsap/react";
import { animate, createScope, svg } from "animejs";

gsap.registerPlugin(useGSAP);

/** GSAP owns page/card entrances. Anime.js owns only the SVG inside the chart. */
export function useDashboardMotion(
  root: RefObject<HTMLElement | null>,
  view: string,
  ready: boolean,
) {
  useGSAP(
    () => {
      if (!ready || !root.current) return;
      const media = gsap.matchMedia();
      media.add("(prefers-reduced-motion: no-preference)", () => {
        const heading = root.current?.querySelector(".page-heading");
        const cards = root.current?.querySelectorAll(
          ".metric-card, .overview-grid > [data-slot='card'], .view-content > [data-slot='card'], .two-columns > [data-slot='card']",
        );
        const timeline = gsap.timeline({ defaults: { ease: "power2.out" } });
        if (heading)
          timeline.fromTo(
            heading,
            { opacity: 0, y: 8 },
            {
              opacity: 1,
              y: 0,
              duration: 0.32,
              clearProps: "opacity,transform",
            },
          );
        if (cards?.length)
          timeline.fromTo(
            cards,
            { opacity: 0, y: 12 },
            {
              opacity: 1,
              y: 0,
              duration: 0.4,
              stagger: 0.045,
              clearProps: "opacity,transform",
            },
            0.07,
          );
      });
      return () => media.revert();
    },
    { scope: root, dependencies: [view, ready], revertOnUpdate: true },
  );
}

export function usePriceChartMotion(
  root: RefObject<HTMLDivElement | null>,
  identity: string,
) {
  useEffect(() => {
    if (!root.current) return;
    const scope = createScope({
      root: root.current,
      mediaQueries: { reducedMotion: "(prefers-reduced-motion: reduce)" },
    }).add((self) => {
      if (self?.matches.reducedMotion) return;
      const line =
        root.current?.querySelector<SVGPathElement>("[data-chart-line]");
      const area =
        root.current?.querySelector<SVGPathElement>("[data-chart-area]");
      if (line)
        animate(svg.createDrawable(line), {
          draw: ["0 0", "0 1"],
          duration: 680,
          ease: "outQuad",
        });
      if (area)
        animate(area, {
          opacity: [0, 1],
          duration: 480,
          delay: 120,
          ease: "outQuad",
        });
    });
    return () => scope.revert();
  }, [root, identity]);
}
