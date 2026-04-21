# Design System Specification: The Kinetic Engine

## 1. Overview & Creative North Star
This design system is built for the high-stakes environment of Industrial IoT. We are moving away from the "SaaS-standard" look toward a **"Kinetic Engine"** aesthetic—a visual language that feels as precise, heavy, and intentional as the machinery it monitors. 

While inspired by the functional clarity of GitHub, we elevate the experience through **Organic Industrialism**. We reject the flat, white-label grid in favor of high-contrast typography, asymmetrical layouts that mimic technical schematics, and deep tonal layering. This system doesn't just display data; it pulses with the life of the factory floor.

**The Creative North Star: "Precision in the Shadows"**
The interface should feel like a high-end physical console. We achieve this through:
*   **Intentional Asymmetry:** Using the grid to create "technical tension"—offsetting headers or utilizing wide margins to evoke blueprints.
*   **Bespoke Depth:** Moving beyond shadows into a world of "glass and gas," where information floats in a tiered, atmospheric dark space.

---

## 2. Colors: Tonal Architecture
The palette is rooted in an OLED-ready dark environment. Colors are used as functional signals, not just decoration.

### The Color Roles
*   **Base (`surface` / #131313):** The infinite void. This is the foundation of the machine.
*   **The Accents:** 
    *   **Success/Healthy (`primary_fixed` / #45FDD2):** A hyper-modern neon cyan. Use this for "Optimal" states and primary action paths.
    *   **Danger/Failure (`secondary_container` / #D30017):** A glowing crimson. This is reserved for critical warnings and imminent failures.
*   **Neutral Elevators:** `surface_container_low` (#1C1B1B) to `surface_container_highest` (#353534).

### The "No-Line" Rule
**Strict Mandate:** Designers are prohibited from using 1px solid borders to section off the UI. Containers must be defined by:
1.  **Tonal Shifts:** Placing a `surface_container_high` card on a `surface` background.
2.  **Negative Space:** Using the spacing scale to create distinct visual groups.
3.  **Background Patterns:** Using subtle industrial schematics or glowing nodes to "frame" a section.

### The "Glass & Gradient" Rule
Standard containers feel static. To create a premium "Cyber-Industrial" feel, use **Glassmorphism** for floating panels (e.g., modals or side-drawers). 
*   **Recipe:** Use a semi-transparent version of `surface_container_high` (#2A2A2A at 70% opacity) with a `backdrop-filter: blur(24px)`.
*   **CTAs:** Apply subtle linear gradients (from `primary` to `primary_fixed`) to main action buttons to give them a tactile, glowing quality.

---

## 3. Typography: The Editorial Tech Scale
We use a dual-typeface system to balance technical precision with high-end editorial authority.

*   **Space Grotesk (Display & Headlines):** This is our "Technical Signature." Its geometric quirks feel industrial and futuristic. Use `display-lg` (3.5rem) for critical KPIs that need to shout.
*   **Inter (Body & Labels):** Our "Workhorse." Use Inter for all functional data, descriptions, and system labels. It provides the legibility needed for dense machine logs.

**Hierarchy Strategy:** 
Create extreme contrast. Pair a `display-sm` headline with a `label-sm` subtitle in `on_surface_variant` (#BACAC3) to create a sophisticated, high-end document feel.

---

## 4. Elevation & Depth: Tonal Layering
Depth in this system is achieved through physical stacking, not synthetic drop shadows.

### The Layering Principle
Think of the UI as layers of frosted industrial glass.
1.  **Level 0 (Floor):** `surface` (#131313).
2.  **Level 1 (Sections):** `surface_container_low` (#1C1B1B).
3.  **Level 2 (Active Cards):** `surface_container_high` (#2A2A2A).
4.  **Level 3 (Pop-overs/Modals):** `surface_container_highest` (#353534).

### Ambient Shadows
When an element must float (e.g., a critical failure alert), use a **Ghost Shadow**. 
*   **Spec:** 0px 24px 48px rgba(0, 0, 0, 0.5). 
*   Avoid hard black shadows; the shadow should feel like a soft occlusion of the light coming from the glowing accents.

### The "Ghost Border" Fallback
If a boundary is absolutely required for accessibility, use a **Ghost Border**:
*   `outline_variant` (#3B4A44) at **15% opacity**. It should be barely perceptible—a hint of an edge, not a cage.

---

## 5. Components

### Buttons: High-Frequency Actions
*   **Primary:** `primary_fixed` background with `on_primary_fixed` text. No border. Subtle glow on hover (box-shadow with primary color at 20% opacity).
*   **Secondary:** Ghost Border style. `on_surface` text with a 10% `outline_variant` edge.
*   **Tertiary:** Pure text using `primary_fixed_dim`. No container.

### Status Nodes (Bespoke IoT Component)
Instead of standard dots, use "Pulse Nodes"—small circular elements using `primary_fixed` (Success) or `secondary_container` (Danger) with a CSS animation that mimics a radar pulse or heart rate.

### Cards & Lists
*   **Cards:** Use `surface_container_high`. Roundedness: `lg` (0.5rem). 
*   **Lists:** **Forbid dividers.** Separate list items using 8px of vertical padding and a background shift to `surface_container_low` on hover. Use `label-sm` for metadata to maintain a clean, technical look.

### Input Fields
*   **Style:** Underline-only or subtle tonal shift. Avoid the "boxed" input.
*   **State:** When focused, the bottom border glows with `primary_fixed` (#45FDD2).

### Predictive Gauges (Bespoke IoT Component)
For machine failure prediction, use thick, non-rounded stroke bars. Use `secondary_container` for the "danger zone" and `primary_fixed` for the "safe zone," separated by a gap of 2px to maintain the "schematic" feel.

---

## 6. Do’s and Don’ts

### Do
*   **Do** use asymmetrical grid columns (e.g., a 4-column side panel with an 8-column main view).
*   **Do** use `monospace` for purely numerical data and timestamps to reinforce the technical vibe.
*   **Do** leverage "Breathing Room." High-density data requires large margins (32px+) to prevent cognitive overload.

### Don’t
*   **Don’t** use 100% opaque borders. They clutter the industrial aesthetic.
*   **Don’t** use standard blue for links. Use `primary_fixed` (Mint) for all "positive" interactions.
*   **Don’t** use "Soft" or "Playful" iconography. Use sharp, geometric, 2px stroke icons that match the `outline` weight.
*   **Don’t** settle for a flat background. Always layer at least two tiers of `surface_container` to create depth.