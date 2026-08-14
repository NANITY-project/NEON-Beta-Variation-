#pragma once
// =============================================================================
// nectar_vision.hpp — Vision module (NECTAR idle-loop doc §3/§8 step 4).
//
// Scope for this build step: OS-signal captioning + a coarse frame-diff
// heuristic, standalone and testable on its own — NOT yet spliced into
// run_idle_loop()'s generation stream. That splice is §7/§8 step 5
// (mid-stream injection), deliberately separate: the doc calls the splice
// "the part most likely to be fragile," so this module is built and
// validated in isolation first, exactly as step 4 asks ("start with slow
// full-frame polling to validate captioning, then move to event-driven
// diffing").
//
// Two caption sources, matched to the "both" plan (OS signals now, real
// captioning later):
//   - WindowSignal: focused app-id/title via the Wayland compositor. There
//     is no cross-compositor Wayland API for this (unlike X11's
//     wmctrl/xdotool) — every compositor does it differently. This targets
//     sway/wlroots via `swaymsg -t get_tree`, since it's the most common
//     scriptable case, and degrades to "unavailable" cleanly on anything
//     else rather than guessing or crashing. If your compositor isn't
//     sway, this needs a second backend (KWin script, GNOME extension,
//     Hyprland's hyprctl, etc.) — the WindowSignal interface below is
//     built so adding one is a new function, not a redesign.
//   - Captioner: an abstract hook for real image captioning. NullCaptioner
//     (the only implementation right now) always returns "unavailable" —
//     it exists so a real vision-model-backed Captioner can be dropped in
//     later without touching VisionModule's polling/diffing logic at all.
//
// Frame diffing: `grim` (the standard wlroots/sway screenshot CLI) captures
// a frame; instead of decoding pixels, this hashes the raw output bytes
// (FNV-1a). That's not pixel-accurate (a 1-pixel change and a full redraw
// both just look "different"), but it's cheap and it's the right level of
// precision for "did the idle loop's visual grounding change at all" — the
// doc's own event-driven-diffing goal, not a general vision task.
//
// Neither backend is available in a typical CI/container/headless
// environment; both fail closed (available=false, no throw) so the module
// is safe to construct and poll() anywhere, including where swaymsg/grim
// don't exist.
// =============================================================================
#include <array>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <string>

#include "rawllm_json.hpp"

namespace vision {

// ---- shell capture ---------------------------------------------------------
// Runs `cmd`, captures stdout as raw bytes. Returns nullopt if the command
// couldn't be launched or exited non-zero — never throws, since "the tool
// isn't installed" is an expected, common outcome here, not an error.
inline std::optional<std::string> run_capture(const std::string& cmd) {
    // Redirect stderr to /dev/null so a missing binary's shell error
    // doesn't leak into stdout capture or the terminal.
    std::string full_cmd = cmd + " 2>/dev/null";
    FILE* pipe = popen(full_cmd.c_str(), "r");
    if (!pipe) return std::nullopt;

    std::string out;
    char buf[4096];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), pipe)) > 0)
        out.append(buf, n);

    int status = pclose(pipe);
    if (status != 0) return std::nullopt;
    return out;
}

// ---- FNV-1a 64-bit hash -----------------------------------------------------
inline uint64_t fnv1a(const std::string& data) {
    uint64_t h = 1469598103934665603ULL;   // offset basis
    for (unsigned char c : data) {
        h ^= c;
        h *= 1099511628211ULL;             // prime
    }
    return h;
}

// ---- window signal (sway/wlroots backend) ----------------------------------
struct WindowInfo {
    bool        available = false;
    std::string app_id;     // e.g. "kitty", "firefox"
    std::string title;      // window title, if the compositor exposes one
};

// Recursively walks swaymsg's `get_tree` JSON for the focused node. sway's
// tree is a nested container structure (outputs -> workspaces -> windows),
// so this just depth-first searches for `"focused": true` rather than
// assuming a fixed depth.
inline bool find_focused(const json::Value& node, WindowInfo& out) {
    if (!node.is_object()) return false;
    if (node["focused"].get_bool(false)) {
        out.available = true;
        out.app_id = node["app_id"].get_str();
        if (out.app_id.empty())
            out.app_id = node["window_properties"]["class"].get_str();
        out.title = node["name"].get_str();
        return true;
    }
    const auto& nodes = node["nodes"];
    if (nodes.is_array())
        for (size_t i = 0; i < nodes.size(); ++i)
            if (find_focused(nodes[i], out)) return true;
    const auto& floating = node["floating_nodes"];
    if (floating.is_array())
        for (size_t i = 0; i < floating.size(); ++i)
            if (find_focused(floating[i], out)) return true;
    return false;
}

inline WindowInfo get_focused_window_sway() {
    WindowInfo info;
    auto raw = run_capture("swaymsg -t get_tree");
    if (!raw) return info;   // swaymsg missing or failed — not sway/wlroots, or not installed

    json::Value tree = json::try_parse(*raw);
    if (tree.is_null()) return info;   // malformed output — degrade, don't throw

    find_focused(tree, info);
    return info;
}

// ---- frame capture + hash (grim backend) -----------------------------------
// Returns the FNV-1a hash of a full-screen PPM capture, or nullopt if grim
// isn't available. PPM (not PNG) deliberately — no compression means the
// hash reflects pixel content directly rather than also being sensitive to
// the PNG encoder's own noise.
inline std::optional<uint64_t> capture_frame_hash() {
    auto raw = run_capture("grim -t ppm -");
    if (!raw || raw->empty()) return std::nullopt;
    return fnv1a(*raw);
}

// ---- pluggable captioner (real vision model — not implemented yet) --------
// Abstract on purpose: this build step ships only NullCaptioner. A real
// implementation (e.g. wrapping a small vision-language model through
// NEON, or shelling out to an external captioning service) plugs in later
// by subclassing this — VisionModule itself never changes.
class Captioner {
public:
    virtual ~Captioner() = default;
    virtual bool available() const { return false; }
    // frame_ppm: raw PPM bytes from capture_frame_hash()'s same source, if
    // the caller wants to pass real pixels through for captioning (kept
    // separate from the hash, which never needs full pixels).
    virtual std::string caption(const std::string& /*frame_ppm*/) { return ""; }
};

class NullCaptioner : public Captioner {
public:
    bool available() const override { return false; }
};

// ---- VisionModule: ties the above together into short text deltas --------
// poll() is meant to be called on a timer by the caller (slow full-frame
// polling for this step, per §8 step 4 — event-driven diffing is the
// natural next step once this is validated, but that's a scheduling change
// in the caller, not a change to this module's logic). Returns a short
// delta string (matching the doc's "user opened a terminal" style) only
// when something is judged to have changed; nullopt otherwise, so a caller
// doing event-driven diffing can just skip emitting anything on a nullopt.
class VisionModule {
public:
    explicit VisionModule(Captioner* captioner = nullptr)
        : captioner_(captioner ? captioner : &default_captioner_) {}

    // Returns a short delta describing what changed since the last poll(),
    // or nullopt if nothing did (or neither backend is available).
    std::optional<std::string> poll() {
        std::optional<std::string> delta;

        WindowInfo win = get_focused_window_sway();
        if (win.available && (win.app_id != last_window_.app_id ||
                               win.title  != last_window_.title)) {
            std::string desc = "user switched to " +
                (win.app_id.empty() ? std::string("an application") : win.app_id);
            if (!win.title.empty()) desc += " (\"" + win.title + "\")";
            delta = desc;
        }
        if (win.available) last_window_ = win;

        auto hash = capture_frame_hash();
        if (hash) {
            if (last_frame_hash_ && *hash != *last_frame_hash_ && !delta) {
                // Window didn't change but the screen did (scrolling,
                // typing, a redraw) — coarser signal, only surfaced when
                // the window signal didn't already explain the change.
                delta = "screen content changed";
            }
            last_frame_hash_ = hash;
        }

        return delta;
    }

    bool window_signal_available() const { return get_focused_window_sway().available; }
    bool frame_signal_available()  const { return capture_frame_hash().has_value(); }
    bool captioner_available()     const { return captioner_->available(); }

private:
    Captioner*         captioner_;
    NullCaptioner       default_captioner_;
    WindowInfo          last_window_;
    std::optional<uint64_t> last_frame_hash_;
};

} // namespace vision
