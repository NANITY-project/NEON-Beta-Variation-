#pragma once
// =============================================================================
// nectar_vision.hpp — Vision module (NECTAR idle-loop doc §3/§8 step 4).
//
// Scope for this build step: OS-signal captioning + a real content
// captioner, standalone and testable on its own — NOT yet spliced into
// run_idle_loop()'s generation stream. That splice is §7/§8 step 5
// (mid-stream injection), deliberately separate: the doc calls the splice
// "the part most likely to be fragile," so this module is built and
// validated in isolation first, exactly as step 4 asks.
//
// Backend: Hyprland only (not sway/generic wlroots). Wayland has no
// cross-compositor window-info API — GNOME needs a shell extension, KDE
// needs a KWin script, sway needs its own IPC — so "generic Wayland" isn't
// really a thing; each compositor needs its own backend. This targets
// Hyprland's `hyprctl activewindow -j`, which — unlike sway's nested
// get_tree — returns one flat JSON object for whatever window is
// currently focused (or `{}` if nothing is), so there's no tree-walk here,
// just a direct parse.
//
// Two caption sources:
//   - WindowSignal: focused app class/title via `hyprctl activewindow -j`.
//   - Captioner: now backed by real OCR (OcrCaptioner, via `tesseract`)
//     instead of NullCaptioter — it describes actual on-screen text
//     content rather than just flagging "something changed." This is a
//     genuinely useful signal for a dev-facing idle companion (most of
//     what's on screen is code/terminal text), but it is NOT general image
//     captioning — it says nothing about non-text visual content (a photo,
//     a diagram, a video). NullCaptioner is kept as a fallback/base case
//     for anywhere OCR isn't wanted or tesseract isn't installed.
//
// Frame capture: `grim` (the standard wlroots/Hyprland/sway screenshot
// CLI) — this part is compositor-agnostic within wlroots and didn't need
// to change with the Hyprland swap.
//
// None of grim/hyprctl/tesseract are guaranteed to be present; each
// backend fails closed (available=false, no throw) so the module is safe
// to construct and poll() anywhere.
// =============================================================================
#include <array>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <optional>
#include <string>
#include <unistd.h>

#include "rawllm_json.hpp"

namespace vision {

// ---- shell capture ---------------------------------------------------------
// Runs `cmd`, captures stdout as raw bytes. Returns nullopt if the command
// couldn't be launched or exited non-zero — never throws, since "the tool
// isn't installed" is an expected, common outcome here, not an error.
inline std::optional<std::string> run_capture(const std::string& cmd) {
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

// ---- window signal (Hyprland backend) --------------------------------------
struct WindowInfo {
    bool        available = false;
    std::string app_id;     // hyprctl's "class", e.g. "kitty", "firefox"
    std::string title;      // hyprctl's "title"
};

inline WindowInfo get_focused_window_hyprland() {
    WindowInfo info;
    auto raw = run_capture("hyprctl activewindow -j");
    if (!raw) return info;   // hyprctl missing or failed — not Hyprland, or not installed

    json::Value w = json::try_parse(*raw);
    if (w.is_null() || !w.is_object() || w.o.empty())
        return info;   // malformed, or `{}` (Hyprland's "nothing focused") — degrade, don't throw

    info.app_id = w["class"].get_str();
    info.title  = w["title"].get_str();
    info.available = !info.app_id.empty() || !info.title.empty();
    return info;
}

// ---- frame capture + hash (grim backend) -----------------------------------
struct FrameCapture {
    bool        available = false;
    std::string ppm;     // raw PPM bytes, empty if unavailable
    uint64_t    hash = 0;
};

// PPM (not PNG) deliberately — no compression means the hash reflects pixel
// content directly, and OcrCaptioner can feed the same raw bytes straight
// to tesseract without a decode step.
inline FrameCapture capture_frame() {
    FrameCapture fc;
    auto raw = run_capture("grim -t ppm -");
    if (!raw || raw->empty()) return fc;
    fc.available = true;
    fc.ppm = std::move(*raw);
    fc.hash = fnv1a(fc.ppm);
    return fc;
}

inline std::optional<uint64_t> capture_frame_hash() {
    FrameCapture fc = capture_frame();
    if (!fc.available) return std::nullopt;
    return fc.hash;
}

// ---- pluggable captioner ----------------------------------------------------
class Captioner {
public:
    virtual ~Captioner() = default;
    virtual bool available() const { return false; }
    virtual std::string caption(const std::string& /*frame_ppm*/) { return ""; }
};

// Always-unavailable base case — kept as an explicit "no captioning" option
// (e.g. for a headless test, or a machine without tesseract) rather than
// leaving VisionModule with no fallback if OcrCaptioner::available() is
// false.
class NullCaptioner : public Captioner {
public:
    bool available() const override { return false; }
};

// Real captioner: OCR via `tesseract`. Describes actual on-screen text
// content — "screen shows text: ..." — not just a change flag. Scope is
// genuinely limited to text: a screen that's mostly a photo, video, or
// diagram will caption as empty or near-empty, since there's no OCR
// content to extract. That's an honest limitation of this backend, not a
// bug — true general image captioning needs a vision-language model,
// which isn't part of this build step.
class OcrCaptioner : public Captioner {
public:
    bool available() const override {
        return run_capture("tesseract --version").has_value();
    }

    std::string caption(const std::string& frame_ppm) override {
        if (frame_ppm.empty()) return "";

        std::string tmp_path = "/tmp/nectar_vision_frame_" + std::to_string(getpid()) + ".ppm";
        {
            std::ofstream f(tmp_path, std::ios::binary);
            if (!f) return "";
            f.write(frame_ppm.data(), (std::streamsize)frame_ppm.size());
        }
        // --psm 3 (default): fully automatic page segmentation, no OSD.
        // Fine for arbitrary screen content (mixed terminal/editor/browser
        // text) where we don't know layout ahead of time.
        auto text = run_capture("tesseract " + tmp_path + " stdout --psm 3");
        std::remove(tmp_path.c_str());
        if (!text) return "";
        return summarize(*text);
    }

private:
    static std::string summarize(const std::string& raw_text) {
        // Collapse whitespace/newlines to single spaces and trim — this
        // feeds a short idle-loop delta ("screen shows: ..."), not a full
        // page dump.
        std::string collapsed;
        collapsed.reserve(raw_text.size());
        bool last_was_space = false;
        for (char c : raw_text) {
            char out_c = (c == '\n' || c == '\r' || c == '\t') ? ' ' : c;
            if (out_c == ' ') {
                if (last_was_space) continue;
                last_was_space = true;
            } else {
                last_was_space = false;
            }
            collapsed += out_c;
        }
        size_t start = collapsed.find_first_not_of(' ');
        if (start == std::string::npos) return "";   // OCR found nothing (non-text content)
        size_t end = collapsed.find_last_not_of(' ');
        collapsed = collapsed.substr(start, end - start + 1);

        constexpr size_t kMaxLen = 160;
        if (collapsed.size() > kMaxLen) {
            collapsed.resize(kMaxLen);
            collapsed += "...";
        }
        return "screen shows text: \"" + collapsed + "\"";
    }
};

// ---- VisionModule: ties the above together into short text deltas --------
class VisionModule {
public:
    explicit VisionModule(Captioner* captioner = nullptr)
        : captioner_(captioner ? captioner : &default_captioner_) {}

    std::optional<std::string> poll() {
        std::optional<std::string> delta;

        WindowInfo win = get_focused_window_hyprland();
        if (win.available && (win.app_id != last_window_.app_id ||
                               win.title  != last_window_.title)) {
            std::string desc = "user switched to " +
                (win.app_id.empty() ? std::string("an application") : win.app_id);
            if (!win.title.empty()) desc += " (\"" + win.title + "\")";
            delta = desc;
        }
        if (win.available) last_window_ = win;

        FrameCapture frame = capture_frame();
        if (frame.available) {
            bool changed = last_frame_hash_.has_value() && frame.hash != *last_frame_hash_;
            if (changed && !delta) {
                if (captioner_->available()) {
                    std::string cap = captioner_->caption(frame.ppm);
                    delta = !cap.empty() ? cap : std::string("screen content changed");
                } else {
                    delta = "screen content changed";
                }
            }
            last_frame_hash_ = frame.hash;
        }

        return delta;
    }

    bool window_signal_available() const { return get_focused_window_hyprland().available; }
    bool frame_signal_available()  const { return capture_frame().available; }
    bool captioner_available()     const { return captioner_->available(); }

private:
    Captioner*  captioner_;
    OcrCaptioner default_captioner_;
    WindowInfo   last_window_;
    std::optional<uint64_t> last_frame_hash_;
};

} // namespace vision
