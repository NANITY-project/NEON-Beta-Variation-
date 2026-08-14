#pragma once
// =============================================================================
// nectar_diskmem.hpp — On-disk memory module (NECTAR idle-loop doc §5.1).
//
// Scope, deliberately narrow: this is ONLY the write path described in
// §5.1 — batched, source-tagged append-only records. It is NOT the answer
// path's retrieval/pruning/tiering system (§2's "answer path" column) —
// that's a separate, larger module that reads this same store back. This
// header has no read/query function on purpose: "the idle loop never reads
// back from disk; only the answer path retrieves" (§2, §5.1) is enforced
// structurally, not just by convention, by not exposing one here.
//
// Two guardrails from §5.1, both required from the first write, not
// retrofitted later:
//   - Batch the writes: flush() only fires on a token-count or time
//     threshold, never per generated token.
//   - Tag source at write time: every record carries source: idle vs.
//     source: interaction. There is no code path to write an untagged
//     entry — Source is a required constructor argument, not a default.
// =============================================================================
#include <chrono>
#include <deque>
#include <fstream>
#include <mutex>
#include <string>

namespace diskmem {

enum class Source { Idle, Interaction };

inline const char* source_str(Source s) {
    return s == Source::Idle ? "idle" : "interaction";
}

// Minimal JSON string escaping — good enough for our own generated text
// (arbitrary UTF-8 bytes, no embedded NULs expected). Not a general JSON
// escaper; if this store ever needs to round-trip attacker-controlled
// input, revisit.
inline std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (c < 0x20) { /* drop other control chars */ }
                else out += (char)c;
        }
    }
    return out;
}

// Batches text before it hits disk, then flushes one JSONL record at a
// time: {"ts":<epoch_s>,"source":"idle"|"interaction","tokens":<n>,"text":"..."}
//
// Thread-safe (a single mutex around the small pending buffer) since the
// idle loop and, later, an interaction handler may run on different
// threads sharing one process — cheap enough here that it's not worth
// making lock-free.
class WriteBuffer {
public:
    WriteBuffer(std::string path, Source source,
                size_t flush_every_tokens = 300,
                std::chrono::seconds flush_interval = std::chrono::seconds(30))
        : path_(std::move(path)), source_(source),
          flush_every_tokens_(flush_every_tokens), flush_interval_(flush_interval),
          last_flush_(std::chrono::steady_clock::now()) {}

    // Call once per unit of text that has "aged out" of the live window
    // (idle loop: once ensure_cache_room() reports it as part of a
    // discarded range — see run_idle_loop()). This only grows the
    // in-memory buffer; flush_locked() decides when it actually hits disk.
    void add(const std::string& text_piece) {
        std::lock_guard<std::mutex> lock(mu_);
        pending_ += text_piece;
        ++pending_tokens_;
        maybe_flush_locked();
    }

    // Force a flush regardless of thresholds. Call on clean shutdown so a
    // partial buffer isn't silently dropped when the process exits.
    void flush() {
        std::lock_guard<std::mutex> lock(mu_);
        flush_locked();
    }

    size_t pending_tokens() const {
        std::lock_guard<std::mutex> lock(mu_);
        return pending_tokens_;
    }

private:
    void maybe_flush_locked() {
        auto now = std::chrono::steady_clock::now();
        bool by_count = pending_tokens_ >= flush_every_tokens_;
        bool by_time  = pending_tokens_ > 0 && (now - last_flush_) >= flush_interval_;
        if (by_count || by_time) flush_locked();
    }

    void flush_locked() {
        if (pending_.empty()) return;
        std::ofstream f(path_, std::ios::app);
        if (f) {
            auto epoch_s = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            f << "{\"ts\":" << epoch_s
              << ",\"source\":\"" << source_str(source_) << "\""
              << ",\"tokens\":" << pending_tokens_
              << ",\"text\":\"" << json_escape(pending_) << "\"}\n";
        }
        pending_.clear();
        pending_tokens_ = 0;
        last_flush_ = std::chrono::steady_clock::now();
    }

    std::string           path_;
    Source                source_;
    size_t                flush_every_tokens_;
    std::chrono::seconds  flush_interval_;
    std::chrono::steady_clock::time_point last_flush_;

    mutable std::mutex mu_;
    std::string pending_;
    size_t      pending_tokens_ = 0;
};

} // namespace diskmem
