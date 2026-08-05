#pragma once
// =============================================================================
// rawllm_json.hpp  —  Minimal correct recursive-descent JSON parser
//
// FIX: replaces four hand-rolled json_get_str/int/float/bool helpers that:
//   • matched "role" and "content" substrings inside string values
//   • broke on nested objects / arrays
//   • had no escape-sequence awareness in the fast path
//   • were not re-entrant / context-free
//
// Usage:
//   auto v = json::parse(body);
//   std::string role = v["messages"][0]["role"].get_str();
//   int max_tok      = v["max_tokens"].get_int(256);
//   bool stream      = v["stream"].get_bool(false);
// =============================================================================
#include <string>
#include <vector>
#include <unordered_map>
#include <stdexcept>
#include <cstdlib>     // strtod
#include <cstring>
#include <cstdio>      // snprintf (used in escape())

namespace json {

// ── Value ─────────────────────────────────────────────────────────────────────
struct Value {
    enum class Type { Null, Bool, Number, String, Array, Object };
    Type type = Type::Null;

    bool        b   = false;
    double      n   = 0.0;
    std::string s;
    std::vector<Value>                     a;
    std::unordered_map<std::string, Value> o;

    Value() = default;

    bool is_null()   const noexcept { return type == Type::Null;   }
    bool is_bool()   const noexcept { return type == Type::Bool;   }
    bool is_number() const noexcept { return type == Type::Number; }
    bool is_string() const noexcept { return type == Type::String; }
    bool is_array()  const noexcept { return type == Type::Array;  }
    bool is_object() const noexcept { return type == Type::Object; }

    bool get_bool(bool def = false) const noexcept {
        if (is_bool())   return b;
        if (is_number()) return n != 0.0;
        return def;
    }
    int get_int(int def = 0) const noexcept {
        if (is_number()) return static_cast<int>(n);
        return def;
    }
    float get_float(float def = 0.f) const noexcept {
        if (is_number()) return static_cast<float>(n);
        return def;
    }
    int64_t get_int64(int64_t def = 0) const noexcept {
        if (is_number()) return static_cast<int64_t>(n);
        return def;
    }
    const std::string& get_str() const noexcept {
        static const std::string empty;
        return is_string() ? s : empty;
    }
    std::string get_str(const std::string& def) const noexcept {
        return is_string() ? s : def;
    }

    // Object subscript — returns a null Value for missing keys (never throws).
    const Value& operator[](const std::string& key) const noexcept {
        static const Value null_val;
        if (!is_object()) return null_val;
        auto it = o.find(key);
        return (it != o.end()) ? it->second : null_val;
    }
    // Array subscript — returns a null Value for out-of-range (never throws).
    const Value& operator[](std::size_t i) const noexcept {
        static const Value null_val;
        return (is_array() && i < a.size()) ? a[i] : null_val;
    }

    std::size_t size() const noexcept {
        if (is_array())  return a.size();
        if (is_object()) return o.size();
        return 0;
    }
    bool contains(const std::string& key) const noexcept {
        return is_object() && o.count(key) > 0;
    }
};

// ── Parser internals ─────────────────────────────────────────────────────────
namespace detail {

struct Parser {
    const char* p;
    const char* end;

    void skip_ws() noexcept {
        while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r'))
            ++p;
    }
    char peek() noexcept { skip_ws(); return p < end ? *p : '\0'; }

    char consume() noexcept { return p < end ? *p++ : '\0'; }

    void expect(char c) {
        skip_ws();
        if (p >= end || *p != c) {
            std::string msg = "JSON: expected '";
            msg += c; msg += "'";
            if (p < end) { msg += " got '"; msg += *p; msg += "'"; }
            throw std::runtime_error(msg);
        }
        ++p;
    }

    // Returns a decoded JSON string (cursor must be positioned at the opening '"').
    std::string parse_string() {
        expect('"');
        std::string res;
        res.reserve(32);
        while (p < end) {
            unsigned char c = (unsigned char)*p++;
            if (c == '"') return res;
            if (c == '\\') {
                if (p >= end) break;
                unsigned char esc = (unsigned char)*p++;
                switch (esc) {
                    case '"':  res += '"';  break;
                    case '\\': res += '\\'; break;
                    case '/':  res += '/';  break;
                    case 'n':  res += '\n'; break;
                    case 'r':  res += '\r'; break;
                    case 't':  res += '\t'; break;
                    case 'b':  res += '\b'; break;
                    case 'f':  res += '\f'; break;
                    case 'u': {
                        if (p + 4 > end)
                            throw std::runtime_error("JSON: truncated \\uXXXX escape");
                        char hex[5] = {p[0],p[1],p[2],p[3],0};
                        p += 4;
                        unsigned cp = static_cast<unsigned>(std::strtoul(hex, nullptr, 16));
                        // Handle surrogate pairs (UTF-16 high surrogate 0xD800–0xDBFF)
                        if (cp >= 0xD800 && cp <= 0xDBFF) {
                            if (p + 6 <= end && p[0]=='\\' && p[1]=='u') {
                                p += 2;
                                char hex2[5] = {p[0],p[1],p[2],p[3],0};
                                p += 4;
                                unsigned low = static_cast<unsigned>(
                                    std::strtoul(hex2, nullptr, 16));
                                cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
                            }
                        }
                        // Emit UTF-8
                        if (cp < 0x80) {
                            res += static_cast<char>(cp);
                        } else if (cp < 0x800) {
                            res += static_cast<char>(0xC0 | (cp >> 6));
                            res += static_cast<char>(0x80 | (cp & 0x3F));
                        } else if (cp < 0x10000) {
                            res += static_cast<char>(0xE0 | (cp >> 12));
                            res += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                            res += static_cast<char>(0x80 | (cp & 0x3F));
                        } else {
                            res += static_cast<char>(0xF0 | (cp >> 18));
                            res += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
                            res += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                            res += static_cast<char>(0x80 | (cp & 0x3F));
                        }
                        break;
                    }
                    default: res += static_cast<char>(esc); break;
                }
            } else {
                res += static_cast<char>(c);
            }
        }
        throw std::runtime_error("JSON: unterminated string");
    }

    Value parse_value() {
        switch (peek()) {
            case '"': { Value v; v.type=Value::Type::String; v.s=parse_string(); return v; }
            case '{': return parse_object();
            case '[': return parse_array();
            case 't': {
                // "true"
                if (p+4 <= end && p[0]=='t'&&p[1]=='r'&&p[2]=='u'&&p[3]=='e') {
                    p+=4; Value v; v.type=Value::Type::Bool; v.b=true; return v; }
                throw std::runtime_error("JSON: invalid token");
            }
            case 'f': {
                if (p+5 <= end && p[0]=='f'&&p[1]=='a'&&p[2]=='l'&&p[3]=='s'&&p[4]=='e') {
                    p+=5; Value v; v.type=Value::Type::Bool; v.b=false; return v; }
                throw std::runtime_error("JSON: invalid token");
            }
            case 'n': {
                if (p+4 <= end && p[0]=='n'&&p[1]=='u'&&p[2]=='l'&&p[3]=='l') {
                    p+=4; return Value(); }
                throw std::runtime_error("JSON: invalid token");
            }
            default: {
                // Number (integer or float, optional leading '-')
                skip_ws();
                if (p >= end) throw std::runtime_error("JSON: unexpected end");
                char* ep = nullptr;
                double n = std::strtod(p, &ep);
                if (ep == p) throw std::runtime_error("JSON: expected number");
                p = ep;
                Value v; v.type = Value::Type::Number; v.n = n; return v;
            }
        }
    }

    Value parse_object() {
        expect('{');
        Value v; v.type = Value::Type::Object;
        if (peek() == '}') { ++p; return v; }
        while (true) {
            skip_ws();
            std::string key = parse_string();
            expect(':');
            v.o[std::move(key)] = parse_value();
            char c = peek();
            if (c == '}') { ++p; return v; }
            if (c == ',') { ++p; continue; }
            throw std::runtime_error("JSON: expected ',' or '}' in object");
        }
    }

    Value parse_array() {
        expect('[');
        Value v; v.type = Value::Type::Array;
        if (peek() == ']') { ++p; return v; }
        while (true) {
            v.a.push_back(parse_value());
            char c = peek();
            if (c == ']') { ++p; return v; }
            if (c == ',') { ++p; continue; }
            throw std::runtime_error("JSON: expected ',' or ']' in array");
        }
    }
};

} // namespace detail

// ── Public API ────────────────────────────────────────────────────────────────

// Parse a JSON string; throws std::runtime_error on malformed input.
inline Value parse(const std::string& s) {
    if (s.empty()) return Value{};
    detail::Parser p{ s.c_str(), s.c_str() + s.size() };
    return p.parse_value();
}

// Parse without throwing; returns a null Value on error.
inline Value try_parse(const std::string& s) noexcept {
    try { return parse(s); } catch (...) { return Value{}; }
}

// Produce a minimal JSON string escape of 's' for use in hand-built responses.
inline std::string escape(const std::string& s) {
    std::string o;
    o.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
            case '"':  o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n";  break;
            case '\r': o += "\\r";  break;
            case '\t': o += "\\t";  break;
            default:
                if (c < 0x20) {
                    char buf[8]; std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    o += buf;
                } else {
                    o += static_cast<char>(c);
                }
        }
    }
    return o;
}

} // namespace json
