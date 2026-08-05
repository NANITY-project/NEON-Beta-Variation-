// Standalone loader smoke test — doesn't need NEON.cpp's forward-pass
// plumbing at all, so it works today, before the templating integration
// described in rawllm_nctr_loader.hpp's bottom-of-file notes is done.
#include "rawllm_nctr_loader.hpp"
#include <iostream>

int main(int argc, char** argv) {
    if (argc < 2) { std::cerr << "usage: " << argv[0] << " <file.nctr>\n"; return 1; }
    try {
        loader::NCTRLoader model;
        model.open(argv[1]);

        engine::Config cfg;
        model.validate_config(cfg);

        std::cout << "OK — parsed " << model.tensors.size() << " tensors\n"
                  << "  n_vocab="   << cfg.n_vocab
                  << " n_embd="    << cfg.n_embd
                  << " n_layer="   << cfg.n_layer
                  << " n_head="    << cfg.n_head
                  << " n_kv_head=" << cfg.n_kv_head
                  << " head_dim="  << cfg.head_dim
                  << " n_ff="      << cfg.n_ff
                  << " ctx_len="   << cfg.ctx_len
                  << " use_swiglu=" << cfg.use_swiglu << "\n"
                  << "  tokenizer vocab=" << model.tok_meta.tokens.size()
                  << " bos=" << model.tok_meta.bos_id
                  << " eos=" << model.tok_meta.eos_id << "\n"
                  << "  manifest: " << model.manifest_json.substr(0, 80) << "...\n";

        // Spot-check a couple of tensor entries by name+shape+pointer validity.
        for (const auto& want : {"token_embd.weight", "blk.0.ffn_gate.weight"}) {
            bool found = false;
            for (const auto& t : model.tensors) {
                if (t.name == want) {
                    found = true;
                    std::cout << "  " << t.name << " shape=[";
                    for (size_t i = 0; i < t.shape.size(); ++i)
                        std::cout << t.shape[i] << (i+1<t.shape.size() ? "," : "");
                    std::cout << "] nbytes=" << t.nbytes
                              << " data_ptr_valid=" << (t.data_ptr != nullptr) << "\n";
                }
            }
            if (!found) { std::cerr << "MISSING TENSOR: " << want << "\n"; return 1; }
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << "\n";
        return 1;
    }
}
