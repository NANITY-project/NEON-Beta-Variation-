#pragma once
// =============================================================================
// rawllm_util.hpp  —  NUMA-aware ThreadPool
// =============================================================================
#include "rawllm_common.hpp"

namespace util {

inline void bind_thread_numa(int node) {
#ifdef USE_NUMA
    if (numa_available() < 0) return;
    int max_node = numa_max_node();
    node = node % (max_node + 1);
    struct bitmask* m = numa_bitmask_alloc(max_node + 1);
    numa_bitmask_setbit(m, node);
    numa_set_membind(m);
    numa_run_on_node(node);
    numa_bitmask_free(m);
#else
    (void)node;
#endif
}

class ThreadPool {
public:
    explicit ThreadPool(int n_workers, bool numa_bind = false) {
        n_workers = std::max(1, n_workers);
        workers_.reserve(n_workers);
        for (int i = 0; i < n_workers; ++i) {
            workers_.emplace_back([this, i, numa_bind] {
                if (numa_bind) bind_thread_numa(i);
                for (;;) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lk(mu_);
                        cv_.wait(lk, [this]{ return stop_ || !q_.empty(); });
                        if (stop_ && q_.empty()) return;
                        task = std::move(q_.front()); q_.pop();
                        active_.fetch_add(1, std::memory_order_relaxed);
                    }
                    task();
                    // FIX (correctness — confirmed via gdb thread dump showing
                    // all 12 workers idle while the main thread stayed
                    // permanently blocked in ThreadPool::wait()): this used
                    // to be `active_.fetch_sub(...); done_cv_.notify_all();`
                    // with NEITHER statement holding mu_. wait()'s predicate
                    // (`q_.empty() && active_==0`) is only safe from a lost
                    // wakeup if every mutation that predicate depends on is
                    // made under the SAME mutex the waiter holds while
                    // checking it — otherwise a worker can complete the
                    // decrement AND the notify_all() entirely within the
                    // narrow window between the waiter's predicate
                    // evaluating false and it actually entering the OS-level
                    // condition_variable block, and that notify is then
                    // simply gone (condition variables don't queue signals).
                    // The waiter then blocks forever even though the real
                    // condition is already true. This was always possible in
                    // principle; it only became practically likely once
                    // individual tasks dropped from hundreds of ms to
                    // microseconds (the fused Q4_0 kernel), which turned a
                    // vanishingly rare race into one hit within single-digit
                    // tokens. The notify itself can stay outside the lock —
                    // only the mutation needs to be inside it.
                    {
                        std::lock_guard<std::mutex> lk(mu_);
                        active_.fetch_sub(1, std::memory_order_relaxed);
                    }
                    done_cv_.notify_all();
                }
            });
        }
    }

    ~ThreadPool() {
        wait();
        { std::lock_guard<std::mutex> lk(mu_); stop_ = true; }
        cv_.notify_all();
        for (auto& w : workers_) w.join();
    }

    template<typename F>
    void submit(F&& f) {
        { std::lock_guard<std::mutex> lk(mu_); q_.emplace(std::forward<F>(f)); }
        cv_.notify_one();
    }

    void wait() {
        std::unique_lock<std::mutex> lk(mu_);
        done_cv_.wait(lk, [this]{ return q_.empty() && active_.load() == 0; });
    }

    int size() const { return static_cast<int>(workers_.size()); }

private:
    std::vector<std::thread>          workers_;
    std::queue<std::function<void()>> q_;
    std::mutex                        mu_;
    std::condition_variable           cv_, done_cv_;
    std::atomic<int>                  active_{0};
    bool                              stop_{false};
};

} // namespace util