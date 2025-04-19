#pragma once

#include <mutex>
#include <condition_variable>

/**
 * A simple Semaphore class.
 */
class Semaphore {
	std::mutex mutex;
	std::condition_variable condition;
	unsigned long count;

public:
	Semaphore(int count = 0)
		: count(count){}

	void release() {
		std::lock_guard<decltype(mutex)> lock(mutex);
		++count;
		condition.notify_one();
	}

    void acquire() {
        std::unique_lock<decltype(mutex)> lock(mutex);
        if (!count)
            condition.wait(lock);
        --count;
    }

    void acquireAll() {
        std::unique_lock<decltype(mutex)> lock(mutex);
        if (!count)
            condition.wait(lock);
        count = 0;
    }

    void wait(){
        std::unique_lock<decltype(mutex)> lock(mutex);
        if (!count)
            condition.wait(lock);
    }

	bool try_acquire() {
		std::lock_guard<decltype(mutex)> lock(mutex);
		if (count) {
			--count;
			return true;
		}
		return false;
	}
};
