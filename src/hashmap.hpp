#pragma once

#include <algorithm>
#include <iostream>
#include <list>
#include <ranges>

template <typename KeyType, typename ValueType,
          typename Hash = std::hash<KeyType>>
class HashMap {
  private:
    struct Entry {
        KeyType key;
        ValueType value;
    };

    using Bucket = std::list<Entry>;
    std::vector<Bucket> buckets;
    size_t size;
    Hash hashFunction;

    size_t hash(const KeyType& key) const {
        return hashFunction(key) % buckets.size();
    }

  public:
    explicit HashMap(size_t bucketCount = 10) : buckets(bucketCount), size(0) {}

    void insert(const KeyType& key, const ValueType& value) {
        size_t index = hash(key);
        auto& bucket = buckets[index];

        auto found = std::ranges::find_if(
            bucket, [&](const Entry& entry) { return entry.key == key; });

        if (found != bucket.end()) {
            std::cerr << "Key already exists. Update not supported in this "
                         "implementation."
                      << std::endl;
            return;
        }

        bucket.push_back({key, value});
        ++size;
    }

    bool contains(const KeyType& key) const {
        size_t index = hash(key);
        const auto& bucket = buckets[index];

        return std::ranges::any_of(
            bucket, [&](const Entry& entry) { return entry.key == key; });
    }

    const ValueType& get(const KeyType& key) const {
        size_t index = hash(key);
        const auto& bucket = buckets[index];

        auto found = std::ranges::find_if(
            bucket, [&](const Entry& entry) { return entry.key == key; });

        if (found != bucket.end()) {
            return found->value;
        }
        throw std::out_of_range("Key not found in HashMap");
    }

    void remove(const KeyType& key) {
        size_t index = hash(key);
        auto& bucket = buckets[index];

        auto found = std::ranges::find_if(
            bucket, [&](const Entry& entry) { return entry.key == key; });

        if (found != bucket.end()) {
            bucket.erase(found);
            --size;
        }
    }

    size_t getSize() const { return size; }

    size_t getBucketCount() const { return buckets.size(); }

    void read() const {
        std::cout << "All key-value pairs in the HashMap:" << std::endl;
        for (const auto& bucket : buckets) {
            for (const auto& entry : bucket) {
                std::cout << "Key: " << entry.key << ", Value: " << entry.value
                          << std::endl;
            }
        }
    }
};
