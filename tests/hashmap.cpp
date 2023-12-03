#include <algorithm>
#include <functional>
#include <iostream>
#include <list>
#include <ranges>
#include <unordered_map>
#include <sstream> 

#include <catch2/catch_test_macros.hpp>

#include "../src/hashmap.hpp"


TEST_CASE("HashMap basic operations", "[HashMap]") {
    HashMap<std::string, int> myMap;

    SECTION("Insertion and retrieval") {
        myMap.insert("apple", 5);
        myMap.insert("banana", 10);

        REQUIRE(myMap.contains("apple") == true);
        REQUIRE(myMap.contains("banana") == true);
        REQUIRE(myMap.contains("orange") == false);

        REQUIRE(myMap.get("apple") == 5);
        REQUIRE(myMap.get("banana") == 10);
    }

    SECTION("Removal") {
        myMap.insert("apple", 5);
        myMap.insert("banana", 10);

        myMap.remove("apple");
        REQUIRE(myMap.contains("apple") == false);
        REQUIRE(myMap.getSize() == 1);

        // Removing non-existing key shouldn't throw error
        myMap.remove("orange");
        REQUIRE(myMap.getSize() == 1);
    }

    SECTION("Read all key-value pairs") {
        myMap.insert("apple", 5);
        myMap.insert("banana", 10);

        std::stringstream buffer;
        std::streambuf* old = std::cout.rdbuf(buffer.rdbuf());

        myMap.read();

        std::string output = buffer.str();
        REQUIRE(output.find("Key: apple, Value: 5") != std::string::npos);
        REQUIRE(output.find("Key: banana, Value: 10") != std::string::npos);

        std::cout.rdbuf(old);
    }

    SECTION("HashMap size and bucket count") {
        myMap.insert("apple", 5);
        myMap.insert("banana", 10);

        REQUIRE(myMap.getSize() == 2);
        REQUIRE(myMap.getBucketCount() == 10);

        HashMap<std::string, int> anotherMap(5);
        REQUIRE(anotherMap.getBucketCount() == 5);
    }
}
