// common_aliases.h
#ifndef COMMON_ALIASES_H
#define COMMON_ALIASES_H

#include <memory>

template <typename T, typename Deleter = std::default_delete<T>>
using uptr = std::unique_ptr<T, Deleter>;

template <typename T>
using sptr = std::shared_ptr<T>;

#endif // COMMON_ALIASES_H
