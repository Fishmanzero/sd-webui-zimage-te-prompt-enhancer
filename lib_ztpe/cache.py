# -*- coding: utf-8 -*-
"""有界 LRU 结果缓存。

旧版的 `_cache` 是无上限的模块级 dict，长时间使用会持续增长；且 key 用
`template[:64]`，三套内置预设的前 64 字符完全相同，导致跨预设误命中。
这里两处都修掉：整模板进哈希，容量固定并做 LRU 淘汰。
"""

import hashlib
from collections import OrderedDict

_MISS = object()


def make_key(*parts):
    """按顺序把各部分写进 sha256，用 \\0 分隔避免拼接歧义。"""
    hasher = hashlib.sha256()
    for part in parts:
        hasher.update(repr(part).encode("utf-8"))
        hasher.update(b"\x00")
    return hasher.hexdigest()


class LRUCache:
    def __init__(self, maxsize=64):
        if maxsize < 1:
            raise ValueError("maxsize must be >= 1")
        self.maxsize = int(maxsize)
        self._data = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key):
        if key not in self._data:
            self.misses += 1
            return _MISS
        self.hits += 1
        self._data.move_to_end(key)
        return self._data[key]

    def put(self, key, value):
        if key in self._data:
            self._data.move_to_end(key)
        self._data[key] = value
        while len(self._data) > self.maxsize:
            self._data.popitem(last=False)

    def clear(self):
        self._data.clear()
        self.hits = 0
        self.misses = 0

    def __len__(self):
        return len(self._data)

    def __contains__(self, key):
        return key in self._data


def is_miss(value):
    return value is _MISS
