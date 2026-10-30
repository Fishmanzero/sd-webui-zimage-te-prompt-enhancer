# -*- coding: utf-8 -*-
"""LRU 缓存与 key 组合自检。"""

from lib_ztpe import cache


def run(check):
    lru = cache.LRUCache(maxsize=4)
    check("初始为空", len(lru) == 0)
    check("未命中返回 sentinel", cache.is_miss(lru.get("a")))

    for i in range(4):
        lru.put(f"k{i}", i)
    check("装到上限", len(lru) == 4)
    check("命中返回值", lru.get("k1") == 1)

    lru.put("k4", 4)
    check("超限淘汰最旧", "k0" not in lru and len(lru) == 4)
    check("刚命中的 k1 未被淘汰", "k1" in lru)

    lru.get("k1")
    lru.put("k5", 5)
    check("命中会刷新热度", "k1" in lru and "k2" not in lru)

    lru.clear()
    check("clear 后为空", len(lru) == 0)

    check("maxsize < 1 拒绝", _raises(lambda: cache.LRUCache(0)))

    key_a = cache.make_key("model", "一只猫", "模板A", "zh", 42)
    check("key 稳定", key_a == cache.make_key("model", "一只猫", "模板A", "zh", 42))
    check("模板变化则 key 变化", key_a != cache.make_key("model", "一只猫", "模板B", "zh", 42))
    check("种子变化则 key 变化", key_a != cache.make_key("model", "一只猫", "模板A", "zh", 43))
    check("语种变化则 key 变化", key_a != cache.make_key("model", "一只猫", "模板A", "en", 42))
    check("拼接歧义被消除",
          cache.make_key("ab", "c") != cache.make_key("a", "bc"))

    long_a = "X" * 64 + "A"
    long_b = "X" * 64 + "B"
    check("长模板不再因前 64 字符相同而碰撞",
          cache.make_key(long_a) != cache.make_key(long_b),
          "旧版 template[:64] 回归")


def _raises(fn):
    try:
        fn()
    except Exception:
        return True
    return False
