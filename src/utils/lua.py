"""Redis Lua scripts — the two operations that must be atomic server-side.

Both close a check-then-act race that separate commands cannot:

- **budget**: two concurrent reservations must not both slip past the last paid
  slot, so the compare-against-limit and the increment happen in one script.
- **lock release**: only the holder may release, so the token compare and the
  delete happen in one script — a lock that expired and was re-acquired
  elsewhere can't be deleted out from under the new owner.

Scripts are registered once per store via ``register_script`` (EVALSHA with an
automatic EVAL fallback after a Redis restart flushes the script cache).
"""

from __future__ import annotations

# KEYS[1] = budget counter, ARGV[1] = limit, ARGV[2] = ttl seconds.
# Returns 1 if a slot was reserved (counter incremented), 0 if the limit is hit.
# The TTL is refreshed on every reservation: the counter must outlive the
# session, never the other way around.
BUDGET_CHECK_AND_INCR = """
local used = tonumber(redis.call('GET', KEYS[1]) or '0')
if used >= tonumber(ARGV[1]) then
  return 0
end
redis.call('INCR', KEYS[1])
redis.call('EXPIRE', KEYS[1], ARGV[2])
return 1
"""

# KEYS[1] = lock key, ARGV[1] = holder token.
# Returns 1 if the lock was held by this token and is now released, else 0.
RELEASE_LOCK_IF_HELD = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
  return redis.call('DEL', KEYS[1])
end
return 0
"""
