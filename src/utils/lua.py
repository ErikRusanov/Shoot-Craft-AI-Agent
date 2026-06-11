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

# KEYS[1] = budget counter (micro-USD spent), ARGV[1] = estimate micro-USD,
# ARGV[2] = limit micro-USD, ARGV[3] = ttl seconds.
# Returns 1 if the estimate was reserved (counter += estimate), 0 if it would
# exceed the limit. Reserve-then-settle: the conservative estimate is added up
# front so concurrent reservations can never both slip past the last dollar; the
# caller settles the delta to the real cost afterwards via BUDGET_ADJUST.
# The TTL is refreshed on every reservation: the counter outlives the session.
BUDGET_RESERVE = """
local used = tonumber(redis.call('GET', KEYS[1]) or '0')
local estimate = tonumber(ARGV[1])
if used + estimate > tonumber(ARGV[2]) then
  return 0
end
redis.call('INCRBY', KEYS[1], estimate)
redis.call('EXPIRE', KEYS[1], ARGV[3])
return 1
"""

# KEYS[1] = budget counter (micro-USD spent), ARGV[1] = signed delta micro-USD,
# ARGV[2] = ttl seconds. Applies the delta (settle to actual cost, or a refund),
# floors the counter at zero so a refund can never drive it negative, and returns
# the new value. The floor matters because settle subtracts (actual < reserved
# estimate is the common case) and a cancel subtracts the whole estimate.
BUDGET_ADJUST = """
local v = tonumber(redis.call('GET', KEYS[1]) or '0') + tonumber(ARGV[1])
if v < 0 then v = 0 end
redis.call('SET', KEYS[1], v)
redis.call('EXPIRE', KEYS[1], ARGV[2])
return v
"""

# KEYS[1] = lock key, ARGV[1] = holder token.
# Returns 1 if the lock was held by this token and is now released, else 0.
RELEASE_LOCK_IF_HELD = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
  return redis.call('DEL', KEYS[1])
end
return 0
"""
