#!/usr/bin/env python3
"""Route raw sub-tile DRAM reads through a Blackhole-congruent path.

tt-metal requires (l1_addr % A) == (dram_addr % A) for DRAM reads, with
A = NOC_DRAM_READ_ALIGNMENT_BYTES: 64 on Blackhole, 32 on Wormhole. The
generated vector loads put 32-byte chunks whose DRAM offsets are 32 apart into
L1 slots >=512 apart, so the phases agree mod 32 but not mod 64.
"""
import re, sys
HELPER = r'''
template <typename ACC>
inline void loom_read_congruent(const ACC &acc, uint32_t page, uint32_t off,
                                uint32_t l1_dst, uint32_t len) {
  uint64_t direct = acc.get_noc_addr(page, off);
  uint32_t sp = ((uint32_t) direct) & 63u;
  uint32_t dp = l1_dst & 63u;
  if (sp == dp) { noc_async_read(direct, l1_dst, len); return; }
  uint32_t span = (sp + len + 63u) & ~63u;
  uint32_t base_l1 = l1_dst - dp;
  noc_async_read(direct - sp, base_l1, span);
  noc_async_read_barrier();
  volatile uint8_t *b = (volatile uint8_t *) ((uintptr_t) base_l1);
  volatile uint8_t *d = (volatile uint8_t *) ((uintptr_t) l1_dst);
  for (uint32_t i = 0; i < len; i += 1) { d[i] = b[sp + i]; }
}
'''
p = sys.argv[1]; s = open(p).read()
sites = {m.group(1): (m.group(2), m.group(3), m.group(4)) for m in
         re.finditer(r"uint64_t (temp_\d+) = (v\d+)\.get_noc_addr\(([^,]+), ([^)]+)\);", s)}
def repl(m):
    t, dst, ln = m.groups()
    if t not in sites: return m.group(0)
    acc, page, off = sites[t]
    return f"loom_read_congruent({acc}, {page}, {off}, {dst}, {ln});"
s, n = re.subn(r"noc_async_read\((temp_\d+), (\w+), (\w+)\);", repl, s)
s = s.replace("void kernel_main() {", HELPER + "\nvoid kernel_main() {", 1)
open(p, "w").write(s); print(f"congruence shim applied at {n} raw DRAM reads")
