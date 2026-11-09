import capstone, subprocess, bisect, sys, re

data = open('/tmp/rb2.dylib','rb').read()
base = 0x100000000
md = capstone.Cs(capstone.CS_ARCH_ARM64, capstone.CS_MODE_ARM)

# symbols
syms = []
for line in subprocess.run(["nm","-arch","arm64","/tmp/rb2.dylib"],
                           capture_output=True, text=True).stdout.splitlines():
    p = line.split()
    if len(p) == 3:
        try: syms.append((int(p[0],16), p[2]))
        except ValueError: pass
syms.sort()
addrs = [a for a,_ in syms]
cache = {}
def name(a):
    if a in cache: return cache[a]
    i = bisect.bisect_right(addrs, a) - 1
    n = syms[i][1] if i >= 0 else "?"
    n = subprocess.run(["c++filt", n], capture_output=True, text=True).stdout.strip()
    cache[a] = n
    return n

# one linear pass, attributing BL targets to the enclosing symbol
edge = {}
lo, hi = 0x100f00000, 0x100f70000     # the analyser region
cur = None
for ins in md.disasm(data[lo-base:hi-base], lo):
    i = bisect.bisect_right(addrs, ins.address) - 1
    cur = syms[i][0] if i >= 0 else None
    if ins.mnemonic == "bl":
        try: t = int(ins.op_str.replace("#",""),16)
        except ValueError: continue
        edge.setdefault(cur, set()).add(t)

def dfs(root, maxdepth=6, seen=None, depth=0, out=None):
    if seen is None: seen = set()
    if out is None: out = []
    if root in seen or depth > maxdepth: return out
    seen.add(root)
    kids = sorted(edge.get(root, ()))
    out.append((depth, root, len(kids)))
    for k in kids:
        if lo <= k <= hi:
            dfs(k, maxdepth, seen, depth+1, out)
    return out

roots = [0x100f32bb0, 0x100f32f60]   # Analyzer::setSample, Analyzer::analyze
for r in roots:
    tree = dfs(r)
    print(f"\n=== DFS from {hex(r)} {name(r)[:70]}  ({len(tree)} reachable functions)")
    for depth, f, nk in tree:
        print("   " + "  "*depth + f"{name(f)[:74]}  (callees in region: {nk})")
