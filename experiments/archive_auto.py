"""Archive the current MULTI-auto results of the main experiment (metrics and
histories) under results/ablation/<name>.json before they are re-computed."""
import glob, json, os, sys
import common as C
name = sys.argv[1]
out = {}
for f in sorted(glob.glob(os.path.join(C.RESULTS, "main", "*.json"))):
    m = json.load(open(f))
    a = dict(m["MULTI-auto"]); a["sigma_hat"] = m["_problem"]["sigma_hat"]
    out[os.path.basename(f)[:-5]] = a
json.dump(out, open(os.path.join(C.RESULTS, "ablation", name + ".json"), "w"), indent=1, default=float)
print("archived", len(out), "->", name)
