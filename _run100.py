import subprocess, sys

# Clean partial rows from aborted batch
import csv
with open("logs/matches.csv") as f:
    rows = list(csv.DictReader(f))
clean = [r for r in rows if "sims=120" not in r.get("ai1","") + r.get("ai2","")]
with open("logs/matches.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(clean)
print(f"Cleaned. Ready. Base rows: {len(clean)}")

total = 0
for batch in range(10):
    start_id = batch * 10
    print(f"\n=== Batch {batch+1}/10 (matches {start_id+1}-{start_id+10}) ===")
    r = subprocess.run(
        [sys.executable, "src/scripts/compare.py",
         "--matches", "10",
         "--start-id", str(start_id),
         "--depth", "3",
         "--rl-model", "models/rl_agent.pth",
         "--mcts-sims", "120",
         "--c-puct", "1.4",
         "--num-res-blocks", "5",
         "--channels", "64"],
        capture_output=True, text=True, timeout=900)
    # Show last line of stdout
    out_lines = [l for l in (r.stdout or "").split("\n") if l.strip()]
    if out_lines:
        print("  Last:", out_lines[-1])
    if r.stderr:
        print("  ERR:", r.stderr[-100:])
    total += 10
    print(f"  Progress: {total}/100")

print(f"\nAll {total} matches complete!")
