import os
import subprocess
import sys

def main():
    cmd = [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"]
    print(f"Running: {' '.join(cmd)}")
    
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )
    
    print(proc.stdout)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)
        
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        try:
            with open(summary_path, "a", encoding="utf-8") as f:
                f.write("### SubNeutralize CI Test Run Output\n\n```text\n")
                # Write the trailing summary / failures
                out = proc.stdout
                if len(out) > 8000:
                    out = out[-8000:]
                f.write(out)
                if proc.stderr:
                    f.write("\n--- STDERR ---\n")
                    f.write(proc.stderr[-2000:] if len(proc.stderr) > 2000 else proc.stderr)
                f.write("\n```\n")
        except Exception as e:
            print(f"Warning: Failed to write to GITHUB_STEP_SUMMARY: {e}", file=sys.stderr)

    sys.exit(proc.returncode)

if __name__ == "__main__":
    main()
