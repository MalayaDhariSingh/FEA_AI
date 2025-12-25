import os
import random
import time
import glob # Helps find files

# 1. SETUP
# Use the correct path you found earlier
abaqus_cmd = r"C:\SIMULIA\Commands\abaqus.bat"

# Clean start
if os.path.exists("data_log.txt"):
    os.remove("data_log.txt")

with open('data_log.txt', 'w') as f:
    f.write("Radius,Load,MaxStress\n")

print("Starting CLEAN Data Generation Pipeline...")

# 2. THE LOOP (500 Samples)
for i in range(500):
    
    # Generate Parameters
    r = round(random.uniform(5.0, 15.0), 2)
    l = round(random.uniform(50.0, 200.0), 2)
    
    print(f"--- Sim {i+1}/500: Radius={r}, Load={l} ---")
    
    # Run Abaqus
    command = f"\"{abaqus_cmd}\" cae noGUI=worker.py -- {r} {l}"
    os.system(command)
    
    # 3. THE JANITOR (Cleanup Logic)
    # The worker names jobs like: Sim_R{int}_L{int}
    # We reconstruct that name to find the files
    job_base_name = f"Sim_R{int(r)}_L{int(l)}"
    
    # Find all files starting with this name (odb, dat, msg, etc.)
    # and DELETE them
    garbage_files = glob.glob(f"{job_base_name}.*")
    
    for file in garbage_files:
        try:
            os.remove(file)
        except:
            pass # Sometimes Windows locks a file for a ms, just skip it
            
    print(f"   (Cleaned up {len(garbage_files)} temp files)")

print("Batch Finished. Check data_log.txt!")