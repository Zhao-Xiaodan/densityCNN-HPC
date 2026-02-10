# PBS Queue Error Troubleshooting

## Error: "Queue is not enabled"

```bash
$ qsub pbs_calibration_experimental_study_DEBUG.sh
qsub: Queue is not enabled
```

---

## What This Means

The PBS queue is **temporarily disabled**, usually due to:
- 🔧 System maintenance
- ⚠️ Scheduler issues
- 🔄 Queue reconfiguration
- 🚫 Administrative restrictions

---

## Immediate Troubleshooting Steps

### 1. Check Queue Status

```bash
# Check all queues and their status
qstat -Q

# Expected output:
# Queue              Max   Tot Ena Str   Que   Run   Hld   Wat   Trn   Ext Type
# ---------------- ----- ----- --- --- ----- ----- ----- ----- ----- ----- ----
# workq                0     0 yes yes     0     0     0     0     0     0 Exec
# gpu                  0     0 yes yes     0     0     0     0     0     0 Exec
```

**Look for**:
- `Ena` column = `yes` (queue is enabled)
- `Str` column = `yes` (queue can start jobs)

### 2. Check Queue Details

```bash
# Get detailed queue information
qstat -Qf

# or for specific queue
qmgr -c "list queue workq"
```

### 3. Check System Status

```bash
# Check PBS server status
qstat -B

# Check if there are any running jobs
qstat -a

# Check your jobs specifically
qstat -u phyzxi
```

### 4. Check for Maintenance Announcements

```bash
# Look for system messages
cat /etc/motd

# Check if there's a maintenance file
ls -la /etc/nologin 2>/dev/null
```

---

## Solutions

### Option 1: Wait for Queue to be Enabled

If maintenance is ongoing:

```bash
# Check status periodically
watch -n 60 'qstat -Q'

# Or set up a simple loop
while true; do
    qstat -Q | grep -i "yes.*yes" && echo "Queue is up!" && break
    echo "Queue still disabled, checking again in 5 minutes..."
    sleep 300
done
```

### Option 2: Try Alternative Queue

Some HPC systems have multiple queues:

```bash
# List all available queues
qstat -Q

# Common queue names:
# - workq (default work queue)
# - gpu (GPU queue)
# - express (fast queue, shorter walltime)
# - batch (batch jobs)
# - debug (debugging queue, limited resources)
```

**Modify PBS script** to use a different queue:

```bash
# Add this line near the top of your PBS script:
#PBS -q gpu          # Specify GPU queue
# or
#PBS -q workq        # Specify work queue
# or
#PBS -q batch        # Specify batch queue
```

### Option 3: Check with HPC Support

```bash
# Email HPC support
# For Vanda cluster (NSCC Singapore):
# Email: help@nscc.sg
# Subject: PBS Queue Disabled - User phyzxi

# Or check HPC status page:
# https://www.nscc.sg/service-status
```

---

## Quick Fixes to Try

### Fix 1: Add Queue Specification

Create a modified PBS script with explicit queue:

```bash
# Edit the PBS script
nano pbs_calibration_experimental_study_DEBUG.sh

# Add this line after the other #PBS directives (around line 8):
#PBS -q gpu
```

Then resubmit:
```bash
qsub pbs_calibration_experimental_study_DEBUG.sh
```

### Fix 2: Check Your Account Status

```bash
# Verify your project allocation
sacct -u phyzxi 2>/dev/null || echo "Not using SLURM"

# Check PBS account
qstat -u phyzxi

# Verify you have access to submit jobs
qmgr -c "list queue workq" | grep "acl_users"
```

### Fix 3: Try Submitting to Different Queue Explicitly

```bash
# Submit with explicit queue specification
qsub -q gpu pbs_calibration_experimental_study_DEBUG.sh

# or
qsub -q workq pbs_calibration_experimental_study_DEBUG.sh

# or try default queue
qsub -q batch pbs_calibration_experimental_study_DEBUG.sh
```

---

## Modified PBS Script with Queue Specification

I'll create a version with explicit queue specification:

```bash
#!/bin/bash
#PBS -l walltime=48:00:00
#PBS -j oe
#PBS -k oed
#PBS -N Calibration_Experimental_Study_DEBUG
#PBS -l select=1:ncpus=36:mpiprocs=1:ompthreads=36:ngpus=1:mem=240gb
#PBS -M phyzxi@nus.edu.sg
#PBS -m abe
#PBS -q gpu              # ← ADD THIS LINE (try gpu, workq, or batch)

cd $PBS_O_WORKDIR
...
```

---

## Diagnostic Commands Summary

```bash
# 1. Check queue status
qstat -Q

# 2. Check server status
qstat -B

# 3. Check your jobs
qstat -u phyzxi

# 4. List all queues with details
qstat -Qf

# 5. Check if scheduler is running
ps aux | grep pbs

# 6. Check system messages
cat /etc/motd
```

---

## Common Queue States

| State | Meaning | Action |
|-------|---------|--------|
| `Ena=yes, Str=yes` | ✅ Queue accepting jobs | Submit normally |
| `Ena=no, Str=yes` | ⚠️ Queue disabled, jobs run | Wait or use alt queue |
| `Ena=yes, Str=no` | ⚠️ Queue enabled, jobs paused | Wait for restart |
| `Ena=no, Str=no` | ❌ Queue fully disabled | Wait or contact support |

---

## Expected Output When Working

```bash
$ qsub pbs_calibration_experimental_study_DEBUG.sh
555555.stdct-mgmt-02      # ← Job ID assigned successfully
```

---

## If Problem Persists

### Contact HPC Support

**For NSCC Vanda:**
- **Email**: help@nscc.sg
- **Phone**: +65 6419 1500
- **Portal**: https://user.nscc.sg

**Information to provide:**
- Username: phyzxi
- Error message: "qsub: Queue is not enabled"
- Output of: `qstat -Q`
- Output of: `qstat -B`
- Job script name: pbs_calibration_experimental_study_DEBUG.sh

### Alternative: Use SLURM (if available)

Some HPC systems use SLURM instead of PBS:

```bash
# Check if SLURM is available
which sbatch

# If yes, I can convert the PBS script to SLURM format
```

---

## Temporary Workaround: Run Locally (NOT RECOMMENDED)

If urgent and queue is down for extended period:

```bash
# Run directly (bypasses queue, uses current node - NOT IDEAL for GPU)
cd ~/scratch/densityCNN-HPC
bash pbs_calibration_experimental_study_DEBUG.sh
```

⚠️ **WARNING**: This runs on login node, may:
- Violate HPC policies
- Not have GPU access
- Be killed by admins
- Interfere with other users

**Only use as last resort for testing!**

---

## Next Steps

1. **Check queue status**: `qstat -Q`
2. **If maintenance**: Wait (check `/etc/motd` for timeline)
3. **If specific queue down**: Try adding `#PBS -q gpu` to script
4. **If persistent**: Contact help@nscc.sg

Once queue is enabled, resubmit:
```bash
qsub pbs_calibration_experimental_study_DEBUG.sh
```

---

**Date**: February 10, 2026
**Status**: Queue disabled (check with `qstat -Q`)
**Action**: Wait for queue to be re-enabled or contact HPC support
