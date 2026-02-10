# Queue Access Denied - Troubleshooting

## Error: "Access to queue is denied"

```bash
$ qsub pbs_calibration_experimental_study_DEBUG.sh
qsub: Access to queue is denied
```

**This is different from "Queue is not enabled"!**

This means:
- ✅ Queue IS working
- ❌ Your account doesn't have permission to use this queue

---

## Root Cause

The `#PBS -q gpu` directive I added is **causing the access denial** because:
1. Your account may not have GPU queue access
2. GPU queue requires special allocation/permission
3. You were using the default queue before (which worked)

---

## IMMEDIATE FIX: Remove Queue Specification

Let PBS use the **default queue** (which you've been using successfully):

### Option 1: Remove the Line (Recommended)

Edit the PBS script and **delete or comment out** line 8:

```bash
# DELETE OR COMMENT THIS LINE:
#PBS -q gpu

# Or change to:
##PBS -q gpu    (commented out)
```

### Option 2: Try Default Queue Explicitly

Change line 8 to:
```bash
#PBS -q workq    # Default work queue
```

---

## Quick Fix Commands (On HPC)

```bash
cd ~/scratch/densityCNN-HPC

# Method 1: Submit without queue specification
sed '/#PBS -q gpu/d' pbs_calibration_experimental_study_DEBUG.sh > pbs_temp.sh
qsub pbs_temp.sh

# Method 2: Comment out the queue line
sed 's/^#PBS -q gpu$/##PBS -q gpu/' pbs_calibration_experimental_study_DEBUG.sh > pbs_temp.sh
qsub pbs_temp.sh

# Method 3: Submit with workq explicitly
qsub -q workq pbs_calibration_experimental_study_DEBUG.sh

# Method 4: Submit without any queue specification (let PBS choose)
qsub pbs_calibration_experimental_study_DEBUG.sh
# (but remove #PBS -q gpu from script first)
```

---

## Diagnostic Commands

### 1. Check Which Queues You Can Access

```bash
# List queues and check access
qstat -Q

# Check queue ACLs (access control lists)
qstat -Qf gpu | grep -i acl

# Check if you're in the allowed users list
qstat -Qf gpu | grep -A 5 "acl_users"

# Check your groups
groups

# Check default queue
qmgr -c "list server default_queue"
```

### 2. Check Your Previous Successful Jobs

```bash
# Look at your previous successful jobs to see what queue they used
qstat -xf $(qstat -xu phyzxi | tail -1 | awk '{print $1}') | grep queue

# Or check output files from previous runs
grep "Queue" Calibration_Experimental_Study_DEBUG.o* 2>/dev/null
```

### 3. Check Available Queues for Your Project

```bash
# Check your project allocation
qstat -Qf | grep -E "(Queue:|acl_user_enable|acl_users)"

# Check server defaults
qstat -Bf | grep default
```

---

## Solution: Use Default Queue

Your **previous successful jobs** (like job 555208) worked because they **didn't specify a queue**.

### Fixed PBS Script (Remove Queue Line)

I'll create a version without the queue specification:

```bash
#!/bin/bash
#PBS -l walltime=48:00:00
#PBS -j oe
#PBS -k oed
#PBS -N Calibration_Experimental_Study_DEBUG
#PBS -l select=1:ncpus=36:mpiprocs=1:ompthreads=36:ngpus=1:mem=240gb
#PBS -M phyzxi@nus.edu.sg
#PBS -m abe
# NO #PBS -q line - use default queue

cd $PBS_O_WORKDIR
...
```

---

## Why This Happened

1. **Before**: Your original job 555208 had **no** `#PBS -q` line → Used default queue → Worked
2. **I added**: `#PBS -q gpu` to try to fix the "queue not enabled" error
3. **Now**: PBS denies access because your account doesn't have GPU queue permissions
4. **Fix**: Remove `#PBS -q gpu` and use default queue (which assigns GPU automatically based on `ngpus=1`)

---

## Understanding PBS Queue vs GPU Access

**Important distinction**:
- `#PBS -l select=...ngpus=1` ← Requests GPU hardware ✅ (you have access)
- `#PBS -q gpu` ← Requests GPU queue ❌ (you don't have access)

**You don't need `#PBS -q gpu`** because:
- Your resource request already includes `ngpus=1`
- PBS will automatically assign you to a GPU-capable node
- The default queue handles GPU jobs fine

---

## Quick Commands Summary

```bash
# On HPC, try these in order:

# 1. Check what queue you should use
qmgr -c "list server default_queue"

# 2. Remove the queue line and resubmit
cd ~/scratch/densityCNN-HPC
cp pbs_calibration_experimental_study_DEBUG.sh pbs_calibration_experimental_study_DEBUG_NOQUEUE.sh
nano pbs_calibration_experimental_study_DEBUG_NOQUEUE.sh
# (Delete or comment line 8: #PBS -q gpu)
qsub pbs_calibration_experimental_study_DEBUG_NOQUEUE.sh

# OR quick one-liner:
grep -v "^#PBS -q gpu" pbs_calibration_experimental_study_DEBUG.sh > pbs_temp.sh && qsub pbs_temp.sh
```

---

## Check Previous Working Jobs

```bash
# Find a previous successful job
qstat -xu phyzxi | grep -E "555208|Completed"

# Check what queue it used
qstat -xf 555208 | grep "queue ="

# Expected: queue = workq  (or similar, but NOT gpu)
```

---

## Alternative: Request GPU Queue Access

If you **need** the GPU queue specifically, contact support:

**Email**: help@nscc.sg

**Request**:
```
Subject: GPU Queue Access Request - User phyzxi

Dear NSCC Support,

I need access to the GPU queue for my project.

Current error: "qsub: Access to queue is denied" when using #PBS -q gpu

Account: phyzxi
Project: [your project name]
Justification: Deep learning training requiring GPU resources

Note: I can successfully submit jobs with ngpus=1 to the default queue,
but would like explicit GPU queue access.

Thank you.
```

---

## What You Should Do NOW

**Best solution** (works immediately):

```bash
# On HPC
cd ~/scratch/densityCNN-HPC

# Create version without queue specification
grep -v "^#PBS -q" pbs_calibration_experimental_study_DEBUG.sh > pbs_calibration_experimental_study_DEBUG_fixed.sh

# Submit
qsub pbs_calibration_experimental_study_DEBUG_fixed.sh

# Monitor
tail -f Calibration_Experimental_Study_DEBUG.o*
```

This should work because:
- ✅ You've submitted jobs before (555208 worked)
- ✅ Your resource request includes `ngpus=1` (GPU hardware access)
- ✅ Default queue can handle GPU jobs
- ❌ You just don't have explicit "gpu" queue access

---

## Summary

| Issue | Cause | Fix |
|-------|-------|-----|
| "Queue is not enabled" | Queue disabled | Wait or contact support |
| "Access to queue is denied" | No permission for GPU queue | **Remove `#PBS -q gpu` line** |

**Action**: Remove the `#PBS -q gpu` line and resubmit!

---

**Date**: February 10, 2026
**Status**: Ready to fix - remove queue specification
**Expected**: Job should submit successfully without `#PBS -q` line
