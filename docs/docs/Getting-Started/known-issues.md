# Known Issues

---

## NumPy Installation Failure on RHEL/Amazon Linux with GCC < 9.3

### Steps to Reproduce

1. Attempt to install `stickler-eval` on Amazon Linux or RHEL-based system
2. System has GCC version < 9.3 (e.g., GCC 7.3.1)
3. Run: `pip install stickler-eval`

### Actual Behavior

Installation fails with error:
```
ERROR: Problem encountered: NumPy requires GCC >= 9.3
```

The build process attempts to compile NumPy 2.3.4 which requires GCC >= 9.3, but the system has an older GCC version (e.g., GCC 7.3.1 on Amazon Linux).

### Expected Behavior

`stickler-eval` should install successfully without compilation errors.

### Root Cause

NumPy 2.3.4+ requires GCC >= 9.3 for compilation. Older RHEL-based systems (including SageMaker Jupyter Notebooks Classic on Amazon Linux) ship with GCC 7.3.1, which is incompatible with recent NumPy versions.

### Workaround

Install on a system with GCC >= 9.3, such as:
- SageMaker Studio notebooks (Ubuntu 22.04 with GCC 11.4.0)
- Ubuntu-based environments
- Update GCC on the target system to version 9.3 or higher

---
