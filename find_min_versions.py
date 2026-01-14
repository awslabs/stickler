#!/usr/bin/env python3
"""
Systematically find minimum working versions by testing progressively older versions.
Usage: python find_min_versions.py [--package PACKAGE] [--all]
"""
import subprocess
import sys
import re
import argparse
from packaging import version as pkg_version

PACKAGES = [
    "pydantic",
    "rapidfuzz", 
    "munkres",
    "numpy",
    "scipy",
    "psutil",
    "pandas",
    "jsonschema"
]

def run_cmd(cmd, check=True):
    """Run shell command."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        return None
    return result.stdout.strip()

def get_available_versions(package):
    """Get all available versions from PyPI."""
    print(f"Fetching versions for {package}...")
    output = run_cmd(f"pip index versions {package}", check=False)
    if not output:
        return []
    
    versions = []
    for line in output.split('\n'):
        if 'Available versions:' in line:
            version_str = line.split('Available versions:')[1].strip()
            versions = [v.strip() for v in version_str.split(',')]
            break
    
    # Parse and sort versions
    parsed = []
    for v in versions:
        try:
            parsed.append(pkg_version.parse(v))
        except:
            continue
    
    parsed.sort()
    return [str(v) for v in parsed]

def test_version(package, ver):
    """Test if tests pass with specific version."""
    print(f"\n  Testing {package}=={ver}...", end=" ", flush=True)
    
    # Install version
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", f"{package}=={ver}"],
        capture_output=True
    )
    if result.returncode != 0:
        print("❌ (install failed)")
        return False
    
    # Run tests
    result = subprocess.run(
        ["pytest", "tests/", "-x", "-q", "--tb=no"],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("✅")
        return True
    else:
        print("❌")
        return False

def find_minimum_version(package, current_min_major):
    """Binary search to find minimum working version."""
    print(f"\n{'='*60}")
    print(f"Finding minimum version for {package}")
    print(f"Current constraint starts at major version {current_min_major}")
    print('='*60)
    
    versions = get_available_versions(package)
    if not versions:
        print(f"⚠️  Could not fetch versions")
        return None
    
    # Filter to versions in the major version range we care about
    major_versions = [v for v in versions if v.startswith(f"{current_min_major}.")]
    
    if not major_versions:
        print(f"⚠️  No versions found for major version {current_min_major}")
        return None
    
    print(f"Found {len(major_versions)} versions in {current_min_major}.x range")
    print(f"Range: {major_versions[0]} to {major_versions[-1]}")
    
    # Binary search
    left, right = 0, len(major_versions) - 1
    min_working = None
    
    while left <= right:
        mid = (left + right) // 2
        ver = major_versions[mid]
        
        if test_version(package, ver):
            min_working = ver
            right = mid - 1  # Try older versions
        else:
            left = mid + 1  # Need newer version
    
    if min_working:
        print(f"\n✅ Minimum working version: {min_working}")
    else:
        print(f"\n❌ No working version found in {current_min_major}.x range")
    
    return min_working

def get_current_constraints():
    """Parse current version constraints from pyproject.toml."""
    with open("pyproject.toml") as f:
        content = f.read()
    
    constraints = {}
    for package in PACKAGES:
        pattern = rf'"{package}>=(\d+)\.\d+\.\d+,<(\d+)\.\d+\.\d+"'
        match = re.search(pattern, content)
        if match:
            constraints[package] = {
                'min_major': int(match.group(1)),
                'max_major': int(match.group(2))
            }
    
    return constraints

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", help="Test specific package")
    parser.add_argument("--all", action="store_true", help="Test all packages")
    
    args = parser.parse_args()
    
    constraints = get_current_constraints()
    
    if args.package:
        if args.package not in PACKAGES:
            print(f"Unknown package. Choose from: {', '.join(PACKAGES)}")
            return
        
        pkg = args.package
        min_ver = find_minimum_version(pkg, constraints[pkg]['min_major'])
        
        if min_ver:
            max_major = constraints[pkg]['max_major']
            print(f"\n📋 Suggested constraint: {pkg}>={min_ver},<{max_major}.0.0")
    
    elif args.all:
        results = {}
        
        for pkg in PACKAGES:
            min_ver = find_minimum_version(pkg, constraints[pkg]['min_major'])
            results[pkg] = min_ver
            
            # Reinstall latest to avoid conflicts
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", "--upgrade", pkg],
                capture_output=True
            )
        
        print("\n" + "="*60)
        print("RESULTS - Suggested version constraints:")
        print("="*60)
        
        for pkg in PACKAGES:
            min_ver = results[pkg]
            max_major = constraints[pkg]['max_major']
            if min_ver:
                print(f'{pkg:15} >= {min_ver},<{max_major}.0.0')
            else:
                print(f'{pkg:15} ❌ Could not determine')
    
    else:
        print("Usage:")
        print("  --package NAME    Find minimum version for specific package")
        print("  --all             Find minimum versions for all packages")
        print("\nExample: python find_min_versions.py --package pydantic")

if __name__ == "__main__":
    main()
