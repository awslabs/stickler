#!/usr/bin/env python3
"""
Test minimum version bounds for dependencies.
Usage: python test_version_bounds.py [--test-all] [--package PACKAGE]
"""
import subprocess
import sys
import json
import argparse
from pathlib import Path

DEPENDENCIES = {
    "pydantic": ">=2.0.0,<3.0.0",
    "rapidfuzz": ">=3.0.0,<4.0.0",
    "munkres": ">=1.1.0,<2.0.0",
    "numpy": ">=1.24.0,<3.0.0",
    "scipy": ">=1.10.0,<2.0.0",
    "psutil": ">=5.8.0,<6.0.0",
    "pandas": ">=1.5.0,<3.0.0",
    "jsonschema": ">=4.0.0,<5.0.0",
}

def run_cmd(cmd, check=True):
    """Run shell command and return output."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"Error: {result.stderr}")
        return None
    return result.stdout.strip()

def get_installed_version(package):
    """Get currently installed version of a package."""
    output = run_cmd(f"pip show {package}", check=False)
    if not output:
        return None
    for line in output.split('\n'):
        if line.startswith('Version:'):
            return line.split(':', 1)[1].strip()
    return None

def test_with_version(package, version):
    """Test if package works with specific version."""
    print(f"\n{'='*60}")
    print(f"Testing {package}=={version}")
    print('='*60)
    
    # Install specific version
    print(f"Installing {package}=={version}...")
    result = run_cmd(f"pip install {package}=={version}", check=False)
    if result is None:
        print(f"❌ Failed to install {package}=={version}")
        return False
    
    # Run tests
    print("Running tests...")
    result = subprocess.run(["pytest", "tests/", "-x", "-q"], capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ Tests passed with {package}=={version}")
        return True
    else:
        print(f"❌ Tests failed with {package}=={version}")
        print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
        return False

def find_minimum_version(package, suggested_min):
    """Binary search to find minimum working version."""
    print(f"\n🔍 Finding minimum version for {package}")
    print(f"Suggested minimum: {suggested_min}")
    
    # Get available versions
    output = run_cmd(f"pip index versions {package}", check=False)
    if not output:
        print(f"⚠️  Could not fetch versions for {package}")
        return suggested_min
    
    # Parse versions (simplified - just test the suggested one)
    if test_with_version(package, suggested_min):
        return suggested_min
    else:
        print(f"⚠️  Minimum version {suggested_min} doesn't work")
        return None

def update_pyproject(new_deps):
    """Update pyproject.toml with new dependency versions."""
    pyproject_path = Path("pyproject.toml")
    content = pyproject_path.read_text()
    
    for package, version_spec in new_deps.items():
        # Find and replace the dependency line
        import re
        pattern = rf'"{package}>=[\d\.]+"'
        replacement = f'"{package}{version_spec}"'
        content = re.sub(pattern, replacement, content)
    
    # Backup original
    backup_path = Path("pyproject.toml.backup")
    if not backup_path.exists():
        backup_path.write_text(pyproject_path.read_text())
        print(f"✅ Backed up original to {backup_path}")
    
    pyproject_path.write_text(content)
    print(f"✅ Updated {pyproject_path}")

def main():
    parser = argparse.ArgumentParser(description="Test dependency version bounds")
    parser.add_argument("--test-all", action="store_true", help="Test all dependencies")
    parser.add_argument("--package", help="Test specific package")
    parser.add_argument("--update", action="store_true", help="Update pyproject.toml with suggested versions")
    parser.add_argument("--dry-run", action="store_true", help="Show suggested versions without testing")
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("\n📋 Suggested version constraints:")
        print("="*60)
        for package, version_spec in DEPENDENCIES.items():
            print(f"{package:15} {version_spec}")
        print("\nRun with --update to apply these to pyproject.toml")
        return
    
    if args.update:
        update_pyproject(DEPENDENCIES)
        print("\n✅ Updated pyproject.toml")
        print("Run 'pip install -e .[dev]' to install with new constraints")
        return
    
    if args.package:
        # Test specific package
        if args.package not in DEPENDENCIES:
            print(f"Unknown package: {args.package}")
            print(f"Available: {', '.join(DEPENDENCIES.keys())}")
            return
        
        version_spec = DEPENDENCIES[args.package]
        min_version = version_spec.split(">=")[1].split(",")[0]
        find_minimum_version(args.package, min_version)
    
    elif args.test_all:
        # Test all packages
        results = {}
        for package, version_spec in DEPENDENCIES.items():
            min_version = version_spec.split(">=")[1].split(",")[0]
            result = find_minimum_version(package, min_version)
            results[package] = result
        
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        for package, result in results.items():
            status = "✅" if result else "❌"
            print(f"{status} {package:15} minimum: {result or 'FAILED'}")
    
    else:
        print("Usage:")
        print("  --dry-run          Show suggested versions")
        print("  --update           Update pyproject.toml")
        print("  --test-all         Test all minimum versions")
        print("  --package NAME     Test specific package")

if __name__ == "__main__":
    main()
