#Requires -Version 5.1
##===- utils/fetch-hrx-release.ps1 --------------------------*- Script -*-===##
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##
#
# Windows-native port of utils/fetch-hrx-release.sh.
#
# Download, checksum-verify, and extract the pinned HRX (amdxdna) release so the
# HRX runtime path has a libhrx (hrx.dll) to dispatch through -- without
# cloning/building HRX from source. Both HRX entry points consume the same
# library provisioned here: the IRON/Python flow selects it with
# IRON_RUNTIME=hrx and the C++ example `make` flow selects it with RUNTIME=hrx.
#
# Usage:
#   ./utils/fetch-hrx-release.ps1              # fetch + extract (idempotent)
#   ./utils/fetch-hrx-release.ps1 -PrintEnv    # print the path to env.ps1
#   . (./utils/fetch-hrx-release.ps1)          # fetch, then dot-source env.ps1
#
# The pinned coordinates live in utils/hrx-release.env; any of
# HRX_RELEASE_{REPO,TAG,ASSET_WINDOWS,SHA256_WINDOWS} may be overridden from the
# environment. Set HRX_RELEASE_DIR to change where the asset is unpacked
# (default: third_party/.hrx-release). Pass -Force to re-download over an
# existing extraction.
#
# Auth: the release repo is private, so a token is needed. The script prefers
# `gh release download` (which honors the machine's gh auth / GH_TOKEN);
# otherwise it falls back to Invoke-WebRequest and, if set, sends
# GH_TOKEN/GITHUB_TOKEN as a bearer token.
#
# On success it prints the absolute path to the extracted tree's env.ps1
# (dot-source it) which sets HRX_DIR / PATH / CMAKE_PREFIX_PATH / LIBHRX_DIR.
#
##===----------------------------------------------------------------------===##

[CmdletBinding()]
param([switch]$PrintEnv, [switch]$Force)

$ErrorActionPreference = "Stop"

# Human-facing progress goes to the information/host stream (stderr-equivalent)
# so the single stdout line stays the env.ps1 path, mirroring the bash script's
# stdout/stderr split for `. (fetch-hrx-release.ps1)` composition.
function Log($msg) { Write-Host "[fetch-hrx-release] $msg" }
function Die($msg) { throw "[fetch-hrx-release] ERROR: $msg" }

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot  = Split-Path -Parent $ScriptDir

# --- parse utils/hrx-release.env (KEY="value") ---
$envmap = @{}
Get-Content (Join-Path $ScriptDir "hrx-release.env") | ForEach-Object {
  if ($_ -match '^\s*([A-Za-z0-9_]+)\s*=\s*"?([^"#]*)"?\s*$') {
    $envmap[$matches[1]] = $matches[2].Trim()
  }
}

# Environment overrides win over the pinned defaults.
function Val([string]$key, [string]$fallback) {
  $envVal = [Environment]::GetEnvironmentVariable($key)
  if ($envVal) { return $envVal }
  if ($envmap.ContainsKey($key) -and $envmap[$key]) { return $envmap[$key] }
  return $fallback
}

$repo  = Val 'HRX_RELEASE_REPO'  $null
$tag   = Val 'HRX_RELEASE_TAG'   $null
$asset = Val 'HRX_RELEASE_ASSET_WINDOWS' $null
$sha   = (Val 'HRX_RELEASE_SHA256_WINDOWS' $null)
if (-not $repo)  { Die "HRX_RELEASE_REPO not set" }
if (-not $tag)   { Die "HRX_RELEASE_TAG not set" }
if (-not $asset) { Die "HRX_RELEASE_ASSET_WINDOWS not set in hrx-release.env" }
if (-not $sha)   { Die "HRX_RELEASE_SHA256_WINDOWS not set in hrx-release.env" }
$sha = $sha.ToLower()

# Fetched artifacts live under third_party/ (a build artifact, git-ignored),
# next to the other vendored third-party dependencies.
$destDir = if ($env:HRX_RELEASE_DIR) { $env:HRX_RELEASE_DIR } `
           else { Join-Path $RepoRoot "third_party/.hrx-release" }
New-Item -ItemType Directory -Force -Path $destDir | Out-Null

$zip        = Join-Path $destDir $asset
$stem       = [IO.Path]::GetFileNameWithoutExtension($asset)   # strip .zip
$extractDir = Join-Path $destDir $stem
$envPs1     = Join-Path $extractDir "env.ps1"

function Emit-Env {
  if ($PrintEnv) { Write-Output $envPs1 }
  else {
    Log "HRX release ready. Dot-source its environment with:"
    Log "    . `"$envPs1`""
    Write-Output $envPs1
  }
}

# Idempotent: reuse an existing good extraction unless -Force.
if ((Test-Path $envPs1) -and -not $Force) {
  Log "Already extracted at $extractDir (use -Force to refetch)."
  Emit-Env
  return
}

# --- download (gh preferred, else Invoke-WebRequest) ---
if ((-not (Test-Path $zip)) -or $Force) {
  Log "Downloading $asset from $repo@$tag ..."
  $ok = $false
  $gh = Get-Command gh -ErrorAction SilentlyContinue
  if ($gh) {
    & gh release download $tag --repo $repo --pattern $asset --dir $destDir --clobber
    $ok = ($LASTEXITCODE -eq 0)
    if (-not $ok) { Log "gh download failed; falling back to Invoke-WebRequest." }
  }
  if (-not $ok) {
    $url = "https://github.com/$repo/releases/download/$tag/$asset"
    $headers = @{}
    $token = if ($env:GH_TOKEN) { $env:GH_TOKEN } else { $env:GITHUB_TOKEN }
    if ($token) { $headers['Authorization'] = "Bearer $token" }
    for ($i = 1; $i -le 3; $i++) {
      try { Invoke-WebRequest -Uri $url -Headers $headers -OutFile $zip; $ok = $true; break }
      catch {
        if ($i -eq 3) {
          Die "download failed: $url (private release? set GH_TOKEN or install gh) -- $($_.Exception.Message)"
        }
        Start-Sleep -Seconds ($i * 5)
      }
    }
  }
} else {
  Log "Reusing cached $zip."
}

# --- checksum ---
$actual = (Get-FileHash $zip -Algorithm SHA256).Hash.ToLower()
if ($actual -ne $sha) { Die "checksum mismatch for ${asset}: got $actual, expected $sha" }
Log "Checksum OK ($sha)."

# --- extract ---
Log "Extracting into $destDir ..."
if (Test-Path $extractDir) { Remove-Item -Recurse -Force $extractDir }
Expand-Archive -Path $zip -DestinationPath $destDir -Force

# The Windows asset unpacks to a single top-level folder named after the asset
# stem (matching the Linux tarball convention). If a future asset unpacks
# differently, normalize by walking up from include/hrx/hrx_runtime.h so
# $extractDir is always the relocatable install prefix.
if (-not (Test-Path (Join-Path $extractDir "include\hrx\hrx_runtime.h"))) {
  $hdr = Get-ChildItem -Recurse $destDir -Filter hrx_runtime.h -File -ErrorAction SilentlyContinue |
         Select-Object -First 1
  if (-not $hdr) { Die "hrx_runtime.h not found after extraction of $asset" }
  # include/hrx/hrx_runtime.h -> prefix is two levels up from the header dir.
  $extractDir = (Resolve-Path (Join-Path $hdr.Directory.FullName "..\..")).Path
  $envPs1 = Join-Path $extractDir "env.ps1"
}

# --- find the DLL dir (bin first, then lib) ---
$dll = $null
foreach ($sub in @("bin", "lib")) {
  $cand = Get-ChildItem -Path (Join-Path $extractDir $sub) -Include "hrx.dll","libhrx.dll" `
            -File -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($cand) { $dll = $cand; break }
}
if (-not $dll) {
  $dll = Get-ChildItem -Recurse $extractDir -Include "hrx.dll","libhrx.dll" -File `
           -ErrorAction SilentlyContinue | Select-Object -First 1
}
$dllDir = if ($dll) { $dll.Directory.FullName } else { Join-Path $extractDir "bin" }
if (-not $dll) { Log "WARNING: no hrx.dll/libhrx.dll found under $extractDir; defaulting to $dllDir" }

# --- synthesize env.ps1 for the install-prefix layout ---
# $root is resolved at dot-source time so the tree stays relocatable; the DLL
# dir is baked in relative to $root for the same reason.
$dllRel = $dllDir.Substring($extractDir.Length).TrimStart('\','/')
@"
# Auto-generated by fetch-hrx-release.ps1 for the install-prefix release layout.
`$root = Split-Path -Parent `$MyInvocation.MyCommand.Path
`$env:HRX_DIR = `$root
`$env:CMAKE_PREFIX_PATH = "`$root;`$env:CMAKE_PREFIX_PATH"
`$_hrx_dll_dir = Join-Path `$root "$dllRel"
`$env:PATH = "`$_hrx_dll_dir;`$env:PATH"
# Hints for the Python discovery once it learns to look for a .dll on Windows.
`$env:LIBHRX_DIR = `$_hrx_dll_dir
`$env:HRX_LIBHRX = Join-Path `$_hrx_dll_dir "$($dll.Name)"
"@ | Set-Content -Encoding UTF8 $envPs1

Emit-Env
