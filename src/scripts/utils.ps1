# Copyright (c) Microsoft Corporation.
# Modified by Oxford Quantum Circuits Ltd
# Licensed under the MIT License.

Properties {
    $llvm_releases_url = "https://github.com/llvm/llvm-project/releases"
    $feature2releaseprefix = @{ "llvm11-0" = "/download/llvmorg-11.0.0/clang+llvm-11.0.0-x86_64-linux-gnu-ubuntu-20.04.tar.xz";
                                "llvm12-0" = "/download/llvmorg-12.0.0/clang+llvm-12.0.0-x86_64-linux-gnu-ubuntu-20.04.tar.xz";
                                "llvm13-0" = "/download/llvmorg-13.0.0/clang+llvm-13.0.0-x86_64-linux-gnu-ubuntu-20.04.tar.xz";
                                "llvm14-0" = "/download/llvmorg-14.0.0/clang+llvm-14.0.0-x86_64-linux-gnu-ubuntu-18.04.tar.xz"
    }
}

if (!(Test-Path function:\Get-RepoRoot)) {
    # pin working directory to this repo in case
    # we are ever in a submodule
    function Get-RepoRoot {
        exec -workingDirectory $PSScriptRoot {
            git rev-parse --show-toplevel
        }
    }
}

# Fix temp path for non-windows platforms if missing
if (!(Test-Path env:\TEMP)) {
    $env:TEMP = [System.IO.Path]::GetTempPath()
}

####
# Utilities
####

# Writes an Azure DevOps message with default debug severity
function Write-BuildLog {
    param (
        [Parameter(Mandatory = $true)]
        [string]$message,
        [Parameter(Mandatory = $false)]
        [ValidateSet("group", "warning", "error", "section", "debug", "command", "endgroup")]
        [string]$severity = "debug"
    )
    Write-Host "##[$severity]$message"
}

# Returns true if a command with the specified name exists.
function Test-CommandExists($name) {
    $null -ne (Get-Command $name -ErrorAction SilentlyContinue)
}

# Sets the LLVM path in the env section of the .cargo/config.toml
# Configures vscode rust analyzer to the correct features
function Use-LlvmInstallation {
    param (
        [string]$path
    )
    Write-BuildLog "Setting LLVM installation to: $path"

    $llvm_config_options = @(Get-Command (Join-Path $path "bin" "llvm-config*"))
    Assert ($llvm_config_options.Length -gt 0) "llvm config not found in $path"

    $llvm_config = $llvm_config_options[0].Source
    Write-BuildLog "Found llvm-config : $llvm_config"

    $version = [Version]::Parse("$(&$llvm_config --version)")
    $prefix = "LLVM_SYS_$($version.Major)0_PREFIX"

    Write-BuildLog "Setting $prefix set to: $path"

    if ($IsWindows) {
        # we have to escape '\'
        $path = $path.Replace('\', '\\')
    }

    # Create the workspace cofig.toml and set the LLVM_SYS env var
    New-Item -ItemType File -Path $CargoConfigToml -Force
    Add-Content -Path $CargoConfigToml -Value "[env]"
    Add-Content -Path $CargoConfigToml -Value "$($prefix) = `"$($path)`""
}

function Test-LlvmConfig {
    param (
        [string]$path
    )

    $llvm_config_options = @(Get-Command (Join-Path $path "bin" "llvm-config*"))
    if ($llvm_config_options.Length -eq 0) {
        return $false
    }
    $llvm_config = $llvm_config_options[0].Source
    try {
        exec {
            & $llvm_config --version | Out-Null
        }
    }
    catch {
        return $false
    }
    return $true
}

function Resolve-InstallationDirectory {
    if (Test-Path env:\RSQL_LLVM_EXTERNAL_DIR) {
        return $env:RSQL_LLVM_EXTERNAL_DIR
    }
    else {
        $packagePath = Get-DefaultInstallDirectory
        return $packagePath
    }
}

function Get-DefaultInstallDirectory {
    if (Test-Path env:\RSQL_CACHE_DIR) {
        $env:RSQL_CACHE_DIR
    }
    else {
        Join-Path $Target (Get-LLVMFeatureVersion)
    }
}

# Executes the supplied script block using psake's exec
# Warning: Do not use this command on anything that contains
#          sensitive information!
function Invoke-LoggedCommand {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [scriptblock]$cmd,

        [string]$errorMessage = $null,

        [int]$maxRetries = 0,

        [string]$retryTriggerErrorPattern = $null,

        [Alias("wd")]
        [string]$workingDirectory = $null
    )

    Write-BuildLog "Invoke-LoggedCommand in $workingDirectory`:"
    Write-BuildLog $ExecutionContext.InvokeCommand.ExpandString($cmd).Trim() "command"

    # errorMessage pulls default values from psake. We
    # only want to pass the param if we want to override.
    # all other parameters have safe defaults.
    $extraArgs = $errorMessage ? @{ "errorMessage" = $errorMessage } : @{};
    exec $cmd @extraArgs `
        -maxRetries $maxRetries `
        -retryTriggerErrorPattern $retryTriggerErrorPattern `
        -workingDirectory $workingDirectory
}

function Use-ExternalLlvmInstallation {
    Write-BuildLog "Using LLVM installation specified by RSQL_LLVM_EXTERNAL_DIR"
    Assert (Test-Path $env:RSQL_LLVM_EXTERNAL_DIR) "RSQL_LLVM_EXTERNAL_DIR folder does not exist"
    Use-LlvmInstallation $env:RSQL_LLVM_EXTERNAL_DIR
}

function Test-AllowedToDownloadLlvm {
    # If RSQL_DOWNLOAD_LLVM isn't set, we don't allow for download
    # If it is set, then we use its value
    ((Test-Path env:\RSQL_DOWNLOAD_LLVM) -and ($env:RSQL_DOWNLOAD_LLVM -eq $true))
}

function Test-InCondaEnvironment {
    (Test-Path env:\CONDA_PREFIX)
}

function Test-InVenvEnvironment {
    (Test-Path env:\VIRTUAL_ENV)
}

function Test-InVirtualEnvironment {
    (Test-InCondaEnvironment) -or (Test-InVenvEnvironment)
}

function Get-LLVMFeatureVersion {
    if (Test-Path env:\RSQL_LLVM_FEATURE_VERSION) {
        $env:RSQL_LLVM_FEATURE_VERSION
    }
    else {
        # "llvm11-0", "llvm12-0", "llvm13-0", "llvm14-0"
        "llvm14-0"
    }
}

function Get-CargoArgs {
    @("-vv")
}

function Get-Wheels([string] $project) {
    $name = $project.Replace('-', '_')
    $pattern = Join-Path $Wheels $name-*.whl
    Get-Item -ErrorAction Ignore $pattern
}

function Get-Wheel([string] $project) {
    $wheels = @(Get-Wheels $project)
    Assert ($wheels.Length -gt 0) "Missing wheels for $project."
    Assert ($wheels.Length -le 1) "Multiple wheels for $project ($wheels). Clean the wheels directory."
    $wheels[0]
}

function Resolve-Python() {
    $hasPython = $null -ne (Get-Command python -ErrorAction Ignore)
    if ($hasPython -and ((python --version) -Match "Python 3.*")) {
        Write-BuildLog "Python"
        "python"
    }
    else {
        Write-BuildLog "Python 3"
        "python3"
    }
}

function install-llvm {
    Param(
        [Parameter(Mandatory)]
        [string]$buildllvmDir, # root directory of the Rust `build-llvm` module
        [Parameter(Mandatory)]
        [ValidateSet("download", "build")]
        [string]$operation,
        [Parameter(Mandatory)]
        [ValidateSet("llvm11-0", "llvm12-0", "llvm13-0", "llvm14-0")]
        [string]$feature
    )

    $llvm_release = "$llvm_releases_url/$($feature2releaseprefix[$feature])"
    $llvm_release_file = $llvm_release.split('/')[-1]

    $installationDirectory = Resolve-InstallationDirectory
    Write-BuildLog "installationDirectory: $installationDirectory"
    New-Item -ItemType Directory -Force $installationDirectory | Out-Null
    if (($operation -eq "download")) {
        if ($IsWindows) {
            if (!(Test-Path -Path "$installationDirectory/bin" -PathType Leaf)) {
                Write-BuildLog "Extracting LLVM binaries under $installationDirectory"
                7z x -y "$buildllvmDir/zipped/$feature.7z" -o"$installationDirectory"
            } else {
                Write-BuildLog "Already extracted LLVM binaries"
            }
        } else {
            if (!(Test-Path -Path "$installationDirectory/$llvm_release_file" -PathType Leaf)) {
                Invoke-WebRequest -Uri "$llvm_release" -OutFile "$installationDirectory/$llvm_release_file"
            } else {
                Write-BuildLog "Already downloaded pre-built LLVM binaries"
            }

            if (!(Test-Path -Path "$installationDirectory/bin" -PathType Leaf)) {
                Write-BuildLog "Extracting LLVM binaries under $installationDirectory"
                tar -xvf "$installationDirectory/$llvm_release_file" -C $installationDirectory --strip-components=1
            } else {
                Write-BuildLog "Already extracted LLVM binaries"
            }
        }
    }
    elseif (($operation -eq "build")) {
        Invoke-LoggedCommand -wd $buildllvmDir {
            cargo build --release --no-default-features --features "$operation-llvm,$feature-no-llvm-linking" -vv
        }
    }
}
