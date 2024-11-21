# Copyright (c) Microsoft Corporation.
# Modified by Oxford Quantum Circuits Ltd
# Licensed under the MIT License.

Include utils.ps1

Properties {
    $ProjectRoot = Resolve-Path (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
    $Root = Join-Path $ProjectRoot src
    $Docs = Join-Path $ProjectRoot docs
    $Examples = Join-Path $ProjectRoot examples
    $BuildLlvm = Join-Path $Root build-llvm
    $Rasqal = Join-Path $Root rasqal
    $Target = Join-Path $Root target
    $Wheels = Join-Path $Target wheels
    $CargoConfigToml = Join-Path $Root .cargo config.toml
    $AuditWheelTag = "manylinux_2_31_x86_64"
    $Python = Resolve-Python
}

task default -depends build
task build -depends build-llvm, build-rasqal, test-rasqal, run-examples, format
task check -depends check-licenses
task format -depends format-rust, format-python
task pypi-build -depends build, audit-rasqal, check

task format-python {
    Invoke-LoggedCommand -workingDirectory $Root {
        pip install ruff
        ruff format
        ruff check --fix --silent --exit-zero
    }
}

task format-rust {
    Invoke-LoggedCommand -workingDirectory $Root {
        cargo fmt --all
    }
}

task build-llvm -depends init {
    $env:LLVM_SYS_150_PREFIX = Resolve-InstallationDirectory
    Invoke-LoggedCommand -workingDirectory $BuildLlvm { cargo update }
    Invoke-LoggedCommand -workingDirectory $BuildLlvm { cargo build --release @(Get-CargoArgs) }
    Invoke-LoggedCommand -workingDirectory $BuildLlvm { cargo test --release @(Get-CargoArgs) }
}

task build-rasqal -depends init {
    # Copy over readme and license to embed in Python project.
    Copy-Item -Path (Join-Path $ProjectRoot README.md) -Destination $Rasqal -force
    Copy-Item -Path (Join-Path $ProjectRoot LICENSE) -Destination $Rasqal -force

    $env:MATURIN_PEP517_ARGS = (Get-CargoArgs) -Join " "
    Get-Wheels rasqal | Remove-Item -Verbose
    Invoke-LoggedCommand -workingDirectory $Rasqal { cargo update }
    Invoke-LoggedCommand { pip --verbose wheel --no-deps --wheel-dir $Wheels $Rasqal }
}

task audit-rasqal -depends build-rasqal {
    if ($IsLinux) {
        Invoke-LoggedCommand { & $Python -m pip install auditwheel patchelf }
    }
    if (Test-CommandExists auditwheel) {
       $unauditedWheels = Get-Wheels rasqal
       Invoke-LoggedCommand { auditwheel show $unauditedWheels }
       Invoke-LoggedCommand { auditwheel repair --wheel-dir $Wheels --plat $AuditWheelTag $unauditedWheels }
       $unauditedWheels | Remove-Item
    }
}

task test-rasqal -depends build-rasqal {
    # pyo3 has troubles with cargo test without excluding extension-module, so we do that here.
    Invoke-LoggedCommand -workingDirectory $Rasqal {
        cargo test --release @(Get-CargoArgs) --no-default-features
    }

    # Force reinstall the package if it exists, but not its dependencies.
    $packages = Get-Wheels rasqal
    Invoke-LoggedCommand -workingDirectory $Root {
        pip install $packages
        pip install --force-reinstall --no-deps $packages
    }

    # Run Python tests
    Invoke-LoggedCommand -workingDirectory $Root {
        pip install pytest
        pytest .
    }
}

task setup-examples -depends check-environment {
    # Install required dependencies.
    Invoke-LoggedCommand -workingDirectory $Root {
        pip install qsharp
    }
}

task run-examples -depends setup-examples {
    # Run our Python examples.
    Invoke-LoggedCommand -workingDirectory $Examples {
        python examples.py
        python sandbox.py
    }
}


task check-environment {
    $pyenv = Join-Path $Root ".env"
    if ((Test-Path -Path $pyenv) -eq $false) {
        Write-BuildLog "No virtual environment found."
        Write-BuildLog "Setting up virtual environment in $pyenv"
        & $Python -m venv $pyenv
    }
    else {
        Write-BuildLog "Virtual environment found."
    }

    if ($IsWindows) {
        Write-BuildLog "In Windows"
        . (Join-Path $pyenv Scripts Activate.ps1)
    }
    else {
        Write-BuildLog "Not in Windows"
        . (Join-Path $pyenv bin Activate.ps1)
    }

    $env_message = @(
        "Building LLVM requires a virtualenv or conda environment to build.",
        "Neither the VIRTUAL_ENV nor CONDA_PREFIX environment variables are set.",
        "See https://virtualenv.pypa.io/en/latest/index.html on how to use virtualenv"
    )
    Assert ((Test-InVirtualEnvironment) -eq $true) ($env_message -Join ' ')
}

task init -depends check-environment {
    # build-llvm has this logic built in when compiled on its own
    # but we must have LLVM installed prior to the wheels being built.

    # if an external LLVM is specified, make sure it exist and
    # skip further bootstapping
    if (Test-Path env:\RSQL_LLVM_EXTERNAL_DIR) {
        Use-ExternalLlvmInstallation
    }
    else {
        $packagePath = Resolve-InstallationDirectory
        if (Test-LlvmConfig $packagePath) {
            Write-BuildLog "LLVM target is already installed."
            # LLVM is already downloaded
            Use-LlvmInstallation $packagePath
        }
        else {
            Write-BuildLog "LLVM target is not installed."
			Write-BuildLog "Downloading LLVM Disabled, building from source."
			# We don't have an external LLVM installation specified
			# We are not downloading LLVM
			# So we need to build it.
			Invoke-Task "install-llvm-from-source"
		
            $installationDirectory = Resolve-InstallationDirectory
            Use-LlvmInstallation $installationDirectory
        }
    }
}

task install-llvm-from-source {
    if ($IsWindows) {
        Include vcvars.ps1
    }

    # TODO: Need to make prefix dynamic and also fix to not use env variables.
    $installationDirectory = Resolve-InstallationDirectory
    $env:LLVM_SYS_150_PREFIX = $installationDirectory
    $env:RSQL_CACHE_DIR = $installationDirectory

    install-llvm $BuildLlvm build (Get-LLVMFeatureVersion)
    Assert (Test-LlvmConfig $installationDirectory) "install-llvm-from-source failed to install a usable LLVM installation"
}

task check-licenses {
    # Uses cargo-deny to verify that the linked components
    # only use approved licenses
    # https://github.com/EmbarkStudios/cargo-deny
    Invoke-LoggedCommand -wd $Root {
        cargo install cargo-deny
        cargo deny check licenses
    }

    # patchelf is only pulled in for linux wheel patching, not directly referenced.
    Invoke-LoggedCommand -wd $Root {
        pip install pip-licenses
        pip-licenses --ignore-packages patchelf --allow-only="
        BSD;
        BSD License;
        BSD-3-Clause;
        Mozilla Public License 2.0 (MPL 2.0);
        Python Software Foundation License;
        Apache Software License;
        MIT License;
        MIT No Attribution License (MIT-0);
        MIT OR Apache-2.0;
        MIT;
        Public domain"
    }
}
