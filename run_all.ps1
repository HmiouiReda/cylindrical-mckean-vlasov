param(
    [switch]$Quick,
    [switch]$Force,
    [switch]$Clean,
    [switch]$StopOnError
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# =============================================================================
# COMPLETE PAPER EXPERIMENT PLAN
# =============================================================================
# This is a fixed experiment list. There is NO clock-based cutoff, timeout,
# budget guard, or automatic early stopping in this PowerShell script.
#
# With the full JSON configurations, the plan is designed to be of the order
# of 12--18 hours on the laptop used for the existing experiments. Actual time
# depends on CPU, PyTorch build, thermal throttling, and background processes.
#
# Scientific coverage:
#   1. Main moment-functional results: 6 benchmarks x 5 independent seeds.
#   2. Quantile-and-tail stress test: all five 1D benchmarks at seed 0.
#   3. Single-trajectory training: OU, three seeds.
#   4. K_keep variance ablation: OU, seed 0.
#   5. Signature-depth ablation: double well, levels 1--3, three seeds.
#   6. Ito--Stratonovich consistency check: multiplicative benchmark.
#   7. Targeted sweeps on informative benchmarks:
#        J, Fourier features, learning rate, MDN depth, Psi depth.
#   8. Nonlinear depth sanity check: double well, seed 0.
#
# Default hyperparameter points are copied from the corresponding headline
# runs into the sweep folders. They are not retrained, because the data split,
# seed, configuration and model are exactly the same.
# =============================================================================

$OutputRoot = "outputs/paper_full"
$ManifestPath = Join-Path $OutputRoot "run_manifest.csv"
$TranscriptPath = Join-Path $OutputRoot "run_all_transcript.log"
$StartTime = Get-Date

if ($Clean -and (Test-Path $OutputRoot)) {
    Write-Host "[clean] Removing $OutputRoot"
    Remove-Item -Recurse -Force $OutputRoot
}
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

# Prefer the project virtual environment when it exists.
$PythonCandidates = @(
    (Join-Path $PSScriptRoot ".Mckean/Scripts/python.exe"),
    (Join-Path $PSScriptRoot ".venv/Scripts/python.exe")
)
$Python = "python"
foreach ($Candidate in $PythonCandidates) {
    if (Test-Path $Candidate) {
        $Python = $Candidate
        break
    }
}

$ModeName = if ($Quick) { "quick" } else { "full" }

# Reproducibility and protection against excessive BLAS thread contention.
$env:PYTHONHASHSEED = "0"
$env:OMP_NUM_THREADS = "2"
$env:MKL_NUM_THREADS = "2"
$env:OPENBLAS_NUM_THREADS = "2"
$env:NUMEXPR_NUM_THREADS = "2"

$HeadlineSeeds = @(0, 1, 2, 3, 4)
$SweepSeeds = @(0, 1, 2)
$AllBenchmarks = @("ou", "ou_uniform", "ou_gmm", "double_well", "multiplicative", "ou2d")
$Benchmarks1D = @("ou", "ou_uniform", "ou_gmm", "double_well", "multiplicative")

$JValuesToTrain = @(1, 2, 3, 8)              # J=5 is reused from headline.
$FourierValuesToTrain = @(1, 2, 6, 8)        # P=4 is reused from headline.
$LearningRateValuesToTrain = @(0.0001, 0.0003, 0.003) # 0.001 is reused.
$DepthValuesToTrain = @(1, 3, 4)              # depth=2 is reused.
$SignatureValuesToTrain = @(1, 2)             # level=3 is reused for double well.

$script:FailureCount = 0

function Add-ManifestRow {
    param(
        [string]$Group,
        [string]$Benchmark,
        [string]$Functional,
        [string]$Parameter,
        [string]$Value,
        [int]$Seed,
        [string]$OutDir,
        [string]$Status,
        [double]$RuntimeMinutes,
        [string]$StartedAt,
        [string]$FinishedAt,
        [string]$Message = ""
    )

    $row = [PSCustomObject]@{
        group = $Group
        benchmark = $Benchmark
        functional = $Functional
        parameter = $Parameter
        value = $Value
        seed = $Seed
        outdir = $OutDir
        status = $Status
        runtime_minutes = [Math]::Round($RuntimeMinutes, 3)
        started_at = $StartedAt
        finished_at = $FinishedAt
        message = $Message
    }

    if (Test-Path $ManifestPath) {
        $row | Export-Csv -Path $ManifestPath -Append -NoTypeInformation
    }
    else {
        $row | Export-Csv -Path $ManifestPath -NoTypeInformation
    }
}

function Handle-Failure {
    param([string]$Message)
    $script:FailureCount += 1
    Write-Warning $Message
    if ($StopOnError) {
        throw $Message
    }
}

function Invoke-Experiment {
    param(
        [string]$Group,
        [string]$Benchmark,
        [int]$Seed,
        [string]$OutDir,
        [string[]]$ExtraArgs,
        [string]$Functional = "moments",
        [string]$Parameter = "",
        [string]$Value = "",
        [string]$SuccessFile = "metrics.json"
    )

    $SuccessPath = Join-Path $OutDir $SuccessFile

    if ((Test-Path $SuccessPath) -and (-not $Force)) {
        Write-Host "[skip completed] $Group | $Benchmark | seed=$Seed | $Parameter=$Value"
        Add-ManifestRow -Group $Group -Benchmark $Benchmark -Functional $Functional `
            -Parameter $Parameter -Value $Value -Seed $Seed -OutDir $OutDir `
            -Status "already_complete" -RuntimeMinutes 0.0 `
            -StartedAt "" -FinishedAt (Get-Date).ToString("s")
        return $true
    }

    New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

    $CommandArgs = @(
        "main.py",
        "--benchmark", $Benchmark,
        "--seed", "$Seed",
        "--outdir", $OutDir
    ) + $ExtraArgs

    if ($Quick) {
        $CommandArgs += "--quick"
    }

    Write-Host ""
    Write-Host "================================================================"
    Write-Host "$Group | benchmark=$Benchmark | seed=$Seed | $Parameter=$Value"
    Write-Host "$Python $($CommandArgs -join ' ')"
    Write-Host "================================================================"

    $RunStart = Get-Date
    & $Python @CommandArgs 2>&1 | Tee-Object -FilePath (Join-Path $OutDir "console.log") | Out-Host
    $ExitCode = $LASTEXITCODE
    $RunEnd = Get-Date
    $Runtime = ($RunEnd - $RunStart).TotalMinutes

    if (($ExitCode -ne 0) -or (-not (Test-Path $SuccessPath))) {
        $Message = "Experiment failed: $Group / $Benchmark / seed $Seed / $Parameter=$Value"
        Add-ManifestRow -Group $Group -Benchmark $Benchmark -Functional $Functional `
            -Parameter $Parameter -Value $Value -Seed $Seed -OutDir $OutDir `
            -Status "failed" -RuntimeMinutes $Runtime `
            -StartedAt $RunStart.ToString("s") -FinishedAt $RunEnd.ToString("s") `
            -Message $Message
        Handle-Failure $Message
        return $false
    }

    Add-ManifestRow -Group $Group -Benchmark $Benchmark -Functional $Functional `
        -Parameter $Parameter -Value $Value -Seed $Seed -OutDir $OutDir `
        -Status "complete" -RuntimeMinutes $Runtime `
        -StartedAt $RunStart.ToString("s") -FinishedAt $RunEnd.ToString("s")
    return $true
}

function Register-BaselineAsSweepPoint {
    param(
        [string]$Group,
        [string]$Benchmark,
        [string]$Parameter,
        [string]$Value,
        [int]$Seed
    )

    $SourceDir = Join-Path $OutputRoot "headline_moments/$Benchmark/seed_$Seed"
    $SourceMetrics = Join-Path $SourceDir "metrics.json"
    $OutDir = Join-Path $OutputRoot "$Group/$Benchmark/$Parameter`_$Value/seed_$Seed"
    $TargetMetrics = Join-Path $OutDir "metrics.json"

    if ((Test-Path $TargetMetrics) -and (-not $Force)) {
        Add-ManifestRow -Group $Group -Benchmark $Benchmark -Functional "moments" `
            -Parameter $Parameter -Value $Value -Seed $Seed -OutDir $OutDir `
            -Status "already_complete" -RuntimeMinutes 0.0 `
            -StartedAt "" -FinishedAt (Get-Date).ToString("s") `
            -Message "Default sweep point already present."
        return $true
    }

    if (-not (Test-Path $SourceMetrics)) {
        $Message = "Cannot reuse missing headline metrics: $SourceMetrics"
        Add-ManifestRow -Group $Group -Benchmark $Benchmark -Functional "moments" `
            -Parameter $Parameter -Value $Value -Seed $Seed -OutDir $OutDir `
            -Status "failed" -RuntimeMinutes 0.0 `
            -StartedAt "" -FinishedAt (Get-Date).ToString("s") -Message $Message
        Handle-Failure $Message
        return $false
    }

    New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
    Copy-Item -Force $SourceMetrics $TargetMetrics
    "Reused from $SourceDir" | Set-Content -Encoding UTF8 (Join-Path $OutDir "reused_from.txt")

    Add-ManifestRow -Group $Group -Benchmark $Benchmark -Functional "moments" `
        -Parameter $Parameter -Value $Value -Seed $Seed -OutDir $OutDir `
        -Status "reused_baseline" -RuntimeMinutes 0.0 `
        -StartedAt "" -FinishedAt (Get-Date).ToString("s") `
        -Message "Identical default configuration reused from headline run."
    return $true
}

function Invoke-SweepPoint {
    param(
        [string]$Group,
        [string]$Benchmark,
        [string]$Parameter,
        [string]$CliFlag,
        [object]$Value,
        [int]$Seed
    )

    $OutDir = Join-Path $OutputRoot "$Group/$Benchmark/$Parameter`_$Value/seed_$Seed"
    $Args = @(
        "--functional", "moments",
        "--baselines", "none",
        $CliFlag, "$Value"
    )

    return Invoke-Experiment -Group $Group -Benchmark $Benchmark -Seed $Seed `
        -OutDir $OutDir -ExtraArgs $Args -Functional "moments" `
        -Parameter $Parameter -Value "$Value"
}

Write-Host ""
Write-Host "============================================================"
Write-Host "COMPLETE PAPER EXPERIMENTS"
Write-Host "Mode: $ModeName"
Write-Host "Python: $Python"
Write-Host "Output root: $OutputRoot"
Write-Host "No time cutoff: the complete fixed plan will run to completion."
Write-Host "Resume completed outputs: $(-not $Force)"
Write-Host "============================================================"

Start-Transcript -Path $TranscriptPath -Append | Out-Null

try {
    # =========================================================================
    # 1. HEADLINE RESULTS — MOMENT FUNCTIONAL, SIX BENCHMARKS, FIVE SEEDS
    # =========================================================================
    foreach ($Seed in $HeadlineSeeds) {
        foreach ($Benchmark in $AllBenchmarks) {
            $OutDir = Join-Path $OutputRoot "headline_moments/$Benchmark/seed_$Seed"
            Invoke-Experiment -Group "headline_moments" `
                -Benchmark $Benchmark -Seed $Seed -OutDir $OutDir `
                -ExtraArgs @("--functional", "moments", "--baselines", "full") `
                -Functional "moments" | Out-Null
        }
    }

    # =========================================================================
    # 2. QUANTILE-AND-TAIL STRESS TEST — ALL 1D BENCHMARKS, SEED 0
    # =========================================================================
    foreach ($Benchmark in $Benchmarks1D) {
        $OutDir = Join-Path $OutputRoot "headline_quantile/$Benchmark/seed_0"
        Invoke-Experiment -Group "headline_quantile" `
            -Benchmark $Benchmark -Seed 0 -OutDir $OutDir `
            -ExtraArgs @("--functional", "quantile", "--baselines", "full") `
            -Functional "quantile" | Out-Null
    }

    # =========================================================================
    # 3. SINGLE-TRAJECTORY TRAINING — CONDITIONAL OU, THREE SEEDS
    # =========================================================================
    foreach ($Seed in $SweepSeeds) {
        $OutDir = Join-Path $OutputRoot "single_particle/ou/seed_$Seed"
        Invoke-Experiment -Group "single_particle" `
            -Benchmark "ou" -Seed $Seed -OutDir $OutDir `
            -ExtraArgs @(
                "--functional", "moments",
                "--baselines", "full",
                "--single-particle"
            ) -Functional "moments" | Out-Null
    }

    # =========================================================================
    # 4. K_KEEP ABLATION — OU, IDENTICAL DATA AND INITIALIZATION
    # =========================================================================
    $KKeepOut = Join-Path $OutputRoot "kkeep/ou/seed_0"
    Invoke-Experiment -Group "kkeep" -Benchmark "ou" -Seed 0 `
        -OutDir $KKeepOut `
        -ExtraArgs @(
            "--functional", "moments",
            "--baselines", "none",
            "--sweep-kkeep", "1,8,64,200",
            "--sweep-epochs", "200"
        ) -Functional "moments" | Out-Null

    # =========================================================================
    # 5. SIGNATURE-DEPTH ABLATION — DOUBLE WELL, THREE SEEDS
    # =========================================================================
    foreach ($Seed in $SweepSeeds) {
        foreach ($Value in $SignatureValuesToTrain) {
            Invoke-SweepPoint -Group "sweep_sig_level" -Benchmark "double_well" `
                -Parameter "sig_level" -CliFlag "--sig-level" `
                -Value $Value -Seed $Seed | Out-Null
        }
        Register-BaselineAsSweepPoint -Group "sweep_sig_level" `
            -Benchmark "double_well" -Parameter "sig_level" -Value "3" `
            -Seed $Seed | Out-Null
    }

    # =========================================================================
    # 6. ITO--STRATONOVICH CONSISTENCY CHECK
    # =========================================================================
    $ItoOut = Join-Path $OutputRoot "ito_strat/multiplicative/seed_0"
    Invoke-Experiment -Group "ito_strat" -Benchmark "multiplicative" -Seed 0 `
        -OutDir $ItoOut -ExtraArgs @("--ito-strat-check") `
        -Functional "none" -SuccessFile "ito_strat_check.json" | Out-Null

    # =========================================================================
    # 7. NUMBER OF MIXTURE COMPONENTS J — EXACT MULTIMODAL OU-GMM
    # =========================================================================
    foreach ($Seed in $SweepSeeds) {
        foreach ($Value in $JValuesToTrain) {
            Invoke-SweepPoint -Group "sweep_J" -Benchmark "ou_gmm" `
                -Parameter "J" -CliFlag "--J" -Value $Value -Seed $Seed | Out-Null
        }
        Register-BaselineAsSweepPoint -Group "sweep_J" -Benchmark "ou_gmm" `
            -Parameter "J" -Value "5" -Seed $Seed | Out-Null
    }

    # =========================================================================
    # 8. FOURIER-FEATURE SWEEPS — BOTH NON-GAUSSIAN INITIAL-LAW FAMILIES
    # =========================================================================
    foreach ($Benchmark in @("ou_gmm", "ou_uniform")) {
        foreach ($Seed in $SweepSeeds) {
            foreach ($Value in $FourierValuesToTrain) {
                Invoke-SweepPoint -Group "sweep_fourier" -Benchmark $Benchmark `
                    -Parameter "n_fourier" -CliFlag "--n-fourier" `
                    -Value $Value -Seed $Seed | Out-Null
            }
            Register-BaselineAsSweepPoint -Group "sweep_fourier" `
                -Benchmark $Benchmark -Parameter "n_fourier" -Value "4" `
                -Seed $Seed | Out-Null
        }
    }

    # =========================================================================
    # 9. INITIAL LEARNING-RATE SWEEP — OU-GMM
    # =========================================================================
    foreach ($Seed in $SweepSeeds) {
        foreach ($Value in $LearningRateValuesToTrain) {
            Invoke-SweepPoint -Group "sweep_lr" -Benchmark "ou_gmm" `
                -Parameter "lr" -CliFlag "--lr" -Value $Value -Seed $Seed | Out-Null
        }
        Register-BaselineAsSweepPoint -Group "sweep_lr" -Benchmark "ou_gmm" `
            -Parameter "lr" -Value "0.001" -Seed $Seed | Out-Null
    }

    # =========================================================================
    # 10. MDN DEPTH SWEEP — OU-GMM, THREE SEEDS
    # =========================================================================
    foreach ($Seed in $SweepSeeds) {
        foreach ($Value in $DepthValuesToTrain) {
            Invoke-SweepPoint -Group "sweep_mdn_depth" -Benchmark "ou_gmm" `
                -Parameter "mdn_depth" -CliFlag "--mdn-depth" `
                -Value $Value -Seed $Seed | Out-Null
        }
        Register-BaselineAsSweepPoint -Group "sweep_mdn_depth" `
            -Benchmark "ou_gmm" -Parameter "mdn_depth" -Value "2" `
            -Seed $Seed | Out-Null
    }

    # =========================================================================
    # 11. PSI DEPTH SWEEP — OU-GMM, THREE SEEDS
    # =========================================================================
    foreach ($Seed in $SweepSeeds) {
        foreach ($Value in $DepthValuesToTrain) {
            Invoke-SweepPoint -Group "sweep_psi_depth" -Benchmark "ou_gmm" `
                -Parameter "psi_depth" -CliFlag "--psi-depth" `
                -Value $Value -Seed $Seed | Out-Null
        }
        Register-BaselineAsSweepPoint -Group "sweep_psi_depth" `
            -Benchmark "ou_gmm" -Parameter "psi_depth" -Value "2" `
            -Seed $Seed | Out-Null
    }

    # =========================================================================
    # 12. NONLINEAR DEPTH SANITY CHECK — DOUBLE WELL, SEED 0
    # =========================================================================
    foreach ($Spec in @(
        @{ Group = "sweep_mdn_depth_nonlinear"; Parameter = "mdn_depth"; Flag = "--mdn-depth" },
        @{ Group = "sweep_psi_depth_nonlinear"; Parameter = "psi_depth"; Flag = "--psi-depth" }
    )) {
        foreach ($Value in $DepthValuesToTrain) {
            Invoke-SweepPoint -Group $Spec.Group -Benchmark "double_well" `
                -Parameter $Spec.Parameter -CliFlag $Spec.Flag `
                -Value $Value -Seed 0 | Out-Null
        }
        Register-BaselineAsSweepPoint -Group $Spec.Group `
            -Benchmark "double_well" -Parameter $Spec.Parameter `
            -Value "2" -Seed 0 | Out-Null
    }
}
finally {
    Write-Host ""
    Write-Host "[aggregate] Building CSV/JSON summaries and figures..."
    & $Python "aggregate_full_results.py" --root $OutputRoot
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Aggregation failed; raw metrics and the manifest remain available."
    }

    $ElapsedHours = ((Get-Date) - $StartTime).TotalHours
    Write-Host ""
    Write-Host "============================================================"
    Write-Host "COMPLETE EXPERIMENT PLAN FINISHED"
    Write-Host "Elapsed: $([Math]::Round($ElapsedHours, 2)) hours"
    Write-Host "Failures recorded: $script:FailureCount"
    Write-Host "Results: $OutputRoot"
    Write-Host "Manifest: $ManifestPath"
    Write-Host "============================================================"
    Stop-Transcript | Out-Null
}

if ($script:FailureCount -gt 0) {
    exit 1
}
