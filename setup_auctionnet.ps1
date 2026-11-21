$ErrorActionPreference = 'Stop'
$requiredCommit = 'c20de631a40e24669d1d6a8e2acefd8f492f172d'
$official = Join-Path $PSScriptRoot 'third_party/AuctionNet'
if (-not (Test-Path -LiteralPath $official)) {
    New-Item -ItemType Directory -Path (Split-Path $official -Parent) -Force | Out-Null
    git clone https://github.com/alimama-tech/AuctionNet.git $official
    if ($LASTEXITCODE -ne 0) { throw 'AuctionNet clone failed' }
}
$current = git -C $official rev-parse HEAD
if ($LASTEXITCODE -ne 0) { throw 'AuctionNet directory is not a Git checkout' }
if ($current.Trim() -ne $requiredCommit) {
    $changes = git -C $official status --porcelain
    if ($changes) { throw 'AuctionNet checkout has local changes; resolve them before switching versions' }
    git -C $official checkout $requiredCommit
    if ($LASTEXITCODE -ne 0) { throw 'Required official commit is unavailable' }
}
$assets = Join-Path $official 'simul_bidding_env/strategy/official_agent'
if (-not (Test-Path -LiteralPath (Join-Path $assets 'BCQtest/bcq_model.pth'))) { throw 'BCQ checkpoint missing' }
if (-not (Test-Path -LiteralPath (Join-Path $assets 'IQLtest/iql_model.pth'))) { throw 'IQL checkpoint missing' }
Write-Host "AuctionNet official commit ready: $requiredCommit"
