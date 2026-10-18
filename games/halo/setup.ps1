# Gets the Halo port m8 plays (OpenCE) into halo\ at this repo's root, at the commit m8's changes
# are for, and puts them on it. Then build it there: python configure.py --release; ninja windows
$root = Resolve-Path "$PSScriptRoot\..\.."
$halo = Join-Path $root 'halo'
$patch = Join-Path $PSScriptRoot 'm8.patch'
# (the OpenCE commit the patch was made on)
$commit = '8102e711'

if (-not (Test-Path $halo)) {
    git clone https://github.com/OpenCommunityEdition/OpenCE.git $halo
    if ($LASTEXITCODE) { exit 1 }
}
git -C $halo apply --check --reverse $patch 2>$null
if (-not $LASTEXITCODE) {
    Write-Host "m8's changes are on halo\ already."
    exit 0
}
git -C $halo checkout $commit
git -C $halo apply --whitespace=nowarn $patch
if ($LASTEXITCODE) { Write-Host "The changes didn't apply: is halo\ a clean OpenCE checkout?"; exit 1 }
Write-Host "Done. Build it: cd halo; python configure.py --release; ninja windows"
