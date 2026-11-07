$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$env:PIP_NO_INDEX = '0'
if (!(Test-Path '.venv/Scripts/python.exe')) {
    py -3.11 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Failed to create virtual environment' }
}
& .venv/Scripts/python.exe -m pip install --upgrade pip --index-url https://pypi.org/simple --timeout 120
if ($LASTEXITCODE -ne 0) { throw 'pip upgrade failed' }
& .venv/Scripts/python.exe -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124 --timeout 120 --retries 10 --resume-retries 10
if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed' }
& .venv/Scripts/python.exe -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
& .venv/Scripts/python.exe -m pip freeze | Set-Content environment-lock.txt
