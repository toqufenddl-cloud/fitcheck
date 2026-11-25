# FitCheck 외부용 시연: 앱(127.0.0.1:8501)과 ngrok 고정 도메인 터널을 한 번에 켠다.
# 사용: powershell -ExecutionPolicy Bypass -File .\run_public.ps1 [-Domain 내도메인.ngrok-free.app]
# 준비 (한 번만): ngrok 설치, `ngrok config add-authtoken <내 토큰>`, .streamlit/secrets.toml에 public_password·ngrok_domain
# 끄기: 이 창에서 Ctrl+C (터널이 닫히면 앱도 끈다)
param([string]$Domain = "")

$ErrorActionPreference = "Continue"   # ngrok이 stderr에 쓰는 줄을 오류로 멈추지 않게 (실패는 아래에서 직접 확인)
$Root = $PSScriptRoot
$Py = "C:\Users\1\.venvs\aba-finance\Scripts\python.exe"
$Secrets = Join-Path $Root ".streamlit\secrets.toml"
$Port = 8501

function Read-Secret([string]$Key) {
    if (-not (Test-Path $Secrets)) { return "" }
    $m = Select-String -Path $Secrets -Pattern "^\s*$Key\s*=\s*`"(.*)`"" | Select-Object -First 1
    if ($m) { return $m.Matches[0].Groups[1].Value } else { return "" }
}

# 1. 확인
if (-not (Test-Path $Py)) { Write-Host "venv python이 없습니다: $Py"; exit 1 }
# 새로 설치한 직후에는 PATH가 아직 안 바뀐 창이 있어 winget 설치 위치도 찾는다
$Ngrok = (Get-Command ngrok -ErrorAction SilentlyContinue).Source
if (-not $Ngrok) { $Ngrok = Get-ChildItem "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\Ngrok.Ngrok*\ngrok.exe" -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty FullName }
if (-not $Ngrok) { Write-Host "ngrok이 설치되어 있지 않습니다 (winget install ngrok.ngrok)."; exit 1 }
if (-not (Read-Secret "public_password")) { Write-Host ".streamlit/secrets.toml에 public_password가 없습니다."; exit 1 }
if (-not $Domain) { $Domain = Read-Secret "ngrok_domain" }
if (-not $Domain) { Write-Host "고정 도메인이 없습니다. -Domain 으로 주거나 secrets.toml의 ngrok_domain에 적어 주세요."; exit 1 }
& $Ngrok config check *> $null
if ($LASTEXITCODE -ne 0) { Write-Host "ngrok 설정 파일이 없거나 잘못되었습니다. 먼저 ngrok config add-authtoken <토큰> 을 실행해 주세요."; exit 1 }
if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
    Write-Host "포트 $Port 를 이미 쓰고 있습니다 (로컬 앱이 켜져 있으면 먼저 꺼 주세요)."; exit 1
}

# 2. 앱 (127.0.0.1만, 상세 오류 숨김, 코드 변경 감시 끔)
$appArgs = @("-X", "utf8", "-m", "streamlit", "run", "app/fitcheck_public.py",
    "--server.address", "127.0.0.1", "--server.port", "$Port", "--server.headless", "true",
    "--client.showErrorDetails", "none", "--client.toolbarMode", "minimal",
    "--server.fileWatcherType", "none", "--server.runOnSave", "false",
    "--browser.gatherUsageStats", "false", "--browser.serverAddress", $Domain, "--browser.serverPort", "443")
$app = Start-Process -FilePath $Py -ArgumentList $appArgs -WorkingDirectory $Root -PassThru -WindowStyle Minimized
Write-Host "앱 시작 (pid $($app.Id)), 응답 기다리는 중..."
$ok = $false
for ($i = 0; $i -lt 60; $i++) {
    Start-Sleep -Seconds 1
    try { if ((Invoke-WebRequest "http://127.0.0.1:$Port/_stcore/health" -UseBasicParsing -TimeoutSec 2).StatusCode -eq 200) { $ok = $true; break } } catch {}
}
if (-not $ok) { Write-Host "앱이 응답하지 않습니다."; Stop-Process -Id $app.Id -Force -ErrorAction SilentlyContinue; exit 1 }

# 3. 터널 (ngrok v3 최신은 --url, 구버전은 --domain)
$flag = if ((& $Ngrok http --help 2>&1 | Out-String) -match "--url") { "--url" } else { "--domain" }
Write-Host "터널: https://$Domain  ->  127.0.0.1:$Port  (끄려면 Ctrl+C)"
try {
    & $Ngrok http "$Port" "$flag=$Domain"
} finally {
    Stop-Process -Id $app.Id -Force -ErrorAction SilentlyContinue
    Write-Host "앱을 껐습니다."
}
