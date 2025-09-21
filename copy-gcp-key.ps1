# PowerShell script to copy GCP service account key to clipboard
# This helps set up GitHub Secrets

Write-Host "🔐 Copying GCP Service Account Key to Clipboard..." -ForegroundColor Blue

$keyPath = ".secrets/cement-ops-key.json"

if (Test-Path $keyPath) {
    try {
        # Read the JSON content
        $keyContent = Get-Content $keyPath -Raw
        
        # Copy to clipboard
        $keyContent | Set-Clipboard
        
        Write-Host "✅ Service account key copied to clipboard successfully!" -ForegroundColor Green
        Write-Host ""
        Write-Host "📋 Next steps:" -ForegroundColor Yellow
        Write-Host "1. Go to GitHub repository settings" -ForegroundColor White
        Write-Host "2. Navigate to Secrets and variables → Actions" -ForegroundColor White
        Write-Host "3. Click 'New repository secret'" -ForegroundColor White
        Write-Host "4. Name: GCP_SA_KEY" -ForegroundColor White
        Write-Host "5. Paste the key content (Ctrl+V)" -ForegroundColor White
        Write-Host "6. Click 'Add secret'" -ForegroundColor White
        Write-Host ""
        Write-Host "🌐 GitHub URL: https://github.com/ArjunSeeramsetty/CementPlantAIOptimization/settings/secrets/actions" -ForegroundColor Cyan
        
    } catch {
        Write-Host "❌ Error copying key: $($_.Exception.Message)" -ForegroundColor Red
    }
} else {
    Write-Host "❌ Service account key file not found at: $keyPath" -ForegroundColor Red
    Write-Host "Please ensure the key file exists in the correct location." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Press any key to continue..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
