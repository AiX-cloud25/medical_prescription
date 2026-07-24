# Frees port 8002 if a stale/orphaned server is holding it, then starts the backend.
$port = 8002

Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue |
    Select-Object -ExpandProperty OwningProcess -Unique |
    ForEach-Object {
        Write-Host "Killing stale process $_ holding port $port"
        Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue
    }

python backend.py
