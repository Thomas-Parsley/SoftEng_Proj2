$paths = @(
    "work/train/labels",
    "work/valid/labels"
)

foreach ($path in $paths) {
    Get-ChildItem -Path $path -Recurse -Filter *.txt | ForEach-Object {
        $file = $_.FullName
        $newLines = @()

        foreach ($line in Get-Content $file) {
            $parts = $line.Split(" ")

            if ($parts.Count -gt 0) {
                switch ($parts[0]) {
                    "0" { $parts[0] = "81" }
                    "1" { $parts[0] = "82" }
                }
            }

            $newLines += ($parts -join " ")
        }

        Set-Content -Path $file -Value $newLines
        Write-Host "Updated $file"
    }
}
