param([switch]$Dry)
$py = "C:\Users\rayyan.a\AppData\Local\anaconda3\python.exe"
& $py D:\tmp\mvp-plan\parse_plans.py | Out-String -Width 250
& $py D:\tmp\mvp-plan\check_plans2.py D:\tmp\mvp-plan\indexes_parsed.json | Out-String -Width 250
& $py D:\tmp\mvp-plan\selfreview.py | Out-String -Width 250
if ($Dry) {
    $env:MASTER_OUT = "D:\tmp\mvp-plan\dry\master.md"; $env:APPX_OUT = "D:\tmp\mvp-plan\dry\appendix.md"; $env:REG_DOC = "D:\tmp\mvp-plan\dry\register.md"
}
& $py D:\tmp\mvp-plan\assemble_master.py
& $py D:\tmp\mvp-plan\compile_register.py | Select-Object -Last 2
Remove-Item Env:\MASTER_OUT, Env:\APPX_OUT, Env:\REG_DOC -ErrorAction SilentlyContinue
