-- One-shot actual Terminal capture; SSH key authorization only, no passwords.
on run argv
    set outputPath to item 1 of argv
    set runID to item 2 of argv
    set remoteRoot to item 3 of argv
    set remoteHost to item 4 of argv
    set remotePort to item 5 of argv
    set remoteCommand to "cd " & quoted form of remoteRoot & " && PYTHONPATH=src .venv/bin/python scripts/review_terminal_evidence.py --run-id " & quoted form of runID & " && nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv"
    set evidenceTitle to "MiniMind repaired run evidence"
    if (count of argv) > 5 then
        if item 6 of argv is "vlm-evaluation" then
            set remoteCommand to "cd " & quoted form of remoteRoot & " && .venv/bin/python reporting/review_vlm_evidence.py --run-id " & quoted form of runID
            set evidenceTitle to "MiniMind fixed VLM evaluation"
        else if item 6 of argv is "video-evaluation" then
            set remoteCommand to "cd " & quoted form of remoteRoot & " && .venv/bin/python reporting/review_video_evidence.py --run-id " & quoted form of runID
            set evidenceTitle to "MiniMind fixed video evaluation"
        else
            error "Unsupported evidence view"
        end if
    end if
    -- Let SSH finish before capture, leaving local sleep as the window process.
    -- This keeps connection arguments out of Terminal's active-process title.
    set shellCommand to "clear && ssh -p " & quoted form of remotePort & " -o BatchMode=yes -o ConnectTimeout=15 " & quoted form of remoteHost & " " & quoted form of remoteCommand & "; sleep 12"
    tell application "Terminal"
        set evidenceTab to do script shellCommand
        set evidenceWindow to front window
        set custom title of evidenceTab to evidenceTitle
        set title displays custom title of evidenceTab to true
        set title displays shell path of evidenceTab to false
        set title displays device name of evidenceTab to false
        set title displays window size of evidenceTab to false
        set title displays file name of evidenceTab to false
        set font name of evidenceTab to "Menlo"
        set font size of evidenceTab to 18
        set bounds of evidenceWindow to {80, 60, 1180, 870}
        activate
        delay 8
        set windowID to id of evidenceWindow
    end tell
    do shell script "/usr/sbin/screencapture -x -l " & windowID & " " & quoted form of outputPath
    return windowID
end run
