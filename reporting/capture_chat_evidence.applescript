-- Capture only a new Terminal window showing verified, read-only diagnostic results.
-- Uses existing SSH key authorization. Never accepts a password.
on run argv
    set outputPath to item 1 of argv
    set runID to item 2 of argv
    set evaluationID to item 3 of argv
    set remoteHost to item 4 of argv
    set remotePort to item 5 of argv
    set caseID to "C07"
    if (count of argv) > 5 then set caseID to item 6 of argv
    set remoteCommand to "cd /root/autodl-tmp/wlh_goya/MiniMind && .venv/bin/python reporting/review_chat_evidence.py --run-id " & quoted form of runID & " --eval-id " & quoted form of evaluationID & " --case " & quoted form of caseID
    -- Clear locally, then let SSH exit before capture to avoid connection-title exposure.
    set shellCommand to "clear && ssh -p " & quoted form of remotePort & " -o BatchMode=yes -o ConnectTimeout=15 " & quoted form of remoteHost & " " & quoted form of remoteCommand & "; sleep 12"
    tell application "Terminal"
        set evidenceTab to do script shellCommand
        set evidenceWindow to front window
        set custom title of evidenceTab to "MiniMind chat diagnostic evidence"
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
