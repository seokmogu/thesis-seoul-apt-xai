on run argv
    set docPath to item 1 of argv
    set pdfPath to item 2 of argv
    tell application "Microsoft Word"
        activate
        set theDoc to open file name docPath
    end tell
    delay 1.5
    tell application "System Events"
        tell process "Microsoft Word"
            keystroke "a" using {command down}
            delay 0.4
            -- Update Fields shortcut: F9
            key code 101
            delay 0.8
        end tell
    end tell
    delay 0.5
    tell application "Microsoft Word"
        set theDoc to active document
        save theDoc
        save as theDoc file name pdfPath file format format PDF
        close theDoc saving no
    end tell
end run
