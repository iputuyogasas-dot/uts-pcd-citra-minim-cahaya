import sys

# Konsol Windows (cp1252) tidak bisa mencetak simbol seperti "Δ" dan "±".
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
