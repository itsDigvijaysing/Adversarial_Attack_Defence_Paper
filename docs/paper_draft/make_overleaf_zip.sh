#!/usr/bin/env bash
# Build overleaf_upload.zip: the paper source and nothing else.
#
#   bash docs/paper_draft/make_overleaf_zip.sh            # build the zip
#   bash docs/paper_draft/make_overleaf_zip.sh --check    # build, then compile
#                                                         # the zip in a temp dir
#   bash docs/paper_draft/make_overleaf_zip.sh --with-bbl # also ship main.bbl
#
# The zip holds main.tex, the .bib file(s) named by \bibliography, every file
# main.tex reads through \tabinput / \input / \include, and every figure named
# by \includegraphics. Build outputs (PDFs of the paper, .aux, .log, .bbl
# unless --with-bbl, ...) are never added. A referenced file that does not
# exist is an error, so a zip that builds here is complete.
#
# --check unzips into a fresh temporary directory and runs
# pdflatex, bibtex, pdflatex, pdflatex there (the same pdfLaTeX engine Overleaf
# must be set to). It fails on any LaTeX error, undefined reference or
# undefined citation, and prints the page count and any overfull boxes.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN="main.tex"
OUT="overleaf_upload.zip"
WITH_BBL=0
CHECK=0
for arg in "$@"; do
    case "$arg" in
        --with-bbl) WITH_BBL=1 ;;
        --check) CHECK=1 ;;
        -h|--help) sed -n '2,19p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown option: $arg" >&2; exit 2 ;;
    esac
done
command -v zip >/dev/null || { echo "error: 'zip' is not installed" >&2; exit 1; }
cd "$HERE"
[ -f "$MAIN" ] || { echo "error: $HERE/$MAIN not found" >&2; exit 1; }

# main.tex without comment lines, so commented-out \input lines are ignored.
SRC="$(grep -v '^[[:space:]]*%' "$MAIN")"
files=("$MAIN")
missing=0

add() {   # add <path>: queue a file once, or record it as missing
    local f="$1"
    if [ ! -f "$f" ]; then
        echo "error: $MAIN references '$f', which does not exist" >&2
        missing=1
        return
    fi
    local g
    for g in "${files[@]}"; do [ "$g" = "$f" ] && return; done
    files+=("$f")
}

# \tabinput{x}, \input{x}, \include{x}: x or x.tex
while read -r name; do
    [ -n "$name" ] || continue
    case "$name" in *.tex) add "$name" ;; *) add "$name.tex" ;; esac
done < <(grep -oE '\\(tabinput|input|include)\{[^}]+\}' <<<"$SRC" \
         | sed -E 's/^[^{]*\{(.*)\}$/\1/' || true)

# \includegraphics[...]{x}: x as written, else the first existing extension
while read -r name; do
    [ -n "$name" ] || continue
    if [ -f "$name" ]; then add "$name"; continue; fi
    found=""
    for ext in .pdf .png .jpg .jpeg .eps; do
        if [ -f "$name$ext" ]; then found="$name$ext"; break; fi
    done
    add "${found:-$name}"
done < <(grep -oE '\\includegraphics(\[[^]]*\])?\{[^}]+\}' <<<"$SRC" \
         | sed -E 's/^.*\{(.*)\}$/\1/' || true)

# \bibliography{a,b}: a.bib, b.bib
while read -r list; do
    IFS=',' read -ra bibs <<<"$list"
    for b in "${bibs[@]}"; do
        b="$(echo "$b" | tr -d '[:space:]')"
        [ -n "$b" ] || continue
        case "$b" in *.bib) add "$b" ;; *) add "$b.bib" ;; esac
    done
done < <(grep -oE '\\bibliography\{[^}]+\}' <<<"$SRC" \
         | sed -E 's/^[^{]*\{(.*)\}$/\1/' || true)

if [ "$WITH_BBL" = 1 ]; then add "${MAIN%.tex}.bbl"; fi
[ "$missing" = 0 ] || exit 1

rm -f "$OUT"
zip -X -q "$OUT" "${files[@]}"
echo "wrote $HERE/$OUT (${#files[@]} files):"
unzip -Z1 "$OUT" | sed 's/^/  /'

[ "$CHECK" = 1 ] || exit 0

# == standalone compile check ==============================================
for tool in pdflatex bibtex unzip; do
    command -v "$tool" >/dev/null || { echo "error: --check needs $tool" >&2; exit 1; }
done
TMP="$(mktemp -d "${TMPDIR:-/tmp}/overleaf_check.XXXXXX")"
unzip -q "$OUT" -d "$TMP"
(
    cd "$TMP"
    stem="${MAIN%.tex}"
    pdflatex -interaction=nonstopmode -halt-on-error "$stem" >/dev/null || true
    bibtex "$stem" >bibtex.out 2>&1 || true
    pdflatex -interaction=nonstopmode -halt-on-error "$stem" >/dev/null || true
    pdflatex -interaction=nonstopmode -halt-on-error "$stem" >/dev/null || true
    errors=$(grep -c '^!' "$stem.log" || true)
    undef_ref=$(grep -c 'Reference .* undefined' "$stem.log" || true)
    undef_cit=$(grep -c 'Citation .* undefined' "$stem.log" || true)
    bib_err=$(grep -c 'error message' bibtex.out || true)
    overfull=$(grep -c '^Overfull' "$stem.log" || true)
    pages="?"
    if command -v pdfinfo >/dev/null && [ -f "$stem.pdf" ]; then
        pages=$(pdfinfo "$stem.pdf" | awk '/^Pages:/ {print $2}')
    fi
    echo "check in $TMP:"
    echo "  LaTeX errors: $errors | undefined references: $undef_ref |" \
         "undefined citations: $undef_cit | bibtex errors: $bib_err"
    echo "  pages: $pages | overfull boxes: $overfull"
    grep -A1 '^Overfull' "$stem.log" | sed 's/^/    /' || true
    if [ ! -f "$stem.pdf" ] || [ "$errors" != 0 ] || [ "$undef_ref" != 0 ] \
       || [ "$undef_cit" != 0 ] || [ "$bib_err" != 0 ]; then
        echo "FAILED: see $TMP/$stem.log" >&2
        exit 1
    fi
    echo "OK: the zip compiles standalone with pdfLaTeX + BibTeX."
)
