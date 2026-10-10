#!/usr/bin/env bash
# check_core_surface.sh — keep every package on flutter_edge_ai's public surface.
#
# Core publishes five libraries: flutter_edge_ai.dart (apps), engine.dart
# (engine SPI, re-exports the app library), engine_io.dart (dart:io/ffi
# helpers), engine_web.dart (dart:js_interop helpers) and genai.dart. Everything
# else lives under lib/src and is private. The analyzer enforces part of that
# (`implementation_imports: error` in every analysis_options.yaml), but its lint
# only looks at files inside lib/, and it cannot tell a reviewed exception from
# a careless one. This script closes both gaps.
#
# It scans every .dart file under packages/ and fails on:
#   (a) an import/export of `package:flutter_edge_ai/<x>` where <x> is neither
#       one of the five public libraries nor under src/ (an old deep path);
#   (b) an import/export of another workspace package's `src/` that is not
#       directly preceded by the annotated ignore
#         // ignore: implementation_imports — <reason>
#       That inline comment is the single allowlist: there is no second list to
#       keep in sync, and the reason sits next to the import it excuses;
#   (c) any such `src/` import or export in the lib/ of a PUBLISHED package
#       (packages/<pkg>/lib/), annotated or not. Unpublished example apps
#       (packages/<pkg>/example/lib/) host white-box test screens and may use
#       an annotated import, like tests.
#
# It prints every annotated import with its reason, so the allowlist can be
# reviewed in one place.
#
# Scope: packages/** only. codelabs/** and the website's dart fences still
# document the published 2.1.x API (the old paths); they join the scan when
# they migrate.
#
# Usage: tool/check_core_surface.sh
# Exit:  0 clean · 1 a violation, or nothing was scanned.

set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Owning package of every pubspec under packages/, deepest directory first so
# an example app wins over the package that contains it.
owners_file="$(mktemp)"
files_file="$(mktemp)"
trap 'rm -f "$owners_file" "$files_file"' EXIT

find packages -name pubspec.yaml \
  -not -path '*/.dart_tool/*' -not -path '*/build/*' \
  -not -path '*/ephemeral/*' -not -path '*/Pods/*' -not -path '*/.symlinks/*' |
  while read -r pubspec; do
    name="$(sed -n 's/^name:[[:space:]]*\([A-Za-z0-9_]*\).*/\1/p' "$pubspec" | head -1)"
    [[ -n "$name" ]] && printf '%s\t%s\n' "$(dirname "$pubspec")" "$name"
  done | awk -F'\t' '{ print length($1) "\t" $0 }' | sort -rn | cut -f2- \
  >"$owners_file"

find packages -name '*.dart' \
  -not -path '*/.dart_tool/*' -not -path '*/build/*' \
  -not -path '*/ephemeral/*' -not -path '*/Pods/*' -not -path '*/.symlinks/*' \
  -not -path '*/node_modules/*' | sort >"$files_file"

scanned="$(wc -l <"$files_file" | tr -d ' ')"
if [[ "$scanned" -eq 0 ]]; then
  echo "::error::check_core_surface: found no .dart files under packages/ — a"
  echo "::error::check that examined nothing is not a pass."
  exit 1
fi

# Workspace packages whose src/ is private to them: every pubspec directly
# under packages/ (examples are apps, nobody imports them).
workspace_pkgs="$(awk -F'\t' '$1 ~ /^packages\/[^\/]+$/ { print $2 }' "$owners_file" | tr '\n' ' ')"

awk -v owners_file="$owners_file" -v workspace_pkgs="$workspace_pkgs" '
BEGIN {
  n = 0
  while ((getline line < owners_file) > 0) {
    split(line, parts, "\t")
    n++; owner_dir[n] = parts[1]; owner_name[n] = parts[2]
  }
  owners = n
  split(workspace_pkgs, wp, " ")
  for (i in wp) if (wp[i] != "") workspace[wp[i]] = 1
  public["flutter_edge_ai.dart"] = 1
  public["engine.dart"] = 1
  public["engine_io.dart"] = 1
  public["engine_web.dart"] = 1
  public["genai.dart"] = 1
  violations = 0; annotated = 0; files = 0
  # The annotation: the comment line directly above the directive.
  ann_re = "^[ \t]*// ignore: implementation_imports — [^ \t].*$"
  # A quoted package: URI, single or double quotes (\047 is the single quote).
  uri_re = "[\"\047]package:[^\"\047]+[\"\047]"
}

function owner_of(path,    i) {
  for (i = 1; i <= owners; i++) {
    if (index(path, owner_dir[i] "/") == 1) return owner_name[i]
  }
  return ""
}

function published_lib(path, pkg) {
  # packages/<dir>/lib/... where <dir> is the package itself (not an example).
  return path ~ /^packages\/[^\/]+\/lib\//
}

function violation(msg) {
  printf "  FAIL  %s:%d  %s\n", FILENAME, start_line, msg
  violations++
}

# Checks one quoted URI found inside a directive.
function check_uri(uri,    rest, pkg, slash) {
  if (uri !~ /^package:/) return
  rest = substr(uri, 9)
  slash = index(rest, "/")
  if (slash == 0) return
  pkg = substr(rest, 1, slash - 1)
  rest = substr(rest, slash + 1)

  if (pkg == "flutter_edge_ai" && rest !~ /^src\// && !(rest in public)) {
    violation("(a) " uri " is not a public library — use flutter_edge_ai.dart, engine.dart, engine_io.dart, engine_web.dart or genai.dart")
    return
  }
  if (!(pkg in workspace) || rest !~ /^src\//) return
  if (pkg == file_owner) return

  if (published_lib(FILENAME, file_owner)) {
    violation("(c) " uri " — a published package lib/ must not import another package'"'"'s src/")
    return
  }
  if (prev_line ~ ann_re) {
    reason = prev_line
    sub(/^[ \t]*\/\/ ignore: implementation_imports — /, "", reason)
    allow[++annotated] = sprintf("  %s:%d  %s\n        — %s", FILENAME, start_line, uri, reason)
  } else {
    violation("(b) " uri " needs `// ignore: implementation_imports — <reason>` on the line above")
  }
}

FNR == 1 {
  files++
  file_owner = owner_of(FILENAME)
  in_directive = 0
  before = ""
}

{
  line = $0
  if (!in_directive && line ~ /^[ \t]*(import|export)[ \t]/) {
    in_directive = 1
    start_line = FNR
    prev_line = before
  }
  if (in_directive) {
    rest = line
    while (match(rest, uri_re)) {
      check_uri(substr(rest, RSTART + 1, RLENGTH - 2))
      rest = substr(rest, RSTART + RLENGTH)
    }
    if (line ~ /;[ \t]*(\/\/.*)?$/) in_directive = 0
  }
  before = line
}

END {
  print "==> Annotated cross-package src/ imports (the allowlist): " annotated
  for (i = 1; i <= annotated; i++) print allow[i]
  print ""
  if (violations > 0) {
    printf "==> check_core_surface: %d violation(s) in %d files scanned.\n", violations, files
    exit 1
  }
  printf "==> check_core_surface: OK — %d files scanned, %d annotated src/ imports, 0 violations.\n", files, annotated
}
' $(cat "$files_file")
