#!/usr/bin/env bash
set -euo pipefail

test "$(id -u)" = 0
command -v setpriv
for model in /models/main /models/dspark; do
    test -d "$model"
    test ! -L "$model"
    test -z "$(find "$model" -xdev ! -type d ! -type f -print -quit)"
    test -z "$(find "$model" -xdev ! -uid 1002 -print -quit)"
    chown -R --no-dereference 1000:1000 "$model"
    find "$model" -xdev -type d -exec chmod 775 {} +
    find "$model" -xdev -type f -exec chmod 664 {} +
done

for uid in 1000 1002; do
    setpriv --reuid="$uid" --regid="$uid" --clear-groups bash -euc '
        for model in /models/main /models/dspark; do
            test -r "$model" && test -w "$model" && test -x "$model"
            while IFS= read -r -d "" file; do
                test -r "$file" && test -w "$file"
            done < <(find "$model" -xdev -type f -print0)
        done
        echo "UID $(id -u): all existing files readable and writable"
    '
done

# Check default ACL inheritance and edits to a file created by the other UID.
probe=".codex-minicpm47-access-20261008-${HOSTNAME}"
for model in /models/main /models/dspark; do
    test ! -e "$model/$probe-1000"
    test ! -e "$model/$probe-1002"
    setpriv --reuid=1000 --regid=1000 --clear-groups bash -euc \
        'printf "intel\n" > "$1"' bash "$model/$probe-1000"
    setpriv --reuid=1002 --regid=1002 --clear-groups bash -euc \
        'printf "xiangyu\n" >> "$1"; printf "xiangyu\n" > "$2"' \
        bash "$model/$probe-1000" "$model/$probe-1002"
    setpriv --reuid=1000 --regid=1000 --clear-groups bash -euc \
        'test -r "$1"; printf "intel\n" >> "$1"' bash "$model/$probe-1002"
    test "$(stat -c %u "$model/$probe-1000")" = 1000
    test "$(stat -c %u "$model/$probe-1002")" = 1002
    rm -- "$model/$probe-1000" "$model/$probe-1002"
    echo "$model: cross-user creation and editing passed; probes removed"
done
