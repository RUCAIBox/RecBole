#!/usr/bin/env python3
"""
Extract metric names and values from a log line or text blob containing a (Ordered)Dict
like: "INFO test result: OrderedDict({'recall@5': np.float64(0.0347), ...})"
and output two CSV lines: a header row of names and a data row of values.

Usage examples:
  # Read from stdin and print to stdout
  echo "07 Nov 20:27 INFO test result: OrderedDict({'recall@5': np.float64(0.0347)})" | \\
    python3 scripts/extract_metrics_to_csv.py

  # Read from file and write to CSV
  python3 scripts/extract_metrics_to_csv.py -f path/to/log.txt -o metrics.csv

  # Pass the line directly
  python3 scripts/extract_metrics_to_csv.py --from-string "07 Nov ... OrderedDict({'recall@5': np.float64(0.0347)})"
"""
from __future__ import annotations

import argparse
import ast
import csv
import io
import re
import sys
from typing import Dict, Iterable, List, Tuple, Union


def read_all_input(args: argparse.Namespace) -> str:
    if args.from_string is not None:
        return args.from_string
    if args.file is not None:
        with open(args.file, "r", encoding="utf-8") as f:
            return f.read()
    # stdin
    if not sys.stdin.isatty():
        return sys.stdin.read()
    raise SystemExit("No input provided. Use --from-string, --file, or pipe data via stdin.")


def extract_braced_substring(text: str) -> str:
    """
    Try to extract the substring representing the dict literal by taking the
    first '{' to the last '}'.
    Falls back to original text if braces not found.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def sanitize_wrappers(text: str) -> str:
    """
    Remove wrappers like np.float64(0.123), numpy.float32(0.1), Decimal(0.2), float(0.3).
    Convert them to the inner numeric literal so that ast.literal_eval can parse it.
    """
    patterns = [
        r"np\.float(?:16|32|64)?\(([^)]+)\)",
        r"numpy\.float(?:16|32|64)?\(([^)]+)\)",
        r"Decimal\(([^)]+)\)",
        r"float\(([^)]+)\)",
    ]
    sanitized = text
    for pat in patterns:
        sanitized = re.sub(pat, r"\1", sanitized)
    return sanitized


def try_literal_eval_dict(text: str) -> Union[Dict[str, Union[int, float, str]], None]:
    try:
        value = ast.literal_eval(text)
    except Exception:
        return None
    if isinstance(value, dict):
        return value
    return None


def regex_parse_kv(text: str) -> Tuple[List[str], List[str]]:
    """
    Fallback parser: extract 'key': value pairs using regex.
    Values are kept as strings and later converted to numbers when possible.
    """
    # This matches keys in single or double quotes, followed by colon, then captures
    # a simple value up to the next comma or closing brace.
    pattern = re.compile(r"""(['"])(.*?)\1\s*:\s*([^,}]+)""")
    keys: List[str] = []
    vals: List[str] = []
    for m in pattern.finditer(text):
        keys.append(m.group(2))
        vals.append(m.group(3).strip())
    if not keys:
        raise ValueError("Could not parse any key/value pairs.")
    return keys, vals


def coerce_to_number(value: Union[int, float, str]) -> Union[int, float, str]:
    if isinstance(value, (int, float)):
        return value
    s = str(value).strip()
    # Remove possible quotes
    if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')):
        s = s[1:-1]
    try:
        # Try int first to keep clean integers when applicable
        if re.fullmatch(r"[+-]?\d+", s):
            return int(s)
        return float(s)
    except Exception:
        return s


def format_value_for_csv(value: Union[int, float, str]) -> str:
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        # Avoid excessive precision while keeping significant digits
        return f"{value:.12g}"
    return str(value)


def to_csv_rows(mapping: Dict[str, Union[int, float, str]]) -> Tuple[List[str], List[str]]:
    keys: List[str] = list(mapping.keys())
    vals: List[str] = [format_value_for_csv(coerce_to_number(mapping[k])) for k in keys]
    return keys, vals


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract metrics dict to two-line CSV (header + values).")
    parser.add_argument("-f", "--file", dest="file", help="Input file containing the log/text.")
    parser.add_argument("--from-string", dest="from_string", help="Input text provided directly.")
    parser.add_argument("-o", "--output", dest="output", help="Write CSV to this file. Prints to stdout if omitted.")
    parser.add_argument("-d", "--delimiter", default=",", help="CSV delimiter (default: ',').")
    parser.add_argument("--encoding", default="utf-8", help="Output file encoding (default: utf-8).")
    args = parser.parse_args(argv)

    raw_text = read_all_input(args)
    dict_like = extract_braced_substring(raw_text)
    sanitized = sanitize_wrappers(dict_like)

    mapping = try_literal_eval_dict(sanitized)
    if mapping is None:
        # Fallback: regex parse
        keys, raw_vals = regex_parse_kv(sanitized)
        values = [format_value_for_csv(coerce_to_number(v)) for v in raw_vals]
    else:
        keys, values = to_csv_rows(mapping)

    if args.output:
        with open(args.output, "w", newline="", encoding=args.encoding) as f:
            writer = csv.writer(f, delimiter=args.delimiter)
            writer.writerow(keys)
            writer.writerow(values)
    else:
        # Print to stdout
        stdout = io.StringIO()
        writer = csv.writer(stdout, delimiter=args.delimiter, lineterminator="\n")
        writer.writerow(keys)
        writer.writerow(values)
        sys.stdout.write(stdout.getvalue())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


