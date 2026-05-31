/**
 * Custom 2D dataset utilities.
 *
 * Provides:
 *   parseCSV(text)   — parse "x,y,label" CSV text into Example2D[]
 *   serializeCSV(examples) — serialize Example2D[] back to CSV text
 *   validateExample  — check a single parsed row
 *
 * No external dependencies. No mathjs required.
 */

import {Example2D} from './dataset';

export interface ParseResult {
  examples: Example2D[];
  errors: string[];
}

/**
 * Parse CSV-like text into Example2D[].
 *
 * Format:
 *   - One example per line: x,y,label
 *   - x, y are floats (the playground domain is roughly [-6, 6])
 *   - label must be 1 or -1 (or 0, which is mapped to -1)
 *   - Lines starting with '#' are treated as comments and skipped
 *   - A header row containing non-numeric first column is auto-skipped
 *   - Whitespace around values is trimmed
 *
 * Returns both successfully parsed examples and an array of error messages
 * (one per bad line) so callers can surface warnings to the user.
 */
export function parseCSV(text: string): ParseResult {
  const examples: Example2D[] = [];
  const errors: string[] = [];

  const lines = text.split(/\r?\n/);
  let lineNum = 0;

  for (const rawLine of lines) {
    lineNum++;
    const line = rawLine.trim();

    // Skip blanks and comments
    if (line === '' || line.startsWith('#')) continue;

    const parts = line.split(',');
    if (parts.length < 3) {
      errors.push(`Line ${lineNum}: expected 3 columns (x,y,label), got ${parts.length} — "${line}"`);
      continue;
    }

    const xStr = parts[0].trim();
    const yStr = parts[1].trim();
    const lStr = parts[2].trim();

    // Auto-skip header rows (first column is non-numeric)
    if (lineNum === 1 && isNaN(Number(xStr))) {
      continue;
    }

    const x = Number(xStr);
    const y = Number(yStr);
    let labelRaw = Number(lStr);

    if (isNaN(x)) {
      errors.push(`Line ${lineNum}: x="${xStr}" is not a number`);
      continue;
    }
    if (isNaN(y)) {
      errors.push(`Line ${lineNum}: y="${yStr}" is not a number`);
      continue;
    }
    if (isNaN(labelRaw)) {
      errors.push(`Line ${lineNum}: label="${lStr}" is not a number`);
      continue;
    }

    // Normalise label: 0 → -1, anything positive → 1, anything negative → -1
    const label = labelRaw === 0 ? -1 : (labelRaw > 0 ? 1 : -1);

    examples.push({x, y, label});
  }

  return {examples, errors};
}

/**
 * Serialize an array of Example2D back into CSV text suitable for round-tripping
 * through parseCSV. Produces a header row followed by one data row per example.
 */
export function serializeCSV(examples: Example2D[]): string {
  const lines = ['x,y,label'];
  for (const ex of examples) {
    lines.push(`${ex.x},${ex.y},${ex.label}`);
  }
  return lines.join('\n');
}

/**
 * Validate a single {x, y, label} triple.
 * Returns null if valid, or an error string if not.
 */
export function validateExample(x: number, y: number, label: number): string | null {
  if (typeof x !== 'number' || isNaN(x)) return 'x must be a finite number';
  if (typeof y !== 'number' || isNaN(y)) return 'y must be a finite number';
  if (label !== 1 && label !== -1) return 'label must be 1 or -1';
  return null;
}

/**
 * Split an Example2D array into train/test sets by a given ratio.
 * Shuffles in-place using Fisher-Yates before splitting.
 *
 * @param examples    Full dataset.
 * @param trainRatio  Fraction for training (default 0.8).
 */
export function trainTestSplit(
    examples: Example2D[],
    trainRatio: number = 0.8): {train: Example2D[]; test: Example2D[]} {
  const arr = examples.slice();
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  const cutoff = Math.floor(arr.length * trainRatio);
  return {train: arr.slice(0, cutoff), test: arr.slice(cutoff)};
}
