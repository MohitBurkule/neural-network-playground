/**
 * Tests for src/customdataset.ts: parseCSV, serializeCSV,
 * validateExample, trainTestSplit.
 */

import {
  parseCSV,
  serializeCSV,
  validateExample,
  trainTestSplit,
} from '../src/customdataset';
import {Example2D} from '../src/dataset';

// ─── parseCSV — valid inputs ──────────────────────────────────────────────────

describe('parseCSV — valid inputs', () => {
  test('parses a simple 3-row CSV into Example2D[]', () => {
    const csv = '1.0,2.0,1\n-1.5,0.5,-1\n0.0,-3.0,1';
    const {examples, errors} = parseCSV(csv);
    expect(errors).toHaveLength(0);
    expect(examples).toHaveLength(3);
    expect(examples[0]).toEqual({x: 1.0, y: 2.0, label: 1});
    expect(examples[1]).toEqual({x: -1.5, y: 0.5, label: -1});
    expect(examples[2]).toEqual({x: 0.0, y: -3.0, label: 1});
  });

  test('skips blank lines', () => {
    const csv = '\n1,2,1\n\n-1,-2,-1\n';
    const {examples, errors} = parseCSV(csv);
    expect(errors).toHaveLength(0);
    expect(examples).toHaveLength(2);
  });

  test('skips comment lines starting with #', () => {
    const csv = '# this is a comment\n1,2,1\n# another comment\n-1,-2,-1';
    const {examples, errors} = parseCSV(csv);
    expect(errors).toHaveLength(0);
    expect(examples).toHaveLength(2);
  });

  test('auto-skips header row when first column is non-numeric', () => {
    const csv = 'x,y,label\n1.0,2.0,1\n-1.0,-2.0,-1';
    const {examples, errors} = parseCSV(csv);
    expect(errors).toHaveLength(0);
    expect(examples).toHaveLength(2);
  });

  test('label=0 is mapped to -1', () => {
    const csv = '1,2,0';
    const {examples} = parseCSV(csv);
    expect(examples[0].label).toBe(-1);
  });

  test('positive label values are mapped to 1', () => {
    const csv = '1,2,5';
    const {examples} = parseCSV(csv);
    expect(examples[0].label).toBe(1);
  });

  test('handles whitespace around values', () => {
    const csv = '  1.5 ,  -2.0 ,  1 ';
    const {examples, errors} = parseCSV(csv);
    expect(errors).toHaveLength(0);
    expect(examples[0]).toEqual({x: 1.5, y: -2.0, label: 1});
  });

  test('handles CRLF line endings', () => {
    const csv = '1,2,1\r\n-1,-2,-1\r\n';
    const {examples, errors} = parseCSV(csv);
    expect(errors).toHaveLength(0);
    expect(examples).toHaveLength(2);
  });

  test('empty input produces empty examples', () => {
    const {examples, errors} = parseCSV('');
    expect(examples).toHaveLength(0);
    expect(errors).toHaveLength(0);
  });

  test('parses floating point x,y values accurately', () => {
    const csv = '3.14159,-2.71828,1';
    const {examples} = parseCSV(csv);
    expect(examples[0].x).toBeCloseTo(3.14159);
    expect(examples[0].y).toBeCloseTo(-2.71828);
  });
});

// ─── parseCSV — malformed inputs ─────────────────────────────────────────────

describe('parseCSV — malformed inputs', () => {
  test('line with too few columns produces an error', () => {
    const csv = '1,2';
    const {examples, errors} = parseCSV(csv);
    expect(errors.length).toBeGreaterThan(0);
    expect(examples).toHaveLength(0);
  });

  test('non-numeric x produces an error', () => {
    // Put it on line 2 to avoid the auto-skip-header heuristic (which only applies to line 1)
    const csv = '1,2,1\nabc,2,1';
    const {errors} = parseCSV(csv);
    expect(errors.length).toBeGreaterThan(0);
    expect(errors[0]).toMatch(/x/i);
  });

  test('non-numeric y produces an error', () => {
    const csv = '1,abc,1';
    const {errors} = parseCSV(csv);
    expect(errors.length).toBeGreaterThan(0);
    expect(errors[0]).toMatch(/y/i);
  });

  test('non-numeric label produces an error', () => {
    const csv = '1,2,bad';
    const {errors} = parseCSV(csv);
    expect(errors.length).toBeGreaterThan(0);
    expect(errors[0]).toMatch(/label/i);
  });

  test('mixed valid and invalid lines — valid ones still parse', () => {
    const csv = '1,2,1\nbad\n-1,-2,-1';
    const {examples, errors} = parseCSV(csv);
    expect(examples).toHaveLength(2);
    expect(errors).toHaveLength(1);
  });
});

// ─── serializeCSV ─────────────────────────────────────────────────────────────

describe('serializeCSV', () => {
  test('produces a header row + one row per example', () => {
    const examples: Example2D[] = [
      {x: 1, y: 2, label: 1},
      {x: -1, y: -2, label: -1},
    ];
    const csv = serializeCSV(examples);
    const lines = csv.split('\n');
    expect(lines[0]).toBe('x,y,label');
    expect(lines).toHaveLength(3);
  });

  test('round-trips through parseCSV', () => {
    const original: Example2D[] = [
      {x: 0.5, y: -1.5, label: 1},
      {x: 2.0, y: 3.0, label: -1},
    ];
    const csv = serializeCSV(original);
    const {examples, errors} = parseCSV(csv);
    expect(errors).toHaveLength(0);
    expect(examples).toHaveLength(2);
    expect(examples[0].x).toBeCloseTo(original[0].x);
    expect(examples[0].y).toBeCloseTo(original[0].y);
    expect(examples[0].label).toBe(original[0].label);
  });
});

// ─── validateExample ──────────────────────────────────────────────────────────

describe('validateExample', () => {
  test('returns null for valid example', () => {
    expect(validateExample(1.0, -2.0, 1)).toBeNull();
    expect(validateExample(0, 0, -1)).toBeNull();
  });

  test('returns error string for NaN x', () => {
    const result = validateExample(NaN, 0, 1);
    expect(result).toBeTruthy();
    expect(typeof result).toBe('string');
  });

  test('returns error string for NaN y', () => {
    const result = validateExample(0, NaN, 1);
    expect(result).toBeTruthy();
  });

  test('returns error string for label not in {-1, 1}', () => {
    expect(validateExample(0, 0, 0)).toBeTruthy();
    expect(validateExample(0, 0, 2)).toBeTruthy();
    expect(validateExample(0, 0, 0.5)).toBeTruthy();
  });

  test('accepts label 1 and -1 specifically', () => {
    expect(validateExample(1, 1, 1)).toBeNull();
    expect(validateExample(1, 1, -1)).toBeNull();
  });
});

// ─── trainTestSplit ───────────────────────────────────────────────────────────

describe('trainTestSplit', () => {
  const makeData = (n: number): Example2D[] =>
    Array.from({length: n}, (_, i) => ({x: i, y: i, label: i % 2 === 0 ? 1 : -1}));

  test('default 80/20 split on 10 examples', () => {
    const data = makeData(10);
    const {train, test} = trainTestSplit(data);
    expect(train).toHaveLength(8);
    expect(test).toHaveLength(2);
  });

  test('preserves all examples (no duplicates / no loss)', () => {
    const data = makeData(20);
    const {train, test} = trainTestSplit(data, 0.7);
    expect(train.length + test.length).toBe(20);
    // Check all x values present
    const allX = new Set([...train.map(e => e.x), ...test.map(e => e.x)]);
    expect(allX.size).toBe(20);
  });

  test('trainRatio=1.0 puts all in train', () => {
    const data = makeData(10);
    const {train, test} = trainTestSplit(data, 1.0);
    expect(train).toHaveLength(10);
    expect(test).toHaveLength(0);
  });

  test('trainRatio=0 puts all in test', () => {
    const data = makeData(10);
    const {train, test} = trainTestSplit(data, 0.0);
    expect(train).toHaveLength(0);
    expect(test).toHaveLength(10);
  });

  test('does not mutate original array', () => {
    const data = makeData(10);
    const originalFirst = data[0].x;
    trainTestSplit(data, 0.8);
    expect(data[0].x).toBe(originalFirst);
  });
});
