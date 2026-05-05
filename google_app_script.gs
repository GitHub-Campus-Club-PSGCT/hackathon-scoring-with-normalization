/**
 * Google Apps Script version of score capture logic from app.py.
 *
 * Expected "Scores" sheet columns:
 *   timestamp | teamid | judgeid | <metric1> | <metric2> | ...
 *
 * Required:
 * - teamid (non-empty)
 * - judgeid (non-empty)
 * - metric columns must be numbers >= 1
 *
 * Behavior:
 * - Validates a score object.
 * - Logs errors to console (console.error) when invalid.
 * - Upserts one record per (teamid, judgeid): updates existing row if found,
 *   otherwise appends a new row.
 */

const SCORES_SHEET_NAME = 'Scores';
const NORMALIZED_SHEET_NAME = 'NormalizedResults';
const TEAM_ID_COLUMN = 'teamid';
const JUDGE_ID_COLUMN = 'judgeid';
const TIMESTAMP_COLUMN = 'timestamp';
const DEFAULT_METRIC_MIN = 1;

/**
 * Optional metric-specific bounds.
 * Example:
 * const METRIC_RANGES = {
 *   innovation: { min: 1, max: 10 },
 *   feasibility: { min: 1, max: 5 }
 * };
 */
const METRIC_RANGES = {
  innovation: { min: 1, max: 10 },
  execution: { min: 1, max: 10 },
  impact: { min: 1, max: 10 },
  scalability: { min: 1, max: 10 },
  presentation: { min: 1, max: 10 }
};

/**
 * Optional metric scoring behavior.
 * - type: 'adder' (default) or 'multiplier'
 * - for multiplier metrics, raw value is mapped from [min,max] to [mapMin,mapMax]
 */
const METRIC_CONFIG = {
  innovation: { type: 'adder' },
  execution: { type: 'adder' },
  impact: { type: 'adder' },
  scalability: { type: 'adder' },
  presentation: { type: 'adder' }
};

/**
 * Main API: validate and save one score submission.
 * @param {Object} scoreData Example:
 * {
 *   teamid: "TEAM-01",
 *   judgeid: "JUDGE-02",
 *   innovation: 8,
 *   impact: 7
 * }
 * @return {{ok: boolean, message: string, errors?: string[]}}
 */
function validateAndSaveScore(scoreData) {
  if (scoreData === undefined) {
    console.log(
      'No input object provided. Validating existing rows, then running normalization.'
    );
    const validation = validateExistingRows();
    if (!validation.ok) {
      return {
        ok: false,
        message: `Validation failed. Normalization skipped. ${validation.message}`,
        errors: ['Normalization skipped because sheet contains invalid rows.']
      };
    }
    return recomputeAndSaveNormalizedResults();
  }

  const validation = validateScoreData(scoreData);
  if (!validation.ok) {
    validation.errors.forEach((err) => console.error(err));
    return validation;
  }

  try {
    upsertScoreRow(scoreData);
    const normalization = recomputeAndSaveNormalizedResults();
    return {
      ok: true,
      message: `Score saved successfully. ${normalization.message}`
    };
  } catch (error) {
    const message = `Failed to save score: ${error.message}`;
    console.error(message);
    return { ok: false, message, errors: [message] };
  }
}

/**
 * Validate teamid, judgeid, and metric values.
 * @param {Object} scoreData
 * @return {{ok: boolean, message: string, errors: string[]}}
 */
function validateScoreData(scoreData) {
  const errors = [];

  if (!scoreData || typeof scoreData !== 'object') {
    errors.push('Invalid payload: score data must be an object.');
    return {
      ok: false,
      message: 'Validation failed.',
      errors
    };
  }

  const teamid = safeString(scoreData[TEAM_ID_COLUMN]);
  const judgeid = safeString(scoreData[JUDGE_ID_COLUMN]);

  if (!teamid) {
    errors.push('Validation error: teamid is required.');
  }
  if (!judgeid) {
    errors.push('Validation error: judgeid is required.');
  }

  const metricKeys = Object.keys(scoreData).filter(
    (key) => key !== TEAM_ID_COLUMN && key !== JUDGE_ID_COLUMN && key !== TIMESTAMP_COLUMN
  );

  if (metricKeys.length === 0) {
    errors.push('Validation error: at least one metric field is required.');
  }

  metricKeys.forEach((metric) => {
    const raw = scoreData[metric];
    const value = Number(raw);
    const min = getMetricMin(metric);
    const max = getMetricMax(metric);

    if (!Number.isFinite(value)) {
      errors.push(`Validation error: metric "${metric}" must be a number.`);
      return;
    }
    if (value < min) {
      errors.push(`Validation error: metric "${metric}" must be >= ${min}.`);
      return;
    }
    if (max !== null && value > max) {
      errors.push(`Validation error: metric "${metric}" must be <= ${max}.`);
    }
  });

  if (errors.length > 0) {
    return {
      ok: false,
      message: 'Validation failed.',
      errors
    };
  }

  return {
    ok: true,
    message: 'Validation passed.',
    errors: []
  };
}

/**
 * Upsert row in Scores sheet by (teamid, judgeid).
 * @param {Object} scoreData
 */
function upsertScoreRow(scoreData) {
  const sheet = getOrCreateScoresSheet_();
  const headers = getHeaders_(sheet);

  const teamIdx = getColumnIndex_(headers, TEAM_ID_COLUMN);
  const judgeIdx = getColumnIndex_(headers, JUDGE_ID_COLUMN);
  const timestampIdx = getColumnIndex_(headers, TIMESTAMP_COLUMN);

  if (teamIdx === -1 || judgeIdx === -1) {
    throw new Error(
      `Sheet "${SCORES_SHEET_NAME}" must include "${TEAM_ID_COLUMN}" and "${JUDGE_ID_COLUMN}" headers.`
    );
  }

  const lastRow = sheet.getLastRow();
  const existingData =
    lastRow > 1 ? sheet.getRange(2, 1, lastRow - 1, headers.length).getValues() : [];

  const teamid = safeString(scoreData[TEAM_ID_COLUMN]);
  const judgeid = safeString(scoreData[JUDGE_ID_COLUMN]);
  const now = new Date();

  const rowValues = headers.map((header) => {
    if (header === TIMESTAMP_COLUMN) return now;
    if (header in scoreData) return scoreData[header];
    return '';
  });

  let targetRow = -1;
  for (let i = 0; i < existingData.length; i++) {
    const row = existingData[i];
    if (safeString(row[teamIdx]) === teamid && safeString(row[judgeIdx]) === judgeid) {
      targetRow = i + 2;
      break;
    }
  }

  if (targetRow !== -1) {
    sheet.getRange(targetRow, 1, 1, headers.length).setValues([rowValues]);
    console.log(`Updated existing score for teamid=${teamid}, judgeid=${judgeid}`);
  } else {
    sheet.appendRow(rowValues);
    console.log(`Inserted new score for teamid=${teamid}, judgeid=${judgeid}`);
  }
}

/**
 * Helper for manual testing in Apps Script editor.
 */
function testSaveScore() {
  const sample = {
    teamid: 'TEAM-01',
    judgeid: 'JUDGE-01',
    innovation: 8,
    execution: 7,
    impact: 9
  };

  const result = validateAndSaveScore(sample);
  console.log(JSON.stringify(result));
}

/**
 * Validate all existing rows in the Scores sheet.
 * Useful when data already exists (for example imported CSV).
 * Logs row-level errors to console without modifying sheet values.
 * @return {{ok: boolean, message: string, checkedRows: number, invalidRows: number}}
 */
function validateExistingRows() {
  const sheet = getOrCreateScoresSheet_();
  const headers = getHeaders_(sheet);
  const lastRow = sheet.getLastRow();

  if (lastRow <= 1) {
    const message = 'No data rows found in Scores sheet.';
    console.log(message);
    return { ok: true, message, checkedRows: 0, invalidRows: 0 };
  }

  const values = sheet.getRange(2, 1, lastRow - 1, headers.length).getValues();
  let invalidRows = 0;

  values.forEach((row, idx) => {
    const rowNumber = idx + 2;
    const rowObject = {};

    headers.forEach((header, colIdx) => {
      rowObject[header] = row[colIdx];
    });

    const result = validateScoreData(rowObject);
    if (!result.ok) {
      invalidRows += 1;
      result.errors.forEach((err) => console.error(`Row ${rowNumber}: ${err}`));
    }
  });

  const checkedRows = values.length;
  const ok = invalidRows === 0;
  const message = ok
    ? `Validation passed for ${checkedRows} row(s).`
    : `Validation failed for ${invalidRows} of ${checkedRows} row(s).`;

  console.log(message);
  return { ok, message, checkedRows, invalidRows };
}

/**
 * Recompute normalized rankings from Scores and write to NormalizedResults sheet.
 * @return {{ok: boolean, message: string, teams?: number}}
 */
function recomputeAndSaveNormalizedResults() {
  try {
    const rankings = normalizeScores_();
    const targetSheet = getOrCreateSheet_(NORMALIZED_SHEET_NAME);
    writeNormalizedResults_(targetSheet, rankings);
    const message = `Normalized results saved to "${NORMALIZED_SHEET_NAME}" (${rankings.length} team(s)).`;
    console.log(message);
    return { ok: true, message, teams: rankings.length };
  } catch (error) {
    const message = `Normalization failed: ${error.message}`;
    console.error(message);
    return { ok: false, message };
  }
}

/**
 * Toolbar-friendly runner: only recompute rankings.
 */
function runNormalization() {
  return recomputeAndSaveNormalizedResults();
}

function normalizeScores_() {
  const sheet = getOrCreateScoresSheet_();
  const headers = getHeaders_(sheet);
  const lastRow = sheet.getLastRow();
  if (lastRow <= 1) return [];

  const metricHeaders = headers.filter(
    (h) => h !== TIMESTAMP_COLUMN && h !== TEAM_ID_COLUMN && h !== JUDGE_ID_COLUMN
  );

  if (metricHeaders.length === 0) {
    throw new Error('No metric columns found in Scores sheet.');
  }

  const rows = sheet.getRange(2, 1, lastRow - 1, headers.length).getValues();
  const entries = [];

  rows.forEach((row, idx) => {
    const rowNumber = idx + 2;
    const entry = {};
    headers.forEach((header, colIdx) => {
      entry[header] = row[colIdx];
    });

    const validation = validateScoreData(entry);
    if (!validation.ok) {
      validation.errors.forEach((err) => console.error(`Row ${rowNumber}: ${err}`));
      return;
    }

    const teamid = safeString(entry[TEAM_ID_COLUMN]);
    const judgeid = safeString(entry[JUDGE_ID_COLUMN]);
    const adderSum = metricHeaders.reduce((sum, metric) => {
      if (getMetricType_(metric) !== 'multiplier') {
        return sum + Number(entry[metric]);
      }
      return sum;
    }, 0);

    const multiplierProduct = metricHeaders.reduce((prod, metric) => {
      if (getMetricType_(metric) === 'multiplier') {
        return prod * mapMultiplierMetric_(metric, Number(entry[metric]));
      }
      return prod;
    }, 1);

    entries.push({
      teamid,
      judgeid,
      totalRaw: adderSum * multiplierProduct
    });
  });

  if (entries.length === 0) return [];

  const entriesByJudge = {};
  entries.forEach((e) => {
    if (!entriesByJudge[e.judgeid]) entriesByJudge[e.judgeid] = [];
    entriesByJudge[e.judgeid].push(e);
  });

  const scoredEntries = [];
  Object.keys(entriesByJudge).forEach((judgeid) => {
    const judgeEntries = entriesByJudge[judgeid];
    const rawScores = judgeEntries.map((e) => e.totalRaw);
    const mean = mean_(rawScores);
    const std = stddev_(rawScores);

    judgeEntries.forEach((e) => {
      const z = std > 0 ? (e.totalRaw - mean) / std : 0;
      const normalized = clamp_(((z + 3) / 6) * 100, 0, 100);
      scoredEntries.push({
        teamid: e.teamid,
        judgeid: e.judgeid,
        raw: e.totalRaw,
        normalized: normalized
      });
    });
  });

  const byTeam = {};
  scoredEntries.forEach((e) => {
    if (!byTeam[e.teamid]) byTeam[e.teamid] = [];
    byTeam[e.teamid].push(e);
  });

  const rankings = Object.keys(byTeam).map((teamid) => {
    const teamScores = byTeam[teamid];
    return {
      teamid: teamid,
      avgRawScore: round2_(mean_(teamScores.map((x) => x.raw))),
      avgNormalizedScore: round2_(mean_(teamScores.map((x) => x.normalized))),
      numJudges: teamScores.length
    };
  });

  rankings.sort((a, b) => b.avgNormalizedScore - a.avgNormalizedScore);
  return rankings;
}

function writeNormalizedResults_(sheet, rankings) {
  sheet.clearContents();
  const header = ['rank', 'teamid', 'avg_raw_score', 'avg_normalized_score', 'num_judges'];
  sheet.getRange(1, 1, 1, header.length).setValues([header]);

  if (rankings.length === 0) {
    return;
  }

  const values = rankings.map((r, idx) => [
    idx + 1,
    r.teamid,
    r.avgRawScore,
    r.avgNormalizedScore,
    r.numJudges
  ]);
  sheet.getRange(2, 1, values.length, header.length).setValues(values);
}

function getOrCreateScoresSheet_() {
  const spreadsheet = SpreadsheetApp.getActiveSpreadsheet();
  let sheet = spreadsheet.getSheetByName(SCORES_SHEET_NAME);
  if (!sheet) {
    sheet = spreadsheet.insertSheet(SCORES_SHEET_NAME);
    sheet.appendRow([TIMESTAMP_COLUMN, TEAM_ID_COLUMN, JUDGE_ID_COLUMN]);
  }
  return sheet;
}

function getOrCreateSheet_(name) {
  const spreadsheet = SpreadsheetApp.getActiveSpreadsheet();
  let sheet = spreadsheet.getSheetByName(name);
  if (!sheet) {
    sheet = spreadsheet.insertSheet(name);
  }
  return sheet;
}

function getHeaders_(sheet) {
  const lastColumn = sheet.getLastColumn();
  if (lastColumn === 0) {
    sheet.appendRow([TIMESTAMP_COLUMN, TEAM_ID_COLUMN, JUDGE_ID_COLUMN]);
    return [TIMESTAMP_COLUMN, TEAM_ID_COLUMN, JUDGE_ID_COLUMN];
  }
  return sheet
    .getRange(1, 1, 1, lastColumn)
    .getValues()[0]
    .map((h) => safeString(h));
}

function getColumnIndex_(headers, name) {
  return headers.indexOf(name);
}

function safeString(value) {
  return value === null || value === undefined ? '' : String(value).trim();
}

function getMetricMin(metricName) {
  if (METRIC_RANGES[metricName] && Number.isFinite(Number(METRIC_RANGES[metricName].min))) {
    return Number(METRIC_RANGES[metricName].min);
  }
  return DEFAULT_METRIC_MIN;
}

function getMetricMax(metricName) {
  if (METRIC_RANGES[metricName] && Number.isFinite(Number(METRIC_RANGES[metricName].max))) {
    return Number(METRIC_RANGES[metricName].max);
  }
  return null;
}

function getMetricType_(metricName) {
  const cfg = METRIC_CONFIG[metricName];
  if (cfg && cfg.type === 'multiplier') return 'multiplier';
  return 'adder';
}

function mapMultiplierMetric_(metricName, value) {
  const min = getMetricMin(metricName);
  const max = getMetricMax(metricName);
  const cfg = METRIC_CONFIG[metricName] || {};
  const mapMin = Number.isFinite(Number(cfg.mapMin)) ? Number(cfg.mapMin) : 0.0;
  const mapMax = Number.isFinite(Number(cfg.mapMax)) ? Number(cfg.mapMax) : 1.0;

  if (!Number.isFinite(value)) return mapMin;
  if (max === null || max === min) return mapMin;

  const mapped = mapMin + ((value - min) / (max - min)) * (mapMax - mapMin);
  return clamp_(mapped, mapMin, mapMax);
}

function mean_(arr) {
  if (!arr || arr.length === 0) return 0;
  return arr.reduce((sum, x) => sum + Number(x), 0) / arr.length;
}

function stddev_(arr) {
  if (!arr || arr.length <= 1) return 0;
  const m = mean_(arr);
  const variance = arr.reduce((sum, x) => sum + Math.pow(Number(x) - m, 2), 0) / arr.length;
  return Math.sqrt(variance);
}

function clamp_(value, min, max) {
  return Math.max(min, Math.min(value, max));
}

function round2_(value) {
  return Math.round(Number(value) * 100) / 100;
}
