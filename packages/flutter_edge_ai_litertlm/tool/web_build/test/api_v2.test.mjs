// Bundle API v2 (R5 §7) through the REAL litert_embeddings_api.js, with
// @litertjs/core and the SentencePiece processor stubbed: the bundle knows no
// prefix, reports the model's dimension and input length, never truncates,
// and tokenizes empty text like native. Each case imports a fresh copy of the
// module, because its state is module-level.
//
// Run: npm test (from tool/web_build)
import { register } from 'node:module';
import { test } from 'node:test';
import assert from 'node:assert/strict';

register('./loader.mjs', import.meta.url);

globalThis.window = globalThis;
globalThis.fetch = async () => ({ ok: true, arrayBuffer: async () => new ArrayBuffer(1) });
console.log = () => {};
console.warn = () => {};

let instance = 0;

/// One id per code point (+100, so no id collides with BOS 2 / EOS 1 / PAD 0),
/// and every text the bundle tokenized, in order.
const encoded = [];
const encodeIds = (text) => {
  encoded.push(text);
  return Array.from(text, (c) => c.codePointAt(0) + 100);
};

/// A compiled model with input `[1, seq]` and output `[1, dim]` unless the
/// shapes are given. `runs` holds the ids of every forward run.
function fakeModel({ seq = 8, dim = 4, inputShape, outputShape } = {}) {
  const runs = [];
  return {
    runs,
    options: { accelerator: 'webgpu' },
    isFullyAccelerated: true,
    getInputDetails: () => [{ shape: Int32Array.from(inputShape ?? [1, seq]) }],
    getOutputDetails: () => [{ shape: Int32Array.from(outputShape ?? [1, dim]) }],
    run: async (tensor) => {
      runs.push(Array.from(tensor.data));
      return [{
        accelerator: 'wasm',
        deleted: false,
        toTypedArray: () => Float32Array.from({ length: dim }, (_, i) => i / 10),
        delete() { this.deleted = true; },
      }];
    },
    delete() {},
  };
}

/// Imports a fresh bundle and loads [model] through its window API.
async function loadFresh(model) {
  globalThis.__compile = () => model;
  globalThis.__encodeIds = encodeIds;
  globalThis.__tensors = [];
  encoded.length = 0;
  await import(`../litert_embeddings_api.js?v2=${instance++}`);
  return window.loadLiteRtEmbeddings('model.tflite', 'tokenizer.model', null);
}

test('the bundle announces API version 2, and the v1 prompt functions are gone', async () => {
  await loadFresh(fakeModel());
  assert.equal(window.liteRtEmbeddingsApiVersion, 2);
  for (const v1 of ['generateEmbedding', 'generateDocumentEmbedding', 'generateEmbeddings']) {
    assert.equal(window[v1], undefined, `${v1} still exists`);
  }
});

test('load reports the dimension and input length read from the model', async () => {
  assert.deepEqual(await loadFresh(fakeModel({ seq: 8, dim: 4 })), { dimension: 4, maxInputTokens: 8 });
  assert.deepEqual(
    await loadFresh(fakeModel({ seq: 256, dim: 768 })),
    { dimension: 768, maxInputTokens: 256 },
  );
  assert.equal(window.getLiteRtEmbeddingDimension(), 768, 'the model\'s, not a constant');
});

test('a model whose input length or dimension cannot be read fails the load', async () => {
  await assert.rejects(loadFresh(fakeModel({ inputShape: [1] })), /sequence length/);
  await assert.rejects(loadFresh(fakeModel({ inputShape: [1, -1] })), /sequence length/);
  await assert.rejects(loadFresh(fakeModel({ outputShape: [1, -1] })), /dimension/);
  await assert.rejects(loadFresh(fakeModel({ outputShape: [] })), /dimension/);
});

test('embedLiteRtText tokenizes the text exactly as given — no prefix, no "▁" — '
  + 'between BOS 2 and EOS 1, padded with 0', async () => {
  const model = fakeModel({ seq: 8, dim: 4 });
  await loadFresh(model);
  const result = await window.embedLiteRtText('ab');

  assert.deepEqual(encoded, ['ab']);
  assert.deepEqual(model.runs, [[2, 197, 198, 1, 0, 0, 0, 0]]);
  assert.equal(result.status, 'ok');
  assert.ok(result.values instanceof Float32Array);
  assert.deepEqual(Array.from(result.values), [0, 0.1, 0.2, 0.3].map(Math.fround));
  assert.ok(globalThis.__tensors.every((t) => t.deleted), 'the input tensor was freed');
});

test('exactly maxInputTokens ids run; one more is too_long, untruncated and never run', async () => {
  const model = fakeModel({ seq: 8 });
  await loadFresh(model);

  // 6 code points + BOS + EOS = 8 ids.
  assert.equal((await window.embedLiteRtText('abcdef')).status, 'ok');
  assert.deepEqual(
    await window.embedLiteRtText('abcdefg'),
    { status: 'too_long', actualTokens: 9, maxInputTokens: 8 },
  );
  assert.equal(model.runs.length, 1, 'the over-long input reached the model');
});

test('empty and whitespace-only text embed like native: [2, ...encodeIds(text), 1]', async () => {
  const model = fakeModel({ seq: 8 });
  await loadFresh(model);

  assert.equal((await window.embedLiteRtText('')).status, 'ok');
  assert.equal((await window.embedLiteRtText('  ')).status, 'ok');
  assert.deepEqual(model.runs, [
    [2, 1, 0, 0, 0, 0, 0, 0],
    [2, 132, 132, 1, 0, 0, 0, 0],
  ]);
});

test('tokenizeLiteRtText returns the ids embedLiteRtText would run, unpadded', async () => {
  const model = fakeModel({ seq: 8 });
  await loadFresh(model);
  const ids = window.tokenizeLiteRtText('task: x');

  assert.ok(ids instanceof Int32Array);
  assert.deepEqual(Array.from(ids), [2, ...Array.from('task: x', (c) => c.codePointAt(0) + 100), 1]);
  // It counts past the model's limit instead of cutting there.
  assert.equal(window.tokenizeLiteRtText('0123456789').length, 12);
  assert.equal(model.runs.length, 0, 'tokenizing runs nothing');
});

test('a non-string input is refused, not coerced', async () => {
  await loadFresh(fakeModel());
  await assert.rejects(window.embedLiteRtText(undefined), /string/);
  assert.throws(() => window.tokenizeLiteRtText(42), /string/);
});

test('cleanup forgets the model: embed and tokenize throw until the next load', async () => {
  await loadFresh(fakeModel({ seq: 8, dim: 4 }));
  await window.cleanupLiteRtEmbeddings();

  assert.equal(window.getLiteRtEmbeddingDimension(), null);
  await assert.rejects(window.embedLiteRtText('a'), /not initialized/);
  assert.throws(() => window.tokenizeLiteRtText('a'), /not initialized/);

  // The next load reads its own model's shapes.
  globalThis.__compile = () => fakeModel({ seq: 16, dim: 2 });
  assert.deepEqual(
    await window.loadLiteRtEmbeddings('m', 't', null),
    { dimension: 2, maxInputTokens: 16 },
  );
});
