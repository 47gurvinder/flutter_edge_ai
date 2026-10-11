// The LiteRT.js embedding state is one per PAGE (`tfliteModel`, `tokenizer`
// and `isInitialized` in litert_embeddings.js), while a `LiteRtJsEmbeddingPass`
// is one per loaded embedder. These tests stand in for the JS with stubs on
// `window` that share one flag, the way the real bundle does, and pin the
// order of loads and disposes across passes, plus what the interim pass on
// bundle API v1 serves through the real interop: the query and document
// prompts with the raw text, nothing else, and no input limit. The same rules
// over a fake bundle run on the VM (test/litert_js_embedding_pass_test.dart).
//
// Not run by tool/test_all.sh or CI, which run on the VM. Run:
//   flutter test --platform chrome test/web/litert_js_embedding_pass_page_lane_test.dart
@TestOn('browser')
library;

import 'dart:js_interop';
import 'dart:js_interop_unsafe';
import 'dart:typed_data';

import 'package:flutter_edge_ai/engine.dart';
import 'package:flutter_edge_ai_litertlm/src/embedding/web/litert_js_embedding_pass.dart';
import 'package:flutter_edge_ai_litertlm/src/embedding/web/litert_web_embeddings.dart';
import 'package:flutter_test/flutter_test.dart';

/// The page-global state the real bundle keeps, reduced to the one flag the
/// Dart side consults.
bool _pageLoaded = false;
int _loads = 0;
int _disposes = 0;

/// What reached the bundle: which prompt, and the text it was given.
final _prompts = <({String kind, String text})>[];

/// [loadDelays] is per call, in order; the last one repeats.
void _installStubs({required List<Duration> loadDelays}) {
  _pageLoaded = false;
  _loads = 0;
  _disposes = 0;
  _prompts.clear();

  JSPromise<JSAny?> embed(String kind, JSString text) => () async {
    if (!_pageLoaded) throw StateError('page state was disposed');
    _prompts.add((kind: kind, text: text.toDart));
    return Float32List(4).toJS;
  }().toJS;

  globalContext['loadLiteRtEmbeddings'] =
      ((JSAny? model, JSAny? tokenizer, JSAny? wasm) {
        return () async {
          _loads++;
          await Future<void>.delayed(
            loadDelays[(_loads - 1).clamp(0, loadDelays.length - 1)],
          );
          _pageLoaded = true;
        }().toJS;
      }).toJS;
  globalContext['cleanupLiteRtEmbeddings'] = (() {
    return () async {
      _disposes++;
      _pageLoaded = false;
    }().toJS;
  }).toJS;
  globalContext['isLiteRtEmbeddingsInitialized'] =
      (() => _pageLoaded.toJS).toJS;
  globalContext['generateEmbedding'] = ((JSString text) => embed(
    'query',
    text,
  )).toJS;
  globalContext['generateDocumentEmbedding'] = ((JSString text) => embed(
    'document',
    text,
  )).toJS;
  globalContext['getLiteRtEmbeddingDimension'] = (() => 4.toJS).toJS;
}

LiteRtJsEmbeddingPass _pass() => LiteRtJsEmbeddingPass(
  const LiteRtJsEmbeddingConfig(
    modelPath: 'm',
    tokenizerPath: 't',
    bundle: WindowLiteRtJsBundle(),
  ),
);

void main() {
  test('a pass closing while its replacement loads leaves the replacement '
      'its page state', () async {
    // The old pass's load is the SLOW one and its close is queued behind it;
    // the replacement's load is fast. Without the page lane the replacement
    // loaded first and the old pass's dispose then deleted its state.
    _installStubs(
      loadDelays: const [
        Duration(milliseconds: 200),
        Duration(milliseconds: 20),
      ],
    );

    final a = _pass();
    final aLoad = a.load();
    await Future<void>.delayed(const Duration(milliseconds: 20));
    final aClose = a.close(); // deliberately not awaited before b loads

    final b = _pass();
    final bLoad = b.load();

    await aLoad;
    await aClose;
    final info = await bLoad;

    expect(info.dimension, 4);
    expect(
      await b.embed(EmbeddingInput.text('z'), TaskType.retrievalQuery),
      hasLength(4),
      reason: 'the replacement lost its page state to the old close',
    );
    expect(_disposes, 1);
    await b.close();
  });

  test('a failed load does not block the lane: the next pass loads', () async {
    _installStubs(loadDelays: const [Duration.zero]);
    var failNext = true;
    final realLoad = globalContext['loadLiteRtEmbeddings'] as JSFunction;
    globalContext['loadLiteRtEmbeddings'] =
        ((JSAny? model, JSAny? tokenizer, JSAny? wasm) {
          if (failNext) {
            failNext = false;
            return Future<void>.error(StateError('fetch failed')).toJS;
          }
          return realLoad.callAsFunction(null, model, tokenizer, wasm)
              as JSPromise;
        }).toJS;

    final failed = _pass();
    await expectLater(failed.load(), throwsException);
    await failed.close();
    expect(_disposes, 0, reason: 'a pass that never loaded disposes nothing');

    final next = _pass();
    await next.load();
    expect(_loads, 1, reason: 'only the second pass reached the real load');
    await next.close();
    expect(_disposes, 1);
  });

  test('bundle API v1: the query and document prompts, no input limit, and '
      'the other seven task types refused by name', () async {
    _installStubs(loadDelays: const [Duration.zero]);
    final pass = _pass();
    final info = await pass.load();
    expect(info.maxInputTokens, isNull, reason: 'v1 cannot tell');

    await pass.embed(EmbeddingInput.text('q'), TaskType.retrievalQuery);
    await pass.embed(EmbeddingInput.text('d'), TaskType.retrievalDocument);
    // The RAW text: the v1 bundle prepends the prompt itself, so a prefix
    // added in Dart would be embedded twice.
    expect(_prompts, [
      (kind: 'query', text: 'q'),
      (kind: 'document', text: 'd'),
    ]);

    for (final type in TaskType.values.skip(2)) {
      await expectLater(
        pass.embed(EmbeddingInput.text('x'), type),
        throwsA(
          isA<EmbeddingEngineException>().having(
            (e) => e.message,
            'message',
            contains('TaskType.${type.name}'),
          ),
        ),
      );
    }
    expect(_prompts, hasLength(2), reason: 'nothing reached the bundle');
    await pass.close();
  });
}
