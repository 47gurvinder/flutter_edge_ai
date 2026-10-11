// Real JavaScript errors — a synchronous `throw` and a rejected promise — from
// the bundle's window API, through the REAL path an app takes: the web
// backend's describe(), the LiteRT.js pass over the window bundle, and core's
// inline host. A load failure must reach the caller as an
// `EmbeddingLoadException`, an embed failure as an `EmbeddingEngineException`,
// both tagged 'LiteRT.js' and carrying the JS message. A page without the
// bundle at all fails the load by naming the missing file, and the close
// behind that failure has nothing to dispose. The stubs are defined with
// `eval`, so they are plain JS functions, not Dart closures.
//
// Not run by tool/test_all.sh or CI, which run on the VM. Run:
//   flutter test --platform chrome test/web/litert_js_embedding_js_errors_test.dart
@TestOn('browser')
library;

import 'dart:async';
import 'dart:js_interop';
import 'dart:js_interop_unsafe';

import 'package:flutter_edge_ai/engine.dart';
// ignore: implementation_imports — drives a pass through core's inline host, which is internal to core
import 'package:flutter_edge_ai/src/embedding/embedding_host.dart';
// ignore: implementation_imports — drives a pass through core's inline host, which is internal to core
import 'package:flutter_edge_ai/src/embedding/embedding_host_web.dart'
    as inline;
import 'package:flutter_edge_ai_litertlm/src/embedding/litert_embedding_backend_web.dart';
import 'package:flutter_test/flutter_test.dart';

void _js(String source) => globalContext.callMethod('eval'.toJS, source.toJS);

/// A bundle whose every call succeeds; each test then breaks one function.
void _installWorkingBundle() => _js('''
  window.loadLiteRtEmbeddings = async function (model, tokenizer, wasm) {};
  window.getLiteRtEmbeddingDimension = function () { return 4; };
  window.isLiteRtEmbeddingsInitialized = function () { return true; };
  window.generateEmbedding = async function (text) {
    return new Float32Array([0.5, 0.5, 0.5, 0.5]);
  };
  window.generateDocumentEmbedding = window.generateEmbedding;
  window.cleanupLiteRtEmbeddings = async function () {};
''');

/// What a page without `<script src="litert_embeddings.js">` has: none of
/// the bundle's window functions.
void _removeBundle() => _js('''
  delete window.loadLiteRtEmbeddings;
  delete window.getLiteRtEmbeddingDimension;
  delete window.isLiteRtEmbeddingsInitialized;
  delete window.generateEmbedding;
  delete window.generateDocumentEmbedding;
  delete window.cleanupLiteRtEmbeddings;
''');

Future<EmbeddingHost> _start({
  void Function(Future<void> cleanup)? onFailedLoadCleanup,
}) async {
  final descriptor = await const LiteRtEmbeddingBackend().describe(
    EmbeddingModelSpec(
      name: 'embeddinggemma',
      modelSource: ModelSource.network('https://example.com/gemma.tflite'),
      tokenizerSource: ModelSource.network('https://example.com/t.model'),
    ),
    const RuntimeConfig(
      maxTokens: 0,
      modelPath: 'blob:model',
      tokenizerPath: 'blob:tokenizer',
    ),
  );
  return inline.startEmbeddingHost(
    descriptor,
    declaredMedia: const {},
    label: 'gemma.tflite',
    onFailedLoadCleanup: onFailedLoadCleanup,
  );
}

void main() {
  group('a load that fails in JS is an EmbeddingLoadException', () {
    for (final (how, stub) in [
      (
        'a synchronous throw',
        "window.loadLiteRtEmbeddings = function () { throw new Error('sync load boom'); };",
      ),
      (
        'a rejected promise',
        "window.loadLiteRtEmbeddings = async function () { throw new Error('async load boom'); };",
      ),
    ]) {
      test(how, () async {
        _installWorkingBundle();
        _js(stub);
        await expectLater(
          _start(),
          throwsA(
            isA<EmbeddingLoadException>()
                .having((e) => e.engine, 'engine', 'LiteRT.js')
                .having((e) => e.message, 'message', contains('load boom')),
          ),
        );
      });
    }
  });

  test('a page without litert_embeddings.js: the load names the file, and '
      'the close behind it warns about nothing', () async {
    _removeBundle();
    final printed = <String>[];
    Future<void>? cleanup;
    await runZoned(
      () => expectLater(
        _start(onFailedLoadCleanup: (c) => cleanup = c),
        throwsA(
          isA<EmbeddingLoadException>()
              .having((e) => e.engine, 'engine', 'LiteRT.js')
              .having(
                (e) => e.message,
                'message',
                contains('litert_embeddings.js'),
              ),
        ),
      ),
      zoneSpecification: ZoneSpecification(
        print: (self, parent, zone, line) => printed.add(line),
      ),
    );
    await cleanup;
    expect(
      printed.where((l) => l.contains('failed to close')),
      isEmpty,
      reason: 'there was no page state, so nothing failed to close',
    );
  });

  group('an embed that fails in JS is an EmbeddingEngineException', () {
    for (final (how, stub) in [
      (
        'a synchronous throw',
        "window.generateEmbedding = function () { throw new Error('sync embed boom'); };",
      ),
      (
        'a rejected promise',
        "window.generateEmbedding = async function () { throw new Error('async embed boom'); };",
      ),
    ]) {
      test(how, () async {
        _installWorkingBundle();
        final host = await _start();
        _js(stub);
        try {
          await expectLater(
            host.embed(EmbeddingInput.text('q'), TaskType.retrievalQuery),
            throwsA(
              isA<EmbeddingEngineException>()
                  .having((e) => e.engine, 'engine', 'LiteRT.js')
                  .having((e) => e.message, 'message', contains('embed boom')),
            ),
          );
          // The host keeps serving after a failed request.
          _installWorkingBundle();
          expect(
            await host.embed(EmbeddingInput.text('q'), TaskType.retrievalQuery),
            hasLength(4),
          );
        } finally {
          await host.close();
        }
      });
    }
  });
}
