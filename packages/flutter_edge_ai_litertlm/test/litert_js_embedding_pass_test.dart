// `LiteRtJsEmbeddingPass` (web, LiteRT.js bundle v1) on the VM, over a fake
// bundle that keeps ONE page-global state the way `litert_embeddings.js`
// does. The real interop is the browser suite's
// (test/web/litert_js_embedding_pass_page_lane_test.dart); here: what reaches
// the bundle, the v1 task-type limits, and the page-wide order of loads and
// disposes across passes.

import 'dart:async';

import 'package:flutter_edge_ai/engine.dart';
import 'package:flutter_edge_ai_litertlm/src/embedding/web/litert_js_embedding_pass.dart';
import 'package:flutter_test/flutter_test.dart';

/// The page-global state of the bundle, shared by every pass on the "page".
final class _FakePage implements LiteRtJsBundle {
  _FakePage({this.loadDelays = const [Duration.zero]});

  /// Per load, in order; the last one repeats.
  final List<Duration> loadDelays;

  /// When set, the next load fails after fetching the tokenizer — as v1
  /// does when the model fails to compile.
  bool failNextLoad = false;

  /// When true, `web/litert_embeddings.js` is not on the page: none of its
  /// window functions exist, so every call into them throws, as the real
  /// interop does (`loadLiteRtEmbeddings is not a function`).
  bool missing = false;

  @override
  bool get isPresent => !missing;

  /// Called as a load starts, while the bundle is reading its URLs.
  void Function()? onLoad;

  bool modelLoaded = false;
  bool tokenizerLoaded = false;
  final loads = <({String model, String tokenizer})>[];
  int disposes = 0;
  final prompts = <({String kind, String text})>[];

  @override
  Future<void> load(
    String modelUrl,
    String tokenizerUrl,
    String wasmPath,
  ) async {
    if (missing) throw StateError('loadLiteRtEmbeddings is not a function');
    loads.add((model: modelUrl, tokenizer: tokenizerUrl));
    onLoad?.call();
    await Future<void>.delayed(
      loadDelays[(loads.length - 1).clamp(0, loadDelays.length - 1)],
    );
    tokenizerLoaded = true;
    if (failNextLoad) {
      failNextLoad = false;
      throw StateError('Failed to load LiteRT model: compile failed');
    }
    modelLoaded = true;
  }

  @override
  int get dimension => 4;

  Future<List<double>> _embed(String kind, String text) async {
    if (!modelLoaded) throw StateError('page state was disposed');
    prompts.add((kind: kind, text: text));
    return const [0.5, 0.5, 0.5, 0.5];
  }

  @override
  Future<List<double>> embedQuery(String text) => _embed('query', text);

  @override
  Future<List<double>> embedDocument(String text) => _embed('document', text);

  @override
  Future<void> dispose() async {
    if (missing) throw StateError('cleanupLiteRtEmbeddings is not a function');
    disposes++;
    modelLoaded = false;
    tokenizerLoaded = false;
  }
}

/// What `resolveWebModelUrl` does: `opfs://x` becomes a `blob:` URL,
/// anything else is returned unchanged.
Future<String> _fakeResolve(String path) async => path.startsWith('opfs://')
    ? 'blob:resolved/${path.substring('opfs://'.length)}'
    : path;

LiteRtJsEmbeddingPass _pass(
  _FakePage page, {
  String modelPath = 'blob:model',
  String tokenizerPath = 'blob:tokenizer',
  List<String>? revoked,
}) => LiteRtJsEmbeddingPass(
  LiteRtJsEmbeddingConfig(
    modelPath: modelPath,
    tokenizerPath: tokenizerPath,
    bundle: page,
    resolveUrl: _fakeResolve,
    revokeUrl: (url) => revoked?.add(url),
  ),
);

void main() {
  group('opfs:// paths (WebStorageMode.streaming)', () {
    test('the bundle gets RESOLVED URLs for the model and the tokenizer, never '
        'an opfs:// path its fetch() would reject', () async {
      final page = _FakePage();
      final pass = _pass(
        page,
        modelPath: 'opfs://gemma.tflite',
        tokenizerPath: 'opfs://sentencepiece.model',
      );

      await pass.load();

      expect(page.loads.single, (
        model: 'blob:resolved/gemma.tflite',
        tokenizer: 'blob:resolved/sentencepiece.model',
      ));
      await pass.close();
    });

    test(
      'a URL that needs no resolving reaches the bundle unchanged',
      () async {
        final page = _FakePage();
        final pass = _pass(page);
        await pass.load();
        expect(page.loads.single, (
          model: 'blob:model',
          tokenizer: 'blob:tokenizer',
        ));
        await pass.close();
      },
    );
  });

  group('object URLs the pass created', () {
    test('the blob: URLs made for opfs:// paths are revoked once the bundle '
        'has read them — not before', () async {
      final revoked = <String>[];
      final page = _FakePage()
        ..onLoad = () => expect(revoked, isEmpty, reason: 'still being read');
      final pass = _pass(
        page,
        modelPath: 'opfs://gemma.tflite',
        tokenizerPath: 'opfs://sentencepiece.model',
        revoked: revoked,
      );

      await pass.load();

      expect(
        revoked,
        unorderedEquals([
          'blob:resolved/gemma.tflite',
          'blob:resolved/sentencepiece.model',
        ]),
      );
      await pass.close();
    });

    test('a load that fails still releases both', () async {
      final revoked = <String>[];
      final page = _FakePage()..failNextLoad = true;
      final pass = _pass(
        page,
        modelPath: 'opfs://gemma.tflite',
        tokenizerPath: 'opfs://sentencepiece.model',
        revoked: revoked,
      );

      await expectLater(pass.load(), throwsStateError);
      expect(revoked, hasLength(2));
      await pass.close();
    });

    test('a URL core owns (not opfs://) is never revoked', () async {
      final revoked = <String>[];
      final pass = _pass(_FakePage(), revoked: revoked);
      await pass.load();
      await pass.close();
      expect(revoked, isEmpty);
    });
  });

  group('bundle API v1', () {
    test('the query and document prompts get the RAW text — the bundle adds '
        'the prefix, so the pass must not', () async {
      final page = _FakePage();
      final pass = _pass(page);
      await pass.load();

      await pass.embed(EmbeddingInput.text('q'), TaskType.retrievalQuery);
      await pass.embed(EmbeddingInput.text('d'), TaskType.retrievalDocument);

      expect(page.prompts, [
        (kind: 'query', text: 'q'),
        (kind: 'document', text: 'd'),
      ]);
      await pass.close();
    });

    test(
      'the other seven task types are refused by name and reach nothing',
      () async {
        final page = _FakePage();
        final pass = _pass(page);
        await pass.load();

        for (final type in TaskType.values.skip(2)) {
          await expectLater(
            pass.embed(EmbeddingInput.text('x'), type),
            throwsA(
              isA<EmbeddingEngineException>()
                  .having((e) => e.engine, 'engine', 'LiteRT.js')
                  .having(
                    (e) => e.message,
                    'message',
                    contains('TaskType.${type.name}'),
                  ),
            ),
          );
        }
        expect(page.prompts, isEmpty);
        await pass.close();
      },
    );

    test('load reports the bundle\'s dimension and no input limit (v1 cannot '
        'tell)', () async {
      final info = await _pass(_FakePage()).load();
      expect(info.dimension, 4);
      expect(info.maxInputTokens, isNull);
    });
  });

  group('the page lane', () {
    test('a pass closing while its replacement loads leaves the replacement '
        'its page state', () async {
      // The old pass's load is the SLOW one and its close is queued behind it;
      // the replacement's load is fast. Without the lane the replacement
      // loaded first and the old dispose then deleted its state.
      final page = _FakePage(
        loadDelays: const [
          Duration(milliseconds: 100),
          Duration(milliseconds: 10),
        ],
      );
      final a = _pass(page);
      final aLoad = a.load();
      await Future<void>.delayed(const Duration(milliseconds: 10));
      final aClose = a.close();
      final b = _pass(page);
      final bLoad = b.load();

      await aLoad;
      await aClose;
      await bLoad;

      expect(
        await b.embed(EmbeddingInput.text('z'), TaskType.retrievalQuery),
        hasLength(4),
        reason: 'the replacement lost its page state to the old close',
      );
      expect(page.disposes, 1);
      await b.close();
    });

    test('a failed load leaves nothing on the page: close disposes the '
        'tokenizer v1 loaded before the model failed', () async {
      // v1 loads the tokenizer, then compiles the model; when the compile
      // fails the tokenizer stays allocated, and the next load overwrote it
      // without freeing it — a leaked SentencePiece processor.
      final page = _FakePage()..failNextLoad = true;
      final failed = _pass(page);
      await expectLater(failed.load(), throwsStateError);
      expect(page.tokenizerLoaded, isTrue, reason: 'what v1 leaves behind');

      await failed.close();

      expect(page.disposes, 1);
      expect(page.tokenizerLoaded, isFalse);
    });

    test('a pass that never reached the bundle disposes nothing', () async {
      final page = _FakePage();
      await _pass(page).close();
      final unresolvable = LiteRtJsEmbeddingPass(
        LiteRtJsEmbeddingConfig(
          modelPath: 'opfs://m',
          tokenizerPath: 'opfs://t',
          bundle: page,
          resolveUrl: (_) => Future.error(StateError('no OPFS')),
          revokeUrl: (_) {},
        ),
      );
      await expectLater(unresolvable.load(), throwsStateError);
      await unresolvable.close();
      expect(page.disposes, 0);
    });

    test(
      'a page without web/litert_embeddings.js: the load names the '
      'missing file, nothing is resolved, and close has nothing to dispose',
      () async {
        // The pass used to mark the page touched before it knew the bundle was
        // there, so the close behind the failed load called the missing
        // cleanupLiteRtEmbeddings — and core then printed a second, false
        // warning that a native model "may still be resident".
        final page = _FakePage()..missing = true;
        final resolved = <String>[];
        final pass = LiteRtJsEmbeddingPass(
          LiteRtJsEmbeddingConfig(
            modelPath: 'opfs://m.tflite',
            tokenizerPath: 'opfs://t.model',
            bundle: page,
            resolveUrl: (path) async {
              resolved.add(path);
              return 'blob:$path';
            },
            revokeUrl: (_) {},
          ),
        );

        await expectLater(
          pass.load(),
          throwsA(
            isA<EmbeddingLoadException>()
                .having((e) => e.engine, 'engine', 'LiteRT.js')
                .having(
                  (e) => e.message,
                  'message',
                  contains('litert_embeddings.js'),
                ),
          ),
        );
        expect(resolved, isEmpty, reason: 'no blob: URL made for nothing');
        await pass.close();
        expect(page.disposes, 0);
      },
    );

    test(
      'a failed load does not block the lane: the next pass loads',
      () async {
        final page = _FakePage()..failNextLoad = true;
        final failed = _pass(page);
        await expectLater(failed.load(), throwsStateError);
        await failed.close();

        final next = _pass(page);
        await next.load();
        expect(page.loads, hasLength(2));
        await next.close();
      },
    );
  });
}
