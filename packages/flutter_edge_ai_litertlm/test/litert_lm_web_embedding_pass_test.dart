// `LiteRtLmWebEmbeddingPass` (web, EmbeddingGemma 2 over @litert-lm/core's
// EmbeddingEngine) on the VM, over a fake runtime that models what the real
// one does with the settings and options it is given. The real interop is the
// browser's (example/integration_test/web_eg2_test.dart on the GCE VM); here:
// the named error for a page without the shim line, the settings that reach
// `EmbeddingEngine.create` (CPU, maxInputLength 1024, the vision tower only
// when images are declared, the budget), the load probe (dimension, the limit
// the engine enforces, NaN/zero refused), what reaches `computeEmbedding`
// (`taskType.prefix + text` for all nine task types, image bytes untouched,
// normalize + special tokens + TRUNCATE), the typed overflow with the engine's
// exact counts, `opfs://` resolve and revoke, close, and one case through
// core's inline host.

import 'dart:typed_data';

import 'package:flutter_edge_ai/engine.dart';
// ignore: implementation_imports — the app-facing model over core's inline host, which is internal to core
import 'package:flutter_edge_ai/src/embedding/common_embedding_model.dart';
// ignore: implementation_imports — the app-facing model over core's inline host, which is internal to core
import 'package:flutter_edge_ai/src/embedding/embedding_host_web.dart'
    as inline;
import 'package:flutter_edge_ai_litertlm/src/embedding/litert_lm_embedding_common.dart';
import 'package:flutter_edge_ai_litertlm/src/embedding/web/litert_lm_web_embedding_pass.dart';
import 'package:flutter_test/flutter_test.dart';

/// "Tokens" as the fake engine counts them: one per character of a text part
/// (prefix included), plus BOS and EOS when special tokens are on; an image is
/// [_imageTokens].
const _imageTokens = 140;

/// The unit vector the fake returns when asked to normalise, and the raw one
/// otherwise — so a pass that stops asking for normalisation is visible.
const _unit = [0.5, 0.5, 0.5, 0.5];
const _raw = [1.0, 1.0, 1.0, 1.0];

final class _FakeEngine implements LiteRtLmWebEngine {
  _FakeEngine(this.settings, this.runtime);

  final LiteRtLmWebEngineSettings settings;
  final _FakeRuntime runtime;

  final calls = <(LiteRtLmWebInput, LiteRtLmWebEmbeddingOptions)>[];
  int deletes = 0;

  @override
  Future<LiteRtLmWebEmbedding> computeEmbedding(
    LiteRtLmWebInput input,
    LiteRtLmWebEmbeddingOptions options,
  ) async {
    if (deletes > 0) throw StateError('engine deleted');
    calls.add((input, options));
    if (runtime.computeError case final error?) throw error;
    final special = options.insertSpecialTokens ? 2 : 0;
    final tokens = switch (input) {
      LiteRtLmWebText(:final text) => text.length + special,
      LiteRtLmWebImage() => _imageTokens + special,
    };
    // The largest signature the engine prepared.
    final limit = runtime.maxSignature;
    var vector = options.normalize ? _unit : _raw;
    if (runtime.vectorFor case final override?) vector = override(input);
    if (tokens <= limit || runtime.neverTruncates) {
      return (
        embedding: vector,
        inputLength: tokens,
        truncatedLength: null,
        numChunks: 1,
      );
    }
    return switch (options.inputOverflowStrategy) {
      LiteRtLmWebOverflowStrategy.truncate => (
        embedding: vector,
        inputLength: tokens,
        truncatedLength: limit,
        numChunks: 1,
      ),
      LiteRtLmWebOverflowStrategy.chunkAndAverage => (
        embedding: vector,
        inputLength: tokens,
        truncatedLength: null,
        numChunks: (tokens + limit - 1) ~/ limit,
      ),
      LiteRtLmWebOverflowStrategy.error => throw StateError(
        'INVALID_ARGUMENT: Input sequence length ($tokens) exceeds maximum '
        'supported signature length ($limit).',
      ),
    };
  }

  @override
  Future<void> delete() async => deletes++;
}

final class _FakeRuntime implements LiteRtLmWebRuntime {
  LiteRtLmWebRuntimeState state = LiteRtLmWebRuntimeState.ready;
  Object? readyError;
  Object? createError;
  Object? computeError;

  /// The text signature the engine keeps for `maxInputLength` 1024.
  int maxSignature = 1024;

  /// A model whose limit is beyond the probe: nothing is ever truncated.
  bool neverTruncates = false;

  List<double> Function(LiteRtLmWebInput input)? vectorFor;

  int readyChecks = 0;
  final created = <_FakeEngine>[];

  _FakeEngine get engine => created.single;

  @override
  Future<LiteRtLmWebRuntimeState> ensureReady() async {
    readyChecks++;
    if (readyError case final error?) throw error;
    return state;
  }

  @override
  Future<LiteRtLmWebEngine> create(LiteRtLmWebEngineSettings settings) async {
    if (createError case final error?) throw error;
    final engine = _FakeEngine(settings, this);
    created.add(engine);
    return engine;
  }
}

LiteRtLmWebEmbeddingConfig _config(
  _FakeRuntime runtime, {
  String modelPath = 'blob:eg2',
  Set<EmbeddingModality> media = const {},
  int visionTokensPerImage = 140,
  List<String>? resolved,
  List<String>? revoked,
}) => LiteRtLmWebEmbeddingConfig(
  modelPath: modelPath,
  fileName: 'embeddinggemma-2-text-vision-440m.litertlm',
  runtime: runtime,
  resolveUrl: (path) async {
    resolved?.add(path);
    return path.startsWith('opfs://')
        ? 'blob:resolved/${path.substring('opfs://'.length)}'
        : path;
  },
  revokeUrl: (url) => revoked?.add(url),
  declaredMedia: media,
  visionTokensPerImage: visionTokensPerImage,
);

LiteRtLmWebEmbeddingPass _pass(
  _FakeRuntime runtime, {
  Set<EmbeddingModality> media = const {},
  int visionTokensPerImage = 140,
}) => LiteRtLmWebEmbeddingPass(
  _config(runtime, media: media, visionTokensPerImage: visionTokensPerImage),
);

/// A PNG header, enough for core's sniffer.
final _png = Uint8List.fromList([
  0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 1, 2, 3, //
]);

/// A WebP header (RIFF....WEBP), which the engine cannot decode.
final _webp = Uint8List.fromList([
  ...'RIFF'.codeUnits, 0, 0, 0, 0, ...'WEBPVP8 '.codeUnits, 1, 2, 3, //
]);

TypeMatcher<EmbeddingLoadException> _loadError(Object message) =>
    isA<EmbeddingLoadException>()
        .having((e) => e.engine, 'engine', liteRtLmEngineTag)
        .having((e) => e.message, 'message', message);

void main() {
  group('load', () {
    test('a text-only load: CPU, maxInputLength 1024, no vision tower, the '
        'dimension and limit read from the probe', () async {
      final runtime = _FakeRuntime();
      final pass = _pass(runtime);

      final info = await pass.load();

      final settings = runtime.engine.settings;
      expect(settings.modelUrl, 'blob:eg2');
      expect(settings.backend, LiteRtLmWebBackend.cpu);
      expect(settings.maxInputLength, 1024);
      expect(settings.minInputLength, isNull, reason: 'the runtime default');
      expect(settings.visionBackend, isNull, reason: 'unset = no vision tower');
      expect(settings.visionTokensPerImage, isNull);
      expect(info.dimension, 4);
      expect(info.maxInputTokens, 1024);
      expect(info.modalities, {EmbeddingModality.text});
      expect(info.imageFormats, isEmpty);
      await pass.close();
    });

    test('the probe is longer than any EmbeddingGemma 2 signature (8192) and '
        'runs under TRUNCATE, so the limit it reports is the one the engine '
        'enforces', () async {
      final runtime = _FakeRuntime()..maxSignature = 2048;
      final pass = _pass(runtime);

      final info = await pass.load();

      final (probe, options) = runtime.engine.calls.single;
      final words = (probe as LiteRtLmWebText).text.split(' ');
      expect(words.length, greaterThan(8192));
      expect(
        options.inputOverflowStrategy,
        LiteRtLmWebOverflowStrategy.truncate,
      );
      expect(info.maxInputTokens, 2048, reason: 'what the engine cut it at');
      await pass.close();
    });

    test('a model the probe does not overflow reports no limit', () async {
      final runtime = _FakeRuntime()..neverTruncates = true;
      final pass = _pass(runtime);
      expect((await pass.load()).maxInputTokens, isNull);
      await pass.close();
    });

    for (final budget in [140, 70]) {
      test('images declared: the vision tower on the same backend, '
          '$budget tokens per image, {text, image} and the four '
          'containers', () async {
        final runtime = _FakeRuntime();
        final pass = _pass(
          runtime,
          media: {EmbeddingModality.image},
          visionTokensPerImage: budget,
        );

        final info = await pass.load();

        final settings = runtime.engine.settings;
        expect(settings.visionBackend, LiteRtLmWebBackend.cpu);
        expect(settings.visionTokensPerImage, budget);
        expect(info.modalities, {
          EmbeddingModality.text,
          EmbeddingModality.image,
        });
        expect(info.imageFormats, {
          EncodedMediaFormat.jpeg,
          EncodedMediaFormat.png,
          EncodedMediaFormat.gif,
          EncodedMediaFormat.bmp,
        });
        expect(info.imageFormats, liteRtLmImageFormats);
        await pass.close();
      });
    }

    test('a page without @litert-lm/core: the named error, nothing resolved '
        'or created', () async {
      final runtime = _FakeRuntime()
        ..state = LiteRtLmWebRuntimeState.loaderMissing;
      final resolved = <String>[];
      final pass = LiteRtLmWebEmbeddingPass(
        _config(runtime, resolved: resolved),
      );
      await expectLater(
        pass.load(),
        throwsA(
          _loadError(
            allOf(
              contains('web/index.html does not load @litert-lm/core'),
              contains('window.EmbeddingEngine = m.EmbeddingEngine;'),
            ),
          ),
        ),
      );
      expect(resolved, isEmpty);
      expect(runtime.created, isEmpty);
      await pass.close();
    });

    test('a page set up for the LLM engine only (no window.EmbeddingEngine): '
        'the error names the missing line', () async {
      final runtime = _FakeRuntime()
        ..state = LiteRtLmWebRuntimeState.embeddingEngineMissing;
      final pass = _pass(runtime);
      await expectLater(
        pass.load(),
        throwsA(
          _loadError(
            allOf(
              startsWith('window.EmbeddingEngine is not set'),
              contains(
                'Add window.EmbeddingEngine = m.EmbeddingEngine; next to '
                'window.Engine = m.Engine;',
              ),
            ),
          ),
        ),
      );
      expect(runtime.created, isEmpty);
      await pass.close();
    });

    test('a loader that failed (the CDN import rejected) is a load error '
        'carrying the cause', () async {
      final failure = StateError('Failed to fetch dynamically imported module');
      final runtime = _FakeRuntime()..readyError = failure;
      await expectLater(
        _pass(runtime).load(),
        throwsA(
          _loadError(
            contains('@litert-lm/core did not load on this page'),
          ).having((e) => e.cause, 'cause', failure),
        ),
      );
    });

    test('audio declared: refused before the page is touched', () async {
      final runtime = _FakeRuntime();
      await expectLater(
        _pass(runtime, media: {EmbeddingModality.audio}).load(),
        throwsA(_loadError(liteRtLmNoAudioRefusal)),
      );
      expect(runtime.readyChecks, 0);
      expect(runtime.created, isEmpty);
    });

    test('create fails, text only: the runtime\'s message, no vision '
        'hint', () async {
      final runtime = _FakeRuntime()..createError = StateError('out of memory');
      await expectLater(
        _pass(runtime).load(),
        throwsA(
          _loadError(
            allOf(
              contains('could not create the embedding engine'),
              contains('out of memory'),
              isNot(contains('vision encoder')),
            ),
          ),
        ),
      );
    });

    test('create fails with images declared (the 270m has no vision '
        'encoder): the message names the bundle that embeds images', () async {
      final runtime = _FakeRuntime()
        ..createError = StateError('NOT_FOUND: vision encoder');
      await expectLater(
        _pass(runtime, media: {EmbeddingModality.image}).load(),
        throwsA(
          _loadError(
            allOf(
              contains('NOT_FOUND: vision encoder'),
              contains('install embeddinggemma-2-text-vision-440m'),
            ),
          ),
        ),
      );
    });

    test('an opfs:// path reaches the engine as the blob: URL made for it, '
        'which the pass revokes once the engine read it — on success and on '
        'failure; any other path is never revoked', () async {
      for (final fail in [false, true]) {
        final runtime = _FakeRuntime();
        if (fail) runtime.createError = StateError('create failed');
        final revoked = <String>[];
        final pass = LiteRtLmWebEmbeddingPass(
          _config(runtime, modelPath: 'opfs://eg2.litertlm', revoked: revoked),
        );
        if (fail) {
          await expectLater(
            pass.load(),
            throwsA(isA<EmbeddingLoadException>()),
          );
        } else {
          await pass.load();
          expect(
            runtime.engine.settings.modelUrl,
            'blob:resolved/eg2.litertlm',
          );
        }
        expect(revoked, ['blob:resolved/eg2.litertlm']);
        await pass.close();
      }

      final runtime = _FakeRuntime();
      final revoked = <String>[];
      final pass = LiteRtLmWebEmbeddingPass(
        _config(runtime, modelPath: 'blob:core-owned', revoked: revoked),
      );
      await pass.load();
      expect(revoked, isEmpty, reason: 'core owns that URL');
      await pass.close();
    });

    for (final (name, vector) in [
      ('NaN', [double.nan, 0.5, 0.5, 0.5]),
      ('infinity', [double.infinity, 0.5, 0.5, 0.5]),
      ('all zeros', [0.0, 0.0, 0.0, 0.0]),
    ]) {
      test('a probe vector with $name (the fp16 failure) refuses the load, '
          'and the close behind it deletes the engine', () async {
        final runtime = _FakeRuntime()..vectorFor = (_) => vector;
        final pass = _pass(runtime);
        await expectLater(
          pass.load(),
          throwsA(_loadError(contains('NaN, infinity or only zeros'))),
        );
        await pass.close();
        expect(runtime.engine.deletes, 1);
      });
    }

    test('a probe that throws refuses the load; the engine is deleted on '
        'close', () async {
      final runtime = _FakeRuntime()..computeError = StateError('wasm trap');
      final pass = _pass(runtime);
      await expectLater(
        pass.load(),
        throwsA(_loadError(contains('wasm trap'))),
      );
      await pass.close();
      expect(runtime.engine.deletes, 1);
    });

    test('load runs once, and never after close', () async {
      final runtime = _FakeRuntime();
      final pass = _pass(runtime);
      await pass.load();
      await expectLater(pass.load(), throwsStateError);
      await pass.close();

      final closed = _pass(_FakeRuntime());
      await closed.close();
      await expectLater(closed.load(), throwsStateError);
    });
  });

  group('embed', () {
    test('text: every task type reaches the engine as prefix + text, with '
        'normalize, special tokens and TRUNCATE', () async {
      final runtime = _FakeRuntime();
      final pass = _pass(runtime);
      await pass.load();
      for (final type in TaskType.values) {
        expect(
          await pass.embed(EmbeddingInput.text('Mount Everest'), type),
          _unit,
        );
      }
      final calls = runtime.engine.calls.skip(1).toList(); // after the probe
      expect(
        [for (final (input, _) in calls) (input as LiteRtLmWebText).text],
        [for (final type in TaskType.values) '${type.prefix}Mount Everest'],
      );
      for (final (_, options) in calls) {
        expect(options.normalize, isTrue);
        expect(options.insertSpecialTokens, isTrue);
        expect(
          options.inputOverflowStrategy,
          LiteRtLmWebOverflowStrategy.truncate,
        );
      }
      await pass.close();
    });

    test('empty text embeds the prefix alone, as on native', () async {
      final runtime = _FakeRuntime();
      final pass = _pass(runtime);
      await pass.load();
      await pass.embed(EmbeddingInput.text(''), TaskType.retrievalQuery);
      expect(
        (runtime.engine.calls.last.$1 as LiteRtLmWebText).text,
        TaskType.retrievalQuery.prefix,
      );
      await pass.close();
    });

    test('at the limit it embeds; one token over is '
        'EmbeddingInputTooLongException with the engine\'s exact counts, the '
        'cut vector is never returned, and the pass keeps working', () async {
      final runtime = _FakeRuntime()..maxSignature = 64;
      final pass = _pass(runtime);
      await pass.load();
      // retrievalQuery's prefix is 30 characters; + BOS and EOS.
      final prefix = TaskType.retrievalQuery.prefix.length;
      final fits = 'x' * (64 - prefix - 2);
      expect(
        await pass.embed(EmbeddingInput.text(fits), TaskType.retrievalQuery),
        _unit,
      );
      await expectLater(
        pass.embed(EmbeddingInput.text('${fits}y'), TaskType.retrievalQuery),
        throwsA(
          isA<EmbeddingInputTooLongException>()
              .having((e) => e.maxInputTokens, 'maxInputTokens', 64)
              .having((e) => e.actualTokens, 'actualTokens', 65),
        ),
      );
      await expectLater(
        pass.embed(EmbeddingInput.text('z' * 3000), TaskType.none),
        throwsA(
          isA<EmbeddingInputTooLongException>()
              .having((e) => e.maxInputTokens, 'maxInputTokens', 64)
              .having((e) => e.actualTokens, 'actualTokens', 3002),
        ),
      );
      expect(await pass.embed(EmbeddingInput.text('ok'), TaskType.none), _unit);
      await pass.close();
    });

    test('an image reaches the engine as its own bytes, without a '
        'prefix', () async {
      final runtime = _FakeRuntime();
      final pass = _pass(runtime, media: {EmbeddingModality.image});
      await pass.load();
      expect(
        await pass.embed(EmbeddingInput.image(_png), TaskType.retrievalQuery),
        _unit,
      );
      final (input, options) = runtime.engine.calls.last;
      expect((input as LiteRtLmWebImage).bytes, _png);
      expect(options.normalize, isTrue);
      await pass.close();
    });

    test('an image on a text-only pass, two modalities, or audio never reach '
        'the engine (a host bug)', () async {
      final runtime = _FakeRuntime();
      final textOnly = _pass(runtime);
      await textOnly.load();
      for (final input in [
        EmbeddingInput.image(_png),
        EmbeddingInput.audio(Uint8List.fromList([1, 2, 3])),
      ]) {
        await expectLater(
          textOnly.embed(input, TaskType.none),
          throwsA(isA<EmbeddingEngineException>()),
        );
      }
      expect(runtime.engine.calls, hasLength(1), reason: 'the probe only');
      await textOnly.close();
    });

    test('an image the engine fails on is EmbeddingEngineException with the '
        'runtime\'s message; a cut image is never "too long"', () async {
      final runtime = _FakeRuntime();
      final pass = _pass(runtime, media: {EmbeddingModality.image});
      await pass.load();
      runtime.computeError = StateError('INVALID_ARGUMENT: outofdata');
      await expectLater(
        pass.embed(EmbeddingInput.image(_png), TaskType.none),
        throwsA(
          isA<EmbeddingEngineException>()
              .having((e) => e.engine, 'engine', liteRtLmEngineTag)
              .having(
                (e) => e.message,
                'message',
                allOf(contains('image'), contains('outofdata')),
              ),
        ),
      );
      runtime
        ..computeError = null
        ..maxSignature = 100;
      await expectLater(
        pass.embed(EmbeddingInput.image(_png), TaskType.none),
        throwsA(isA<EmbeddingEngineException>()),
      );
      await pass.close();
    });

    test('a vector with NaN or only zeros, or of the wrong length, is '
        'EmbeddingEngineException', () async {
      final runtime = _FakeRuntime();
      final pass = _pass(runtime);
      await pass.load();
      for (final vector in [
        [double.nan, 0.5, 0.5, 0.5],
        [0.0, 0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5],
      ]) {
        runtime.vectorFor = (_) => vector;
        await expectLater(
          pass.embed(EmbeddingInput.text('x'), TaskType.none),
          throwsA(isA<EmbeddingEngineException>()),
        );
      }
      await pass.close();
    });

    test('close deletes the engine once; embed after it fails', () async {
      final runtime = _FakeRuntime();
      final pass = _pass(runtime);
      await pass.load();
      await pass.close();
      await pass.close();
      expect(runtime.engine.deletes, 1);
      await expectLater(
        pass.embed(EmbeddingInput.text('x'), TaskType.none),
        throwsA(isA<EmbeddingEngineException>()),
      );
    });
  });

  test('through core\'s inline host: the model reports {text, image}, the '
      'limit and CPU; a WebP is refused on the caller before the engine sees '
      'it; text overflows with exact counts', () async {
    final runtime = _FakeRuntime();
    final host = await inline.startEmbeddingHost(
      EmbeddingPassDescriptor<LiteRtLmWebEmbeddingConfig>(
        engineTag: liteRtLmEngineTag,
        factory: buildLiteRtLmWebEmbeddingPass,
        config: _config(runtime, media: {EmbeddingModality.image}),
        activeBackend: PreferredBackend.cpu,
      ),
      declaredMedia: {EmbeddingModality.image},
      label: 'embeddinggemma-2-text-vision-440m.litertlm',
    );
    final model = CommonEmbeddingModel.withHost(
      host,
      engine: liteRtLmEngineTag,
      activeBackend: PreferredBackend.cpu,
    );
    try {
      expect(model.dimension, 4);
      expect(model.maxInputTokens, 1024);
      expect(model.modalities, {
        EmbeddingModality.text,
        EmbeddingModality.image,
      });
      expect(model.activeBackend, PreferredBackend.cpu);
      final calls = runtime.engine.calls.length;
      await expectLater(
        model.embed(EmbeddingInput.image(_webp), taskType: TaskType.none),
        throwsA(
          isA<EmbeddingFormatNotSupportedException>().having(
            (e) => e.detected,
            'detected',
            EncodedMediaFormat.webp,
          ),
        ),
      );
      expect(runtime.engine.calls, hasLength(calls), reason: 'never sent');
      expect(
        await model.embed(EmbeddingInput.image(_png), taskType: TaskType.none),
        _unit,
      );
      await expectLater(
        model.embed(
          EmbeddingInput.text('w' * 1100),
          taskType: TaskType.retrievalDocument,
        ),
        throwsA(
          isA<EmbeddingInputTooLongException>()
              .having((e) => e.maxInputTokens, 'maxInputTokens', 1024)
              .having(
                (e) => e.actualTokens,
                'actualTokens',
                TaskType.retrievalDocument.prefix.length + 1100 + 2,
              ),
        ),
      );
    } finally {
      await model.close();
    }
    expect(runtime.engine.deletes, 1);
  });
}
