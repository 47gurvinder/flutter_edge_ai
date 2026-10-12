// The EmbeddingGemma 2 web pass over the REAL interop — `WindowLiteRtLmWebRuntime`
// and the page's `window.litertLmReady` / `window.EmbeddingEngine` — with a
// stub engine of `@litert-lm/core` 0.18.0's shape defined in plain JS (`eval`),
// through the path an app takes: the web backend's describe(), the pass, and
// core's inline host. What the VM suite cannot see: the JS objects themselves.
// - the loader states: no loader, a loader without `EmbeddingEngine` (an
//   index.html written for the LLM only), a loader still importing, a loader
//   that failed;
// - the settings object `EmbeddingEngine.create` receives: `backend: 3` (CPU),
//   `maxInputLength: 1024`, and NO `visionBackend` key unless images are
//   declared (the runtime tests `!== undefined`; a null would ask for a tower
//   without a backend);
// - each request: a text part is the prefixed string, an image part is
//   `{type: 'image', data: ArrayBuffer}` of exactly the file's bytes, options
//   `{normalize: true, insertSpecialTokens: true, inputOverflowStrategy: 1}`;
// - `truncatedLength` absent vs present, and JS rejections as typed errors.
//
// Not run by tool/test_all.sh or CI, which run on the VM. Run on the GCE VM:
//   flutter test --platform chrome test/web/litert_lm_web_embedding_runtime_test.dart
@TestOn('browser')
library;

import 'dart:js_interop';
import 'dart:js_interop_unsafe';
import 'dart:typed_data';

import 'package:flutter_edge_ai/engine.dart';
// ignore: implementation_imports — drives a pass through core's inline host, which is internal to core
import 'package:flutter_edge_ai/src/embedding/common_embedding_model.dart';
// ignore: implementation_imports — drives a pass through core's inline host, which is internal to core
import 'package:flutter_edge_ai/src/embedding/embedding_host_web.dart'
    as inline;
import 'package:flutter_edge_ai_litertlm/src/embedding/litert_embedding_backend_web.dart';
import 'package:flutter_edge_ai_litertlm/src/embedding/web/litert_lm_web_embedding_pass.dart';
import 'package:flutter_edge_ai_litertlm/src/embedding/web/litert_lm_web_embedding_runtime.dart';
import 'package:flutter_test/flutter_test.dart';

void _js(String source) => globalContext.callMethod('eval'.toJS, source.toJS);

T _eval<T extends JSAny?>(String source) =>
    globalContext.callMethod<T>('eval'.toJS, source.toJS);

/// Removes everything a test may have put on the page.
void _clearPage() => _js('''
  delete window.EmbeddingEngine;
  delete window.litertLmReady;
  delete window.__eg2;
''');

/// A stub `EmbeddingEngine` of 0.18.0's shape that records what it is given.
/// A text "token" is a space-separated word, plus BOS and EOS; an image is
/// 140 tokens. Inputs over `__eg2.maxSeq` under TRUNCATE come back with
/// `truncatedLength`, as the runtime's do.
void _installEngine() => _js('''
  window.__eg2 = { settings: [], inputs: [], options: [], deletes: 0,
                   maxSeq: 1024, createError: null, computeError: null };
  window.EmbeddingEngine = function () {};
  window.EmbeddingEngine.create = async function (settings) {
    const s = window.__eg2;
    s.settings.push(settings);
    if (s.createError) throw new Error(s.createError);
    return {
      computeEmbedding: async function (input, options) {
        s.inputs.push(input);
        s.options.push(options);
        if (s.computeError) throw new Error(s.computeError);
        const tokens = (typeof input === 'string'
            ? input.split(' ').length : 140) + 2;
        const response = { embedding: [0.5, 0.5, 0.5, 0.5],
                           inputLength: tokens, numChunks: 1 };
        if (tokens > s.maxSeq && options.inputOverflowStrategy === 1) {
          response.truncatedLength = s.maxSeq;
        }
        return response;
      },
      delete: async function () { s.deletes++; },
    };
  };
''');

final _bundle440m = EmbeddingModelSpec(
  name: 'eg2-440m',
  modelSource: ModelSource.network(
    'https://example.com/embeddinggemma-2-text-vision-440m.litertlm',
  ),
);

RuntimeConfig _config({bool supportImage = false, bool supportAudio = false}) =>
    RuntimeConfig(
      maxTokens: 0,
      modelPath: 'blob:https://example.com/eg2',
      supportImage: supportImage,
      supportAudio: supportAudio,
    );

/// The model an app gets: describe() → the pass → core's inline host.
Future<EmbeddingModel> _start({bool images = false}) async {
  final descriptor = await const LiteRtEmbeddingBackend().describe(
    _bundle440m,
    _config(supportImage: images),
  );
  final media = {if (images) EmbeddingModality.image};
  final host = await inline.startEmbeddingHost(
    descriptor,
    declaredMedia: media,
    label: 'embeddinggemma-2-text-vision-440m.litertlm',
  );
  return CommonEmbeddingModel.withHost(
    host,
    engine: descriptor.engineTag,
    activeBackend: descriptor.activeBackend,
  );
}

JSObject get _state => globalContext['__eg2']! as JSObject;

JSAny _recorded(String list, int index) =>
    (_state[list]! as JSArray<JSAny>).toDart[index];

JSObject _recordedObject(String list, int index) =>
    _recorded(list, index) as JSObject;

int _count(String list) => (_state[list]! as JSArray).length;

/// A PNG header, enough for core's sniffer.
final _png = Uint8List.fromList([
  0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 7, 8, 9, //
]);

void main() {
  tearDown(_clearPage);

  group('ensureReady', () {
    const runtime = WindowLiteRtLmWebRuntime();

    test('no loader and no EmbeddingEngine: loaderMissing', () async {
      expect(
        await runtime.ensureReady(),
        LiteRtLmWebRuntimeState.loaderMissing,
      );
    });

    test(
      'a loader that sets only window.Engine: embeddingEngineMissing',
      () async {
        _js('''
        window.litertLmReady = Promise.resolve().then(() => {
          window.Engine = function () {};
          return window.Engine;
        });
      ''');
        expect(
          await runtime.ensureReady(),
          LiteRtLmWebRuntimeState.embeddingEngineMissing,
        );
      },
    );

    test(
      'a loader still importing: ready once it has set EmbeddingEngine',
      () async {
        _js('''
        window.litertLmReady = new Promise((resolve) => setTimeout(() => {
          window.EmbeddingEngine = function () {};
          resolve(null);
        }, 50));
      ''');
        expect(await runtime.ensureReady(), LiteRtLmWebRuntimeState.ready);
      },
    );

    test('a loader that failed: the rejection propagates', () async {
      _js('''
        window.litertLmReady = Promise.reject(
            new Error('Failed to fetch dynamically imported module'));
        window.litertLmReady.catch(() => {});
      ''');
      await expectLater(runtime.ensureReady(), throwsA(anything));
    });
  });

  test('a page without the shim line: the app gets the named '
      'EmbeddingLoadException', () async {
    _js('''
      window.litertLmReady = Promise.resolve().then(() => {
        window.Engine = function () {};
        return window.Engine;
      });
    ''');
    await expectLater(
      _start(),
      throwsA(
        isA<EmbeddingLoadException>()
            .having((e) => e.engine, 'engine', 'LiteRT-LM')
            .having(
              (e) => e.message,
              'message',
              contains('window.EmbeddingEngine = m.EmbeddingEngine;'),
            ),
      ),
    );
  });

  test('text only: create gets CPU, maxInputLength 1024 and no vision keys; '
      'the probe and each text are strings; the options are normalize, '
      'special tokens and TRUNCATE (1)', () async {
    _installEngine();
    final model = await _start();
    try {
      expect(model.dimension, 4);
      expect(model.maxInputTokens, 1024);
      expect(model.activeBackend, PreferredBackend.cpu);

      final settings = _recordedObject('settings', 0);
      expect(
        (settings['model']! as JSString).toDart,
        'blob:https://example.com/eg2',
      );
      expect((settings['backend']! as JSNumber).toDartInt, 3);
      expect((settings['maxInputLength']! as JSNumber).toDartInt, 1024);
      for (final key in [
        'visionBackend',
        'visionTokensPerImage',
        'minInputLength',
      ]) {
        expect(settings.has(key), isFalse, reason: '$key must be absent');
      }

      await model.embed(
        EmbeddingInput.text('a red stop sign'),
        taskType: TaskType.retrievalQuery,
      );
      final probe = _recorded('inputs', 0);
      expect(probe.typeofEquals('string'), isTrue);
      final text = _recorded('inputs', 1);
      expect(
        (text as JSString).toDart,
        '${TaskType.retrievalQuery.prefix}a red stop sign',
      );
      final options = _recordedObject('options', 1);
      expect((options['normalize']! as JSBoolean).toDart, isTrue);
      expect((options['insertSpecialTokens']! as JSBoolean).toDart, isTrue);
      expect((options['inputOverflowStrategy']! as JSNumber).toDartInt, 1);
    } finally {
      await model.close();
    }
    expect((_state['deletes']! as JSNumber).toDartInt, 1);
  });

  test('images: create gets visionBackend 3 and 140 tokens; an image reaches '
      'the engine as {type: "image", data: ArrayBuffer} of exactly its '
      'bytes', () async {
    _installEngine();
    final model = await _start(images: true);
    try {
      final settings = _recordedObject('settings', 0);
      expect((settings['visionBackend']! as JSNumber).toDartInt, 3);
      expect((settings['visionTokensPerImage']! as JSNumber).toDartInt, 140);
      expect(model.modalities, {
        EmbeddingModality.text,
        EmbeddingModality.image,
      });

      expect(
        await model.embed(EmbeddingInput.image(_png), taskType: TaskType.none),
        [0.5, 0.5, 0.5, 0.5],
      );
      final part = _recordedObject('inputs', _count('inputs') - 1);
      expect((part['type']! as JSString).toDart, 'image');
      final data = part['data']!;
      expect(
        _eval<JSBoolean>(
          'window.__eg2.inputs[window.__eg2.inputs.length - 1].data '
          'instanceof ArrayBuffer',
        ).toDart,
        isTrue,
      );
      expect((data as JSArrayBuffer).toDart.asUint8List(), _png);
    } finally {
      await model.close();
    }
  });

  test('an over-long text: EmbeddingInputTooLongException with the engine\'s '
      'counts, then the model keeps working', () async {
    _installEngine();
    final model = await _start();
    try {
      final words = List.filled(1500, 'w').join(' ');
      await expectLater(
        model.embed(EmbeddingInput.text(words), taskType: TaskType.none),
        throwsA(
          isA<EmbeddingInputTooLongException>()
              .having((e) => e.maxInputTokens, 'maxInputTokens', 1024)
              .having((e) => e.actualTokens, 'actualTokens', 1502),
        ),
      );
      expect(
        await model.embed(EmbeddingInput.text('ok'), taskType: TaskType.none),
        hasLength(4),
      );
    } finally {
      await model.close();
    }
  });

  test('JS rejections: create → EmbeddingLoadException naming the 440m when '
      'images were declared; computeEmbedding → EmbeddingEngineException with '
      'the JS message', () async {
    _installEngine();
    _js("window.__eg2.createError = 'NOT_FOUND: no vision encoder';");
    await expectLater(
      _start(images: true),
      throwsA(
        isA<EmbeddingLoadException>()
            .having((e) => e.message, 'message', contains('NOT_FOUND'))
            .having(
              (e) => e.message,
              'message',
              contains('install embeddinggemma-2-text-vision-440m'),
            ),
      ),
    );

    _installEngine();
    final model = await _start();
    try {
      _js("window.__eg2.computeError = 'wasm trap';");
      await expectLater(
        model.embed(EmbeddingInput.text('x'), taskType: TaskType.none),
        throwsA(
          isA<EmbeddingEngineException>()
              .having((e) => e.engine, 'engine', 'LiteRT-LM')
              .having((e) => e.message, 'message', contains('wasm trap')),
        ),
      );
    } finally {
      await model.close();
    }
  });
}
