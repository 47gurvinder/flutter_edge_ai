// The web `LiteRtEmbeddingBackend`: what it claims and what describe() hands
// core. Web routing is on the spec — the web arm claims only two-file specs, so
// a `.litertlm` bundle is refused by `installEmbedder()` before any download.
//
// Not run by tool/test_all.sh or CI, which run on the VM. Run:
//   flutter test --platform chrome test/web/litert_embedding_backend_web_test.dart
@TestOn('browser')
library;

import 'package:flutter_edge_ai/engine.dart';
import 'package:flutter_edge_ai_litertlm/src/embedding/litert_embedding_backend_web.dart';
import 'package:flutter_edge_ai_litertlm/src/embedding/web/litert_js_embedding_pass.dart';
import 'package:flutter_edge_ai_litertlm/src/embedding/web/litert_web_embeddings.dart';
import 'package:flutter_test/flutter_test.dart';

final _tflite = EmbeddingModelSpec(
  name: 'embeddinggemma',
  modelSource: ModelSource.network('https://example.com/gemma.tflite'),
  tokenizerSource: ModelSource.network('https://example.com/t.model'),
);

RuntimeConfig _config({bool supportImage = false, bool supportAudio = false}) =>
    RuntimeConfig(
      maxTokens: 0,
      modelPath: 'blob:model',
      tokenizerPath: 'blob:tokenizer',
      supportImage: supportImage,
      supportAudio: supportAudio,
    );

void main() {
  const backend = LiteRtEmbeddingBackend();

  test('claims a two-file spec, never a .litertlm bundle', () {
    expect(backend.canHandle(_tflite), isTrue);
    expect(
      backend.canHandle(
        EmbeddingModelSpec(
          name: 'eg2',
          modelSource: ModelSource.network('https://example.com/eg2.litertlm'),
        ),
      ),
      isFalse,
    );
  });

  test('describe() hands core the LiteRT.js pass over the resolved paths, '
      'with no backend claimed', () async {
    final descriptor = await backend.describe(_tflite, _config());
    final config = descriptor.config as LiteRtJsEmbeddingConfig;

    expect(descriptor.engineTag, 'LiteRT.js');
    expect(config.modelPath, 'blob:model');
    expect(config.tokenizerPath, 'blob:tokenizer');
    expect(config.bundle, isA<WindowLiteRtJsBundle>());
    expect(descriptor.activeBackend, isNull, reason: 'unknown, never CPU');
    expect(descriptor.build(), isA<LiteRtJsEmbeddingPass>());
  });

  test('describe() refuses declared images or audio', () async {
    for (final (image, audio) in [(true, false), (false, true)]) {
      await expectLater(
        backend.describe(
          _tflite,
          _config(supportImage: image, supportAudio: audio),
        ),
        throwsA(
          isA<EmbeddingLoadException>().having(
            (e) => e.message,
            'message',
            'gemma.tflite is a text-only model',
          ),
        ),
      );
    }
  });
}
