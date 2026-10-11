// LiteRT embedding backend — native arm. It only DESCRIBES what to run
// (`describe()` is pure: no native library, no isolate); core builds the pass
// in its embedding worker, hosts it, checks every request and closes it.
//
// Routing is by the file core actually opens (`config.modelPath`): a
// `.litertlm` bundle (EmbeddingGemma 2) or a `.tflite` graph with a separate
// tokenizer (EmbeddingGemma 1, Gecko).

import 'package:flutter_edge_ai/engine.dart';

import 'litert_embedding_pass.dart';

/// LiteRT embedding backend: `.tflite` embedders (EmbeddingGemma 1, Gecko)
/// over the LiteRT C API, and `.litertlm` embedding bundles.
final class LiteRtEmbeddingBackend extends EmbeddingBackendProvider {
  const LiteRtEmbeddingBackend();

  @override
  String get name => 'LiteRT Embedding';

  @override
  int get priority => 0;

  /// A two-file spec (model + tokenizer), or a `.litertlm` bundle. A future
  /// self-contained spec of another kind (an ONNX directory) is not claimed.
  @override
  bool canHandle(EmbeddingModelSpec spec) =>
      spec.tokenizerSource != null ||
      isLiteRtLmFilename(spec.files.first.filename);

  @override
  Future<EmbeddingPassDescriptor<Object>> describe(
    EmbeddingModelSpec spec,
    RuntimeConfig config,
  ) async {
    final file = spec.files.first.filename;
    final opensBundle = isLiteRtLmFilename(config.modelPath);
    if (opensBundle != spec.isSelfContained) {
      throw EmbeddingLoadException(
        'identity corrupt: the active embedder ${spec.name} opens '
        '${config.modelPath}, which ${opensBundle ? 'is' : 'is not'} a '
        '.litertlm bundle, while its spec '
        '${spec.isSelfContained ? 'carries no' : 'names a'} tokenizer. '
        'Install the embedder again.',
        engine: opensBundle ? _liteRtLm : _liteRt,
      );
    }
    if (opensBundle) return _describeBundle(file, config);
    return _describeTflite(spec, file, config);
  }
}

const _liteRt = 'LiteRT';
const _liteRtLm = 'LiteRT-LM';

/// A `.litertlm` embedding bundle never asks the tokenizer registry: the
/// bundle carries its own.
Never _describeBundle(String file, RuntimeConfig config) {
  if (config.supportImage || config.supportAudio) {
    throw const EmbeddingLoadException(
      'flutter_edge_ai_litertlm 1.12.0 embeds text only; images arrive in '
      '1.13.0, audio in 1.15.0',
      engine: _liteRtLm,
    );
  }
  // Until the EmbeddingGemma 2 pass lands (the next PR of the 2.2.0 train),
  // a .litertlm embedder is refused here, before any isolate exists.
  throw EmbeddingLoadException(
    '$file is a .litertlm embedding bundle, which this build of '
    'flutter_edge_ai_litertlm cannot run yet.',
    engine: _liteRtLm,
  );
}

EmbeddingPassDescriptor<LiteRtEmbeddingConfig> _describeTflite(
  EmbeddingModelSpec spec,
  String file,
  RuntimeConfig config,
) {
  if (config.supportImage || config.supportAudio) {
    throw EmbeddingLoadException('$file is a text-only model', engine: _liteRt);
  }
  final tokenizerPath =
      config.tokenizerPath ??
      (throw EmbeddingLoadException(
        'identity corrupt: the active embedder ${spec.name} names a tokenizer, '
        'but core resolved no tokenizer path for it. Install the embedder '
        'again.',
        engine: _liteRt,
      ));
  return EmbeddingPassDescriptor<LiteRtEmbeddingConfig>(
    engineTag: _liteRt,
    factory: buildLiteRtEmbeddingPass,
    config: LiteRtEmbeddingConfig(
      modelPath: config.modelPath,
      tokenizerPath: tokenizerPath,
      // Asked for, not named. Which family this model needs is a fact about
      // the model, not about LiteRT — and hardcoding Gemma SentencePiece here
      // is what made a catch-all `canHandle` a trap: a WordPiece model was
      // accepted and then tokenized with the wrong convention.
      tokenizerFactory: EmbeddingTokenizerRegistry.instance.resolveFor(spec),
    ),
    // CPU, and by decision rather than omission: the GPU delegate compiles
    // and then returns all-zero vectors for EmbeddingGemma's int4 weights
    // (removed in ab3df2bf). Surfaced as EmbeddingModel.activeBackend so a
    // caller who asked for GPU can see what happened in a release build.
    activeBackend: PreferredBackend.cpu,
  );
}
