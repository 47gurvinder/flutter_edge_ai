// Web LiteRT embedding backend. It builds [WebEmbeddingModel], which runs on
// the LiteRT.js bundle in this package's `web/` — so unlike the native arm it
// never asks core's tokenizer registry: tokenization happens inside
// `sentencepiece.js`, not in Dart.

import 'package:flutter_edge_ai/engine.dart';
import 'web/web_embedding_model.dart';

/// Web LiteRT embedding backend — builds [WebEmbeddingModel] (LiteRT.js).
class LiteRtEmbeddingBackend implements EmbeddingBackendProvider {
  const LiteRtEmbeddingBackend();

  @override
  String get name => 'LiteRT Embedding';

  @override
  int get priority => 0;

  @override
  bool canHandle(EmbeddingModelSpec spec) => true;

  @override
  Future<EmbeddingModel> createModel(
    EmbeddingModelSpec spec,
    RuntimeConfig config,
  ) async {
    return WebEmbeddingModel(
      modelPath: config.modelPath,
      tokenizerPath: config.tokenizerPath,
      onClose: () {}, // core resets its state via addCloseListener
    );
  }
}
