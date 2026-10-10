/// The embedding tokenizers for flutter_edge_ai.
///
/// Gemma SentencePiece and BERT-family WordPiece, task-type prefixing, and the
/// routing that picks between them — registered as a [GemmaEmbeddingTokenizers]
/// provider. The seam engine packages implement, the isolate worker and the
/// pooling live in `flutter_edge_ai` itself, so no engine depends on this.
///
/// To actually run embeddings, add an engine package that provides an
/// `EmbeddingBackendProvider` — e.g. `flutter_edge_ai_litertlm`'s
/// `LiteRtEmbeddingBackend` — and register it:
///
/// ```dart
/// import 'package:flutter_edge_ai/flutter_edge_ai.dart';
/// import 'package:flutter_edge_ai_litertlm/flutter_edge_ai_litertlm.dart';
///
/// await FlutterEdgeAi.initialize(
///   embeddingBackends: [LiteRtEmbeddingBackend()],
///   embeddingTokenizers: [GemmaEmbeddingTokenizers()],
/// );
/// ```
///
/// The seam exists so `flutter_edge_ai_litertlm` and `flutter_edge_ai_onnx`
/// share one tokenizer/worker/pooling implementation instead of each
/// reimplementing the isolate facade.
library;

// The embedding seam core owns, re-exported so a single import still covers
// the whole embedding surface for app and test code:
// - the runtime-agnostic `ForwardPass` seam (design doc §2, §11 D3/D4): engine
//   packages (flutter_edge_ai_litertlm, flutter_edge_ai_onnx) implement
//   `EmbeddingForwardPass` and build a `ForwardPassDescriptor` from a
//   top-level factory tear-off to plug into the common embedder;
// - the tokenizer seam (design D-T1): engine packages implement
//   `EmbeddingTokenizer` and build a `ForwardPassDescriptor.tokenizerFactory`
//   from a top-level factory tear-off;
// - the common embedder: background-isolate worker + `EmbeddingModel` facade
//   over any `ForwardPassDescriptor`. Core's engine library selects the web
//   stub under `dart.library.js_interop`; web engine packages build their own
//   `EmbeddingModel` directly and never reach it.
export 'package:flutter_edge_ai/engine.dart'
    show
        CommonEmbeddingModel,
        EmbeddingForwardPass,
        EmbeddingForwardPassFactory,
        EmbeddingOutputContract,
        EmbeddingTokenizer,
        EmbeddingTokenizerFactory,
        ForwardPassDescriptor,
        ForwardResult,
        TokenizedInput,
        meanPoolAndNormalize;
// The tokenizer families this package implements, as a registrable
// provider. This is what lets an engine ask for a tokenizer instead of
// naming one — and therefore what lets it stop depending on this package.
export 'src/tokenizer_provider.dart';

// NOTE: `src/embedding_tokenizer.dart` and `src/wordpiece_embedding_tokenizer.dart`
// are native-only leaves (`dart:io`, and for the former
// `dart_sentencepiece_tokenizer`, which imports `dart:io`/`dart:isolate`
// unconditionally) — deliberately NOT exported from this barrel so importing
// this package can never break a web build. They're used internally by
// engine backends (reached only via core's native-only
// `CommonEmbeddingModel`); native-only engine/test code
// that needs them directly imports
// `package:flutter_edge_ai_embeddings/src/embedding_tokenizer.dart` /
// `package:flutter_edge_ai_embeddings/src/wordpiece_embedding_tokenizer.dart`.
