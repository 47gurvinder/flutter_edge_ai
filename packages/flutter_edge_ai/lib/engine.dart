/// Engine SPI of flutter_edge_ai: everything an engine, backend, tokenizer,
/// resolver or skill-executor package needs, in one web-safe import. Re-exports
/// the app library (`flutter_edge_ai.dart`).
///
/// Native-only helpers (`dart:io` / `dart:ffi`) are in `engine_io.dart`,
/// web-only ones (`dart:js_interop`) in `engine_web.dart`. Nothing under
/// `lib/src` is public: import these libraries, never a `src/` path.
///
/// The namespace is pinned by the symbol golden in `test/public_surface/`.
library;

export 'flutter_edge_ai.dart';

// Registries engines consult.
export 'src/registry/embedding_tokenizer_registry.dart'
    show EmbeddingTokenizerRegistry;
export 'src/registry/skill_executor_registry.dart' show SkillExecutorRegistry;
export 'src/registry/hugging_face_resolver_source.dart'
    show HuggingFaceResolverSource;
export 'src/registry/runtime_config.dart' show RuntimeConfig;

// Lifecycle and logging.
export 'src/lifecycle/close_notifier.dart' show CloseNotifier;
export 'src/utils/edge_ai_log.dart' show edgeAiLog, edgeAiLogLevel;

// File naming (top-level functions over FileNameUtils, which stays internal).
export 'src/utils/file_names.dart' show isLiteRtLmFilename, sanitizeHfDirName;

// Media sniffing for LLM web arms. Pure Dart, so it is web-safe and lives here
// rather than in engine_web.dart.
export 'src/web/web_image_format.dart' show detectImageMimeType;

// Inference-engine helpers. (`supportedLoraRanks` and
// `defaultMaxFunctionBufferLength` reach engines through the app library.)
export 'src/extensions.dart' show MessageExtension;
export 'src/parsing/sdk_response_parser.dart' show SdkResponseParser;
export 'src/parsing/sdk_text_extractor.dart' show SdkTextExtractor;
export 'src/model_management/model_specs.dart' show TtsModelTypeManifest;

// Embedding seam: the contracts an embedding engine implements, the pooling
// helper, and the worker-isolate facade core hosts them in.
export 'src/embedding/forward_pass.dart'
    show
        EmbeddingForwardPass,
        EmbeddingForwardPassFactory,
        EmbeddingOutputContract,
        ForwardPassDescriptor,
        ForwardResult;
export 'src/embedding/pooling.dart' show meanPoolAndNormalize;
export 'src/embedding/tokenizer_adapter.dart'
    show EmbeddingTokenizer, EmbeddingTokenizerFactory, TokenizedInput;
// Real arm by default, web stub under js_interop: the worker needs
// dart:isolate, and web engine arms build their own EmbeddingModel.
export 'src/embedding/common_embedding_model.dart'
    if (dart.library.js_interop) 'src/embedding/common_embedding_model_stub.dart'
    show CommonEmbeddingModel;
