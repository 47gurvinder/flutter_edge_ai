import 'package:flutter_edge_ai/src/registry/runtime_config.dart';
import 'package:flutter_edge_ai/src/interface.dart' show EmbeddingModel;
import 'package:flutter_edge_ai/src/model_management/model_specs.dart'
    show EmbeddingModelSpec;

/// A pluggable embedding backend (LiteRT C API + Gecko/EmbeddingGemma, or a
/// third-party backend). Same probe-chain shape as [InferenceEngineProvider]:
/// selected by probing the embedding model spec, highest-priority first match.
///
/// Passed to `FlutterEdgeAi.initialize` via `embeddingBackends:`.
abstract class EmbeddingBackendProvider {
  /// Human-readable name for diagnostics / error messages.
  String get name;

  /// Selection precedence on overlap. Core backends use 0.
  int get priority => 0;

  /// Whether this backend can serve [spec]. Probed by the registry.
  bool canHandle(EmbeddingModelSpec spec);

  /// Build a runtime [EmbeddingModel] for [spec] + [config].
  Future<EmbeddingModel> createModel(
    EmbeddingModelSpec spec,
    RuntimeConfig config,
  );
}
