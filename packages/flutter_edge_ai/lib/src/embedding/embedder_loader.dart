// The one path every shell (mobile, desktop, web) builds the active embedder
// through (R5 §4.2, §5.5). The shells keep only their platform wiring — the
// manager they own, the cache they own, the debug notice — and call this
// inside `EmbedderCache.serialize`, so resolving paths, deciding on reuse and
// building stay one indivisible step on every platform.

import 'package:flutter_edge_ai/src/domain/platform_types.dart'
    show PreferredBackend;
import 'package:flutter_edge_ai/src/model_management/active_embedding_identity.dart'
    show noActiveEmbedderError;
import 'package:flutter_edge_ai/src/model_file_manager_interface.dart'
    show ModelFileManager;
import 'package:flutter_edge_ai/src/model_management/constants/preferences_keys.dart';
import 'package:flutter_edge_ai/src/model_management/model_specs.dart'
    show EmbeddingModelSpec;
import 'package:flutter_edge_ai/src/registry/embedding_backend_provider.dart';
import 'package:flutter_edge_ai/src/registry/embedding_registry.dart';
import 'package:flutter_edge_ai/src/registry/runtime_config.dart';
import 'package:flutter_edge_ai/src/utils/edge_ai_log.dart';

import 'common_embedding_model.dart';
import 'embedder_cache.dart';
import 'embedding_exceptions.dart';
import 'embedding_modality.dart';
import 'embedding_model.dart';

/// Returns the embedder for [manager]'s active embedding spec — the cached one
/// when it serves the request, else a new one built through the registered
/// backend and recorded in [cache]. Run it inside [EmbedderCache.serialize].
///
/// [supportImage] / [supportAudio] declare the media the caller will send;
/// text is implied whenever the model has a text tower. A rebuild widens them
/// to everything the cached embedder already served (R5 §4.6).
///
/// [verifyInstalled] re-checks that the active spec's files are still
/// installed before building (the mobile shell's long-standing check).
///
/// Throws [StateError] when no embedding spec is active or its files cannot be
/// resolved, or when no registered backend can run it;
/// [EmbeddingLoadException] when the backend refuses the request or the model
/// fails to load; [EmbedderCloseTimeoutException] when the embedder it
/// replaces is still closing after the cache's bound.
Future<EmbeddingModel> loadActiveEmbedder({
  required ModelFileManager manager,
  required EmbedderCache cache,
  required PreferredBackend? preferredBackend,
  required bool supportImage,
  required bool supportAudio,
  bool verifyInstalled = false,
}) async {
  final active = manager.activeEmbeddingModel;
  if (active == null) throw noActiveEmbedderError();
  if (active is! EmbeddingModelSpec) {
    throw StateError(
      'The active embedding slot holds a ${active.runtimeType}, not an '
      'EmbeddingModelSpec.',
    );
  }

  // §4.2 step 1: paths. A self-contained spec has no tokenizer, and its
  // tokenizerPath is null — never the model path.
  final paths = await manager.getModelFilePaths(active);
  if (paths == null || paths.isEmpty) {
    throw StateError(
      'The files of the active embedding model (${active.name}) were not '
      'found. Install it with `FlutterEdgeAi.installEmbedder()` first.',
    );
  }
  final modelPath =
      paths[PreferencesKeys.embeddingModelFile] ??
      (throw StateError(
        'Could not find the model path of the active embedding model '
        '(${active.name}).',
      ));
  final tokenizerPath = active.isSelfContained
      ? null
      : (paths[PreferencesKeys.embeddingTokenizerFile] ??
            (throw StateError(
              'Could not find the tokenizer path of the active embedding '
              'model (${active.name}).',
            )));
  edgeAiLog(
    'Using active embedding model: $modelPath'
    '${tokenizerPath == null ? '' : ', tokenizer: $tokenizerPath'}',
  );

  // Step 2: the cache decides — reuse, or what to build (widened media).
  final requested = ActiveEmbedderParams(
    modelPath: modelPath,
    tokenizerPath: tokenizerPath,
    preferredBackend: preferredBackend,
    media: {
      if (supportImage) EmbeddingModality.image,
      if (supportAudio) EmbeddingModality.audio,
    },
  );
  final decision = await cache.reuseOrInvalidate(requested, label: active.name);
  final reused = decision.reuse;
  if (reused != null) return reused;
  final build = decision.build;

  if (verifyInstalled && !await manager.isModelInstalled(active)) {
    throw StateError(
      'The active embedding model (${active.name}) is no longer installed. '
      'Install it again with `FlutterEdgeAi.installEmbedder()`.',
    );
  }

  try {
    // Step 3: the backend describes the pass; refusals surface here, before
    // any isolate exists.
    final backends = EmbeddingRegistry.instance;
    final backend = backends.findFor(active);
    if (backend == null) {
      throw StateError(
        backends.hasAny
            ? 'No registered embedding backend can run '
                  '${active.files.first.filename}. Registered backends: '
                  '${backends.registered.map((b) => b.name).join(', ')}.'
            : 'No embedding backend registered. Add flutter_edge_ai_litertlm '
                  'to pubspec.yaml and pass LiteRtEmbeddingBackend() in '
                  'embeddingBackends: of FlutterEdgeAi.initialize(...).',
      );
    }
    final config = RuntimeConfig(
      maxTokens: 0,
      modelPath: modelPath,
      tokenizerPath: tokenizerPath,
      preferredBackend: preferredBackend,
      supportImage: build.media.contains(EmbeddingModality.image),
      supportAudio: build.media.contains(EmbeddingModality.audio),
    );
    final descriptor = await backend.describe(active, config);

    // Step 4: core builds, hosts and checks the pass.
    // A pass whose load fails is closed behind the error; the cache holds
    // the next build until that close has ended (never two engines).
    void holdNextBuild(Future<void> cleanup) =>
        cache.trackFailedLoadCleanup(cleanup, build);
    final EmbeddingModel model;
    if (descriptor != null) {
      model = await CommonEmbeddingModel.start(
        descriptor,
        declaredMedia: build.media,
        label: modelPath,
        onFailedLoadCleanup: holdNextBuild,
      );
    } else {
      model = await _buildThroughCreateModel(
        backend,
        active,
        config,
        build.media,
        modelPath,
        holdNextBuild,
      );
    }

    // Core owns the lifecycle: the cache tracks the model, forgets it when it
    // closes (identity-guarded) and records what it was built from.
    await cache.adopt(model, build);
    return model;
  } catch (_) {
    // FIX #170: leave no bookkeeping behind, so a retry is a fresh decision
    // rather than a replay of this failure.
    cache.invalidate();
    rethrow;
  }
}

/// A backend that predates `describe()` builds its own model; core hosts it.
// TRANSITIONAL(C stage 3): remove — every engine describe()s in 2.2.0.
Future<EmbeddingModel> _buildThroughCreateModel(
  EmbeddingBackendProvider backend,
  EmbeddingModelSpec spec,
  RuntimeConfig config,
  Set<EmbeddingModality> media,
  String label,
  void Function(Future<void> cleanup) onFailedLoadCleanup,
) async {
  if (media.isNotEmpty) {
    throw EmbeddingLoadException(
      '${backend.name} predates the embedding pass API and embeds text '
      'only; it cannot load ${media.map((m) => m.name).join(' and ')} for '
      '$label.',
      engine: backend.name,
    );
  }
  final legacy = await backend.createModel(spec, config);
  return CommonEmbeddingModel.hostLegacyModel(
    legacy,
    engineTag: backend.name,
    label: label,
    onFailedLoadCleanup: onFailedLoadCleanup,
  );
}
