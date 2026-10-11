// The web `.tflite` embedding pass (EmbeddingGemma 1) over the LiteRT.js
// bundle in this package's `web/litert_embeddings.js` — what the web
// `LiteRtEmbeddingBackend.describe` hands core. Core hosts it inline on the
// page, checks every request and closes it.
//
// Pure Dart: the bundle's window API arrives in the config as a
// [LiteRtJsBundle] (the `dart:js_interop` one from the web backend, a fake in
// the VM tests), so nothing here imports `dart:js_interop`.
//
// Interim, on bundle API v1: the bundle tokenizes and prefixes inside the JS,
// and knows two prompts — `generateEmbedding` (the query prompt, which it uses
// for every type but a document) and `generateDocumentEmbedding`. So this pass
// serves `TaskType.retrievalQuery` and `.retrievalDocument` only and refuses
// the other seven by name rather than embed them with the wrong prefix, and it
// reports `maxInputTokens: null` because v1 cannot tell.
//
// Until then, an over-long input is TRUNCATED SILENTLY on web: v1 cuts the
// ids (prefix, BOS, text, EOS) to the model's sequence length, which also
// drops the EOS, and returns a vector of the cut text with no error — where
// native throws `EmbeddingInputTooLongException`. Bundle API v2 (Dart builds
// the prefix; overflow reported, never truncated) replaces all three.

import 'dart:async';

import 'package:flutter/foundation.dart' show kDebugMode;
import 'package:flutter_edge_ai/engine.dart';

import 'web_runtime.dart';

/// The `web/litert_embeddings.js` window API (bundle v1), as the pass uses it.
abstract interface class LiteRtJsBundle {
  /// Whether `web/litert_embeddings.js` is on the page at all — its window
  /// API exists. When it is not, no call below can do anything but throw.
  bool get isPresent;

  /// `loadLiteRtEmbeddings`: fetches the tokenizer, then fetches and compiles
  /// the model. Page-global state.
  Future<void> load(String modelUrl, String tokenizerUrl, String wasmPath);

  /// `getLiteRtEmbeddingDimension`.
  int get dimension;

  /// `generateEmbedding`: the bundle prepends the query prompt.
  Future<List<double>> embedQuery(String text);

  /// `generateDocumentEmbedding`: the bundle prepends the document prompt.
  Future<List<double>> embedDocument(String text);

  /// `cleanupLiteRtEmbeddings`: frees the model and the tokenizer.
  Future<void> dispose();
}

/// What the web `LiteRtEmbeddingBackend.describe` hands core: the model and
/// tokenizer paths core registered for the active spec, the bundle, and how to
/// turn those paths into URLs the bundle can fetch.
///
/// Hosted inline, never sent across an isolate.
final class LiteRtJsEmbeddingConfig {
  const LiteRtJsEmbeddingConfig({
    required this.modelPath,
    required this.tokenizerPath,
    required this.bundle,
    required this.resolveUrl,
    required this.revokeUrl,
  });

  /// The paths core registered for the installed files — under
  /// `WebStorageMode.streaming` an `opfs://` path the bundle's `fetch()`
  /// cannot open.
  final String modelPath;
  final String tokenizerPath;

  final LiteRtJsBundle bundle;

  /// `resolveWebModelUrl`: `opfs://` → `blob:`, anything else unchanged.
  final Future<String> Function(String path) resolveUrl;

  /// `URL.revokeObjectURL`: releases a URL [resolveUrl] created. The pass
  /// owns those (the `blob:` URL of an `opfs://` path) and revokes them once
  /// the bundle has read them; a path it did not resolve from `opfs://` is
  /// never revoked.
  final void Function(String url) revokeUrl;
}

/// The descriptor factory.
EmbeddingPass buildLiteRtJsEmbeddingPass(LiteRtJsEmbeddingConfig config) =>
    LiteRtJsEmbeddingPass(config);

/// The engine tag in the errors this pass raises.
const liteRtJsEngine = 'LiteRT.js';

/// The LiteRT.js embedding pass. Text only.
final class LiteRtJsEmbeddingPass extends EmbeddingPass {
  LiteRtJsEmbeddingPass(this._config);

  final LiteRtJsEmbeddingConfig _config;

  LiteRtJsBundle get _bundle => _config.bundle;

  /// Where the page's LiteRT.js runtime was actually loaded from, if at all.
  /// Static because the runtime is one per page, not one per pass.
  static String? _loadedWasmPath;

  /// Every load and every dispose of the page's LiteRT.js embedding state, in
  /// the order they were CALLED, across all passes.
  ///
  /// That state is one per page (`tfliteModel`, `tokenizer` and
  /// `isInitialized` in litert_embeddings.js). Without a page-wide order, a
  /// pass still disposing and its replacement loading race: the old dispose
  /// can land after the new load and delete the state the new pass just
  /// loaded. Only the owner of that state can order the two.
  static Future<void> _pageLane = Future<void>.value();

  /// Runs [op] on the page lane. The lane advances on a gate of its own, so
  /// [op]'s failure still reaches whoever awaits it and never blocks the next
  /// op.
  static Future<T> _onPage<T>(Future<T> Function() op) {
    final previous = _pageLane;
    final gate = Completer<void>();
    _pageLane = gate.future;
    return previous.then((_) => op()).whenComplete(gate.complete);
  }

  /// Whether this pass's load reached the bundle — and its close must
  /// therefore dispose the page state, even when the load FAILED: v1 loads the
  /// tokenizer before it compiles the model, so a failed compile leaves the
  /// tokenizer allocated, and the next load would overwrite it without freeing
  /// it. `cleanupLiteRtEmbeddings` checks each piece and is safe on partial
  /// state.
  bool _touchedPage = false;
  bool _closed = false;

  @override
  Future<EmbeddingPassInfo> load() => _onPage(() async {
    if (_closed) {
      throw StateError('LiteRtJsEmbeddingPass was closed before it loaded');
    }
    // Before anything is resolved or the page counts as touched: without the
    // bundle there is no page state, so the close behind this failure has
    // nothing to dispose — and must not call the missing cleanup, which core
    // would report as a native model that "may still be resident".
    if (!_bundle.isPresent) {
      throw const EmbeddingLoadException(
        'web/litert_embeddings.js is not loaded on this page. Copy '
        'packages/flutter_edge_ai_litertlm/web/litert_embeddings.js (and the '
        "three files it imports) into your app's web/ and add "
        '<script type="module" src="litert_embeddings.js"></script> to '
        'web/index.html.',
        engine: liteRtJsEngine,
      );
    }
    // Configurable since 2.2.0; the old hardcoded '/wasm/' was served by
    // nothing, so this call always failed. See [LiteRtWebRuntime].
    final wasmPath = LiteRtWebRuntime.wasmPath;
    // The runtime is a page-level singleton behind a flag in
    // litert_embeddings.js: once it has loaded, a different prefix is
    // discarded there without a word. Say so here instead.
    final loadedFrom = _loadedWasmPath;
    if (loadedFrom != null && loadedFrom != wasmPath) {
      edgeAiLog(
        'LiteRtWebRuntime.wasmPath changed to $wasmPath after the runtime '
        'was loaded from $loadedFrom — the new value is ignored. Set it '
        'before the first embedder loads.',
      );
    }
    // Under WebStorageMode.streaming both are `opfs://`, which the bundle's
    // fetch() rejects: every LiteRT.js embedder failed to load there.
    final tokenizerUrl = await _config.resolveUrl(_config.tokenizerPath);
    try {
      final modelUrl = await _config.resolveUrl(_config.modelPath);
      try {
        _touchedPage = true;
        await _bundle.load(modelUrl, tokenizerUrl, wasmPath);
      } finally {
        // The bundle has read the model by now (or failed to).
        _releaseIfCreated(_config.modelPath, modelUrl);
      }
    } finally {
      _releaseIfCreated(_config.tokenizerPath, tokenizerUrl);
    }
    _loadedWasmPath ??= wasmPath;
    if (kDebugMode) edgeAiLog('LiteRT.js embeddings loaded');
    // v1 cannot report the model's input length — and truncates an input
    // over it silently, EOS included (see the header). Bundle v2 fixes both.
    return EmbeddingPassInfo(dimension: _bundle.dimension);
  });

  /// Revokes [url] when this pass created it — the `blob:` URL of an
  /// `opfs://` [rawPath] — and never a URL core owns. Each holds a
  /// model-sized Blob until the page unloads otherwise.
  void _releaseIfCreated(String rawPath, String url) {
    if (rawPath.startsWith('opfs://')) _config.revokeUrl(url);
  }

  @override
  Future<List<double>> embed(EmbeddingInput input, TaskType taskType) async {
    final text = input.text;
    if (text == null) {
      throw StateError('LiteRtJsEmbeddingPass embeds text only');
    }
    return switch (taskType) {
      // The raw text: the v1 bundle prepends the prompt itself.
      TaskType.retrievalQuery => _bundle.embedQuery(text),
      TaskType.retrievalDocument => _bundle.embedDocument(text),
      // The v1 bundle builds the prefix itself and knows only the query and
      // document prompts; it would embed every other type with the query
      // prompt. Refused instead of silently embedded with the wrong prefix.
      _ => throw EmbeddingEngineException(
        'web/litert_embeddings.js (bundle API v1) predates '
        'TaskType.${taskType.name}; it embeds only TaskType.retrievalQuery '
        'and TaskType.retrievalDocument.',
        engine: liteRtJsEngine,
      ),
    };
  }

  @override
  Future<void> close() {
    if (_closed) return Future<void>.value();
    _closed = true;
    // Queued NOW: behind any load already on the lane — this pass's own
    // included, so that one is never leaked — and ahead of any load a
    // replacement starts after this call.
    return _onPage(() async {
      if (!_touchedPage) return;
      _touchedPage = false;
      await _bundle.dispose();
      if (kDebugMode) edgeAiLog('LiteRT.js embeddings disposed');
    });
  }
}
