// TRANSITIONAL(C stage 3): remove this file — it exists only while
// embedding backends may still build their own EmbeddingModel through the
// pre-2.2 `EmbeddingBackendProvider.createModel` (litertlm and onnx until C
// stage 2). Core wraps such a model in this pass and hosts it inline, so the
// model an app receives has the 2.2 API, core's request checks and core's
// close semantics whichever way the engine built it.

import 'dart:async';

import 'embedding_exceptions.dart';
import 'embedding_host.dart' show warnEmbeddingHost;
import 'embedding_input.dart';
import 'embedding_model.dart';
import 'embedding_pass.dart';
import 'task_type.dart';

/// What [buildLegacyModelPass] wraps. Not sendable: hosted inline only.
final class LegacyModelConfig {
  LegacyModelConfig(this.model, {required this.engine});

  /// The model the engine built.
  final EmbeddingModel model;

  /// The engine's name, for messages.
  final String engine;

  /// Completes, with a reason, when [model] closes without the pass closing
  /// it — its worker died. The inline host then behaves as an isolate host
  /// whose worker exited (W6), so the cache evicts it.
  final died = Completer<String>();
}

/// The [EmbeddingPassDescriptor] factory for a [LegacyModelConfig].
EmbeddingPass buildLegacyModelPass(LegacyModelConfig config) =>
    _LegacyModelPass(config);

final class _LegacyModelPass extends EmbeddingPass {
  _LegacyModelPass(this._config) {
    // Also fires when the host closes this pass, which closes the model; the
    // inline host ignores a death reported while it is closing.
    try {
      _config.model.addCloseListener(() {
        if (_config.died.isCompleted) return;
        _config.died.complete('the ${_config.engine} model closed itself');
      });
    } catch (e) {
      // `addCloseListener` is abstract on the interface, so a third-party
      // model may refuse it. Core's own model (the one the cache tracks) still
      // takes listeners; only a self-close of this one goes unnoticed.
      warnEmbeddingHost(
        'the ${_config.engine} model refused a close listener ($e); if it '
        'closes on its own, core will not notice until a request fails.',
      );
    }
  }

  final LegacyModelConfig _config;

  /// Text only, no declared limit: these models cannot report one, and the
  /// pre-2.2 engines behind them truncate instead of refusing.
  @override
  Future<EmbeddingPassInfo> load() async {
    // ignore: deprecated_member_use_from_same_package
    final dimension = await _config.model.getDimension();
    return EmbeddingPassInfo(dimension: dimension);
  }

  @override
  Future<List<double>> embed(EmbeddingInput input, TaskType taskType) {
    // These engines were written against the two 2.1 task types. litertlm's
    // web bundle, for one, maps every type but retrievalDocument to the query
    // prefix — refused rather than silently embedded with the wrong prefix.
    if (taskType.index > TaskType.retrievalDocument.index) {
      throw EmbeddingEngineException(
        '${_config.engine} predates TaskType.${taskType.name}; it embeds only '
        'TaskType.retrievalQuery and TaskType.retrievalDocument.',
        engine: _config.engine,
      );
    }
    // ignore: deprecated_member_use_from_same_package
    return _config.model.generateEmbedding(input.text!, taskType: taskType);
  }

  @override
  Future<void> close() => _config.model.close();
}
