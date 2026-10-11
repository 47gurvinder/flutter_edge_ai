// The inline embedding host (R5 §4.5): the pass runs on the calling isolate —
// the web page on web, where there is no Isolate.spawn. Same RequestPump, same
// load check and failure encoding as the isolate host, so the web behaves like
// native: text before media, queued requests fail on close, the pass is
// closed once.
//
// Pure Dart (no dart:isolate, no dart:js_interop): the VM tests run it, and on
// native the TRANSITIONAL pre-2.2 seam path hosts its models here.

import 'dart:async';

import 'package:meta/meta.dart' show visibleForTesting;

import 'embedding_exceptions.dart';
import 'embedding_host.dart';
import 'embedding_input.dart';
import 'embedding_modality.dart';
import 'embedding_pass.dart';
import 'load_check.dart';
import 'request_lanes.dart';
import 'task_type.dart';
import 'wire.dart';

const _slowLoadNotice = Duration(seconds: 30);
const _slowCloseNotice = Duration(seconds: 30);

/// Builds [descriptor]'s pass, loads it and checks it against
/// [declaredMedia]; the conditional host entry point `CommonEmbeddingModel`
/// calls on web. [label] names the model file in messages.
///
/// Fails with `EmbeddingLoadException`. A pass whose load failed is closed
/// behind the error, not before it: a close that never returns must not hold
/// the caller, which may be the embedder cache's lane.
Future<EmbeddingHost> startEmbeddingHost(
  EmbeddingPassDescriptor<Object> descriptor, {
  required Set<EmbeddingModality> declaredMedia,
  required String label,
  @visibleForTesting Duration slowLoadNotice = _slowLoadNotice,
  @visibleForTesting Duration slowCloseNotice = _slowCloseNotice,
}) => startInlineEmbeddingHost(
  descriptor,
  declaredMedia: declaredMedia,
  label: label,
  slowLoadNotice: slowLoadNotice,
  slowCloseNotice: slowCloseNotice,
);

/// [startEmbeddingHost], callable on every platform.
///
/// [passDied] completes, with a reason, when the pass's engine stopped on its
/// own; the host then behaves as an isolate host whose worker exited.
// TRANSITIONAL(C stage 3): remove [passDied] — only the pre-2.2 seam's
// models (hosted inline on native too) can die under the host.
Future<EmbeddingHost> startInlineEmbeddingHost(
  EmbeddingPassDescriptor<Object> descriptor, {
  required Set<EmbeddingModality> declaredMedia,
  required String label,
  Duration slowLoadNotice = _slowLoadNotice,
  Duration slowCloseNotice = _slowCloseNotice,
  Future<String>? passDied,
}) async {
  final engine = descriptor.engineTag;
  final EmbeddingPass pass;
  try {
    pass = descriptor.build();
  } catch (e) {
    throw _loadFailure(e, engine);
  }
  final notice = Timer(
    slowLoadNotice,
    () => warnEmbeddingHost(
      'loading the $engine embedder for $label has taken '
      '${describeHostDuration(slowLoadNotice)} and is still running.',
    ),
  );
  try {
    final info = await pass.load();
    final modalities = checkLoadedPass(
      info,
      declaredMedia,
      engine: engine,
      label: label,
    );
    return _InlineEmbeddingHost(
      pass,
      info,
      modalities,
      engine: engine,
      label: label,
      slowCloseNotice: slowCloseNotice,
      passDied: passDied,
    );
  } catch (e) {
    unawaited(_closeAfterFailedLoad(pass, engine, label));
    throw _loadFailure(e, engine);
  } finally {
    notice.cancel();
  }
}

EmbeddingException _loadFailure(Object error, String engine) =>
    EmbeddingFailure.fromError(
      error,
      engine: engine,
      loading: true,
      causeInMessage: false,
    ).toException(cause: error is EmbeddingException ? error.cause : null);

Future<void> _closeAfterFailedLoad(
  EmbeddingPass pass,
  String engine,
  String label,
) async {
  final error = await closePassReportingFailure(pass);
  if (error != null) {
    warnEmbeddingHost(
      'the $engine embedding pass for $label failed to close after its load '
      'failed; its native model may still be resident: $error',
    );
  }
}

final class _Request {
  _Request(this.input, this.taskType);

  final EmbeddingInput input;
  final TaskType taskType;
  final completer = Completer<List<double>>();
}

final class _InlineEmbeddingHost implements EmbeddingHost {
  _InlineEmbeddingHost(
    this._pass,
    this.info,
    this.modalities, {
    required this._engine,
    required this._label,
    required this._slowCloseNotice,
    Future<String>? passDied,
  }) {
    _pump = RequestPump<_Request>(_serve);
    _runDone = _pump.run();
    if (passDied != null) unawaited(passDied.then(_died));
  }

  final EmbeddingPass _pass;
  final String _engine;
  final String _label;
  final Duration _slowCloseNotice;
  late final RequestPump<_Request> _pump;
  late final Future<void> _runDone;
  final _unexpectedExit = Completer<String>();
  Future<void>? _closeFuture;
  String? _deathReason;

  @override
  final EmbeddingPassInfo info;

  @override
  final Set<EmbeddingModality> modalities;

  @override
  Future<String> get unexpectedExit => _unexpectedExit.future;

  @override
  Future<List<double>> embed(EmbeddingInput input, TaskType taskType) {
    final request = _Request(input, taskType);
    if (_closeFuture != null ||
        _deathReason != null ||
        !_pump.file(request, media: isMediaRequest(input))) {
      return Future.error(_closedError());
    }
    return request.completer.future;
  }

  EmbeddingException _closedError() {
    final reason = _deathReason;
    return reason == null
        ? closedBeforeRun
        : EmbedderClosedException(
            'The embedding model is closed: its $_engine engine stopped on its '
            'own ($reason).',
          );
  }

  Future<void> _serve(_Request request) async {
    try {
      request.completer.complete(
        await _pass.embed(request.input, request.taskType),
      );
    } catch (e) {
      request.completer.completeError(
        EmbeddingFailure.fromError(
          e,
          engine: _engine,
          causeInMessage: false,
        ).toException(cause: e is EmbeddingException ? e.cause : null),
      );
    }
  }

  @override
  Future<void> close() => _closeFuture ??= _shutDown();

  Future<void> _shutDown() async {
    for (final request in _pump.requestClose()) {
      request.completer.completeError(closedBeforeRun);
    }
    final notice = Timer(
      _slowCloseNotice,
      () => warnEmbeddingHost(
        'closing the $_engine embedder for $_label has taken '
        '${describeHostDuration(_slowCloseNotice)}: the embedding request in '
        "flight, or the pass's own close(), has not returned. It keeps "
        'waiting rather than abandon the native model.',
      ),
    );
    try {
      await _runDone;
      final error = await closePassReportingFailure(_pass);
      if (error != null) {
        warnEmbeddingHost(
          'the $_engine embedding pass for $_label failed to close; its '
          'native model may still be resident: $error',
        );
      }
    } finally {
      notice.cancel();
    }
  }

  /// The engine stopped under the host: fail what has not started, tell the
  /// owner, and release what is left.
  // TRANSITIONAL(C stage 3): remove with startInlineEmbeddingHost's passDied.
  void _died(String reason) {
    if (_closeFuture != null || _deathReason != null) return;
    _deathReason = reason;
    final failure = EmbeddingEngineException(
      'The $_engine embedder for $_label stopped unexpectedly: $reason',
      engine: _engine,
    );
    for (final request in _pump.requestClose()) {
      request.completer.completeError(failure);
    }
    warnEmbeddingHost(
      'the $_engine embedder for $_label stopped unexpectedly: $reason',
    );
    _unexpectedExit.complete(reason);
    _closeFuture = _shutDown();
  }
}
