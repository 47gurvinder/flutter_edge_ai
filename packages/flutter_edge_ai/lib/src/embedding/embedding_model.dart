import 'dart:async';

import 'package:flutter_edge_ai/src/domain/platform_types.dart'
    show PreferredBackend;
import 'package:flutter_edge_ai/src/model_management/cancel_token.dart'
    show CancelToken;

import 'embedding_exceptions.dart';
import 'embedding_input.dart';
import 'embedding_modality.dart';
import 'task_type.dart';

/// An embedding model, as returned by `FlutterEdgeAi.getActiveEmbedder()`.
///
/// Core builds every model an app receives: it hosts the engine's pass, checks
/// each request before it is sent, and owns the close. Extend this class only
/// for test fakes — `final class FakeEmbedder extends EmbeddingModel` (add
/// `with CloseNotifier` from `engine.dart` for close listeners) implementing
/// [dimension], [isClosed], [embed], [addCloseListener] and [close].
// TRANSITIONAL(C stage 3): remove — becomes `abstract base class` (so
// `implements`, and therefore mockito/mocktail mocks, stop compiling outside
// core) once no engine package builds its own model any more. A `base` class
// may still be extended from another library; what breaks is that every
// subclass must then be `base`, `final` or `sealed`, and the litertlm and onnx
// web models (until C stage 2) and the repo's test fakes are plain `class`.
abstract class EmbeddingModel {
  /// Length of every vector this model returns. Known when the model is
  /// returned: every engine loads eagerly.
  // TRANSITIONAL(C stage 3): remove the body — abstract in 2.2.0. It exists
  // only so models built through the pre-2.2 seam (which answer
  // `getDimension()` instead) keep compiling; core never hands one out
  // unwrapped.
  int get dimension => throw UnsupportedError(
    '$runtimeType predates EmbeddingModel.dimension; '
    'use the model FlutterEdgeAi.getActiveEmbedder() returns.',
  );

  /// What this model accepts: the media declared at load (and confirmed by
  /// the engine), plus text when the model has a text tower. A request
  /// carrying anything else fails with `EmbeddingModalityNotLoadedException`.
  Set<EmbeddingModality> get modalities => const {EmbeddingModality.text};

  /// Longest input in tokens — task prefix, BOS and EOS included — as the
  /// engine read it from the MODEL FILE at load. Null when the file declares
  /// no limit. Longer inputs fail with `EmbeddingInputTooLongException`; they
  /// are never truncated.
  int? get maxInputTokens => null;

  /// Backend this embedder actually uses, when known — the counterpart of
  /// `InferenceModel.activeBackend`, and the answer to
  /// `getActiveEmbedder(preferredBackend:)` that survives a release build.
  ///
  /// Every native backend answers [PreferredBackend.cpu] whatever was asked:
  /// LiteRT because its GPU delegate returns all-zero vectors for
  /// EmbeddingGemma's int4 weights, ONNX because it appends no execution
  /// provider. So a caller who passed `gpu` can read here that it did not
  /// happen, instead of learning it from a debug-only log line.
  ///
  /// Null on web, where the runtime picks for itself and does not report
  /// back. Null also means "this engine did not say". Treat null as unknown,
  /// never as CPU.
  PreferredBackend? get activeBackend => null;

  /// Whether [close] has been called, or the model can no longer be used
  /// because its worker exited.
  // TRANSITIONAL(C stage 3): remove the body — abstract in 2.2.0 (it was
  // defaulted to false in 2.1).
  bool get isClosed => false;

  /// The vector for [input].
  ///
  /// [taskType] prefixes TEXT parts only; [TaskType.none] embeds the text raw.
  ///
  /// Every error is delivered as a FAILED FUTURE, never thrown synchronously:
  /// `EmbedderClosedException`, `EmbeddingModalityNotLoadedException`,
  /// `UnsupportedEmbeddingCombinationException`,
  /// `EmbeddingFormatNotSupportedException`,
  /// `EmbeddingInputTooLongException`, `EmbeddingEngineException`.
  // TRANSITIONAL(C stage 3): remove the body — abstract in 2.2.0. Models
  // built through the pre-2.2 seam override `generateEmbedding` instead, and
  // core never hands one out unwrapped.
  Future<List<double>> embed(
    EmbeddingInput input, {
    required TaskType taskType,
  }) => Future.error(
    UnsupportedError(
      '$runtimeType predates EmbeddingModel.embed; '
      'use the model FlutterEdgeAi.getActiveEmbedder() returns.',
    ),
  );

  /// The vectors for [inputs], in input order.
  ///
  /// The first failure fails the call. [cancelToken] stops sending and fails
  /// the call with `EmbeddingCancelledException`; partial results are
  /// discarded. Text and media alike.
  ///
  /// The models core returns also validate every input before the first is
  /// sent, and keep at most two requests of the call outstanding; requests
  /// already outstanding when the call fails or is cancelled finish and are
  /// dropped. This default body — for fakes — embeds one input at a time.
  Future<List<List<double>>> embedAll(
    List<EmbeddingInput> inputs, {
    required TaskType taskType,
    CancelToken? cancelToken,
  }) async {
    final out = <List<double>>[];
    for (final input in inputs) {
      if (cancelToken?.isCancelled ?? false) {
        throw const EmbeddingCancelledException();
      }
      out.add(await embed(input, taskType: taskType));
    }
    return out;
  }

  /// Register a callback fired once when this model is closed — by [close],
  /// or because its worker exited. Core's embedder cache evicts through it.
  void addCloseListener(void Function() listener);

  /// Closes the model. Idempotent. Requests not yet started fail with
  /// `EmbedderClosedException`; the one in flight finishes; then the engine is
  /// released. Waits as long as that takes — nothing is ever killed, since a
  /// killed worker never frees its native model.
  Future<void> close();

  /// Deprecated: use `embed(EmbeddingInput.text(text), taskType: taskType)`.
  @Deprecated(
    'Use embed(EmbeddingInput.text(text), taskType: taskType). '
    'Removed in 2.4.0.',
  )
  // TRANSITIONAL(C stage 3): add @nonVirtual — pre-2.2 seam models still
  // override this forwarder.
  Future<List<double>> generateEmbedding(
    String text, {
    TaskType taskType = TaskType.retrievalQuery,
  }) => embed(EmbeddingInput.text(text), taskType: taskType);

  /// Deprecated: use `embedAll` over `EmbeddingInput.text` inputs.
  @Deprecated(
    'Use embedAll([for (final t in texts) EmbeddingInput.text(t)], '
    'taskType: taskType). Removed in 2.4.0.',
  )
  // TRANSITIONAL(C stage 3): add @nonVirtual — pre-2.2 seam models still
  // override this forwarder.
  Future<List<List<double>>> generateEmbeddings(
    List<String> texts, {
    TaskType taskType = TaskType.retrievalQuery,
  }) => embedAll([
    for (final t in texts) EmbeddingInput.text(t),
  ], taskType: taskType);

  /// Deprecated: use [dimension].
  @Deprecated('Use dimension. Removed in 2.4.0.')
  // TRANSITIONAL(C stage 3): add @nonVirtual — pre-2.2 seam models still
  // override this forwarder.
  Future<int> getDimension() => isClosed
      ? Future.error(const EmbedderClosedException())
      : Future.sync(() => dimension);
}
