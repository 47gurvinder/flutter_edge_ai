// How a failure crosses from an embedding host to the facade (R5 §4.7).
//
// The isolate host never sends exception objects — their `cause` may not be
// sendable — so it encodes them as an [EmbeddingFailure] and the main side
// rebuilds the typed exception. The inline host runs the same encoding, so a
// pass that throws gets the same exception type and message on web as on
// native. Pure Dart.

import 'embedding_exceptions.dart';

/// The four kinds of failure a host reports.
enum EmbeddingFailureKind {
  /// The request never ran: the model closed first.
  closed,

  /// The input exceeded the model's limit.
  tooLong,

  /// The engine failed while embedding.
  engine,

  /// The engine failed to load or refused the configuration.
  load,
}

/// A sendable description of an embedding failure.
final class EmbeddingFailure {
  const EmbeddingFailure({
    required this.kind,
    required this.message,
    required this.engine,
    this.nativeCode,
    this.maxInputTokens,
    this.actualTokens,
  });

  /// Encodes [error] thrown by an engine's pass.
  ///
  /// Typed embedding exceptions keep their kind, code and counts. Anything
  /// else — a `StateError`, a native wrapper's exception, another
  /// `EmbeddingException` subtype a pass has no business throwing — becomes an
  /// engine failure ([loading]: a load failure) whose message is the error's
  /// text.
  factory EmbeddingFailure.fromError(
    Object error, {
    required String engine,
    bool loading = false,
  }) => switch (error) {
    EmbedderClosedException(:final message) => EmbeddingFailure(
      kind: EmbeddingFailureKind.closed,
      message: message,
      engine: engine,
    ),
    EmbeddingInputTooLongException(
      :final message,
      :final maxInputTokens,
      :final actualTokens,
    ) =>
      EmbeddingFailure(
        kind: EmbeddingFailureKind.tooLong,
        message: message,
        engine: engine,
        maxInputTokens: maxInputTokens,
        actualTokens: actualTokens,
      ),
    EmbeddingLoadException(
      :final message,
      engine: final reported,
      :final nativeCode,
    ) =>
      EmbeddingFailure(
        kind: EmbeddingFailureKind.load,
        message: message,
        engine: reported,
        nativeCode: nativeCode,
      ),
    EmbeddingEngineException(
      :final message,
      engine: final reported,
      :final nativeCode,
    ) =>
      EmbeddingFailure(
        kind: loading ? EmbeddingFailureKind.load : EmbeddingFailureKind.engine,
        message: message,
        engine: reported,
        nativeCode: nativeCode,
      ),
    EmbeddingException(:final message) => EmbeddingFailure(
      kind: loading ? EmbeddingFailureKind.load : EmbeddingFailureKind.engine,
      message: message,
      engine: engine,
    ),
    _ => EmbeddingFailure(
      kind: loading ? EmbeddingFailureKind.load : EmbeddingFailureKind.engine,
      message: '$error',
      engine: engine,
    ),
  };

  final EmbeddingFailureKind kind;
  final String message;
  final String engine;
  final int? nativeCode;
  final int? maxInputTokens;
  final int? actualTokens;

  /// The typed exception this failure stands for. [cause] is attached where
  /// the original error is still at hand (the inline host); across the
  /// isolate boundary it is not.
  EmbeddingException toException({Object? cause}) => switch (kind) {
    EmbeddingFailureKind.closed => EmbedderClosedException(message),
    EmbeddingFailureKind.tooLong => EmbeddingInputTooLongException(
      message,
      // Never null for this kind: fromError copies it from the exception.
      maxInputTokens: maxInputTokens ?? 0,
      actualTokens: actualTokens,
    ),
    EmbeddingFailureKind.engine => EmbeddingEngineException(
      message,
      engine: engine,
      nativeCode: nativeCode,
      cause: cause,
    ),
    EmbeddingFailureKind.load => EmbeddingLoadException(
      message,
      engine: engine,
      nativeCode: nativeCode,
      cause: cause,
    ),
  };
}
