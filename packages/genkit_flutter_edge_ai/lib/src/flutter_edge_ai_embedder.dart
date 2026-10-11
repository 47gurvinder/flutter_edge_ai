import 'package:flutter_edge_ai/flutter_edge_ai.dart' as gemma;
import 'package:genkit/plugin.dart';

import 'backend_parse.dart';
import 'flutter_edge_ai_options.dart';
import 'flutter_edge_ai_runtime.dart';

/// Creates a Genkit [Embedder] action backed by flutter_edge_ai's embedding model.
///
/// Every request resolves the model through [FlutterEdgeAiRuntime.getActiveEmbedder]
/// — a cache hit in flutter_edge_ai — instead of keeping one: the model a
/// previous request got may have been closed or replaced since. When the model
/// is closed between that lookup and the request (`EmbedderClosedException`,
/// which means the request never ran), it is resolved again and the request
/// retried once.
Embedder<FlutterEdgeAiEmbedConfig> createFlutterEdgeAiEmbedder({
  required String name,
  required FlutterEdgeAiRuntime runtime,
}) {
  return Embedder<FlutterEdgeAiEmbedConfig>(
    name: name,
    fn: (request, _) async {
      // Parse optional backend preference. The getter is read inside the try:
      // it casts lazily, so a non-string value must fail as INVALID_ARGUMENT
      // here rather than as a TypeError later.
      String? configPreferredBackend;
      if (request.options != null) {
        try {
          configPreferredBackend = FlutterEdgeAiEmbedConfig.fromJson(
            request.options!,
          ).preferredBackend;
        } catch (e) {
          throw GenkitException(
            'Invalid embed options: $e',
            status: StatusCode.invalidArgument,
          );
        }
      }

      // Parse preferredBackend string to enum.
      final backend = parsePreferredBackend(configPreferredBackend);

      // One text input per document.
      final inputs = [
        for (final doc in request.input)
          gemma.EmbeddingInput.text(_documentToText(doc)),
      ];

      final vectors = await _embedAll(runtime, backend, inputs);

      return EmbedResponse(
        embeddings: vectors
            .asMap()
            .entries
            .map(
              (entry) => Embedding(
                embedding: entry.value,
                metadata: request.input[entry.key].metadata,
              ),
            )
            .toList(growable: false),
      );
    },
  );
}

/// Embeds [inputs] with the active embedder, resolving it again — once — when
/// it was closed before the request ran.
Future<List<List<double>>> _embedAll(
  FlutterEdgeAiRuntime runtime,
  gemma.PreferredBackend? backend,
  List<gemma.EmbeddingInput> inputs,
) async {
  Future<List<List<double>>> attempt() async {
    final model = await runtime.getActiveEmbedder(preferredBackend: backend);
    // retrievalQuery: what 0.8.0's `generateEmbeddings(texts)` defaulted to,
    // so the same documents keep the same vectors.
    return model.embedAll(inputs, taskType: gemma.TaskType.retrievalQuery);
  }

  try {
    return await attempt();
  } on gemma.EmbedderClosedException {
    return attempt();
  }
}

/// Extracts plain text from a [DocumentData] by joining its text parts.
String _documentToText(DocumentData doc) {
  final buffer = StringBuffer();
  for (final part in doc.content) {
    if (part.isText) {
      if (buffer.isNotEmpty) buffer.write(' ');
      buffer.write(part.text);
    }
  }
  return buffer.toString();
}
