import 'package:flutter_edge_ai/flutter_edge_ai.dart';

/// The profile id `FlutterEdgeAiActiveEmbedder.forActiveModel` derives:
/// `'<namespace>@<installed file names joined by +>'`, in the spec's file
/// order (model first, then its tokenizer when it has one). Not exported; a
/// function of its own so the derivation is tested without an installed
/// model.
///
/// Installed file names are portable basenames (core validates them) and
/// carry no control characters, so the result is a valid profile id whenever
/// [namespace] is.
String activeModelProfileId({
  required String namespace,
  required EmbeddingModelSpec spec,
}) => '$namespace@${[for (final file in spec.files) file.filename].join('+')}';
