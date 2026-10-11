// The load-time check of what an engine reported against what the caller
// declared (R5 §4.2 step 5, D4). Pure; both hosts run it right after
// EmbeddingPass.load(), so an engine that cannot serve a declared modality is
// refused at load — once, with one error type for every engine — instead of on
// the first request.

import 'embedding_exceptions.dart';
import 'embedding_modality.dart';
import 'embedding_pass.dart';

/// Checks [info] against [declaredMedia] and returns the model's modalities:
/// the declared media plus text when the engine has a text tower.
///
/// Throws [EmbeddingLoadException] when the dimension is not positive, when a
/// declared modality is not one the engine serves for this file, or when the
/// engine claims a media modality without naming a single format it decodes
/// (an engine bug). [label] names the model file in the message.
Set<EmbeddingModality> checkLoadedPass(
  EmbeddingPassInfo info,
  Set<EmbeddingModality> declaredMedia, {
  required String engine,
  required String label,
}) {
  if (info.dimension <= 0) {
    throw EmbeddingLoadException(
      '$engine reported dimension ${info.dimension} for $label; a model must '
      'produce vectors of positive length.',
      engine: engine,
    );
  }
  for (final modality in declaredMedia) {
    if (modality == EmbeddingModality.text) continue;
    if (!info.modalities.contains(modality)) {
      throw EmbeddingLoadException(
        '$engine cannot embed ${_plural(modality)} for $label: the engine '
        'serves ${_describe(info.modalities)} for this file.',
        engine: engine,
      );
    }
  }
  if (info.modalities.contains(EmbeddingModality.image) &&
      info.imageFormats.isEmpty) {
    throw EmbeddingLoadException(
      '$engine reports image support for $label but no image format it '
      'decodes (an engine bug).',
      engine: engine,
    );
  }
  if (info.modalities.contains(EmbeddingModality.audio) &&
      info.audioFormats.isEmpty) {
    throw EmbeddingLoadException(
      '$engine reports audio support for $label but no audio format it '
      'decodes (an engine bug).',
      engine: engine,
    );
  }
  return Set.unmodifiable({
    if (info.modalities.contains(EmbeddingModality.text))
      EmbeddingModality.text,
    for (final modality in declaredMedia)
      if (modality != EmbeddingModality.text) modality,
  });
}

String _plural(EmbeddingModality modality) => switch (modality) {
  EmbeddingModality.text => 'text',
  EmbeddingModality.image => 'images',
  EmbeddingModality.audio => 'audio',
};

String _describe(Set<EmbeddingModality> modalities) => modalities.isEmpty
    ? 'nothing'
    : [
        for (final m in EmbeddingModality.values)
          if (modalities.contains(m)) _plural(m),
      ].join(' and ');
