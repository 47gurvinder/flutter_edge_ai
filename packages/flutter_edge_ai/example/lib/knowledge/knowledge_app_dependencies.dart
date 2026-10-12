import 'dart:async';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:flutter/widgets.dart' show ImageProvider;
import 'package:flutter_edge_ai/flutter_edge_ai.dart';
import 'package:flutter_edge_ai_rag/flutter_edge_ai_rag.dart';
import 'package:flutter_edge_ai_sqlite/flutter_edge_ai_sqlite.dart';
import 'package:image_picker/image_picker.dart';
import 'package:just_audio/just_audio.dart';
import 'package:record/record.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../gemma_bootstrap.dart' show exampleRag;
import '../image_search/photo_store.dart';
import '../models/embedding_model.dart' as catalog;
import '../models/stt_model.dart';
import '../rag_demo/rag_storage_location.dart';
import '../utils/installed_model_lookup.dart';
import 'knowledge_audio.dart';
import 'knowledge_dependencies.dart';
import 'knowledge_item.dart';
import 'knowledge_platform.dart';

/// The SharedPreferences key the library is saved under.
const kKnowledgeLibraryKey = 'knowledge_library_v1';

/// How much of the answer the language model may write.
const kAnswerMaxTokens = 220;

/// The app's wiring of the Knowledge screen: the models the three model
/// lists left active, SQLite indexes (one location per kind and embedding
/// space), photos and memos kept in the app's documents (photos in OPFS on
/// the web), `just_audio` playback and `record` capture.
KnowledgeDependencies knowledgeAppDependencies() {
  final photos = createPhotoStore(directoryName: 'knowledge_photos');
  final speech = _Speech();
  final picker = ImagePicker();
  return KnowledgeDependencies(
    readSetup: readKnowledgeSetup,
    embedderFor: (kind, info) => FlutterEdgeAiActiveEmbedder(
      profileId: knowledgeProfileId(kind, info),
      preferredBackend: info.backend,
      // Every index declares every medium the screen uses, so core loads the
      // embedder once with all of them instead of once per index.
      supportImage: info.kinds.contains(KnowledgeKind.photo),
      supportAudio: info.kinds.contains(KnowledgeKind.memo),
    ),
    openIndex: (kind, embedder, info) async {
      final location = await resolveRagStorageLocation(
        'knowledge_${kind.modality.name}_${knowledgeProfileId(kind, info)}.db',
      );
      return exampleRag.open(
        spec: VectorStoreSpec(
          providerId: SqliteVectorStoreProvider.providerId,
          location: location,
          documents: kind.modality,
        ),
        embedder: embedder,
      );
    },
    library: const _PreferencesLibraryStore(),
    media: _AppMedia(photos),
    transcribe: speech.transcribe,
    answerer: _LlmAnswerer(),
    player: _JustAudioPlayer(),
    recorder: _Recorder(),
    pickPhoto: () async {
      final file = await picker.pickImage(source: ImageSource.gallery);
      if (file == null) return null;
      return (name: file.name, bytes: await file.readAsBytes());
    },
    beforeEmbedding: kIsWeb ? _releaseLanguageModel : () async => false,
    beforeAnswer: kIsWeb ? _releaseVisionEncoder : () async {},
    release: speech.close,
  );
}

/// The profile of [kind]'s index: the text id for notes and memos (audio
/// vectors do not depend on the vision budget, and the store's modality
/// keeps an audio index apart from a text one), the image id — the text id
/// plus the vision budget — for photos.
String knowledgeProfileId(KnowledgeKind kind, KnowledgeEmbedderInfo info) =>
    switch (kind) {
      KnowledgeKind.photo => catalog.imageIndexProfileId(info.textProfileId),
      KnowledgeKind.note || KnowledgeKind.memo => info.textProfileId,
    };

/// What the three model lists left active.
Future<KnowledgeSetup> readKnowledgeSetup() async {
  const audioSupported = !kIsWeb;
  final (embedder, problem) = await _embedder(audioSupported: audioSupported);
  return KnowledgeSetup(
    embedder: embedder,
    embedderProblem: problem,
    stt: kIsWeb ? null : _stt(),
    llm: _llm()?.info,
    audioSupported: audioSupported,
    preloadLlm: !kIsWeb,
  );
}

Future<(KnowledgeEmbedderInfo?, String?)> _embedder({
  required bool audioSupported,
}) async {
  const install =
      'Install EmbeddingGemma 2 Text + Vision + Audio (740M) from the '
      'Embedding Models list.';
  final spec = FlutterEdgeAi.activeEmbedderSpec;
  if (spec == null) return (null, 'No embedding model is active. $install');
  final profileId = await activeEmbeddingProfileId();
  final files = {for (final file in spec.files) file.filename};
  final model = catalog.EmbeddingModel.values
      .where((m) => m.ragProfileId == profileId && files.contains(m.filename))
      .firstOrNull;
  if (profileId == null || model == null) {
    return (
      null,
      'The active embedding model is not one of the example\'s, so its '
          'embedding space has no name. $install',
    );
  }
  return (
    KnowledgeEmbedderInfo(
      // "EmbeddingGemma 2 (740M)": the card lists what it embeds.
      name: model.displayName.replaceFirst(
        RegExp(r' Text( \+ Vision)?( \+ Audio)?'),
        '',
      ),
      textProfileId: model.ragProfileId,
      kinds: {
        KnowledgeKind.note,
        if (model.embedsImages) KnowledgeKind.photo,
        if (model.embedsAudio && audioSupported) KnowledgeKind.memo,
      },
      backend: model.embedderBackend,
    ),
    null,
  );
}

KnowledgeSttInfo? _stt() {
  final spec = FlutterEdgeAi.activeSttSpec;
  if (spec == null) return null;
  final file = spec.files.first.filename;
  final model = SttModel.values
      .where((m) => Uri.parse(m.modelUrl).pathSegments.last == file)
      .firstOrNull;
  return KnowledgeSttInfo(
    name: model?.displayName ?? file,
    window: switch (spec.sttModelType) {
      SttModelType.whisper => const Duration(seconds: 30),
      SttModelType.moonshine ||
      SttModelType.parakeet => const Duration(seconds: 5),
    },
  );
}

/// The active language model and how the screen loads it: the catalog's
/// context window and backend for a catalog model.
({
  KnowledgeLlmInfo info,
  int maxTokens,
  PreferredBackend? backend,
  bool supportImage,
})?
_llm() {
  final spec = FlutterEdgeAi.activeModelSpec;
  if (spec == null) return null;
  final id = activeInferenceModelId();
  final model = switch (id == null ? null : resolveCatalog(id)) {
    InferenceMatch(:final model) => model,
    _ => null,
  };
  final supportImage =
      model?.supportImage ?? (spec.modelType == ModelType.gemma4 && !kIsWeb);
  return (
    info: KnowledgeLlmInfo(
      name: model?.displayName ?? id ?? spec.name,
      seesImages: supportImage,
    ),
    maxTokens: model?.maxTokens ?? 4096,
    backend: model?.preferredBackend,
    supportImage: supportImage,
  );
}

/// The web: the embedder's vision encoder (about 3.3 GB) and a `.litertlm`
/// language model do not share the page's 4 GB of WebAssembly memory, so
/// they take turns. True when the language model was released.
Future<bool> _releaseLanguageModel() async {
  final model = FlutterEdgeAiPlugin.instance.initializedModel;
  if (model == null || model.isClosed) return false;
  debugPrint('[Knowledge] web: releasing the language model to embed');
  await model.close();
  return true;
}

Future<void> _releaseVisionEncoder() async {
  final embedder = FlutterEdgeAiPlugin.instance.initializedEmbeddingModel;
  if (embedder == null || embedder.isClosed) return;
  if (!embedder.modalities.contains(EmbeddingModality.image)) return;
  debugPrint('[Knowledge] web: releasing the vision encoder for the LLM');
  await embedder.close();
}

final class _PreferencesLibraryStore implements KnowledgeLibraryStore {
  const _PreferencesLibraryStore();

  @override
  Future<String?> read() async =>
      (await SharedPreferences.getInstance()).getString(kKnowledgeLibraryKey);

  @override
  Future<void> write(String json) async {
    final saved = await (await SharedPreferences.getInstance()).setString(
      kKnowledgeLibraryKey,
      json,
    );
    if (!saved) throw StateError('Could not save the knowledge library.');
  }
}

final class _AppMedia implements KnowledgeMedia {
  _AppMedia(this._photos);

  final PhotoStore _photos;

  @override
  Future<Uint8List> read(String reference) async {
    if (assetKeyOf(reference) case final key?) {
      final data = await rootBundle.load(key);
      return data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes);
    }
    if (reference.endsWith('.wav')) return readMemoFile(reference);
    return _photos.read(reference);
  }

  @override
  ImageProvider imageFor(String reference) => _photos.imageFor(reference);

  @override
  Future<String> keepPhoto(Uint8List bytes, {required String name}) =>
      _photos.keep(bytes, name: name);

  @override
  Future<String> keepMemo(Uint8List wav) => keepMemoFile(wav);

  @override
  Future<void> forget(String reference) async {
    if (assetKeyOf(reference) != null) return;
    if (reference.endsWith('.wav')) return forgetMemoFile(reference);
    await _photos.forget(reference);
  }
}

/// The active speech-to-text model, loaded on first use and closed with the
/// screen.
final class _Speech {
  SpeechRecognizer? _recognizer;

  Future<String> transcribe(Uint8List pcm, KnowledgeSttInfo stt) async {
    final recognizer = _recognizer ??= await FlutterEdgeAi.getActiveStt();
    final pieces = splitForSpeechWindow(pcm, window: stt.window);
    final text = StringBuffer();
    for (final piece in pieces) {
      final words = (await recognizer.transcribe(piece)).trim();
      if (words.isEmpty) continue;
      if (text.isNotEmpty) text.write(' ');
      text.write(words);
    }
    return text.toString();
  }

  Future<void> close() async {
    final recognizer = _recognizer;
    _recognizer = null;
    await recognizer?.close();
  }
}

/// The active language model. It stays loaded (it is the app's singleton);
/// each answer runs in a fresh session that is closed when it ends.
final class _LlmAnswerer implements KnowledgeAnswerer {
  InferenceModelSession? _session;

  /// What the model is asked to run on: the catalog's backend, CPU on the
  /// iOS Simulator (it has no GPU for LiteRT-LM).
  Future<PreferredBackend?> _backend() async =>
      await isIosSimulator() ? PreferredBackend.cpu : _llm()?.backend;

  Future<InferenceModel> _model() async {
    final llm = _llm();
    if (llm == null) throw StateError('No language model is active.');
    return FlutterEdgeAi.getActiveModel(
      maxTokens: llm.maxTokens,
      preferredBackend: await _backend(),
      supportImage: llm.supportImage,
      maxNumImages: llm.supportImage ? 1 : null,
    );
  }

  @override
  Future<String> warmUp() async {
    final model = await _model();
    final backend = model.activeBackend?.name ?? 'webgpu';
    debugPrint(
      '[Knowledge] language model on $backend (asked for '
      '${(await _backend())?.name ?? 'the default'})',
    );
    return backend;
  }

  @override
  Stream<String> answer({
    required String systemInstruction,
    required String prompt,
    Uint8List? image,
  }) async* {
    final model = await _model();
    final session = await model.createSession(
      systemInstruction: systemInstruction,
      enableVisionModality: image != null,
      maxOutputTokens: kAnswerMaxTokens,
      // Low: answers read figures and dates off the sources.
      temperature: 0.2,
    );
    _session = session;
    try {
      await session.addQueryChunk(
        Message(text: prompt, isUser: true, imageBytes: image),
      );
      yield* session.getResponseAsync();
    } finally {
      _session = null;
      await session.close();
    }
  }

  @override
  Future<void> stop() async => _session?.stopGeneration();
}

final class _JustAudioPlayer implements KnowledgeMemoPlayer {
  final _player = AudioPlayer();

  /// Bumped by every play and stop: a play that was overtaken leaves the
  /// player to the one that overtook it.
  int _generation = 0;

  @override
  Future<void> play(String reference) async {
    final generation = ++_generation;
    if (assetKeyOf(reference) case final key?) {
      await _player.setAsset(key);
    } else {
      await _player.setFilePath(reference);
    }
    if (generation != _generation) return;
    // Completes when the memo has played to its end or was stopped.
    await _player.play();
    if (generation == _generation) await _player.stop();
  }

  @override
  Future<void> stop() {
    _generation++;
    return _player.stop();
  }

  @override
  Future<void> dispose() => _player.dispose();
}

final class _Recorder implements KnowledgeRecorder {
  final _recorder = AudioRecorder();

  @override
  Future<void> start() async {
    // Asks for the microphone where the platform asks (Android, iOS, macOS).
    if (!await _recorder.hasPermission()) {
      throw const KnowledgeRecorderException(
        'Recording a memo needs the microphone: allow it in Settings.',
      );
    }
    await _recorder.start(
      const RecordConfig(
        encoder: AudioEncoder.wav,
        sampleRate: 16000,
        numChannels: 1,
      ),
      path: await temporaryRecordingPath(),
    );
  }

  @override
  Future<Uint8List> stop() async {
    final path = await _recorder.stop();
    if (path == null) throw StateError('The recorder returned no file.');
    return readAndDeleteRecording(path);
  }

  @override
  Future<void> cancel() => _recorder.cancel();

  @override
  Future<void> dispose() => _recorder.dispose();
}
