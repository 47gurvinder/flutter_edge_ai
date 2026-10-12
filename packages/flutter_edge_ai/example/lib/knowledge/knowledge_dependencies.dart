import 'dart:typed_data';

import 'package:flutter/widgets.dart' show ImageProvider;
import 'package:flutter_edge_ai/flutter_edge_ai.dart' show PreferredBackend;
import 'package:flutter_edge_ai_rag/flutter_edge_ai_rag.dart';

import '../image_search/photo_preparation.dart';
import 'knowledge_index.dart';
import 'knowledge_item.dart';

/// The active embedding model, as the Embedding Models list left it.
final class KnowledgeEmbedderInfo {
  const KnowledgeEmbedderInfo({
    required this.name,
    required this.textProfileId,
    required this.kinds,
    this.backend,
  });

  final String name;

  /// Its catalog TEXT profile id; image and audio indexes derive theirs.
  final String textProfileId;

  /// What it can index on this platform: notes always, photos with a vision
  /// encoder, voice memos with an audio encoder (not on the web).
  final Set<KnowledgeKind> kinds;

  /// npu for a per-SoC bundle, null (CPU natively) for every other.
  final PreferredBackend? backend;
}

/// The active speech-to-text model, as the Speech-to-Text list left it.
final class KnowledgeSttInfo {
  const KnowledgeSttInfo({required this.name, required this.window});

  final String name;

  /// How much audio it hears at once; longer memos are transcribed in
  /// pieces.
  final Duration window;
}

/// The active language model, as the Inference Models list left it.
final class KnowledgeLlmInfo {
  const KnowledgeLlmInfo({required this.name, required this.seesImages});

  final String name;

  /// Whether a retrieved photo is sent along with the question.
  final bool seesImages;
}

/// What is installed and active. A null part is missing: the screen says
/// which model list installs it.
final class KnowledgeSetup {
  const KnowledgeSetup({
    this.embedder,
    this.embedderProblem,
    this.stt,
    this.llm,
    required this.audioSupported,
    this.preloadLlm = true,
  });

  final KnowledgeEmbedderInfo? embedder;

  /// Why there is no usable embedder, ready to show; null when there is one.
  final String? embedderProblem;
  final KnowledgeSttInfo? stt;
  final KnowledgeLlmInfo? llm;

  /// Whether this platform records and plays voice memos (not the web).
  final bool audioSupported;

  /// Whether the language model may load in the background while the user
  /// searches: not on the web, where it and the vision encoder take turns.
  final bool preloadLlm;
}

/// The language model behind Ask.
abstract interface class KnowledgeAnswerer {
  /// Loads the active language model unless it is loaded; returns what it
  /// runs on, e.g. "gpu".
  Future<String> warmUp();

  /// Streams the answer to [prompt], with [image] (encoded) when one is sent.
  /// A fresh session per answer, closed when the stream ends.
  Stream<String> answer({
    required String systemInstruction,
    required String prompt,
    Uint8List? image,
  });

  /// Stops the answer being generated.
  Future<void> stop();
}

/// Plays voice memos.
abstract interface class KnowledgeMemoPlayer {
  /// Plays [reference]; completes when it has finished or was stopped.
  Future<void> play(String reference);

  Future<void> stop();

  Future<void> dispose();
}

/// Records voice memos: 16 kHz mono WAV.
abstract interface class KnowledgeRecorder {
  /// Starts recording; throws [KnowledgeRecorderException] when the
  /// microphone is not available or not allowed.
  Future<void> start();

  /// Stops and returns the WAV bytes.
  Future<Uint8List> stop();

  Future<void> cancel();

  Future<void> dispose();
}

final class KnowledgeRecorderException implements Exception {
  const KnowledgeRecorderException(this.message);
  final String message;

  @override
  String toString() => message;
}

/// Where photos and memos live: bundled assets (`asset:` references) and
/// copies the app kept.
abstract interface class KnowledgeMedia {
  /// The encoded bytes [reference] points to.
  Future<Uint8List> read(String reference);

  /// What a photo reference shows.
  ImageProvider imageFor(String reference);

  /// Keeps a picked photo; returns its reference.
  Future<String> keepPhoto(Uint8List bytes, {required String name});

  /// Keeps a recorded memo (WAV); returns its reference.
  Future<String> keepMemo(Uint8List wav);

  /// Deletes a copy the app kept (never an asset).
  Future<void> forget(String reference);
}

/// Saves the library (user items, transcripts, what is indexed).
abstract interface class KnowledgeLibraryStore {
  Future<String?> read();
  Future<void> write(String json);
}

/// Everything the Knowledge screen needs from outside. The app's wiring is
/// `knowledgeAppDependencies()`; tests pass fakes.
final class KnowledgeDependencies {
  const KnowledgeDependencies({
    required this.readSetup,
    required this.embedderFor,
    required this.openIndex,
    required this.library,
    required this.media,
    required this.transcribe,
    required this.answerer,
    required this.player,
    required this.recorder,
    required this.pickPhoto,
    this.preparePhoto = preparePhotoForEmbedding,
    this.beforeEmbedding = _released,
    this.beforeAnswer = _nothing,
    this.release = _nothing,
    this.now = DateTime.now,
  });

  /// What is installed and active on the model lists.
  final Future<KnowledgeSetup> Function() readSetup;

  /// The RAG adapter over the active embedder for one kind's index.
  final RagEmbedder Function(KnowledgeKind kind, KnowledgeEmbedderInfo info)
  embedderFor;

  /// Opens one kind's index for the active embedder.
  final Future<RagIndex> Function(
    KnowledgeKind kind,
    RagEmbedder embedder,
    KnowledgeEmbedderInfo info,
  )
  openIndex;

  final KnowledgeLibraryStore library;
  final KnowledgeMedia media;

  /// Transcribes 16 kHz mono PCM16 with the active speech-to-text model, in
  /// pieces of its window.
  final Future<String> Function(Uint8List pcm16kMono, KnowledgeSttInfo stt)
  transcribe;

  final KnowledgeAnswerer answerer;
  final KnowledgeMemoPlayer player;
  final KnowledgeRecorder recorder;

  /// A photo from the library; null when the user cancelled.
  final Future<({String name, Uint8List bytes})?> Function() pickPhoto;

  /// Upright, at most 1024 px, as a BMP (what the model gets).
  final Future<PreparedPhoto> Function(Uint8List encoded) preparePhoto;

  /// Runs before anything is embedded, and before Ask: on the web, where
  /// the vision encoder and the language model do not fit in one page's
  /// WebAssembly memory, this releases the language model. True when it
  /// did (the next question loads it again).
  final Future<bool> Function() beforeEmbedding;

  /// Runs before the language model loads: on the web it releases the
  /// embedder's vision encoder.
  final Future<void> Function() beforeAnswer;

  /// Closes what the dependencies opened (the speech recognizer).
  final Future<void> Function() release;

  final DateTime Function() now;

  /// Opens the knowledge index for [info]'s kinds.
  Future<KnowledgeIndex> openKnowledgeIndex(KnowledgeEmbedderInfo info) =>
      KnowledgeIndex.open(
        kinds: info.kinds,
        embedderFor: (kind) => embedderFor(kind, info),
        openIndex: (kind, embedder) => openIndex(kind, embedder, info),
      );
}

Future<void> _nothing() async {}

Future<bool> _released() async => false;
