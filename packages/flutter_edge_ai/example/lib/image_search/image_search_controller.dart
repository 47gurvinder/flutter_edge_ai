import 'dart:async';

import 'package:flutter/foundation.dart';
import 'package:flutter_edge_ai/flutter_edge_ai.dart';
import 'package:flutter_edge_ai_rag/flutter_edge_ai_rag.dart';

import '../models/embedding_model.dart' show imageIndexProfileId;
import 'eg2_vision_samples.dart';
import 'photo_store.dart';

/// One encoded image the user picked (or a bundled sample).
typedef PickedPhoto = ({String name, Uint8List bytes});

/// Everything the Image Search screen needs from outside: the embedder, the
/// RAG index, where photos live, and where they come from. The app's wiring is
/// `imageSearchAppDependencies()`; tests pass fakes.
final class ImageSearchDependencies {
  const ImageSearchDependencies({
    required this.hasActiveEmbedder,
    required this.textProfileId,
    required this.loadEmbedder,
    required this.ragEmbedder,
    required this.openIndex,
    required this.photos,
    required this.pickPhotos,
    required this.pickQueryPhoto,
    required this.loadSample,
    this.closeEmbedderOnDispose = true,
  });

  /// Whether an embedder is installed and active at all.
  final bool Function() hasActiveEmbedder;

  /// The active embedder's catalog profile id (its TEXT space), or null when
  /// it is not a catalog model.
  final Future<String?> Function() textProfileId;

  /// `FlutterEdgeAi.getActiveEmbedder(supportImage: true)`.
  final Future<EmbeddingModel> Function() loadEmbedder;

  /// The RAG adapter over the active embedder for an image index.
  final RagEmbedder Function(String imageProfileId) ragEmbedder;

  /// Opens (or reopens) the image index bound to [embedder]'s profile.
  final Future<RagIndex> Function(RagEmbedder embedder) openIndex;

  final PhotoStore photos;

  /// Photos to add; empty when the user cancelled.
  final Future<List<PickedPhoto>> Function() pickPhotos;

  /// A photo to search by; null when the user cancelled.
  final Future<PickedPhoto?> Function() pickQueryPhoto;

  /// The bytes of a bundled asset.
  final Future<Uint8List> Function(String assetKey) loadSample;

  /// Close the embedder when the screen goes: the vision tower holds about
  /// 385 MB once it has embedded an image.
  final bool closeEmbedderOnDispose;
}

enum ImageSearchStatus { loading, ready, failed }

/// A hit: the photo's reference, its cosine similarity, and a label.
typedef ImageSearchHit = ({String reference, double score, String label});

/// A photo indexed in this session, with what embedding it cost.
typedef IndexedPhoto = ({String reference, String label, Duration embed});

/// The Image Search screen's state and operations: text → image and image →
/// image search over a RAG image index, on the active EmbeddingGemma 2 bundle
/// with a vision encoder.
///
/// Every failure is kept in [error] (or, per photo while adding, in
/// [addFailures]) for the screen to show — never swallowed.
final class ImageSearchController extends ChangeNotifier {
  ImageSearchController(this._deps);

  final ImageSearchDependencies _deps;
  _TimedRagEmbedder? _embedder;
  RagIndex? _index;
  EmbeddingModel? _model;
  bool _disposed = false;

  ImageSearchStatus status = ImageSearchStatus.loading;

  /// What failed last, ready to show; null when nothing did.
  String? error;

  /// True when [error] is fixed by choosing another model on the list.
  bool suggestModelList = false;

  /// An operation is running.
  bool busy = false;

  /// What [busy] is doing, e.g. "Embedding 3 of 10".
  String? progress;

  PreferredBackend? activeBackend;
  Duration? loadTime;
  int? dimension;
  Set<EmbeddingModality> modalities = const {};
  String? profileId;
  int documentCount = 0;

  /// Photos indexed in this session, newest first.
  final List<IndexedPhoto> added = [];

  /// Photos that could not be added in the last batch, with the reason.
  final List<String> addFailures = [];

  List<ImageSearchHit> results = const [];

  /// What [results] answer: the query text, or the query photo's name.
  String? resultsFor;

  /// How long embedding the last query took.
  Duration? queryEmbed;

  /// The median embedding time of [added].
  Duration? get medianEmbed {
    if (added.isEmpty) return null;
    final times = [for (final photo in added) photo.embed]..sort();
    return times[times.length ~/ 2];
  }

  /// Loads the embedder with images declared and opens the image index.
  /// Called again after the user chose another model.
  Future<void> open() async {
    _set(() {
      status = ImageSearchStatus.loading;
      error = null;
      suggestModelList = false;
    });
    final previous = _index;
    _index = null;
    _embedder = null;
    try {
      await previous?.dispose();
      if (!_deps.hasActiveEmbedder()) {
        throw const _Refusal(
          'No embedding model is active. Choose EmbeddingGemma 2 Text + '
          'Vision (440M) on the Embedding Models list, then come back.',
        );
      }
      final textId = await _deps.textProfileId();
      if (textId == null) {
        throw const _Refusal(
          'The active embedding model is not from the example catalog, so this '
          'screen cannot name its embedding space. Choose EmbeddingGemma 2 '
          'Text + Vision (440M) on the Embedding Models list.',
        );
      }
      final watch = Stopwatch()..start();
      // Throws EmbeddingLoadException when the model has no vision encoder
      // (the 270M names the 440M bundle) or is not EmbeddingGemma 2 at all.
      final model = await _deps.loadEmbedder();
      watch.stop();
      if (_disposed) {
        // The screen went while the model loaded: nobody else will close it.
        if (_deps.closeEmbedderOnDispose) await model.close();
        return;
      }
      _model = model;
      final id = imageIndexProfileId(textId);
      final embedder = _TimedRagEmbedder(_deps.ragEmbedder(id));
      final index = await _deps.openIndex(embedder);
      if (_disposed) {
        await index.dispose();
        return;
      }
      _embedder = embedder;
      _index = index;
      final stats = await index.stats();
      _set(() {
        loadTime = watch.elapsed;
        activeBackend = model.activeBackend;
        dimension = model.dimension;
        modalities = model.modalities;
        profileId = id;
        documentCount = stats.documentCount;
        status = ImageSearchStatus.ready;
      });
      debugPrint(
        '[ImageSearch] ready: backend=${model.activeBackend}, load '
        '${watch.elapsedMilliseconds} ms, profile $id, '
        '${stats.documentCount} photos indexed',
      );
    } catch (e, st) {
      debugPrint('[ImageSearch] open failed: $e\n$st');
      _set(() {
        status = ImageSearchStatus.failed;
        error = _describe(e);
        suggestModelList =
            e is _Refusal || e is EmbeddingLoadException || e is StateError;
      });
    }
  }

  /// Adds the ten bundled scenes (replacing them if they were added before).
  Future<void> addSamples() => _run('Adding samples', () async {
    final samples = <({String id, String reference, PickedPhoto photo})>[
      for (final scene in eg2VisionSampleCaptions.keys)
        (
          id: 'sample-$scene',
          reference: assetPhotoReference(eg2VisionSampleAsset(scene)),
          photo: (
            name: '$scene.png',
            bytes: await _deps.loadSample(eg2VisionSampleAsset(scene)),
          ),
        ),
    ];
    await _add(samples);
  });

  /// Lets the user pick photos and adds each.
  Future<void> addPhotos() => _run('Adding photos', () async {
    final picked = await _deps.pickPhotos();
    if (picked.isEmpty) return;
    final stamp = DateTime.now().microsecondsSinceEpoch;
    final photos = <({String id, String reference, PickedPhoto photo})>[];
    for (var i = 0; i < picked.length; i++) {
      final photo = picked[i];
      photos.add((
        id: 'photo-$stamp-$i',
        reference: await _deps.photos.keep(photo.bytes, name: photo.name),
        photo: photo,
      ));
    }
    await _add(photos);
  });

  Future<void> _add(
    List<({String id, String reference, PickedPhoto photo})> photos,
  ) async {
    final index = _requireIndex();
    final embedder = _embedder!;
    addFailures.clear();
    for (var i = 0; i < photos.length; i++) {
      final (:id, :reference, :photo) = photos[i];
      _set(() => progress = 'Embedding ${i + 1} of ${photos.length}');
      try {
        await index.add(
          id: id,
          document: EmbeddingInput.image(photo.bytes),
          content: reference,
        );
        added.insert(0, (
          reference: reference,
          label: photo.name,
          embed: embedder.last ?? Duration.zero,
        ));
      } on EmbeddingException catch (e) {
        // One photo the model cannot take (HEIC, WebP, a broken file) does
        // not stop the others; the reason is shown next to its name.
        debugPrint('[ImageSearch] ${photo.name} not added: $e');
        addFailures.add('${photo.name}: ${e.message}');
        await _deps.photos.forget(reference);
      }
    }
    await index.flush();
    documentCount = (await index.stats()).documentCount;
  }

  /// Text → image: the photos that best match [query].
  Future<void> searchText(String query) {
    final text = query.trim();
    if (text.isEmpty) return Future.value();
    return _run('Searching', () async {
      final hits = await _requireIndex().search(
        query: EmbeddingInput.text(text),
        topK: 10,
      );
      _show(hits, '"$text"');
    });
  }

  /// Image → image: lets the user pick a photo and finds the closest ones.
  Future<void> searchByPhoto() => _run('Searching', () async {
    final photo = await _deps.pickQueryPhoto();
    if (photo == null) return;
    await _searchByImage(photo);
  });

  /// Image → image with [photo] as the query.
  Future<void> searchByImage(PickedPhoto photo) =>
      _run('Searching', () => _searchByImage(photo));

  Future<void> _searchByImage(PickedPhoto photo) async {
    final hits = await _requireIndex().search(
      query: EmbeddingInput.image(photo.bytes),
      topK: 10,
    );
    _show(hits, photo.name);
  }

  void _show(List<RetrievalResult> hits, String query) {
    results = [
      for (final hit in hits)
        (
          reference: hit.content,
          score: hit.similarity,
          label: photoLabel(hit.content),
        ),
    ];
    resultsFor = query;
    queryEmbed = _embedder?.last;
  }

  /// Empties the index and deletes the photos it kept.
  Future<void> clearIndex() => _run('Clearing', () async {
    final index = _requireIndex();
    await index.clear();
    await index.flush();
    await _deps.photos.forgetAll();
    added.clear();
    results = const [];
    resultsFor = null;
    documentCount = 0;
  });

  RagIndex _requireIndex() =>
      _index ?? (throw StateError('The image index is not open.'));

  Future<void> _run(String what, Future<void> Function() body) async {
    if (busy) return;
    _set(() {
      busy = true;
      progress = what;
      error = null;
    });
    try {
      await body();
    } catch (e, st) {
      debugPrint('[ImageSearch] $what failed: $e\n$st');
      error = _describe(e);
    } finally {
      _set(() {
        busy = false;
        progress = null;
      });
    }
  }

  static String _describe(Object error) => switch (error) {
    _Refusal(:final message) => message,
    EmbeddingException(:final message) => message,
    StateError(:final message) => message,
    ArgumentError(:final message) => '$message',
    _ => '$error',
  };

  void _set(void Function() change) {
    change();
    if (!_disposed) notifyListeners();
  }

  @override
  void dispose() {
    _disposed = true;
    final index = _index;
    final model = _deps.closeEmbedderOnDispose ? _model : null;
    _index = null;
    _model = null;
    // dispose() cannot await. The index waits for accepted operations before
    // it closes its store; only then is the borrowed embedder closed.
    unawaited(
      () async {
        await index?.dispose();
        await model?.close();
      }().catchError((Object e, StackTrace st) {
        debugPrint('[ImageSearch] closing failed: $e\n$st');
      }),
    );
    super.dispose();
  }
}

/// A refusal of this screen's own, with a message ready to show.
final class _Refusal implements Exception {
  const _Refusal(this.message);
  final String message;

  @override
  String toString() => message;
}

/// Times each embedding the index asks for, so the screen can show what one
/// image costs without leaving the RAG API.
final class _TimedRagEmbedder implements RagEmbedder {
  _TimedRagEmbedder(this._inner);

  final RagEmbedder _inner;

  /// How long the last embedding took.
  Duration? last;

  @override
  Set<EmbeddingModality> get modalities => _inner.modalities;

  @override
  Future<EmbeddingProfile> resolveProfile() => _inner.resolveProfile();

  @override
  Future<List<double>> embedDocument(EmbeddingInput document) =>
      _timed(() => _inner.embedDocument(document));

  @override
  Future<List<double>> embedQuery(
    EmbeddingInput query, {
    required EmbeddingModality documentModality,
  }) => _timed(
    () => _inner.embedQuery(query, documentModality: documentModality),
  );

  Future<List<double>> _timed(Future<List<double>> Function() embed) async {
    final watch = Stopwatch()..start();
    final vector = await embed();
    last = watch.elapsed;
    return vector;
  }
}
