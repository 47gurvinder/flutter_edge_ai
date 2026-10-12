import 'dart:async';

import 'package:flutter/foundation.dart';
import 'package:flutter/widgets.dart' show ImageProvider;
import 'package:flutter_edge_ai_rag/flutter_edge_ai_rag.dart';

import '../image_search/photo_preparation.dart';
import 'knowledge_audio.dart';
import 'knowledge_dependencies.dart';
import 'knowledge_index.dart';
import 'knowledge_item.dart';
import 'knowledge_library.dart';
import 'knowledge_prompt.dart';
import 'knowledge_ranking.dart';

enum KnowledgeStatus { loading, ready, failed }

/// Where the language model is.
enum LlmState { missing, idle, loading, ready, failed }

/// A search hit with its item.
typedef KnowledgeResult = ({KnowledgeItem item, RankedHit hit});

/// The longest memo recorded in the app: EmbeddingGemma 2 embeds about 40 s
/// of audio (1024 tokens at about 25 a second).
const kMaxMemo = Duration(seconds: 30);

/// How many retrieved items an answer is given.
const kAnswerSources = 4;

/// The Knowledge screen's state and commands: a knowledge base of notes,
/// photos and voice memos in three RAG indexes over one EmbeddingGemma 2
/// space (the active embedder), searched by text across all three, and
/// questions answered by the active language model from what the search
/// retrieves.
///
/// Every failure lands in [error] (or, per item, in [failureOf]) for the
/// screen to show; nothing is swallowed.
final class KnowledgeViewModel extends ChangeNotifier {
  KnowledgeViewModel(this._deps);

  final KnowledgeDependencies _deps;
  KnowledgeIndex? _index;
  KnowledgeLibrary _library = KnowledgeLibrary.seeded();
  bool _disposed = false;
  Timer? _recordingLimit;
  Future<void>? _warmUp;

  // ── Setup ────────────────────────────────────────────────────────────────

  KnowledgeStatus status = KnowledgeStatus.loading;
  KnowledgeSetup? setup;

  /// How long opening the indexes took (loading the embedder included).
  Duration? openTook;

  /// The last failure, ready to show; null when nothing failed.
  String? error;

  // ── Library and index ────────────────────────────────────────────────────

  final Map<String, String> _failures = {};
  final Set<String> _queued = {};
  String? indexingId;
  bool indexing = false;

  /// Embedding time of the last indexing run, for the summary line.
  Duration? lastRunTook;

  /// Items of the kinds this platform has (memos are not on the web).
  List<KnowledgeItem> get items => [
    for (final item in _library.items)
      if (item.kind != KnowledgeKind.memo || (setup?.audioSupported ?? false))
        item,
  ];

  List<KnowledgeItem> itemsOf(KnowledgeKind kind) => [
    for (final item in items)
      if (item.kind == kind) item,
  ];

  /// What indexing [id] cost, or null when it is not indexed.
  IndexCost? costOf(String id) => switch (setup?.embedder) {
    final embedder? => _library.indexedIn(embedder.textProfileId)[id],
    null => null,
  };

  String? failureOf(String id) => _failures[id];

  bool isQueued(String id) => _queued.contains(id);

  /// What a photo reference shows.
  ImageProvider imageFor(String reference) => _deps.media.imageFor(reference);

  /// Whether the active embedder indexes [kind] at all.
  bool indexes(KnowledgeKind kind) =>
      setup?.embedder?.kinds.contains(kind) ?? false;

  int get indexedCount => items.where((item) => costOf(item.id) != null).length;

  /// Items the active embedder could index that are not indexed yet.
  List<KnowledgeItem> get pending => [
    for (final item in items)
      if (indexes(item.kind) && costOf(item.id) == null) item,
  ];

  bool get busy => indexing || recording || transcribingMemo;

  // ── Search ───────────────────────────────────────────────────────────────

  String? resultsFor;
  List<KnowledgeResult> results = const [];
  Duration? searchTook;
  bool searching = false;

  // ── Ask ──────────────────────────────────────────────────────────────────

  LlmState llmState = LlmState.missing;
  String? llmBackend;
  Duration? llmLoadTook;

  /// The answer, as it streams.
  final ValueNotifier<String> answer = ValueNotifier('');
  String? question;
  List<KnowledgeSource> answerSources = const [];
  bool asking = false;
  Duration? retrievalTook;
  Duration? firstWordsAfter;
  Duration? answerTook;

  /// The numbers of the sources the finished answer cites.
  List<int> get cited => citedSources(answer.value);

  // ── Memos ────────────────────────────────────────────────────────────────

  /// The memo playing, or null.
  final ValueNotifier<String?> playing = ValueNotifier(null);
  bool recording = false;
  DateTime? recordingSince;
  bool transcribingMemo = false;

  // ── Commands ─────────────────────────────────────────────────────────────

  /// Reads what the model lists left active, loads the embedder and opens
  /// the indexes. Called again after the user went to a model list.
  Future<void> open() async {
    _set(() {
      status = KnowledgeStatus.loading;
      error = null;
    });
    final previous = _index;
    _index = null;
    try {
      await previous?.dispose();
      final setup = await _deps.readSetup();
      if (_disposed) return;
      this.setup = setup;
      _library = KnowledgeLibrary.decode(await _deps.library.read());
      llmState = setup.llm == null ? LlmState.missing : LlmState.idle;
      final embedder = setup.embedder;
      if (embedder == null) {
        _set(() {
          status = KnowledgeStatus.failed;
          error = setup.embedderProblem;
        });
        return;
      }
      await _beforeEmbedding();
      final watch = Stopwatch()..start();
      final index = await _deps.openKnowledgeIndex(embedder);
      if (_disposed) {
        await index.dispose();
        return;
      }
      _index = index;
      _reconcile(index, embedder);
      _set(() {
        openTook = watch.elapsed;
        status = KnowledgeStatus.ready;
      });
      debugPrint(
        '[Knowledge] ready: ${embedder.name}, ${index.counts} documents, '
        'opened in ${watch.elapsedMilliseconds} ms',
      );
      if (pending.isEmpty) _startWarmUp();
    } catch (e, st) {
      debugPrint('[Knowledge] open failed: $e\n$st');
      _set(() {
        status = KnowledgeStatus.failed;
        error = _describe(e);
      });
    }
  }

  /// Drops saved index state that the index itself contradicts: when a
  /// kind's index holds another number of documents than the library thinks
  /// (another app version, an interrupted run), that kind is indexed again —
  /// `add` replaces a document with the same id.
  void _reconcile(KnowledgeIndex index, KnowledgeEmbedderInfo embedder) {
    final indexed = _library.indexedIn(embedder.textProfileId);
    for (final kind in index.kinds) {
      final saved = indexed.keys
          .where((id) => _library.byId(id)?.kind == kind)
          .length;
      if (saved != index.counts[kind]) {
        _library.forgetIndexed(embedder.textProfileId, kind: kind);
      }
    }
  }

  /// Indexes every item that is not indexed yet: notes as text, photos as
  /// pixels, memos as sound (transcribed first when a speech-to-text model
  /// is active). One item at a time; each item's cost is kept.
  Future<void> indexLibrary() => _indexItems(pending);

  Future<void> _indexItems(List<KnowledgeItem> items) async {
    final index = _index;
    final embedder = setup?.embedder;
    if (index == null || embedder == null || indexing || items.isEmpty) {
      return;
    }
    _set(() {
      indexing = true;
      error = null;
      _queued.addAll([for (final item in items) item.id]);
      for (final item in items) {
        _failures.remove(item.id);
      }
    });
    final run = Stopwatch()..start();
    try {
      await _beforeEmbedding();
      for (final item in items) {
        if (_disposed) return;
        _set(() => indexingId = item.id);
        try {
          final cost = await _indexOne(index, item);
          _library.markIndexed(embedder.textProfileId, item.id, cost);
        } on Object catch (e, st) {
          debugPrint('[Knowledge] ${item.id} not indexed: $e\n$st');
          _failures[item.id] = _describe(e);
        } finally {
          _queued.remove(item.id);
        }
      }
      await index.flush();
      await _save();
      debugPrint('[Knowledge] ${_runSummary(items, embedder, run.elapsed)}');
    } catch (e, st) {
      debugPrint('[Knowledge] indexing failed: $e\n$st');
      error = _describe(e);
    } finally {
      _set(() {
        indexing = false;
        indexingId = null;
        _queued.clear();
        lastRunTook = run.elapsed;
      });
    }
    if (!_disposed) _startWarmUp();
  }

  /// "indexed 21 of 21 items in 6123 ms: 8 notes embed 412 ms, …", for the
  /// log.
  String _runSummary(
    List<KnowledgeItem> items,
    KnowledgeEmbedderInfo embedder,
    Duration took,
  ) {
    final costs = _library.indexedIn(embedder.textProfileId);
    int ms(Iterable<Duration> durations) =>
        durations.fold(0, (sum, d) => sum + d.inMilliseconds);
    final parts = <String>[];
    for (final kind in KnowledgeKind.values) {
      final done = [
        for (final item in items)
          if (item.kind == kind) ?costs[item.id],
      ];
      if (done.isEmpty) continue;
      final stt = ms([for (final cost in done) ?cost.transcribe]);
      parts.add(
        '${done.length} ${kind.plural.toLowerCase()} embed '
        '${ms(done.map((cost) => cost.embed))} ms'
        '${stt == 0 ? '' : ' + speech-to-text $stt ms'}',
      );
    }
    return 'indexed ${items.length - _failures.length} of ${items.length} '
        'items in ${took.inMilliseconds} ms: ${parts.join(', ')}';
  }

  Future<IndexCost> _indexOne(KnowledgeIndex index, KnowledgeItem item) async {
    switch (item) {
      case NoteItem():
        return (embed: await index.addNote(item), transcribe: null);
      case PhotoItem(:final reference):
        final prepared = await _deps.preparePhoto(
          await _deps.media.read(reference),
        );
        return (
          embed: await index.addPhoto(item, prepared.bytes),
          transcribe: null,
        );
      case MemoItem(:final reference):
        final audio = await _deps.media.read(reference);
        final transcribed = await _transcribeIfNeeded(item, audio);
        return (
          embed: await index.addMemo(transcribed.memo, audio),
          transcribe: transcribed.took,
        );
    }
  }

  /// Transcribes [memo] when it has no transcript and a speech-to-text
  /// model is active; keeps the transcript in the library.
  Future<({MemoItem memo, Duration? took})> _transcribeIfNeeded(
    MemoItem memo,
    Uint8List wav,
  ) async {
    final stt = setup?.stt;
    if (memo.transcript != null || stt == null) return (memo: memo, took: null);
    final watch = Stopwatch()..start();
    final text = await _deps.transcribe(pcm16kMonoFromWav(wav), stt);
    final updated = memo.withTranscript(text.trim());
    _library.replace(updated);
    notifyListeners();
    return (memo: updated, took: watch.elapsed);
  }

  /// Empties the indexes; the library keeps its items (Index adds them
  /// again).
  Future<void> clearIndex() async {
    final index = _index;
    final embedder = setup?.embedder;
    if (index == null || embedder == null || busy) return;
    try {
      await index.clear();
      _library.forgetIndexed(embedder.textProfileId);
      _failures.clear();
      await _save();
      _set(() {
        results = const [];
        resultsFor = null;
        lastRunTook = null;
      });
    } catch (e, st) {
      debugPrint('[Knowledge] clear failed: $e\n$st');
      _set(() => error = _describe(e));
    }
  }

  /// Text → notes, photos and voice memos, ranked together.
  Future<void> search(String query) async {
    final text = query.trim();
    final index = _index;
    if (text.isEmpty || index == null || searching) return;
    _set(() {
      searching = true;
      error = null;
      resultsFor = text;
    });
    try {
      await _beforeEmbedding();
      final watch = Stopwatch()..start();
      final hits = await index.search(text);
      watch.stop();
      _set(() {
        results = _withItems(hits);
        searchTook = watch.elapsed;
      });
      debugPrint(
        '[Knowledge] "$text": ${watch.elapsedMilliseconds} ms, '
        '${[for (final r in results) '${r.item.id} ${r.hit.lift.toStringAsFixed(3)}'].join(', ')}',
      );
    } catch (e, st) {
      debugPrint('[Knowledge] search failed: $e\n$st');
      _set(() => error = _describe(e));
    } finally {
      _set(() => searching = false);
    }
  }

  List<KnowledgeResult> _withItems(List<RankedHit> hits) => [
    for (final hit in hits)
      if (_library.byId(hit.id) case final item?) (item: item, hit: hit),
  ];

  /// Retrieves the items closest to [text] across the three kinds, then
  /// streams the language model's answer, which cites them by number. The
  /// best photo among them is sent along when the model sees images.
  Future<void> ask(String text) async {
    final q = text.trim();
    final index = _index;
    final llm = setup?.llm;
    if (q.isEmpty || index == null || asking) return;
    if (llm == null) {
      _set(() => error = 'No language model is active.');
      return;
    }
    _set(() {
      asking = true;
      error = null;
      question = q;
      answer.value = '';
      answerSources = const [];
      retrievalTook = firstWordsAfter = answerTook = null;
    });
    try {
      await _beforeEmbedding();
      final watch = Stopwatch()..start();
      final hits = await index.search(q, limit: kAnswerSources);
      retrievalTook = watch.elapsed;
      final retrieved = _withItems(hits);
      final photo = llm.seesImages
          ? retrieved.map((r) => r.item).whereType<PhotoItem>().firstOrNull
          : null;
      final sources = [
        for (final (i, result) in retrieved.indexed)
          (
            number: i + 1,
            item: result.item,
            attached: identical(result.item, photo),
          ),
      ];
      _set(() => answerSources = sources);
      Uint8List? image;
      if (photo != null) {
        image = (await _deps.preparePhoto(
          await _deps.media.read(photo.reference),
        )).bytes;
      }
      await _deps.beforeAnswer();
      await _ensureWarm();
      watch.reset();
      final stream = _deps.answerer.answer(
        systemInstruction: kKnowledgeSystemInstruction,
        prompt: buildKnowledgePrompt(q, sources),
        image: image,
      );
      await for (final chunk in stream) {
        if (_disposed) return;
        firstWordsAfter ??= watch.elapsed;
        answer.value += chunk;
      }
      answerTook = watch.elapsed;
      debugPrint(
        '[Knowledge] answered "$q" from ${sources.length} sources'
        '${image == null ? '' : ' + 1 image'}: retrieval '
        '${retrievalTook?.inMilliseconds} ms, first words after '
        '${firstWordsAfter?.inMilliseconds} ms, '
        '${answerTook?.inMilliseconds} ms in all',
      );
    } catch (e, st) {
      debugPrint('[Knowledge] ask failed: $e\n$st');
      error = _describe(e);
    } finally {
      _set(() => asking = false);
    }
  }

  Future<void> stopAnswer() async {
    try {
      await _deps.answerer.stop();
    } catch (e, st) {
      debugPrint('[Knowledge] stop failed: $e\n$st');
      _set(() => error = _describe(e));
    }
  }

  /// Loads the language model in the background, once nothing is waiting to
  /// be indexed, so the first question does not wait for it. Not on the web,
  /// where it would push the vision encoder out of memory
  /// ([KnowledgeSetup.preloadLlm]).
  void _startWarmUp() {
    if (setup?.preloadLlm != true) return;
    if (llmState != LlmState.idle && llmState != LlmState.failed) return;
    unawaited(
      _ensureWarm().catchError((Object e, StackTrace st) {
        debugPrint('[Knowledge] language model failed to load: $e\n$st');
      }),
    );
  }

  Future<void> _ensureWarm() {
    if (setup?.llm == null) {
      return Future.error(StateError('No language model is active.'));
    }
    if (llmState == LlmState.ready) return Future.value();
    return _warmUp ??= () async {
      _set(() => llmState = LlmState.loading);
      final watch = Stopwatch()..start();
      try {
        final backend = await _deps.answerer.warmUp();
        _set(() {
          llmBackend = backend;
          llmLoadTook = watch.elapsed;
          llmState = LlmState.ready;
        });
      } catch (e) {
        _set(() => llmState = LlmState.failed);
        rethrow;
      } finally {
        _warmUp = null;
      }
    }();
  }

  /// On the web the language model is released to embed (see
  /// [KnowledgeDependencies.beforeEmbedding]); it loads again on the next
  /// question.
  Future<void> _beforeEmbedding() async {
    if (await _deps.beforeEmbedding() && llmState == LlmState.ready) {
      _set(() => llmState = LlmState.idle);
    }
  }

  /// Plays [memo], or stops it when it is playing.
  Future<void> togglePlay(MemoItem memo) async {
    if (playing.value == memo.id) {
      await _deps.player.stop();
      return;
    }
    try {
      if (playing.value != null) await _deps.player.stop();
      playing.value = memo.id;
      await _deps.player.play(memo.reference);
    } catch (e, st) {
      debugPrint('[Knowledge] playback failed: $e\n$st');
      _set(() => error = _describe(e));
    } finally {
      if (!_disposed && playing.value == memo.id) playing.value = null;
    }
  }

  /// Adds a note and indexes it.
  Future<void> addNote({required String title, required String body}) async {
    final note = NoteItem(
      id: 'note-${_deps.now().microsecondsSinceEpoch}',
      title: title.trim().isEmpty ? 'Untitled note' : title.trim(),
      body: body.trim(),
    );
    await _addAndIndex(note);
  }

  /// Lets the user pick a photo, keeps a copy and indexes it.
  Future<void> addPhoto() async {
    if (busy) return;
    try {
      final picked = await _deps.pickPhoto();
      if (picked == null || _disposed) return;
      // Decoded before it is kept: a format this platform cannot decode is
      // refused here, before a copy exists.
      await _deps.preparePhoto(picked.bytes);
      final reference = await _deps.media.keepPhoto(
        picked.bytes,
        name: picked.name,
      );
      final number = itemsOf(KnowledgeKind.photo).length + 1;
      await _addAndIndex(
        PhotoItem(
          id: 'photo-${_deps.now().microsecondsSinceEpoch}',
          title: 'Photo $number',
          reference: reference,
        ),
      );
    } catch (e, st) {
      debugPrint('[Knowledge] photo not added: $e\n$st');
      _set(() => error = _describe(e));
    }
  }

  /// Starts recording a voice memo (at most [kMaxMemo]).
  Future<void> startRecording() async {
    if (busy || !(setup?.audioSupported ?? false)) return;
    try {
      await _deps.recorder.start();
      _set(() {
        recording = true;
        recordingSince = _deps.now();
        error = null;
      });
      _recordingLimit = Timer(kMaxMemo, () => unawaited(stopRecording()));
    } catch (e, st) {
      debugPrint('[Knowledge] recording failed: $e\n$st');
      _set(() => error = _describe(e));
    }
  }

  /// Stops recording, keeps the memo, transcribes it and indexes it.
  Future<void> stopRecording() async {
    if (!recording) return;
    _recordingLimit?.cancel();
    _recordingLimit = null;
    _set(() {
      recording = false;
      recordingSince = null;
      transcribingMemo = true;
    });
    try {
      final wav = await _deps.recorder.stop();
      final pcm = pcm16kMonoFromWav(wav);
      if (pcm.isEmpty) throw StateError('The recording is empty.');
      final reference = await _deps.media.keepMemo(wav);
      final number = itemsOf(KnowledgeKind.memo).length + 1;
      final memo = MemoItem(
        id: 'memo-${_deps.now().microsecondsSinceEpoch}',
        title: 'Voice memo $number',
        reference: reference,
        duration: pcmDuration(pcm),
      );
      _set(() => transcribingMemo = false);
      await _addAndIndex(memo);
    } catch (e, st) {
      debugPrint('[Knowledge] memo not added: $e\n$st');
      _set(() => error = _describe(e));
    } finally {
      if (transcribingMemo) _set(() => transcribingMemo = false);
    }
  }

  Future<void> _addAndIndex(KnowledgeItem item) async {
    _library.add(item);
    await _save();
    notifyListeners();
    if (indexes(item.kind)) await _indexItems([item]);
  }

  void clearError() => _set(() => error = null);

  Future<void> _save() => _deps.library.write(_library.encode());

  static String _describe(Object error) => switch (error) {
    PhotoDecodeException(:final message) =>
      'Could not decode that photo on this platform ($message).',
    EmbeddingException(:final message) => message,
    KnowledgeRecorderException(:final message) => message,
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
    _recordingLimit?.cancel();
    final index = _index;
    _index = null;
    // dispose() cannot await: the indexes wait for accepted operations, then
    // close their stores. The models stay loaded (they are the app's).
    unawaited(
      () async {
        if (recording) await _deps.recorder.cancel();
        await _deps.player.stop();
        if (asking) await _deps.answerer.stop();
        await index?.dispose();
        await _deps.recorder.dispose();
        await _deps.player.dispose();
        await _deps.release();
      }().catchError((Object e, StackTrace st) {
        debugPrint('[Knowledge] closing failed: $e\n$st');
      }),
    );
    answer.dispose();
    playing.dispose();
    super.dispose();
  }
}
